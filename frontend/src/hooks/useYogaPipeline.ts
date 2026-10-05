import { useState, useRef, useEffect } from "react";
import { FrameResponse, SequenceResponse, CalibrationProfile, MotionState } from "../types/yoga";
import { analyseFrame, analyseSequence, recoverOcclusion, generateCorrection } from "../utils/api";
import { computeOrientation } from "../utils/geometry";
import { StickyLabel, Ema, EmaMap } from "../utils/stability";
import { SequenceBuffer } from "../utils/sequenceBuffer";
import { CorrectionEfficacyTracker, EfficacyRecord } from "../utils/correctionEfficacy";

interface UseYogaPipelineProps {
  language?: "en" | "hi" | "bn";
  groqApiKey?: string;
  calibrationProfile?: CalibrationProfile;
  correctnessThreshold?: number; // e.g. 0.70
  // GUIDED mode: the pose the user chose to practise. null/undefined = FREE mode (detect whatever they do).
  targetPose?: string | null;
}

export function useYogaPipeline({
  language = "en",
  groqApiKey,
  calibrationProfile,
  correctnessThreshold = 0.70,
  targetPose = null,
}: UseYogaPipelineProps = {}) {
  const [activePose, setActivePose] = useState<string>("transition/unknown");
  const [correctness, setCorrectness] = useState<number>(1.0);
  const [personalCorrectness, setPersonalCorrectness] = useState<number | null>(null);
  const [motionState, setMotionState] = useState<MotionState>("unknown");
  const [deviations, setDeviations] = useState<{ [joint: string]: number }>({});
  const [poseMismatch, setPoseMismatch] = useState<boolean>(false);
  const [guided, setGuided] = useState<FrameResponse["guided"]>(null);
  const [flowPose, setFlowPose] = useState<string>("transition/unknown");
  const [flowConfidence, setFlowConfidence] = useState<number>(0.0);
  const [correctionText, setCorrectionText] = useState<string>("");
  const [correctionIsSafe, setCorrectionIsSafe] = useState<boolean>(true);
  const [recoveredJoints, setRecoveredJoints] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionTimestamp, setPredictionTimestamp] = useState<number>(0);
  const [lastEfficacy, setLastEfficacy] = useState<EfficacyRecord | null>(null);

  // Display stabilisers. The classifier is right roughly a third to a half of
  // the time on real input, so the RAW label flips several times a second and
  // the readout looks broken even when the model is behaving exactly as
  // measured. These change only when the display updates, never what the
  // model predicts.
  const poseSticky = useRef(new StickyLabel("transition/unknown"));
  const flowSticky = useRef(new StickyLabel("transition/unknown"));
  const correctnessEma = useRef(new Ema(0.35));
  const personalEma = useRef(new Ema(0.35));
  const deviationEma = useRef(new EmaMap(0.35));

  // Closed-loop coaching: measures whether the cue we just spoke actually
  // moved the joint it targeted, and escalates when it did not.
  const efficacy = useRef(new CorrectionEfficacyTracker());

  // Buffers and timers
  // Landmarks for the ST-GCN. Filled from EVERY camera result (pushSequenceFrame), NOT from the
  // throttled API loop, and resampled to the 25 fps / 60-frame window the model was trained on.
  const seqBuffer = useRef(new SequenceBuffer());
  const lastCorrectionTime = useRef<number>(0);
  const DEBOUNCE_MS = 30000; // 30 second throttle for LLM guidance API calls
  const lastPredictionTime = useRef<number>(0);
  const PREDICTION_INTERVAL_MS = 10000; // Real-time monitoring predicts/speaks at most once every 10s

  // Rolling angle history, used purely to measure how fast the body is moving
  // so a genuine pose HOLD can be told apart from a TRANSITION between poses.
  // Kept short (~1.5s at 2fps) so it reacts quickly when the user settles.
  const angleHistory = useRef<{ t: number; angles: number[] }[]>([]);
  const MOTION_WINDOW_MS = 1500;

  /** Mean absolute angular velocity (deg/s) across the recent window. */
  const computeMotion = (angles: number[], now: number): number | undefined => {
    angleHistory.current.push({ t: now, angles });
    angleHistory.current = angleHistory.current.filter((s) => now - s.t <= MOTION_WINDOW_MS);
    if (angleHistory.current.length < 2) return undefined;
    const first = angleHistory.current[0];
    const dtSec = (now - first.t) / 1000;
    if (dtSec <= 0) return undefined;
    let total = 0;
    for (let i = 0; i < angles.length; i++) total += Math.abs(angles[i] - first.angles[i]);
    return total / angles.length / dtSec;
  };

  /** Call on every MediaPipe result (before any API throttling). */
  const pushSequenceFrame = (rawLandmarks: number[][]) => {
    seqBuffer.current.push(rawLandmarks, Date.now());
  };

  const processFrame = async (rawLandmarks: number[][], currentAngles: number[], worldAngles?: number[]) => {
    // rawLandmarks shape: [33, 4] -> [x, y, z, visibility]
    if (rawLandmarks.length !== 33) return;

    setIsLoading(true);
    try {
      // 1. Stage 4: Occlusion Handling (the sequence buffer is filled separately, per camera frame).
      const occRes = await recoverOcclusion({ mp_landmarks: rawLandmarks });
      setRecoveredJoints(occRes.occluded_joints_recovered);
      const fusedCoords = occRes.fused_landmarks; // Shape [33, 4]

      // Measure how fast the body is moving. Done on EVERY frame (not just on
      // the throttled prediction tick) so the hold/transition read stays
      // responsive, which is what makes the transition state feel immediate.
      const motion = computeMotion(currentAngles, Date.now());

      // Gate the heavier classification/LLM/speech cycle to once per PREDICTION_INTERVAL_MS
      const nowTick = Date.now();
      if (nowTick - lastPredictionTime.current < PREDICTION_INTERVAL_MS) {
        return;
      }
      lastPredictionTime.current = nowTick;

      // 2. Stage 7: Sequence Flow Analysis (ST-GCN). It needs a full 2.4 s window at the training frame rate.
      // It is a SECOND OPINION shown in the "Sequence Flow" row, never the source of the headline pose or
      // of the form score: on held-out videos it is right ~3 times in 4 on the poses it knows, while the
      // per-frame cascade names the held poses far more reliably, and a model's confidence says how sure
      // it is of a NAME, not whether the FORM is right.
      const seqWindow = seqBuffer.current.window();
      if (seqWindow) {
        const seqRes = await analyseSequence({ coordinates: seqWindow });
        const confident = !seqRes.requires_static_fallback;
        setFlowPose(confident ? seqRes.sequence_pose : "transition/unknown");
        setFlowConfidence(seqRes.confidence);
      } else {
        setFlowPose("transition/unknown");
      }

      // 3. Stage 6: per-frame classification (MLP cascade): names the pose, scores the form, returns the
      // per-joint deviations. Always runs.
      let currentPoseId = activePose;
      let currentCorrectness = correctness;
      let activeDeviations: { [jointName: string]: number } = {};

      // The calibration profile now goes to the backend, which returns BOTH a
      // universal correctness score and a personalised one, so "wrong" and
      // "just a different body" stay distinguishable.
      // Computed from the OCCLUSION-FUSED landmarks, so a briefly hidden
      // ankle doesn't throw the leg/torso ratio off and flip the pose call.
      const orientation = computeOrientation(
        fusedCoords.map((pt) => ({ x: pt[0], y: pt[1] }))) ?? undefined;

      const frameReq = {
        angles: currentAngles,
        world_angles: worldAngles,
        motion,
        calibration: calibrationProfile,
        orientation,
        target_pose: targetPose || undefined,
      };
      let currentMotionState: MotionState = "unknown";

      const frameRes = await analyseFrame(frameReq);
      const rawDevs = frameRes.calibrated_deviations ?? frameRes.deviations;
      currentPoseId = poseSticky.current.push(frameRes.pose_id);
      currentCorrectness = correctnessEma.current.push(frameRes.correctness_score);
      activeDeviations = deviationEma.current.push(rawDevs);
      currentMotionState = frameRes.motion_state ?? "unknown";

      setActivePose(currentPoseId);
      setCorrectness(currentCorrectness);
      setPersonalCorrectness(
        frameRes.personal_correctness_score == null
          ? null
          : personalEma.current.push(frameRes.personal_correctness_score));
      setMotionState(currentMotionState);
      setDeviations(activeDeviations);
      setGuided(frameRes.guided ?? null);

      // Stage 8 (User Digital Twin range filter) now runs server-side: the
      // backend receives the calibration profile and returns
      // calibrated_deviations plus a personal_correctness_score, already
      // picked up above. Doing it there means the native mobile client gets
      // the same personalisation for free instead of each client
      // reimplementing it.
      
      // 5. Target-pose reconciliation: the pose_head classifies whatever pose is
      // actually being performed, independent of what the user selected to
      // practice. Without this check, selecting one asana and performing a
      // completely different one would still show a high correctness score,
      // since that score only ever describes form quality for the DETECTED
      // pose, never whether it matches the user's chosen target.
      // FREE mode (no targetPose): the app detects whatever the user is doing and scores THAT.
      // GUIDED mode: the form score only describes the DETECTED pose, so if the user is holding a
      // different (recognised) pose than the one they chose, say so instead of praising or coaching it.
      const isMismatch = !!targetPose && currentPoseId !== "transition/unknown" && currentPoseId !== targetPose;
      setPoseMismatch(isMismatch);

      // 6. Stage 9 & 10: LLM Correction Generation (with 30s debounce throttle for API calls)
      const now = Date.now();

      // Close the loop on the PREVIOUS cue before considering a new one: did
      // the joint it targeted actually move? This must run every frame, not
      // only when a cue is due, because the response window is shorter than
      // the correction debounce.
      const verdict = efficacy.current.onFrame(activeDeviations, now);
      if (verdict) setLastEfficacy(verdict);
      if (currentMotionState === "transitioning") {
        // Don't correct alignment while the body is still moving -- it's
        // useless mid-flow and a real instructor waits for the hold. This is
        // only possible now that motion is measured separately from
        // recognition failure.
        const movingMsg = language === "hi"
          ? "प्रवाह जारी रखें… अगली मुद्रा में स्थिर होने पर मार्गदर्शन मिलेगा।"
          : language === "bn"
          ? "প্রবাহ চালিয়ে যান… পরের আসনে স্থির হলে নির্দেশনা পাবেন।"
          : "Flowing… hold your next posture and I'll guide you.";
        setCorrectionText(movingMsg);
        setCorrectionIsSafe(true);
      } else if (isMismatch) {
        // The UI layer owns the mismatch message (it has the localized pose names); no coaching for the wrong pose.
        setCorrectionText("");
      } else if (currentPoseId !== "transition/unknown") {
        if (currentCorrectness < correctnessThreshold) {
          if (now - lastCorrectionTime.current > DEBOUNCE_MS) {
            // Tell the backend how many times this exact cue has already been
            // given without the targeted joint moving, so it escalates instead
            // of repeating itself.
            const attempt = efficacy.current.attemptFor(
              currentPoseId,
              // the joint we expect to be targeted: the worst deviation
              Object.entries(activeDeviations)
                .sort((a, b) => b[1] - a[1])[0]?.[0] ?? "");
            const corrRes = await generateCorrection({
              pose_id: currentPoseId,
              deviations: activeDeviations,
              language,
              groq_api_key: groqApiKey,
              attempt,
            });
            setCorrectionText(corrRes.correction_text);
            setCorrectionIsSafe(corrRes.is_safe);
            lastCorrectionTime.current = now;
            efficacy.current.onCueDelivered(
              currentPoseId, corrRes.target_joint, activeDeviations, now);
          }
        } else {
          // Pose is correct - notify user visually in real-time
          const successMsg = language === "hi"
            ? "अंग संरेखण सही है। निरंतर सांस लेते रहें।"
            : language === "bn"
            ? "আসন ভঙ্গি সঠিক আছে। স্বাভাবিক শ্বাস-প্রশ্বাস বজায় রাখুন।"
            : "Pose alignment correct. Keep breathing steadily.";
          setCorrectionText(successMsg);
          setCorrectionIsSafe(true);
        }
      } else {
        // Held still, but the posture isn't one we can name. Say precisely
        // that, instead of the old ambiguous "align your body" -- the user IS
        // holding something; it just isn't recognised, which is a different
        // situation from being mid-flow or out of frame.
        const unknownMsg = language === "hi"
          ? "यह मुद्रा पहचानी नहीं गई। पूरा शरीर फ्रेम में रखें और स्थिर रहें।"
          : language === "bn"
          ? "এই ভঙ্গি চেনা যায়নি। পুরো শরীর ফ্রেমে রেখে স্থির থাকুন।"
          : "I can't identify this posture yet — keep your full body in frame and hold steady.";
        setCorrectionText(unknownMsg);
        setCorrectionIsSafe(true);
      }

      // Mark this prediction cycle complete — drives the 10s speech cadence
      // even when the displayed text is unchanged from the prior cycle.
      setPredictionTimestamp(nowTick);

    } catch (error: any) {
      if (error.name === "AbortError" || (typeof DOMException !== "undefined" && error instanceof DOMException && error.name === "AbortError") || error.message?.includes("aborted") || error.message?.includes("AbortError")) {
        // Silently ignore aborted requests since a newer frame was sent
        return;
      }
      console.error("Error running yoga posture pipeline:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Re-translate the standing message when the language changes. Skipped
    // while moving, so the pipeline's "flowing" message is not overwritten.
    if (motionState === "transitioning") return;
    if (poseMismatch) return;
    if (activePose !== "transition/unknown") {
      if (correctness >= correctnessThreshold) {
        const successMsg = language === "hi"
          ? "अंग संरेखण सही है। निरंतर सांस लेते रहें।"
          : language === "bn"
          ? "আসন ভঙ্গি সঠিক আছে। স্বাভাবিক শ্বাস-প্রশ্বাস বজায় রাখুন।"
          : "Pose alignment correct. Keep breathing steadily.";
        setCorrectionText(successMsg);
        setCorrectionIsSafe(true);
      }
    } else {
      const alignMsg = language === "hi"
        ? "कैमरे के साथ अपने शरीर को संरेखित करें..."
        : language === "bn"
        ? "ক্যামেরার সাথে আপনার শরীর সারিবদ্ধ করুন..."
        : "Align your body with the camera...";
      setCorrectionText(alignMsg);
      setCorrectionIsSafe(true);
    }
  }, [language, activePose, correctness, correctnessThreshold, motionState, poseMismatch]);

  const resetPipeline = () => {
    seqBuffer.current.clear();
    lastPredictionTime.current = 0;
    setActivePose("transition/unknown");
    setCorrectness(1.0);
    setFlowPose("transition/unknown");
    setFlowConfidence(0.0);
    setCorrectionText("");
    setCorrectionIsSafe(true);
    setPersonalCorrectness(null);
    setMotionState("unknown");
    setPoseMismatch(false);
    setGuided(null);
    setDeviations({});
    angleHistory.current = [];
    setRecoveredJoints([]);
  };

  return {
    activePose,
    correctness,
    flowPose,
    flowConfidence,
    correctionText,
    correctionIsSafe,
    lastEfficacy,
    efficacySummary: () => efficacy.current.summary(),
    motionState,
    poseMismatch,
    guided,
    personalCorrectness,
    deviations,
    predictionTimestamp,
    recoveredJoints,
    isLoading,
    processFrame,
    pushSequenceFrame,
    resetPipeline
  };
}
