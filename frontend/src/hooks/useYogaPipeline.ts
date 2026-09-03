import { useState, useRef, useEffect } from "react";
import { FrameResponse, SequenceResponse, CalibrationProfile, MotionState } from "../types/yoga";
import { analyseFrame, analyseSequence, recoverOcclusion, generateCorrection } from "../utils/api";

interface UseYogaPipelineProps {
  language?: "en" | "hi" | "bn";
  groqApiKey?: string;
  calibrationProfile?: CalibrationProfile;
  correctnessThreshold?: number; // e.g. 0.70
}

export function useYogaPipeline({
  language = "en",
  groqApiKey,
  calibrationProfile,
  correctnessThreshold = 0.70,
}: UseYogaPipelineProps = {}) {
  const [activePose, setActivePose] = useState<string>("transition/unknown");
  const [correctness, setCorrectness] = useState<number>(1.0);
  const [personalCorrectness, setPersonalCorrectness] = useState<number | null>(null);
  const [motionState, setMotionState] = useState<MotionState>("unknown");
  const [deviations, setDeviations] = useState<{ [joint: string]: number }>({});
  const [flowPose, setFlowPose] = useState<string>("transition/unknown");
  const [flowConfidence, setFlowConfidence] = useState<number>(0.0);
  const [correctionText, setCorrectionText] = useState<string>("");
  const [correctionIsSafe, setCorrectionIsSafe] = useState<boolean>(true);
  const [recoveredJoints, setRecoveredJoints] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionTimestamp, setPredictionTimestamp] = useState<number>(0);

  // Buffers and timers
  const coordBuffer = useRef<number[][]>([]); // Holds 60 frames of coordinates [60, 99]
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

  const processFrame = async (rawLandmarks: number[][], currentAngles: number[], worldAngles?: number[]) => {
    // rawLandmarks shape: [33, 4] -> [x, y, z, visibility]
    if (rawLandmarks.length !== 33) return;

    setIsLoading(true);
    try {
      // 1. Stage 4: Occlusion Handling — always kept fresh so the 60-frame
      // sequence buffer isn't stretched thin by the slower prediction cadence below.
      const occRes = await recoverOcclusion({ mp_landmarks: rawLandmarks });
      setRecoveredJoints(occRes.occluded_joints_recovered);
      const fusedCoords = occRes.fused_landmarks; // Shape [33, 4]

      // Flatten fused coordinate space [33 joints * 3 coordinates] to 99 values
      const flatFrameCoords = fusedCoords.map(pt => pt.slice(0, 3)).flat(); // Length 99

      // Update rolling sequence buffer (Stage 5)
      coordBuffer.current.push(flatFrameCoords);
      if (coordBuffer.current.length > 60) {
        coordBuffer.current.shift();
      }

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

      let fallbackRequired = true;
      
      // 2. Stage 7: Sequence Flow Analysis (requires complete 60 frame window)
      if (coordBuffer.current.length === 60) {
        const seqRes = await analyseSequence({ coordinates: coordBuffer.current });
        setFlowPose(seqRes.sequence_pose);
        setFlowConfidence(seqRes.confidence);
        fallbackRequired = seqRes.requires_static_fallback;
      }
      
      // 3. Stage 6: Static Pose Classifier (executes on fallback or alongside sequence)
      let currentPoseId = activePose;
      let currentCorrectness = correctness;
      let activeDeviations: { [jointName: string]: number } = {};
      
      // The calibration profile now goes to the backend, which returns BOTH a
      // universal correctness score and a personalised one, so "wrong" and
      // "just a different body" stay distinguishable.
      const frameReq = {
        angles: currentAngles,
        world_angles: worldAngles,
        motion,
        calibration: calibrationProfile,
      };
      let currentMotionState: MotionState = "unknown";

      if (fallbackRequired) {
        const frameRes = await analyseFrame(frameReq);
        currentPoseId = frameRes.pose_id;
        currentCorrectness = frameRes.correctness_score;
        activeDeviations = frameRes.calibrated_deviations ?? frameRes.deviations;
        currentMotionState = frameRes.motion_state ?? "unknown";

        setActivePose(currentPoseId);
        setCorrectness(currentCorrectness);
        setPersonalCorrectness(frameRes.personal_correctness_score ?? null);
        setMotionState(currentMotionState);
        setDeviations(frameRes.calibrated_deviations ?? frameRes.deviations);
      } else {
        // If sequence model is confident, sync Pose ID with the sequence target
        currentPoseId = flowPose;
        currentCorrectness = flowConfidence;
        setActivePose(flowPose);
        setCorrectness(flowConfidence);

        // Fetch frame deviations for sequence flow to provide rich context to the correction generator
        try {
          const frameRes = await analyseFrame(frameReq);
          activeDeviations = frameRes.calibrated_deviations ?? frameRes.deviations;
          currentMotionState = frameRes.motion_state ?? "unknown";
          setPersonalCorrectness(frameRes.personal_correctness_score ?? null);
          setMotionState(currentMotionState);
          setDeviations(frameRes.calibrated_deviations ?? frameRes.deviations);
        } catch (err) {
          console.error("Error fetching deviations for sequence:", err);
        }
      }
      
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
      // (No target pose any more: the app detects whatever the user is doing.)

      // 6. Stage 9 & 10: LLM Correction Generation (with 30s debounce throttle for API calls)
      const now = Date.now();
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
      } else if (currentPoseId !== "transition/unknown") {
        if (currentCorrectness < correctnessThreshold) {
          if (now - lastCorrectionTime.current > DEBOUNCE_MS) {
            const corrRes = await generateCorrection({
              pose_id: currentPoseId,
              deviations: activeDeviations,
              language,
              groq_api_key: groqApiKey
            });
            setCorrectionText(corrRes.correction_text);
            setCorrectionIsSafe(corrRes.is_safe);
            lastCorrectionTime.current = now;
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
  }, [language, activePose, correctness, correctnessThreshold, motionState]);

  const resetPipeline = () => {
    coordBuffer.current = [];
    lastPredictionTime.current = 0;
    setActivePose("transition/unknown");
    setCorrectness(1.0);
    setFlowPose("transition/unknown");
    setFlowConfidence(0.0);
    setCorrectionText("");
    setCorrectionIsSafe(true);
    setPersonalCorrectness(null);
    setMotionState("unknown");
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
    motionState,
    personalCorrectness,
    deviations,
    predictionTimestamp,
    recoveredJoints,
    isLoading,
    processFrame,
    resetPipeline
  };
}
