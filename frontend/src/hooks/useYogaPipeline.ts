import { useState, useRef, useEffect } from "react";
import { FrameResponse, SequenceResponse, CalibrationProfile } from "../types/yoga";
import { analyseFrame, analyseSequence, recoverOcclusion, generateCorrection } from "../utils/api";

interface UseYogaPipelineProps {
  language?: "en" | "hi" | "bn";
  groqApiKey?: string;
  calibrationProfile?: CalibrationProfile;
  correctnessThreshold?: number; // e.g. 0.70
  targetPose?: string; // user-selected practice pose id, e.g. "warrior_2"
}

export function useYogaPipeline({
  language = "en",
  groqApiKey,
  calibrationProfile,
  correctnessThreshold = 0.70,
  targetPose,
}: UseYogaPipelineProps = {}) {
  const [activePose, setActivePose] = useState<string>("transition/unknown");
  const [correctness, setCorrectness] = useState<number>(1.0);
  const [flowPose, setFlowPose] = useState<string>("transition/unknown");
  const [flowConfidence, setFlowConfidence] = useState<number>(0.0);
  const [correctionText, setCorrectionText] = useState<string>("");
  const [correctionIsSafe, setCorrectionIsSafe] = useState<boolean>(true);
  const [poseMismatch, setPoseMismatch] = useState<boolean>(false);
  const [recoveredJoints, setRecoveredJoints] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [predictionTimestamp, setPredictionTimestamp] = useState<number>(0);

  // Buffers and timers
  const coordBuffer = useRef<number[][]>([]); // Holds 60 frames of coordinates [60, 99]
  const lastCorrectionTime = useRef<number>(0);
  const DEBOUNCE_MS = 30000; // 30 second throttle for LLM guidance API calls
  const lastPredictionTime = useRef<number>(0);
  const PREDICTION_INTERVAL_MS = 10000; // Real-time monitoring predicts/speaks at most once every 10s

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
      
      if (fallbackRequired) {
        const frameRes = await analyseFrame({ angles: currentAngles, world_angles: worldAngles });
        currentPoseId = frameRes.pose_id;
        currentCorrectness = frameRes.correctness_score;
        activeDeviations = frameRes.deviations;
        
        setActivePose(currentPoseId);
        setCorrectness(currentCorrectness);
      } else {
        // If sequence model is confident, sync Pose ID with the sequence target
        currentPoseId = flowPose;
        currentCorrectness = flowConfidence;
        setActivePose(flowPose);
        setCorrectness(flowConfidence);

        // Fetch frame deviations for sequence flow to provide rich context to the correction generator
        try {
          const frameRes = await analyseFrame({ angles: currentAngles, world_angles: worldAngles });
          activeDeviations = frameRes.deviations;
        } catch (err) {
          console.error("Error fetching deviations for sequence:", err);
        }
      }
      
      // 4. Stage 8: User Digital Twin Range Filter
      if (calibrationProfile && activeDeviations) {
        Object.keys(activeDeviations).forEach((joint) => {
          if (calibrationProfile[joint]) {
            const { min, max } = calibrationProfile[joint];
            const angleVal = currentAngles[FEATURE_NAMES_ORDER.indexOf(joint)];
            // If user stays within calibrated safe limits, dismiss warning deviations
            if (angleVal >= min && angleVal <= max) {
              activeDeviations[joint] = 0.0;
            }
          }
        });
      }
      
      // 5. Target-pose reconciliation: the pose_head classifies whatever pose is
      // actually being performed, independent of what the user selected to
      // practice. Without this check, selecting one asana and performing a
      // completely different one would still show a high correctness score,
      // since that score only ever describes form quality for the DETECTED
      // pose, never whether it matches the user's chosen target.
      const isMismatch = !!targetPose && currentPoseId !== "transition/unknown" && currentPoseId !== targetPose;
      setPoseMismatch(isMismatch);

      // 6. Stage 9 & 10: LLM Correction Generation (with 30s debounce throttle for API calls)
      const now = Date.now();
      if (isMismatch) {
        // Wrong pose entirely — form-correction guidance would be meaningless here.
        // The UI layer overrides the displayed/spoken text with a mismatch message.
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
        // Transition/Unknown - prompt user to align body visually in real-time
        const alignMsg = language === "hi"
          ? "कैमरे के साथ अपने शरीर को संरेखित करें..."
          : language === "bn"
          ? "ক্যামেরার সাথে আপনার শরীর সারিবদ্ধ করুন..."
          : "Align your body with the camera...";
        setCorrectionText(alignMsg);
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
    // Mismatch messaging is owned by the UI layer (it has the pose display-name
    // lookup); don't let this reactive re-translation effect stomp on it.
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
  }, [language, activePose, correctness, correctnessThreshold, poseMismatch]);

  const resetPipeline = () => {
    coordBuffer.current = [];
    lastPredictionTime.current = 0;
    setActivePose("transition/unknown");
    setCorrectness(1.0);
    setFlowPose("transition/unknown");
    setFlowConfidence(0.0);
    setCorrectionText("");
    setCorrectionIsSafe(true);
    setPoseMismatch(false);
    setRecoveredJoints([]);
  };

  return {
    activePose,
    correctness,
    flowPose,
    flowConfidence,
    correctionText,
    correctionIsSafe,
    poseMismatch,
    predictionTimestamp,
    recoveredJoints,
    isLoading,
    processFrame,
    resetPipeline
  };
}

const FEATURE_NAMES_ORDER = [
  "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
  "hip_l", "hip_r", "knee_l", "knee_r",
  "ankle_l", "ankle_r", "trunk_l", "trunk_r",
  "neck", "hip_abduct_l", "hip_abduct_r"
];
