import { useState, useRef, useEffect } from "react";
import { FrameResponse, SequenceResponse, CalibrationProfile } from "../types/yoga";
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
  const [flowPose, setFlowPose] = useState<string>("transition/unknown");
  const [flowConfidence, setFlowConfidence] = useState<number>(0.0);
  const [correctionText, setCorrectionText] = useState<string>("");
  const [recoveredJoints, setRecoveredJoints] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  
  // Buffers and timers
  const coordBuffer = useRef<number[][]>([]); // Holds 60 frames of coordinates [60, 99]
  const lastCorrectionTime = useRef<number>(0);
  const DEBOUNCE_MS = 4000; // 4 second throttle for audio guidance

  const processFrame = async (rawLandmarks: number[][], currentAngles: number[]) => {
    // rawLandmarks shape: [33, 4] -> [x, y, z, visibility]
    if (rawLandmarks.length !== 33) return;
    
    setIsLoading(true);
    try {
      // 1. Stage 4: Occlusion Handling
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
        const frameRes = await analyseFrame({ angles: currentAngles });
        currentPoseId = frameRes.pose_id;
        currentCorrectness = frameRes.correctness_score;
        activeDeviations = frameRes.deviations;
        
        setActivePose(currentPoseId);
        setCorrectness(currentCorrectness);
      } else {
        // If sequence model is confident, sync Pose ID with the sequence target
        setActivePose(flowPose);
        setCorrectness(flowConfidence);
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
      
      // 5. Stage 9 & 10: LLM Correction Generation (with debounce throttle)
      const now = Date.now();
      if (
        currentCorrectness < correctnessThreshold && 
        currentPoseId !== "transition/unknown" &&
        now - lastCorrectionTime.current > DEBOUNCE_MS
      ) {
        const corrRes = await generateCorrection({
          pose_id: currentPoseId,
          deviations: activeDeviations,
          language,
          groq_api_key: groqApiKey
        });
        setCorrectionText(corrRes.correction_text);
        lastCorrectionTime.current = now;
      } else if (currentCorrectness >= correctnessThreshold) {
        setCorrectionText(""); // Clear feedback once pose is correct
      }
      
    } catch (error) {
      console.error("Error running yoga posture pipeline:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const resetPipeline = () => {
    coordBuffer.current = [];
    setActivePose("transition/unknown");
    setCorrectness(1.0);
    setFlowPose("transition/unknown");
    setFlowConfidence(0.0);
    setCorrectionText("");
    setRecoveredJoints([]);
  };

  return {
    activePose,
    correctness,
    flowPose,
    flowConfidence,
    correctionText,
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
