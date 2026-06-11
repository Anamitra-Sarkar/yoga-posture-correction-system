import { useState, useEffect, useRef } from "react";
import Head from "next/head";
import Script from "next/script";
import { 
  Volume2, 
  VolumeX, 
  Settings, 
  RefreshCw, 
  ShieldCheck, 
  ShieldAlert, 
  Sliders, 
  Compass, 
  Activity,
  CheckCircle2, 
  HelpCircle,
  Sparkles,
  Camera as CameraIcon,
  VideoOff,
  PanelLeftClose,
  PanelLeftOpen,
  ChevronDown,
  Globe,
  Gauge
} from "lucide-react";
import { useYogaPipeline } from "../hooks/useYogaPipeline";
import { CalibrationProfile } from "../types/yoga";
import { extractAnglesFromLandmarks } from "../utils/geometry";

// Preset poses for the simulation mode fallback
const PRESET_POSES = {
  warrior_2: {
    name: "Warrior II (Virabhadrasana II)",
    angles: [165, 160, 90, 95, 95, 105, 120, 175, 90, 88, 85, 87, 180, 85, 88],
    landmarks: [
      [0.0, -0.6, 0.0, 0.95],   // Nose
      [0.18, -0.4, -0.05, 0.9], // Shoulder L
      [-0.18, -0.4, 0.05, 0.9], // Shoulder R
      [0.4, -0.4, -0.08, 0.9],  // Elbow L
      [-0.4, -0.4, 0.08, 0.9],  // Elbow R
      [0.6, -0.4, -0.1, 0.9],   // Wrist L
      [-0.6, -0.4, 0.1, 0.9],   // Wrist R
      [0.12, 0.1, -0.05, 0.9],  // Hip L
      [-0.12, 0.1, 0.05, 0.9],  // Hip R
      [0.35, 0.4, -0.05, 0.9],  // Knee L
      [-0.45, 0.35, 0.05, 0.9], // Knee R
      [0.35, 0.8, -0.05, 0.9],  // Ankle L
      [-0.75, 0.65, 0.05, 0.9]  // Ankle R
    ]
  },
  plank: {
    name: "Plank Pose (Phalakasana)",
    angles: [175, 177, 88, 90, 175, 176, 178, 179, 90, 90, 180, 180, 180, 0, 0],
    landmarks: [
      [-0.5, -0.2, 0.0, 0.95],
      [-0.3, -0.1, -0.05, 0.9],
      [-0.3, -0.1, 0.05, 0.9],
      [-0.3, 0.2, -0.05, 0.9],
      [-0.3, 0.2, 0.05, 0.9],
      [-0.3, 0.5, -0.05, 0.9],
      [-0.3, 0.5, 0.05, 0.9],
      [0.1, -0.12, -0.05, 0.9],
      [0.1, -0.12, 0.05, 0.9],
      [0.45, -0.15, -0.05, 0.9],
      [0.45, -0.15, 0.05, 0.9],
      [0.8, -0.18, -0.05, 0.9],
      [0.8, -0.18, 0.05, 0.9]
    ]
  },
  tree_pose: {
    name: "Tree Pose (Vrikshasana)",
    angles: [175, 175, 15, 15, 125, 175, 45, 178, 90, 90, 178, 178, 180, 45, 0],
    landmarks: [
      [0.0, -0.7, 0.0, 0.95],
      [0.15, -0.5, -0.05, 0.9],
      [-0.15, -0.5, 0.05, 0.9],
      [0.25, -0.3, -0.05, 0.9],
      [-0.25, -0.3, 0.05, 0.9],
      [0.18, -0.1, -0.05, 0.9],
      [-0.18, -0.1, 0.05, 0.9],
      [0.12, -0.05, -0.05, 0.9],
      [-0.12, -0.05, 0.05, 0.9],
      [0.28, -0.15, -0.05, 0.9],
      [-0.12, 0.3, 0.05, 0.9],
      [0.12, 0.2, -0.05, 0.9],
      [-0.12, 0.65, 0.05, 0.9]
    ]
  }
};

const POSE_TARGET_ANGLES: {
  [poseId: string]: {
    joint: string;
    label: string;
    target: number;
    tolerance: number;
  }[];
} = {
  warrior_2: [
    { joint: "knee_l", label: "Left Knee Angle", target: 90, tolerance: 15 },
    { joint: "shoulder_l", label: "Left Shoulder Angle", target: 90, tolerance: 15 },
    { joint: "knee_r", label: "Right Knee Angle", target: 180, tolerance: 15 },
    { joint: "shoulder_r", label: "Right Shoulder Angle", target: 90, tolerance: 15 }
  ],
  chair_pose: [
    { joint: "knee_l", label: "Left Knee Bend", target: 100, tolerance: 15 },
    { joint: "knee_r", label: "Right Knee Bend", target: 100, tolerance: 15 },
    { joint: "hip_l", label: "Left Hip Bend", target: 100, tolerance: 15 }
  ],
  cobra_pose: [
    { joint: "neck", label: "Neck Extension", target: 140, tolerance: 20 },
    { joint: "trunk_l", label: "Left Trunk Extension", target: 140, tolerance: 20 }
  ],
  plank: [
    { joint: "elbow_l", label: "Left Elbow Extension", target: 180, tolerance: 10 },
    { joint: "elbow_r", label: "Right Elbow Extension", target: 180, tolerance: 10 },
    { joint: "trunk_l", label: "Left Spine Line", target: 180, tolerance: 15 }
  ],
  tree_pose: [
    { joint: "knee_l", label: "Left Knee Angle (Bent)", target: 45, tolerance: 15 },
    { joint: "hip_abduct_l", label: "Left Hip Abduction", target: 45, tolerance: 15 },
    { joint: "knee_r", label: "Right Knee Angle (Standing)", target: 180, tolerance: 15 }
  ],
  mountain_pose: [
    { joint: "knee_l", label: "Left Knee Extension", target: 180, tolerance: 10 },
    { joint: "knee_r", label: "Right Knee Extension", target: 180, tolerance: 10 },
    { joint: "trunk_l", label: "Left Spine Straightness", target: 180, tolerance: 10 }
  ]
};

const FEATURE_NAMES_ORDER = [
  "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
  "hip_l", "hip_r", "knee_l", "knee_r",
  "ankle_l", "ankle_r", "trunk_l", "trunk_r",
  "neck", "hip_abduct_l", "hip_abduct_r"
];

const getPoseJoints = (poseId: string) => {
  const cleanId = poseId.toLowerCase().split('/').pop() || "";
  if (POSE_TARGET_ANGLES[cleanId]) {
    return POSE_TARGET_ANGLES[cleanId];
  }
  return [
    { joint: "knee_l", label: "Left Knee Angle", target: 90, tolerance: 15 },
    { joint: "shoulder_l", label: "Left Shoulder Angle", target: 90, tolerance: 15 }
  ];
};


interface ScoreRingProps {
  correctness: number;
}

function ScoreRing({ correctness }: ScoreRingProps) {
  const percentage = Math.round(correctness * 100);
  const [displayPercentage, setDisplayPercentage] = useState(percentage);
  
  useEffect(() => {
    let start = displayPercentage;
    const end = percentage;
    if (start === end) return;
    
    const duration = 400; // ms
    const startTime = performance.now();
    
    let animationFrameId: number;
    
    const updateCount = (currentTime: number) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function (easeOutQuad)
      const ease = progress * (2 - progress);
      const current = Math.round(start + (end - start) * ease);
      
      setDisplayPercentage(current);
      
      if (progress < 1) {
        animationFrameId = requestAnimationFrame(updateCount);
      }
    };
    
    animationFrameId = requestAnimationFrame(updateCount);
    
    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [percentage]);

  const isSuccess = correctness >= 0.75;
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="gauge-circle">
      <svg width="80" height="80" style={{ transform: "rotate(-90deg)" }}>
        {/* Track */}
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="transparent"
          stroke={isSuccess ? "var(--color-success-muted)" : "var(--color-warning-muted)"}
          strokeWidth="6"
        />
        {/* Fill */}
        <circle
          cx="40"
          cy="40"
          r={radius}
          fill="transparent"
          stroke={isSuccess ? "var(--color-success)" : "var(--color-warning)"}
          strokeWidth="6"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 600ms cubic-bezier(0.16, 1, 0.3, 1), stroke 400ms ease" }}
        />
      </svg>
      <div className={`gauge-percentage-center ${isSuccess ? "success" : "warning"}`}>
        {displayPercentage}%
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [apiURL, setApiURL] = useState("http://localhost:8000/api");
  const [lang, setLang] = useState<"en" | "hi" | "bn">("en");
  const [langDropOpen, setLangDropOpen] = useState(false);
  const [activePreset, setActivePreset] = useState<"warrior_2" | "plank" | "tree_pose">("warrior_2");
  const [speechEnabled, setSpeechEnabled] = useState(true);
  // Digital Twin Calibration States
  const [calibrationState, setCalibrationState] = useState<"idle" | "calibrating" | "complete">("idle");
  const [calibrationCountdown, setCalibrationCountdown] = useState(5);
  const [calibratedProfile, setCalibratedProfile] = useState<CalibrationProfile | null>(null);
  const calibrationDataRef = useRef<number[][]>([]);
  const calibrationTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Camera Facing Mode States
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  // Dynamic metrics computed in real-time
  const [allCurrentAngles, setAllCurrentAngles] = useState<number[]>(new Array(15).fill(180));

  // MediaPipe state
  const [mediaPipeLoaded, setMediaPipeLoaded] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [isInitializingCamera, setIsInitializingCamera] = useState(false);

  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const cameraRef = useRef<any>(null);
  const poseRef = useRef<any>(null);
  const lastSpokenText = useRef("");
  const lastApiCallTime = useRef<number>(0);
  const API_THROTTLE_MS = 500;
  const abortControllerRef = useRef<AbortController | null>(null);
  
  // Dynamic metrics computed in real-time
  const [currentKneeAngle, setCurrentKneeAngle] = useState(180);
  const [currentShoulderAngle, setCurrentShoulderAngle] = useState(0);

  // App Shell Sidebar Drawer Toggle (Mobile)
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Collapsible Groups states
  const [openGroupPose, setOpenGroupPose] = useState(true);
  const [openGroupTwin, setOpenGroupTwin] = useState(true);
  const [openGroupConfig, setOpenGroupConfig] = useState(true);

  // Session timer hook
  const [sessionSeconds, setSessionSeconds] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // PWA Install Prompt State
  const [installPrompt, setInstallPrompt] = useState<any>(null);
  const [showInstallBanner, setShowInstallBanner] = useState(false);

  // Offline status hooks
  const [isOnline, setIsOnline] = useState(true);

  useEffect(() => {
    setIsOnline(navigator.onLine);
    const goOnline = () => setIsOnline(true);
    const goOffline = () => setIsOnline(false);
    window.addEventListener('online', goOnline);
    window.addEventListener('offline', goOffline);
    return () => {
      window.removeEventListener('online', goOnline);
      window.removeEventListener('offline', goOffline);
    };
  }, []);

  useEffect(() => {
    const handler = (e: any) => {
      e.preventDefault();
      setInstallPrompt(e);
      setShowInstallBanner(true);
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => window.removeEventListener('beforeinstallprompt', handler);
  }, []);

  // Intercept window.fetch to support request cancellation (AbortController)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const originalFetch = window.fetch;
      window.fetch = async (input, init) => {
        const urlStr = typeof input === "string" ? input : (input as any).url || "";
        if (
          urlStr.includes("/analyse_frame") ||
          urlStr.includes("/analyse_sequence") ||
          urlStr.includes("/generate_correction") ||
          urlStr.includes("/occlusion_recovery")
        ) {
          if (abortControllerRef.current) {
            init = {
              ...init,
              signal: abortControllerRef.current.signal
            };
          }
        }
        try {
          return await originalFetch(input, init);
        } catch (err: any) {
          if (err.name === "AbortError") {
            // Return a promise that never resolves/rejects to silently ignore aborted requests
            return new Promise(() => {});
          }
          throw err;
        }
      };
      return () => {
        window.fetch = originalFetch;
      };
    }
  }, []);

  const handleInstall = async () => {
    if (!installPrompt) return;
    installPrompt.prompt();
    const { outcome } = await installPrompt.userChoice;
    if (outcome === 'accepted') setShowInstallBanner(false);
    setInstallPrompt(null);
  };

  useEffect(() => {
    if (cameraActive) {
      setSessionSeconds(0);
      timerRef.current = setInterval(() => {
        setSessionSeconds(s => s + 1);
      }, 1000);
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    }
    return () => { if (timerRef.current) clearInterval(timerRef.current); };
  }, [cameraActive]);

  const formatTime = (s: number) => {
    const m = Math.floor(s / 60).toString().padStart(2, '0');
    const sec = (s % 60).toString().padStart(2, '0');
    return `${m}:${sec}`;
  };

  // Detect if device has multiple cameras
  useEffect(() => {
    if (typeof window !== "undefined" && navigator.mediaDevices && navigator.mediaDevices.enumerateDevices) {
      navigator.mediaDevices.enumerateDevices().then((devices) => {
        const videoDevices = devices.filter(device => device.kind === "videoinput");
        setHasMultipleCameras(videoDevices.length > 1);
      }).catch(err => {
        console.error("Enumerate devices failed:", err);
      });
    }
  }, []);

  const calibrationStateRef = useRef<"idle" | "calibrating" | "complete">("idle");

  const updateCalibrationState = (state: "idle" | "calibrating" | "complete") => {
    setCalibrationState(state);
    calibrationStateRef.current = state;
  };

  const announceTTS = (text: string) => {
    if (speechEnabled && typeof window !== "undefined" && window.speechSynthesis) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      if (lang === "hi") {
        utterance.lang = "hi-IN";
      } else if (lang === "bn") {
        utterance.lang = "bn-IN";
      } else {
        utterance.lang = "en-US";
      }
      window.speechSynthesis.speak(utterance);
    }
  };

  const startCalibration = () => {
    if (calibrationTimerRef.current) {
      clearInterval(calibrationTimerRef.current);
    }
    calibrationDataRef.current = [];
    updateCalibrationState("calibrating");
    setCalibrationCountdown(15);
    
    // Announce start of calibration with gentle, peaceful instructions
    const text = lang === "hi" 
      ? "डिजिटल ट्विन कैलिब्रेशन शुरू हो रहा है। कृपया कैमरे के सामने खड़े हों, गहरी सांस लें, और अपनी आरामदायक मुद्रा में रहें।" 
      : lang === "bn"
      ? "ডিজিটাল টুইন ক্যালিব্রেশন শুরু হচ্ছে। অনুগ্রহ করে ক্যামেরার সামনে শান্তভাবে দাঁড়িয়ে গভীর শ্বাস নিন এবং আরামদায়ক ভঙ্গিতে থাকুন।"
      : "Starting digital twin calibration. Please stand peacefully in camera view, breathe deeply, and assume a comfortable resting posture.";
    announceTTS(text);

    let count = 15;
    calibrationTimerRef.current = setInterval(() => {
      count -= 1;
      setCalibrationCountdown(count);
      if (count <= 0) {
        if (calibrationTimerRef.current) clearInterval(calibrationTimerRef.current);
        finishCalibration();
      }
    }, 1000);
  };

  const finishCalibration = () => {
    updateCalibrationState("complete");
    const data = calibrationDataRef.current;
    
    const profile: CalibrationProfile = {};
    FEATURE_NAMES_ORDER.forEach((joint, idx) => {
      const jointAngles = data.map(frame => frame[idx]).filter(val => !isNaN(val) && val !== null);
      if (jointAngles.length === 0) {
        profile[joint] = { min: 80, max: 160, resting: 120 };
        return;
      }
      const minVal = Math.min(...jointAngles);
      const maxVal = Math.max(...jointAngles);
      const avgVal = jointAngles.reduce((a, b) => a + b, 0) / jointAngles.length;
      
      // Pad comfort boundaries by 15 degrees to establish comfort zones
      profile[joint] = {
        min: Math.max(0, Math.round(minVal - 15)),
        max: Math.min(180, Math.round(maxVal + 15)),
        resting: Math.round(avgVal)
      };
    });
    
    setCalibratedProfile(profile);

    const completeText = lang === "hi" 
      ? "डिजिटल ट्विन कैलिब्रेशन पूरा हो गया है। अभ्यास शुरू करें।" 
      : lang === "bn"
      ? "ডিজিটাল টুইন ক্যালিব্রেশন সম্পন্ন হয়েছে। অনুশীলন শুরু করুন।"
      : "Calibration complete. Your digital twin has been established. You can now begin practicing.";
    announceTTS(completeText);
  };

  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_YOGA_API_URL || "http://localhost:8000/api";
    setApiURL(url);
    if (typeof window !== "undefined") {
      (window as any).customApiUrl = url;
    }
  }, []);

  const {
    activePose,
    correctness,
    flowPose,
    flowConfidence,
    correctionText,
    recoveredJoints,
    isLoading,
    processFrame,
    resetPipeline
  } = useYogaPipeline({
    language: lang,
    groqApiKey: undefined,
    calibrationProfile: calibratedProfile || undefined,
    correctnessThreshold: 0.75
  });

  // Audio speech synthesis loop
  useEffect(() => {
    if (speechEnabled && correctionText && correctionText !== lastSpokenText.current) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(correctionText);
      if (lang === "hi") {
        utterance.lang = "hi-IN";
      } else if (lang === "bn") {
        utterance.lang = "bn-IN";
      } else {
        utterance.lang = "en-US";
      }
      window.speechSynthesis.speak(utterance);
      lastSpokenText.current = correctionText;
    }
  }, [correctionText, speechEnabled, lang]);


  // Handle checking scripts loading
  const handleScriptLoad = () => {
    if (typeof window !== "undefined" && (window as any).Pose && (window as any).Camera) {
      setMediaPipeLoaded(true);
    }
  };

  // Check on mount if scripts are already in window
  useEffect(() => {
    handleScriptLoad();
    if (typeof window !== "undefined") {
      if (window.innerWidth >= 768) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    }
  }, []);

  // Initialize MediaPipe Pose Model
  const initMediaPipe = () => {
    if (!mediaPipeLoaded) return;
    if (poseRef.current) return;

    const PoseClass = (window as any).Pose;
    const pose = new PoseClass({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
    });

    pose.setOptions({
      modelComplexity: 1,
      smoothLandmarks: true,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });

    pose.onResults(onPoseResults);
    poseRef.current = pose;
  };

  // Start Live Webcam Video Loop (Custom Stream Implementation for swap/facingMode support)
  const startCamera = async (mode = facingMode) => {
    if (!navigator.onLine) {
      alert("No internet connection. MediaPipe requires internet on first load.");
      return;
    }

    initMediaPipe();
    
    setIsInitializingCamera(true);
    try {
      // Clean up previous streams and request frames
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }

      const constraints = {
        video: {
          facingMode: mode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        }
      };

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      streamRef.current = stream;

      const videoElement = videoRef.current;
      if (!videoElement) return;

      videoElement.srcObject = stream;
      videoElement.setAttribute("playsinline", "true");
      await videoElement.play();

      setCameraActive(true);
      setFacingMode(mode);

      // Start custom rendering loop tick to feed MediaPipe
      const tick = async () => {
        if (!stream.active || !videoRef.current) return;
        if (videoRef.current.readyState >= 3) {
          if (poseRef.current) {
            await poseRef.current.send({ image: videoRef.current });
          }
        }
        animationFrameIdRef.current = requestAnimationFrame(tick);
      };
      animationFrameIdRef.current = requestAnimationFrame(tick);

      // Trigger automatic Calibration Sequence
      startCalibration();
      
    } catch (err) {
      console.error("Camera access failed:", err);
      alert("Failed to access camera. Please check camera permissions.");
    } finally {
      setIsInitializingCamera(false);
    }
  };

  // Stop Camera Feed & destroy Digital Twin calibration
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (calibrationTimerRef.current) {
      clearInterval(calibrationTimerRef.current);
      calibrationTimerRef.current = null;
    }
    setCameraActive(false);
    updateCalibrationState("idle");
    setCalibratedProfile(null);
    resetPipeline();
  };

  // Camera facing mode toggle switcher
  const toggleCamera = () => {
    const nextMode = facingMode === "user" ? "environment" : "user";
    startCamera(nextMode);
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // MediaPipe Result processing callback
  const onPoseResults = (results: any) => {
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    const canvasCtx = canvasElement.getContext("2d");
    if (!canvasCtx) return;

    // Draw raw camera frame onto canvas
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.poseLandmarks) {
      // 1. Draw joints skeleton overlay (with clinical palette aesthetics)
      drawSkeletonOverlay(canvasCtx, results.poseLandmarks);

      // 2. Format landmarks for API pipeline: [33 joints, [x, y, z, visibility]]
      const rawLandmarks = results.poseLandmarks.map((pt: any) => [
        pt.x, 
        pt.y, 
        pt.z, 
        pt.visibility || 0.0
      ]);

      // 3. Compute client-side angles for feature list (15 biomechanical features)
      const points = results.poseLandmarks.map((pt: any) => ({
        x: pt.x,
        y: pt.y,
        z: pt.z
      }));
      const angles = extractAnglesFromLandmarks(points);

      // Update UI real-time angle display
      setAllCurrentAngles(angles);
      setCurrentKneeAngle(Math.round(angles[6])); // Left Knee
      setCurrentShoulderAngle(Math.round(angles[2])); // Left Shoulder

      // If currently in calibration mode, buffer joint features
      if (calibrationStateRef.current === "calibrating") {
        calibrationDataRef.current.push(angles);
      }

      // 4. Overwrite custom URL in window namespace for pipeline helper
      if (typeof window !== "undefined") {
        (window as any).customApiUrl = apiURL;
      }

      // 5. Invoke 13-stage pipeline processing step with 500ms (2fps) throttling
      const now = Date.now();
      if (now - lastApiCallTime.current < API_THROTTLE_MS) {
        canvasCtx.restore();
        return; // skip calling backend — too soon since last call
      }
      lastApiCallTime.current = now;

      // Abort any pending in-flight requests before starting new ones
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      abortControllerRef.current = new AbortController();

      processFrame(rawLandmarks, angles);
    } else {
      // Clear angles when body is out of frame
      setCurrentKneeAngle(180);
      setCurrentShoulderAngle(0);
    }
    canvasCtx.restore();
  };

  // Drawing method for Canvas Overlay
  const drawSkeletonOverlay = (ctx: CanvasRenderingContext2D, landmarks: any[]) => {
    const drawLine = (idx1: number, idx2: number, color = "#edecea", width = 3) => {
      const pt1 = landmarks[idx1];
      const pt2 = landmarks[idx2];
      if (pt1 && pt2) {
        ctx.beginPath();
        ctx.moveTo(pt1.x * 640, pt1.y * 480);
        ctx.lineTo(pt2.x * 640, pt2.y * 480);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.stroke();
      }
    };

    // Connections:
    // Shoulders
    drawLine(11, 12, "#01696f", 4);
    // Left Arm
    drawLine(11, 13, "#edecea", 3);
    drawLine(13, 15, "#edecea", 3);
    // Right Arm
    drawLine(12, 14, "#edecea", 3);
    drawLine(14, 16, "#edecea", 3);
    // Hips
    drawLine(23, 24, "#01696f", 4);
    drawLine(11, 23, "#edecea", 3);
    drawLine(12, 24, "#edecea", 3);
    // Left Leg
    drawLine(23, 25, "#edecea", 3);
    // Color code left knee connection if alert triggered
    const isKneeDeviating = Math.abs(currentKneeAngle - 90) > 15 && activePose === "warrior_2";
    drawLine(25, 27, isKneeDeviating ? "#b03060" : "#edecea", isKneeDeviating ? 5 : 3);
    // Right Leg
    drawLine(24, 26, "#edecea", 3);
    drawLine(26, 28, "#edecea", 3);

    // Draw joints and circle
    landmarks.forEach((pt: any, i: number) => {
      if (pt.visibility > 0.5 && [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i)) {
        ctx.beginPath();
        ctx.arc(pt.x * 640, pt.y * 480, i === 25 && isKneeDeviating ? 8 : 5, 0, 2 * Math.PI);
        ctx.fillStyle = i === 25 && isKneeDeviating ? "#b03060" : "#437a22";
        ctx.fill();
      }
    });
  };

  return (
    <>
      {/* Load MediaPipe SDK from CDN */}
      <Script 
        src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" 
        strategy="lazyOnload"
        onLoad={handleScriptLoad}
      />
      <Script 
        src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js" 
        strategy="lazyOnload"
        onLoad={handleScriptLoad}
      />

      {/* Offline Toast Notification */}
      {!isOnline && (
        <div className="offline-toast">
          <span className="offline-dot" />
          You're offline — please check your internet connection.
        </div>
      )}

      {/* PWA Install Banner */}
      {showInstallBanner && (
        <div className="install-banner">
          <div className="install-banner-content">
            <div className="install-banner-icon">🧘</div>
            <div className="install-banner-text">
              <strong>Add AsanaAI to your home screen</strong>
              <span>Practice yoga with your coach, anytime, offline.</span>
            </div>
          </div>
          <div className="install-banner-actions">
            <button className="btn-install" onClick={handleInstall}>Install</button>
            <button className="btn-dismiss" onClick={() => setShowInstallBanner(false)}>✕</button>
          </div>
        </div>
      )}

      <div className={`app-shell ${sidebarOpen ? "sidebar-open" : "sidebar-closed"}`}>
        <Head>
          <title>AsanaAI — Smart Yoga Coach</title>
        </Head>

        <header className="app-header">
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <button 
              className="mobile-menu-btn" 
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
              title={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            >
              {sidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
            </button>
            <div className="app-logo">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="var(--color-primary)"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                className="logo-mark"
              >
                <path d="M12 2C12 2 15 7 15 11C15 15 12 22 12 22C12 22 9 15 9 11C9 7 12 2 12 2Z" />
                <path d="M12 11C15 9 20 9 21 11C22 13 19 16 12 22" />
                <path d="M12 11C9 9 4 9 3 11C2 13 5 16 12 22" />
              </svg>
              <span>AsanaAI</span>
            </div>
          </div>

          <div className="header-actions">
            <div className={`session-timer ${cameraActive ? "active" : ""}`}>
              {formatTime(sessionSeconds)}
            </div>
            <div className="status-pill">
              <div className={`status-dot ${cameraActive ? "active" : ""}`} />
              <span>{cameraActive ? "Live" : "Ready"}</span>
            </div>
            
            <div 
              className="language-selector-wrapper-custom"
              onBlur={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget)) {
                  setLangDropOpen(false);
                }
              }}
            >
              <button
                className="language-selector-trigger"
                onClick={() => setLangDropOpen(!langDropOpen)}
                aria-expanded={langDropOpen}
                aria-haspopup="listbox"
                aria-label="Select language"
              >
                <Globe size={14} />
                <span>{lang.toUpperCase()}</span>
                <ChevronDown size={14} className={`chevron-icon ${langDropOpen ? "open" : ""}`} />
              </button>

              {langDropOpen && (
                <div className="language-dropdown-menu" role="listbox">
                  <button 
                    role="option" 
                    aria-selected={lang === "en"} 
                    className={`language-dropdown-item ${lang === "en" ? "active" : ""}`}
                    onClick={() => {
                      setLang("en");
                      setLangDropOpen(false);
                    }}
                  >
                    EN
                  </button>
                  <button 
                    role="option" 
                    aria-selected={lang === "hi"} 
                    className={`language-dropdown-item ${lang === "hi" ? "active" : ""}`}
                    onClick={() => {
                      setLang("hi");
                      setLangDropOpen(false);
                    }}
                  >
                    HI
                  </button>
                  <button 
                    role="option" 
                    aria-selected={lang === "bn"} 
                    className={`language-dropdown-item ${lang === "bn" ? "active" : ""}`}
                    onClick={() => {
                      setLang("bn");
                      setLangDropOpen(false);
                    }}
                  >
                    BN
                  </button>
                </div>
              )}
            </div>

            <button
              className={`btn-toggle${speechEnabled ? " active" : ""}`}
              onClick={() => setSpeechEnabled(!speechEnabled)}
              aria-pressed={speechEnabled}
              title={speechEnabled ? "Mute voice guidance" : "Enable voice guidance"}
            >
              {speechEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
            </button>

            <button className="btn-primary btn-sm" onClick={resetPipeline}>
              New Session
            </button>
          </div>
        </header>

        {/* Sidebar overlay behind drawer on mobile */}
        {sidebarOpen && (
          <div className="sidebar-overlay visible" onClick={() => setSidebarOpen(false)} />
        )}

        {/* LEFT COLUMN: SIDEBAR CONTROLS */}
        <aside className={`app-sidebar ${sidebarOpen ? "mobile-open" : ""}`}>
          


          {/* Group 2: Target Pose Cards Selector */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupPose(!openGroupPose)}>
              <span>Target Pose</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupPose ? "collapsed" : ""}`} />
            </div>
            <div className={`sidebar-group-body ${!openGroupPose ? "collapsed" : ""}`}>
              <div className="pose-card-grid">
                <div 
                  className={`pose-card ${activePreset === "warrior_2" ? "active" : ""}`}
                  onClick={() => {
                    setActivePreset("warrior_2");
                    setSidebarOpen(false);
                  }}
                >
                  <span className="pose-card-icon">🧘</span>
                  <span className="pose-card-label">Warrior II</span>
                </div>
                <div 
                  className={`pose-card ${activePreset === "plank" ? "active" : ""}`}
                  onClick={() => {
                    setActivePreset("plank");
                    setSidebarOpen(false);
                  }}
                >
                  <span className="pose-card-icon">🏋️</span>
                  <span className="pose-card-label">Plank</span>
                </div>
                <div 
                  className={`pose-card ${activePreset === "tree_pose" ? "active" : ""}`}
                  onClick={() => {
                    setActivePreset("tree_pose");
                    setSidebarOpen(false);
                  }}
                >
                  <span className="pose-card-icon">🌲</span>
                  <span className="pose-card-label">Tree Pose</span>
                </div>
              </div>
            </div>
          </div>


          {/* Group 4: Digital Twin Profile */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupTwin(!openGroupTwin)}>
              <span>Digital Twin Profile</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupTwin ? "collapsed" : ""}`} />
            </div>
            <div className={`sidebar-group-body ${!openGroupTwin ? "collapsed" : ""}`}>
              {calibratedProfile ? (
                <div style={{ display: "flex", flexDirection: "column", gap: "10px", maxHeight: "200px", overflowY: "auto", paddingRight: "4px" }}>
                  <div style={{ fontSize: "11px", color: "var(--color-success)", fontWeight: "500", marginBottom: "4px" }}>
                    ✓ Active Calibration Profile
                  </div>
                  {Object.keys(calibratedProfile).map((joint) => (
                    <div key={joint} style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: "4px" }}>
                      <span style={{ color: "rgba(255,255,255,0.7)" }}>{joint.replace('_', ' ').toUpperCase()}</span>
                      <span style={{ fontWeight: "600" }}>
                        {calibratedProfile[joint].min}° – {calibratedProfile[joint].max}°
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: "12px", color: "rgba(255,255,255,0.5)", lineHeight: "1.5" }}>
                  {calibrationState === "calibrating" ? (
                    <span style={{ color: "var(--color-warning)" }}>Calibrating... recording joint ranges.</span>
                  ) : (
                    "Digital Twin is uncalibrated. Select a pose and start the camera stream to calibrate your joint ranges."
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Group 3: API Configuration */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupConfig(!openGroupConfig)}>
              <span>API Configuration</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupConfig ? "collapsed" : ""}`} />
            </div>
            <div className={`sidebar-group-body ${!openGroupConfig ? "collapsed" : ""}`}>
              <div className="input-group">
                <label className="input-label">Backend API URL</label>
                <input
                  type="text"
                  className="groq-config-input"
                  style={{ width: "100%", padding: "6px 10px", fontSize: "13px" }}
                  value={apiURL}
                  onChange={(e) => {
                    setApiURL(e.target.value);
                    if (typeof window !== "undefined") {
                      (window as any).customApiUrl = e.target.value;
                    }
                  }}
                  placeholder="http://localhost:8000/api"
                />
                <p className="text-muted" style={{ fontSize: "11px", marginTop: "4px" }}>
                  Point to localhost or your Hugging Face Space URL.
                </p>
              </div>
            </div>
          </div>

        </aside>

        {/* RIGHT COLUMN: MAIN CONTENT */}
        <main className="app-content">
          
          <div className="camera-kpi-layout">
            {/* Camera stream block */}
            <div className="camera-panel">
              <div className="camera-panel-header">
                <div className="camera-panel-title">
                  <CameraIcon size={18} />
                  <span>Camera Stream</span>
                </div>

                <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                  {cameraActive && hasMultipleCameras && (
                    <button
                      className="btn-primary btn-sm"
                      style={{ background: "var(--color-primary-muted)", border: "1px solid var(--color-primary)", display: "flex", alignItems: "center", gap: "6px" }}
                      onClick={toggleCamera}
                      disabled={isInitializingCamera}
                    >
                      <RefreshCw size={16} />
                      <span>Swap Camera</span>
                    </button>
                  )}

                  <button 
                    className={`btn-primary btn-sm ${cameraActive ? "btn-error" : ""} ${isInitializingCamera ? "btn-loading" : ""}`}
                    onClick={cameraActive ? stopCamera : () => startCamera()}
                    disabled={!mediaPipeLoaded || isInitializingCamera}
                  >
                    {isInitializingCamera ? (
                      <RefreshCw className="animate-spin" size={16} />
                    ) : cameraActive ? (
                      <>
                        <VideoOff size={16} />
                        <span>Stop Video</span>
                      </>
                    ) : (
                      <>
                        <CameraIcon size={16} />
                        <span>Start Video</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div className="camera-frame" style={{ position: "relative" }}>
                <video 
                  ref={videoRef} 
                  className="camera-video-element"
                  playsInline 
                  muted 
                />
                <canvas 
                  ref={canvasRef} 
                  width={640} 
                  height={480} 
                  className="camera-canvas"
                />
                {cameraActive && calibrationState === "calibrating" && (
                  <div style={{
                    position: "absolute",
                    inset: 0,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    background: "rgba(10, 15, 30, 0.7)",
                    backdropFilter: "blur(4px)",
                    borderRadius: "12px",
                    zIndex: 10,
                    color: "#fff",
                    textAlign: "center"
                  }}>
                    <div style={{
                      padding: "24px 32px",
                      borderRadius: "16px",
                      background: "rgba(255, 255, 255, 0.1)",
                      border: "1px solid rgba(255, 255, 255, 0.2)",
                      boxShadow: "0 8px 32px rgba(0,0,0,0.37)"
                    }}>
                      <Sparkles style={{ color: "var(--color-primary)", marginBottom: "12px", animation: "pulse-spin 3s linear infinite" }} size={40} />
                      <h3 style={{ fontSize: "20px", fontWeight: "600", margin: "0 0 4px 0" }}>Calibrating Digital Twin</h3>
                      <p style={{ fontSize: "13px", color: "rgba(255,255,255,0.7)", margin: "0 0 16px 0" }}>Stay in camera view...</p>
                      <div style={{ fontSize: "36px", fontWeight: "bold", color: "var(--color-primary)" }}>{calibrationCountdown}s</div>
                    </div>
                  </div>
                )}
                {cameraActive && (
                  <div className="camera-live-badge">
                    <div className="live-dot" />
                    <span>LIVE</span>
                  </div>
                )}
                {!cameraActive && (
                  <div className="camera-placeholder">
                    <div className="camera-empty-icon">
                      <VideoOff size={32} />
                    </div>
                    <p>Camera is inactive. Click "Start Video" to begin.</p>
                  </div>
                )}
              </div>
            </div>

            {/* KPI Cards Column */}
            <div className="kpi-column">
              {isLoading ? (
                <>
                  <div className="kpi-card skeleton">
                    <div className="kpi-label skeleton-text" />
                    <div className="kpi-value skeleton-heading" />
                    <div className="kpi-sub skeleton-text" />
                  </div>
                  <div className="kpi-card skeleton">
                    <div className="kpi-label skeleton-text" />
                    <div className="kpi-value skeleton-heading" />
                    <div className="kpi-sub skeleton-text" />
                  </div>
                  <div className="kpi-card skeleton">
                    <div className="kpi-label skeleton-text" />
                    <div className="kpi-value skeleton-heading" />
                    <div className="kpi-sub skeleton-text" />
                  </div>
                </>
              ) : (
                <>
                  <div className="kpi-card">
                    <span className="kpi-label">Posture Score</span>
                    <span className="kpi-value">{Math.round(correctness * 100)}%</span>
                    <span className="kpi-sub">{correctness >= 0.75 ? "✓ On target" : "Needs adjustment"}</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Detected Pose</span>
                    <span className="kpi-value">{activePose === "transition/unknown" ? "—" : activePose.split('/').pop()?.toUpperCase()}</span>
                    <span className="kpi-sub">{flowPose === "transition/unknown" ? "Static" : "Flow mode"}</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">Fusing</span>
                    <span className="kpi-value">{recoveredJoints.length > 0 ? recoveredJoints.length : "—"}</span>
                    <span className="kpi-sub">{recoveredJoints.length > 0 ? "Occlusion active" : "All visible"}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Real-time Feedback & Guidance */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Gauge size={20} />
              <span>Real-Time Feedback Hub</span>
            </h2>

            <div className="feedback-grid">
              <div className="gauge-container glass-panel">
                <ScoreRing correctness={correctness} />
                <span className="gauge-label">Correctness Score</span>
              </div>

              <div className="metrics-list">
                <div className="deviation-item m-0">
                  <span className="deviation-name">Detected Pose</span>
                  <span className="metric-value primary">
                    {activePose === "transition/unknown" ? "Transition/Unknown" : activePose.toUpperCase()}
                  </span>
                </div>
                <div className="deviation-item m-0">
                  <span className="deviation-name">Sequence Flow</span>
                  <span className="metric-value">
                    {flowPose === "transition/unknown" ? "Static Check" : flowPose.toUpperCase()}
                  </span>
                </div>
                <div className="deviation-item m-0">
                  <span className="deviation-name">Occlusion Fusing</span>
                  <span className={`metric-value ${recoveredJoints.length > 0 ? "warning" : "success"}`}>
                    {recoveredJoints.length > 0 ? `Active (${recoveredJoints.length})` : "Inactive"}
                  </span>
                </div>
              </div>
            </div>

            {/* Occlusion recovery logs */}
            {recoveredJoints.length > 0 && (
              <div className="occlusion-alert">
                <p className="occlusion-alert-text">
                  <ShieldAlert size={14} />
                  <span><strong>Fusing Occluded landmarks:</strong> {recoveredJoints.join(", ")} (Coordinate mirrored dynamically from twin joint)</span>
                </p>
              </div>
            )}

            {/* Stable height status/guidance box */}
            {activePose === "transition/unknown" ? (
              <div className="guidance-box info" key="waiting-pose">
                <Activity size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">System Status</span>
                  <span className="guidance-text">Detecting pose... Align your body with the camera.</span>
                </div>
              </div>
            ) : correctionText ? (
              <div className="guidance-box" key={correctionText}>
                <Volume2 size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">Safety Correction Voice Guidance</span>
                  <span className="guidance-text">{correctionText}</span>
                </div>
              </div>
            ) : (
              <div className="guidance-box success" key="alignment-correct">
                <CheckCircle2 size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">Pose Alignment Correct</span>
                  <span className="guidance-text">Joint angle alignment is correct. Keep breathing steadily.</span>
                </div>
              </div>
            )}
          </div>

          {/* Angle details */}
          <div className="glass-panel">
            <h2 className="section-title">
              <HelpCircle size={20} />
              <span>Angle Alignment Details</span>
            </h2>
            
            {activePose === "transition/unknown" ? (
              <div className="angle-placeholder-text">
                <p>Assume a target yoga pose to view real-time joint angle alignments and corrections.</p>
              </div>
            ) : (
              <>
                {getPoseJoints(activePose).map(({ joint, label, target, tolerance }) => {
                  const jointIdx = FEATURE_NAMES_ORDER.indexOf(joint);
                  const currentAngle = allCurrentAngles[jointIdx] !== undefined 
                    ? Math.round(allCurrentAngles[jointIdx]) 
                    : (joint === "knee_l" ? currentKneeAngle : (joint === "shoulder_l" ? currentShoulderAngle : 180));
                  const diff = currentAngle - target;
                  const isDeviating = Math.abs(diff) > tolerance;

                  return (
                    <div className="deviation-item" key={joint}>
                      <span className="deviation-name">{label} (Target: {target}° for {activePose.split('/').pop()?.toUpperCase()})</span>
                      <div className="deviation-row-detail">
                        <div className="deviation-bar-bg">
                          <div 
                            className={`deviation-bar-fill ${isDeviating ? "error" : "success"}`}
                            style={{ 
                              width: `${Math.min(100, Math.abs(diff) * 1.5)}%` 
                            }} 
                          />
                        </div>
                        <span className={`deviation-value ${isDeviating ? "error" : "success"}`}>
                          {currentAngle}° (Diff: {diff > 0 ? "+" : ""}{diff}°)
                        </span>
                      </div>
                    </div>
                  );
                })}
              </>
            )}
          </div>

        </main>
      </div>
    </>
  );
}
