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
  Menu,
  X,
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
          style={{ transition: "stroke-dashoffset 600ms cubic-bezier(0.16, 1, 0.3, 1)" }}
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
  const [activePreset, setActivePreset] = useState<"warrior_2" | "plank" | "tree_pose">("warrior_2");
  const [speechEnabled, setSpeechEnabled] = useState(true);
  
  // Digital Twin User Calibration Profile
  const [calibKneeMin, setCalibKneeMin] = useState(80);
  const [calibKneeMax, setCalibKneeMax] = useState(160);
  const [calibShoulderMin, setCalibShoulderMin] = useState(30);
  const [calibShoulderMax, setCalibShoulderMax] = useState(150);

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

  useEffect(() => {
    const url = process.env.NEXT_PUBLIC_YOGA_API_URL || "http://localhost:8000/api";
    setApiURL(url);
    if (typeof window !== "undefined") {
      (window as any).customApiUrl = url;
    }
  }, []);

  const calibrationProfile: CalibrationProfile = {
    knee_l: { min: calibKneeMin, max: calibKneeMax, resting: 180 },
    shoulder_l: { min: calibShoulderMin, max: calibShoulderMax, resting: 0 }
  };

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
    calibrationProfile,
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

  // Start Live Webcam Video Loop
  const startCamera = async () => {
    if (!navigator.onLine) {
      alert("No internet connection. MediaPipe requires internet on first load.");
      return;
    }

    initMediaPipe();
    
    setIsInitializingCamera(true);
    try {
      const videoElement = videoRef.current;
      if (!videoElement) return;

      const CameraClass = (window as any).Camera;
      const camera = new CameraClass(videoElement, {
        onFrame: async () => {
          if (poseRef.current) {
            await poseRef.current.send({ image: videoElement });
          }
        },
        width: 640,
        height: 480
      });
      
      await camera.start();
      cameraRef.current = camera;
      setCameraActive(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      alert("Failed to access camera. Please check camera permissions.");
    } finally {
      setIsInitializingCamera(false);
    }
  };

  // Stop Camera Feed
  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop();
      cameraRef.current = null;
    }
    setCameraActive(false);
    resetPipeline();
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

        {/* Navbar Header */}
        <header className="app-header">
          <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
            <button className="mobile-menu-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>
              {sidebarOpen ? <X size={20} /> : <Menu size={20} />}
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
            
            <div className="language-selector-wrapper">
              <Globe size={16} />
              <select 
                value={lang} 
                onChange={(e) => setLang(e.target.value as any)}
                className="language-select-native"
              >
                <option value="en">EN</option>
                <option value="hi">HI</option>
                <option value="bn">BN</option>
              </select>
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
            {openGroupPose && (
              <div className="sidebar-group-body">
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
            )}
          </div>


          {/* Group 4: Digital Twin Limits Calibration */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupTwin(!openGroupTwin)}>
              <span>Calibration Limits</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupTwin ? "collapsed" : ""}`} />
            </div>
            {openGroupTwin && (
              <div className="sidebar-group-body">
                <div className="input-group">
                  <label className="input-label">Left Knee Limits</label>
                  <div className="limits-row">
                    <input 
                      type="number" 
                      className="groq-config-input limits-input" 
                      value={calibKneeMin}
                      onChange={(e) => setCalibKneeMin(Number(e.target.value))}
                    />
                    <span className="text-muted">to</span>
                    <input 
                      type="number" 
                      className="groq-config-input limits-input" 
                      value={calibKneeMax}
                      onChange={(e) => setCalibKneeMax(Number(e.target.value))}
                    />
                  </div>
                </div>

                <div className="input-group">
                  <label className="input-label">Left Shoulder Limits</label>
                  <div className="limits-row">
                    <input 
                      type="number" 
                      className="groq-config-input limits-input" 
                      value={calibShoulderMin}
                      onChange={(e) => setCalibShoulderMin(Number(e.target.value))}
                    />
                    <span className="text-muted">to</span>
                    <input 
                      type="number" 
                      className="groq-config-input limits-input" 
                      value={calibShoulderMax}
                      onChange={(e) => setCalibShoulderMax(Number(e.target.value))}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Group 3: API Configuration */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupConfig(!openGroupConfig)}>
              <span>API Configuration</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupConfig ? "collapsed" : ""}`} />
            </div>
            {openGroupConfig && (
              <div className="sidebar-group-body">
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
            )}
          </div>

        </aside>

        {/* RIGHT COLUMN: MAIN CONTENT */}
        <main className="app-content">
          
          {/* KPI Cards Row */}
          <div className="kpi-row">
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

          {/* Camera stream block */}
          <div className="camera-panel">
            <div className="camera-panel-header">
              <div className="camera-panel-title">
                <CameraIcon size={18} />
                <span>Camera Stream</span>
              </div>

              <button 
                className={`btn-primary btn-sm ${cameraActive ? "btn-error" : ""}`}
                onClick={cameraActive ? stopCamera : startCamera}
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

            <div className="camera-frame">
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
