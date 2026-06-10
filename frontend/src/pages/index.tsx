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
  Gauge, 
  Compass, 
  CheckCircle2, 
  HelpCircle,
  Sparkles,
  Camera as CameraIcon,
  VideoOff
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

interface ScoreRingProps {
  correctness: number;
}

function ScoreRing({ correctness }: ScoreRingProps) {
  const percentage = Math.round(correctness * 100);
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
          style={{ transition: "stroke-dashoffset 0.35s" }}
        />
      </svg>
      <div className={`gauge-percentage-center ${isSuccess ? "success" : "warning"}`}>
        {percentage}%
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [apiURL, setApiURL] = useState("http://localhost:8000/api");
  const [lang, setLang] = useState<"en" | "hi" | "bn">("en");
  
  // Modes: "live" (Webcam) or "simulate" (Sliders)
  const [appMode, setAppMode] = useState<"live" | "simulate">("live");
  const [activePreset, setActivePreset] = useState<"warrior_2" | "plank" | "tree_pose">("warrior_2");
  
  // Simulated Slider Angles (for simulation mode)
  const [simKneeL, setSimKneeL] = useState(120);
  const [simShoulderL, setSimShoulderL] = useState(90);
  const [simElbowL, setSimElbowL] = useState(165);
  
  // Occlusion Simulation Settings (for simulation mode)
  const [occlusionType, setOcclusionType] = useState<"none" | "left_elbow" | "right_knee">("none");
  const [speechEnabled, setSpeechEnabled] = useState(true);
  
  // Digital Twin User Calibration Profile
  const [calibKneeMin, setCalibKneeMin] = useState(80);
  const [calibKneeMax, setCalibKneeMax] = useState(160);
  const [calibShoulderMin, setCalibShoulderMin] = useState(30);
  const [calibShoulderMax, setCalibShoulderMax] = useState(150);

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
  
  // Dynamic metrics computed in real-time
  const [currentKneeAngle, setCurrentKneeAngle] = useState(180);
  const [currentShoulderAngle, setCurrentShoulderAngle] = useState(0);

  useEffect(() => {
    if (process.env.NEXT_PUBLIC_YOGA_API_URL) {
      setApiURL(process.env.NEXT_PUBLIC_YOGA_API_URL);
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

  // Sync sliders to preset values in simulation mode
  useEffect(() => {
    if (appMode === "simulate") {
      const preset = PRESET_POSES[activePreset];
      setSimKneeL(preset.angles[6]);
      setSimShoulderL(preset.angles[2]);
      setSimElbowL(preset.angles[0]);
      setCurrentKneeAngle(preset.angles[6]);
      setCurrentShoulderAngle(preset.angles[2]);
      resetPipeline();
    }
  }, [activePreset, appMode]);

  // Handle checking scripts loading
  const handleScriptLoad = () => {
    if (typeof window !== "undefined" && (window as any).Pose && (window as any).Camera) {
      setMediaPipeLoaded(true);
    }
  };

  // Check on mount if scripts are already in window
  useEffect(() => {
    handleScriptLoad();
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
    if (appMode !== "live") return;
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

  // Cleanup camera on unmount or mode switch
  useEffect(() => {
    if (appMode === "simulate") {
      stopCamera();
    }
    return () => {
      stopCamera();
    };
  }, [appMode]);

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
      setCurrentKneeAngle(Math.round(angles[6])); // Left Knee
      setCurrentShoulderAngle(Math.round(angles[2])); // Left Shoulder

      // 4. Overwrite base URL in window namespace for pipeline helper
      if (typeof window !== "undefined") {
        (window as any).process = {
          env: { NEXT_PUBLIC_YOGA_API_URL: apiURL }
        };
      }

      // 5. Invoke 13-stage pipeline processing step
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

  // Pipeline simulation evaluation button (used in Simulation Mode)
  const triggerSimulationStep = () => {
    const preset = PRESET_POSES[activePreset];
    const angles = [...preset.angles];
    angles[6] = simKneeL;
    angles[2] = simShoulderL;
    angles[0] = simElbowL;

    setCurrentKneeAngle(simKneeL);
    setCurrentShoulderAngle(simShoulderL);

    const landmarks = preset.landmarks.map(pt => [...pt]);
    if (occlusionType === "left_elbow") {
      if (landmarks[3]) {
        landmarks[3][3] = 0.1;
        landmarks[3][0] = 0.0;
        landmarks[3][1] = 0.0;
        landmarks[3][2] = 0.0;
      }
    } else if (occlusionType === "right_knee") {
      if (landmarks[10]) {
        landmarks[10][3] = 0.1;
        landmarks[10][0] = 0.0;
        landmarks[10][1] = 0.0;
        landmarks[10][2] = 0.0;
      }
    }

    if (typeof window !== "undefined") {
      (window as any).process = {
        env: { NEXT_PUBLIC_YOGA_API_URL: apiURL }
      };
    }

    processFrame(landmarks, angles);
  };

  return (
    <div className="app-container">
      <Head>
        <title>Smart Yoga Posture Correction System</title>
      </Head>

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

      {/* Navbar Header */}
      <header className="app-header">
        <div className="app-logo">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M2 22C2 22 10 18 14 12C18 6 22 2 22 2C22 2 18 6 12 10C6 14 2 22 2 22Z" />
            <path d="M2 22L17 7" />
          </svg>
          <span>SmartYoga.AI</span>
        </div>
        <div className="header-actions">
          <div className="status-pill">
            <div className={`status-dot ${cameraActive ? "active" : ""}`} />
            <span>{appMode === "live" ? (cameraActive ? "Live Webcam" : "Cam Off") : "Simulation Mode"}</span>
          </div>
          <button 
            className="btn-toggle" 
            onClick={() => setSpeechEnabled(!speechEnabled)}
          >
            {speechEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        
        {/* LEFT COLUMN: CONTROL & SETTINGS */}
        <section className="grid-column">
          
          {/* Mode Switcher */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Compass size={20} />
              <span>Operation Mode</span>
            </h2>
            <div className="btn-row">
              <button 
                className={`btn-toggle ${appMode === "live" ? "active" : ""}`}
                onClick={() => setAppMode("live")}
              >
                Webcam Feed
              </button>
              <button 
                className={`btn-toggle ${appMode === "simulate" ? "active" : ""}`}
                onClick={() => setAppMode("simulate")}
              >
                Simulation Sliders
              </button>
            </div>
          </div>

          {/* Simulation mode configurations */}
          {appMode === "simulate" && (
            <div className="glass-panel">
              <h2 className="section-title">
                <Sliders size={20} />
                <span>Simulate Joint Angles</span>
              </h2>
              
              <div className="input-group">
                <label className="input-label">Select Base Pose Target</label>
                <select 
                  className="groq-config-input"
                  value={activePreset}
                  onChange={(e) => setActivePreset(e.target.value as any)}
                  style={{ appearance: "auto" }}
                >
                  <option value="warrior_2">Warrior II</option>
                  <option value="plank">Plank</option>
                  <option value="tree_pose">Tree Pose</option>
                </select>
              </div>

              <div className="input-group">
                <label className="input-label">Simulate Left Knee Angle</label>
                <div className="slider-container">
                  <input 
                    type="range" 
                    min="40" 
                    max="180" 
                    value={simKneeL} 
                    onChange={(e) => setSimKneeL(Number(e.target.value))}
                    className="slider-input" 
                  />
                  <span className="slider-value">{simKneeL}°</span>
                </div>
              </div>

              <div className="input-group">
                <label className="input-label">Simulate Left Shoulder Angle</label>
                <div className="slider-container">
                  <input 
                    type="range" 
                    min="0" 
                    max="180" 
                    value={simShoulderL} 
                    onChange={(e) => setSimShoulderL(Number(e.target.value))}
                    className="slider-input" 
                  />
                  <span className="slider-value">{simShoulderL}°</span>
                </div>
              </div>

              <div className="input-group">
                <label className="input-label">Simulate Left Elbow Angle</label>
                <div className="slider-container">
                  <input 
                    type="range" 
                    min="40" 
                    max="180" 
                    value={simElbowL} 
                    onChange={(e) => setSimElbowL(Number(e.target.value))}
                    className="slider-input" 
                  />
                  <span className="slider-value">{simElbowL}°</span>
                </div>
              </div>

              <div className="input-group mt-4">
                <label className="input-label">Simulate occlusion block</label>
                <div className="btn-row">
                  <button 
                    className={`btn-toggle ${occlusionType === "none" ? "active" : ""}`}
                    onClick={() => setOcclusionType("none")}
                  >
                    None
                  </button>
                  <button 
                    className={`btn-toggle ${occlusionType === "left_elbow" ? "active" : ""}`}
                    onClick={() => setOcclusionType("left_elbow")}
                  >
                    Left Elbow
                  </button>
                  <button 
                    className={`btn-toggle ${occlusionType === "right_knee" ? "active" : ""}`}
                    onClick={() => setOcclusionType("right_knee")}
                  >
                    Right Knee
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* User Digital Twin calibration profile */}
          <div className="glass-panel">
            <h2 className="section-title">
              <ShieldCheck size={20} />
              <span>Digital Twin Limits Calibration</span>
            </h2>
            <div className="input-group">
              <label className="input-label">Left Knee Min/Max Limits</label>
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
              <label className="input-label">Left Shoulder Min/Max Limits</label>
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

          {/* Configuration Settings */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Settings size={20} />
              <span>Config Gateway</span>
            </h2>
            <div className="input-group">
              <label className="input-label">FastAPI Backend Endpoint</label>
              <input 
                type="text" 
                className="groq-config-input" 
                value={apiURL} 
                onChange={(e) => setApiURL(e.target.value)}
                placeholder="http://localhost:7860/api" 
              />
            </div>

            <div className="input-group">
              <label className="input-label">Guidance Language</label>
              <select 
                className="groq-config-input"
                value={lang}
                onChange={(e) => setLang(e.target.value as any)}
                style={{ appearance: "auto" }}
              >
                <option value="en">English (default)</option>
                <option value="hi">Hindi (हिंदी)</option>
                <option value="bn">Bengali (বাংলা)</option>
              </select>
            </div>
          </div>

        </section>

        {/* RIGHT COLUMN: REAL VIDEO AND METRICS */}
        <section className="grid-column">
          
          {/* Real-time Video Render block */}
          <div className="glass-panel camera-panel">
            <div className="camera-header">
              <h2 className="section-title mb-0">
                <CameraIcon size={20} />
                <span>Camera Stream</span>
              </h2>

              {appMode === "live" ? (
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
              ) : (
                <button 
                  className="btn-primary btn-sm" 
                  onClick={triggerSimulationStep}
                  disabled={isLoading}
                >
                  {isLoading ? <RefreshCw className="animate-spin" size={16} /> : <Sparkles size={16} />}
                  <span>Evaluate step</span>
                </button>
              )}
            </div>

            {/* Video + Canvas Frame setup */}
            <div className="camera-frame">
              {appMode === "live" ? (
                <>
                  <video 
                    ref={videoRef} 
                    className="camera-video-element"
                    width="640" 
                    height="480" 
                    playsInline 
                    muted 
                  />
                  <canvas 
                    ref={canvasRef} 
                    width="640" 
                    height="480" 
                    className="camera-canvas"
                  />
                  {!cameraActive && (
                    <div className="camera-placeholder">
                      <VideoOff size={48} />
                      <span>Camera is inactive. Click "Start Video" to begin.</span>
                    </div>
                  )}
                </>
              ) : (
                <div className="camera-placeholder">
                  <span>Switching to Simulation mode... Sliders are active.</span>
                </div>
              )}
            </div>
          </div>

          {/* Real-time status Metrics */}
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

            {/* Generated correction text */}
            {correctionText && (
              <div className="guidance-box">
                <Volume2 size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">Safety Correction Voice Guidance</span>
                  <span className="guidance-text">{correctionText}</span>
                </div>
              </div>
            )}
            
            {activePose !== "transition/unknown" && !correctionText && (
              <div className="guidance-box success">
                <CheckCircle2 size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">Pose Alignment Correct</span>
                  <span className="guidance-text">Joint angle alignment is correct. Keep breathing steadily.</span>
                </div>
              </div>
            )}
          </div>

          {/* Real angles comparison list */}
          {activePose !== "transition/unknown" && (
            <div className="glass-panel">
              <h2 className="section-title">
                <HelpCircle size={20} />
                <span>Angle Alignment Details</span>
              </h2>
              
              <div className="deviation-item">
                <span className="deviation-name">Left Knee Angle (Target: 90° for Warrior II)</span>
                <div className="deviation-row-detail">
                  <div className="deviation-bar-bg">
                    <div 
                      className={`deviation-bar-fill ${Math.abs(currentKneeAngle - 90) > 15 ? "error" : "success"}`}
                      style={{ 
                        width: `${Math.min(100, Math.abs(currentKneeAngle - 90) * 1.5)}%` 
                      }} 
                    />
                  </div>
                  <span className={`deviation-value ${Math.abs(currentKneeAngle - 90) > 15 ? "error" : "success"}`}>
                    {currentKneeAngle}° (Diff: {Math.round(currentKneeAngle - 90)}°)
                  </span>
                </div>
              </div>

              <div className="deviation-item">
                <span className="deviation-name">Left Shoulder Angle (Target: 90° for Warrior II)</span>
                <div className="deviation-row-detail">
                  <div className="deviation-bar-bg">
                    <div 
                      className={`deviation-bar-fill ${Math.abs(currentShoulderAngle - 90) > 15 ? "error" : "success"}`}
                      style={{ 
                        width: `${Math.min(100, Math.abs(currentShoulderAngle - 90) * 1.5)}%` 
                      }} 
                    />
                  </div>
                  <span className={`deviation-value ${Math.abs(currentShoulderAngle - 90) > 15 ? "error" : "success"}`}>
                    {currentShoulderAngle}° (Diff: {Math.round(currentShoulderAngle - 90)}°)
                  </span>
                </div>
              </div>
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
