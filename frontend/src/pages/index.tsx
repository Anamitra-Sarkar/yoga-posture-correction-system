import { useState, useEffect, useRef } from "react";
import Head from "next/head";
import Script from "next/script";
import { 
  Activity, 
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
import { extractAnglesFromLandmarks, calculateAngle3D } from "../utils/geometry";

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

export default function Dashboard() {
  const [apiURL, setApiURL] = useState("http://localhost:8000/api");
  const [groqKey, setGroqKey] = useState("");
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
    groqApiKey: groqKey || undefined,
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
      // 1. Draw joints skeleton overlay (with premium glowing aesthetics)
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
    const drawLine = (idx1: number, idx2: number, color = "#f8fafc", width = 3) => {
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
    drawLine(11, 12, "#3b82f6", 4);
    // Left Arm
    drawLine(11, 13, "#f8fafc", 3);
    drawLine(13, 15, "#f8fafc", 3);
    // Right Arm
    drawLine(12, 14, "#f8fafc", 3);
    drawLine(14, 16, "#f8fafc", 3);
    // Hips
    drawLine(23, 24, "#3b82f6", 4);
    drawLine(11, 23, "#f8fafc", 3);
    drawLine(12, 24, "#f8fafc", 3);
    // Left Leg
    drawLine(23, 25, "#f8fafc", 3);
    // Color code left knee connection if alert triggered
    const isKneeDeviating = Math.abs(currentKneeAngle - 90) > 15 && activePose === "warrior_2";
    drawLine(25, 27, isKneeDeviating ? "#ef4444" : "#f8fafc", isKneeDeviating ? 5 : 3);
    // Right Leg
    drawLine(24, 26, "#f8fafc", 3);
    drawLine(26, 28, "#f8fafc", 3);

    // Draw joints glow and circle
    landmarks.forEach((pt: any, i: number) => {
      if (pt.visibility > 0.5 && [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i)) {
        ctx.beginPath();
        ctx.arc(pt.x * 640, pt.y * 480, i === 25 && isKneeDeviating ? 8 : 5, 0, 2 * Math.PI);
        ctx.fillStyle = i === 25 && isKneeDeviating ? "#ef4444" : "#10b981";
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
    <div style={{ minHeight: "100vh" }}>
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
          <Activity className="status-dot active" style={{ color: "#10b981" }} />
          <span>SmartYoga.AI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "15px" }}>
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
        <section style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Mode Switcher */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Compass size={20} style={{ color: "#3b82f6" }} />
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
                <Sliders size={20} style={{ color: "#3b82f6" }} />
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

              <div className="input-group" style={{ marginTop: "15px" }}>
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
              <ShieldCheck size={20} style={{ color: "#10b981" }} />
              <span>Digital Twin Limits Calibration</span>
            </h2>
            <div className="input-group">
              <label className="input-label">Left Knee Min/Max Limits</label>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibKneeMin}
                  onChange={(e) => setCalibKneeMin(Number(e.target.value))}
                />
                <span className="text-muted">to</span>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibKneeMax}
                  onChange={(e) => setCalibKneeMax(Number(e.target.value))}
                />
              </div>
            </div>

            <div className="input-group">
              <label className="input-label">Left Shoulder Min/Max Limits</label>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibShoulderMin}
                  onChange={(e) => setCalibShoulderMin(Number(e.target.value))}
                />
                <span className="text-muted">to</span>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibShoulderMax}
                  onChange={(e) => setCalibShoulderMax(Number(e.target.value))}
                />
              </div>
            </div>
          </div>

          {/* Configuration Settings */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Settings size={20} style={{ color: "#94a3b8" }} />
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
              <label className="input-label">Groq API Key (LLM Guidance)</label>
              <input 
                type="password" 
                className="groq-config-input" 
                value={groqKey} 
                onChange={(e) => setGroqKey(e.target.value)}
                placeholder="gsk_..." 
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
        <section style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Real-time Video Render block */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 className="section-title" style={{ marginBottom: 0 }}>
                <CameraIcon size={20} style={{ color: "#10b981" }} />
                <span>Camera Stream</span>
              </h2>

              {appMode === "live" ? (
                <button 
                  className={`btn-primary ${cameraActive ? "btn-error" : ""}`}
                  onClick={cameraActive ? stopCamera : startCamera}
                  disabled={!mediaPipeLoaded || isInitializingCamera}
                  style={{ 
                    padding: "8px 18px", 
                    fontSize: "0.85rem",
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    background: cameraActive ? "linear-gradient(135deg, #ef4444 0%, #dc2626 100%)" : undefined,
                    boxShadow: cameraActive ? "0 4px 15px rgba(239, 68, 68, 0.3)" : undefined
                  }}
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
                  className="btn-primary" 
                  onClick={triggerSimulationStep}
                  disabled={isLoading}
                  style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 18px", fontSize: "0.85rem" }}
                >
                  {isLoading ? <RefreshCw className="animate-spin" size={16} /> : <Sparkles size={16} />}
                  <span>Evaluate step</span>
                </button>
              )}
            </div>

            {/* Video + Canvas Frame setup */}
            <div style={{ position: "relative", width: "100%", height: "480px", background: "#090d16", borderRadius: "12px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.05)" }}>
              {appMode === "live" ? (
                <>
                  <video 
                    ref={videoRef} 
                    style={{ display: "none" }} 
                    width="640" 
                    height="480" 
                    playsInline 
                    muted 
                  />
                  <canvas 
                    ref={canvasRef} 
                    width="640" 
                    height="480" 
                    style={{ width: "100%", height: "100%", objectFit: "cover", transform: "scaleX(-1)" }} 
                  />
                  {!cameraActive && (
                    <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "10px", color: "var(--text-muted)" }}>
                      <VideoOff size={48} />
                      <span>Camera is inactive. Click "Start Video" to begin.</span>
                    </div>
                  )}
                </>
              ) : (
                <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", color: "var(--text-muted)", fontSize: "0.9rem" }}>
                  <span>Switching to Simulation mode... Sliders are active.</span>
                </div>
              )}
            </div>
          </div>

          {/* Real-time status Metrics */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Gauge size={20} style={{ color: "#3b82f6" }} />
              <span>Real-Time Feedback Hub</span>
            </h2>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
              <div className="gauge-container glass-panel" style={{ background: "rgba(0,0,0,0.2)" }}>
                <div className={`gauge-circle ${correctness >= 0.75 ? "success" : "warning"}`}>
                  <span className="gauge-percentage">
                    {Math.round(correctness * 100)}%
                  </span>
                </div>
                <span className="gauge-label">Correctness Score</span>
              </div>

              <div style={{ display: "flex", flexDirection: "column", gap: "12px", justifyContent: "center" }}>
                <div className="deviation-item" style={{ margin: 0 }}>
                  <span className="text-muted">Detected Pose</span>
                  <span style={{ fontWeight: 600, color: "#60a5fa" }}>
                    {activePose === "transition/unknown" ? "Transition/Unknown" : activePose.toUpperCase()}
                  </span>
                </div>
                <div className="deviation-item" style={{ margin: 0 }}>
                  <span className="text-muted">Sequence Flow</span>
                  <span style={{ fontWeight: 600 }}>
                    {flowPose === "transition/unknown" ? "Static Check" : flowPose.toUpperCase()}
                  </span>
                </div>
                <div className="deviation-item" style={{ margin: 0 }}>
                  <span className="text-muted">Occlusion Fusing</span>
                  <span style={{ fontWeight: 600, color: recoveredJoints.length > 0 ? "#f59e0b" : "#10b981" }}>
                    {recoveredJoints.length > 0 ? `Active (${recoveredJoints.length} joints)` : "Inactive"}
                  </span>
                </div>
              </div>
            </div>

            {/* Occlusion recovery logs */}
            {recoveredJoints.length > 0 && (
              <div style={{ marginTop: "16px", padding: "10px", background: "rgba(245, 158, 11, 0.05)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: "8px" }}>
                <p style={{ fontSize: "0.85rem", color: "#f59e0b", display: "flex", alignItems: "center", gap: "6px" }}>
                  <ShieldAlert size={14} />
                  <span><strong>Fusing Occluded landmarks:</strong> {recoveredJoints.join(", ")} (Coordinate mirrored dynamically from twin joint)</span>
                </p>
              </div>
            )}

            {/* Generated correction text */}
            {correctionText && (
              <div className="guidance-box">
                <Volume2 size={24} className="text-warning" style={{ color: "#f59e0b" }} />
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: 600 }}>Safety Correction Voice Guidance</span>
                  <span className="guidance-text">{correctionText}</span>
                </div>
              </div>
            )}
            
            {activePose !== "transition/unknown" && !correctionText && (
              <div className="guidance-box" style={{ background: "rgba(16, 185, 129, 0.05)", borderColor: "var(--color-success)" }}>
                <CheckCircle2 size={24} style={{ color: "var(--color-success)" }} />
                <div style={{ display: "flex", flexDirection: "column" }}>
                  <span style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: 600 }}>Pose Alignment Correct</span>
                  <span className="guidance-text" style={{ color: "#d1fae5" }}>Joint angle alignment is correct. Keep breathing steadily.</span>
                </div>
              </div>
            )}
          </div>

          {/* Real angles comparison list */}
          {activePose !== "transition/unknown" && (
            <div className="glass-panel">
              <h2 className="section-title">
                <HelpCircle size={20} style={{ color: "#ef4444" }} />
                <span>Angle Alignment Details</span>
              </h2>
              
              <div className="deviation-item">
                <span className="deviation-name">Left Knee Angle (Target: 90° for Warrior II)</span>
                <div style={{ display: "flex", alignItems: "center" }}>
                  <div className="deviation-bar-bg">
                    <div 
                      className="deviation-bar-fill" 
                      style={{ 
                        width: `${Math.min(100, Math.abs(currentKneeAngle - 90) * 1.5)}%`,
                        background: Math.abs(currentKneeAngle - 90) > 15 ? "var(--color-error)" : "var(--color-success)"
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
                <div style={{ display: "flex", alignItems: "center" }}>
                  <div className="deviation-bar-bg">
                    <div 
                      className="deviation-bar-fill" 
                      style={{ 
                        width: `${Math.min(100, Math.abs(currentShoulderAngle - 90) * 1.5)}%`,
                        background: Math.abs(currentShoulderAngle - 90) > 15 ? "var(--color-error)" : "var(--color-success)"
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
