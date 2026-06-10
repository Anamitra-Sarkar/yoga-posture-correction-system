import { useState, useEffect, useRef } from "react";
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
  Sparkles
} from "lucide-react";
import { useYogaPipeline } from "../hooks/useYogaPipeline";
import { CalibrationProfile } from "../types/yoga";

// Predefined mock pose defaults for testing
const PRESET_POSES = {
  warrior_2: {
    name: "Warrior II (Virabhadrasana II)",
    angles: [165, 160, 90, 95, 95, 105, 120, 175, 90, 88, 85, 87, 180, 85, 88], // knee_l = 120 (incorrect, should be ~90)
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
      [0.28, -0.15, -0.05, 0.9],  // Knee L bent out
      [-0.12, 0.3, 0.05, 0.9],   // Knee R straight
      [0.12, 0.2, -0.05, 0.9],   // Ankle L resting on thigh
      [-0.12, 0.65, 0.05, 0.9]   // Ankle R standing
    ]
  }
};

export default function Dashboard() {
  const [apiURL, setApiURL] = useState("http://localhost:8000/api");
  const [groqKey, setGroqKey] = useState("");
  const [lang, setLang] = useState<"en" | "hi" | "bn">("en");
  const [activePreset, setActivePreset] = useState<"warrior_2" | "plank" | "tree_pose">("warrior_2");
  
  // Simulated Slider Angles
  const [simKneeL, setSimKneeL] = useState(120);
  const [simShoulderL, setSimShoulderL] = useState(90);
  const [simElbowL, setSimElbowL] = useState(165);
  
  // Occlusion Simulation Settings
  const [occlusionType, setOcclusionType] = useState<"none" | "left_elbow" | "right_knee">("none");
  const [speechEnabled, setSpeechEnabled] = useState(true);
  
  // Digital Twin User Calibration Profile
  const [calibKneeMin, setCalibKneeMin] = useState(80);
  const [calibKneeMax, setCalibKneeMax] = useState(160);
  const [calibShoulderMin, setCalibShoulderMin] = useState(30);
  const [calibShoulderMax, setCalibShoulderMax] = useState(150);

  // Set environment variable on startup if available
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

  // Keep track of spoken text to avoid repeating
  const lastSpokenText = useRef("");

  // Speech synthesis wrapper for Audio Guidance (Stage 11)
  useEffect(() => {
    if (speechEnabled && correctionText && correctionText !== lastSpokenText.current) {
      // Cancel ongoing speech
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

  // Sync preset changes to sliders
  useEffect(() => {
    const preset = PRESET_POSES[activePreset];
    setSimKneeL(preset.angles[6]);      // index 6 is knee_l
    setSimShoulderL(preset.angles[2]);  // index 2 is shoulder_l
    setSimElbowL(preset.angles[0]);     // index 0 is elbow_l
    resetPipeline();
  }, [activePreset]);

  // Construct current simulated joint angles and landmarks coordinates
  const triggerPipelineStep = () => {
    // 1. Get preset data
    const preset = PRESET_POSES[activePreset];
    
    // 2. Adjust angles based on simulated sliders
    const angles = [...preset.angles];
    angles[6] = simKneeL;
    angles[2] = simShoulderL;
    angles[0] = simElbowL;

    // 3. Adjust visibility landmarks depending on simulated occlusion
    const landmarks = preset.landmarks.map(pt => [...pt]);
    if (occlusionType === "left_elbow") {
      // index 3 is left elbow (MP index 13)
      if (landmarks[3]) {
        landmarks[3][3] = 0.1; // Drop visibility to 10%
        landmarks[3][0] = 0.0; // Clear coordinate to simulate raw occlusion
        landmarks[3][1] = 0.0;
        landmarks[3][2] = 0.0;
      }
    } else if (occlusionType === "right_knee") {
      // index 10 is right knee (MP index 26)
      if (landmarks[10]) {
        landmarks[10][3] = 0.1; // Drop visibility
        landmarks[10][0] = 0.0;
        landmarks[10][1] = 0.0;
        landmarks[10][2] = 0.0;
      }
    }

    // 4. Update the endpoint base URL in local storage
    if (typeof window !== "undefined") {
      (window as any).process = {
        env: { NEXT_PUBLIC_YOGA_API_URL: apiURL }
      };
    }

    // 5. Send to hook
    processFrame(landmarks, angles);
  };

  // SVG skeleton rendering helper coordinates
  const renderSkeleton = () => {
    const preset = PRESET_POSES[activePreset];
    // Scale and translate coordinates to fit 100% SVG box
    const getCoords = (idx: number) => {
      const pt = preset.landmarks[idx];
      if (!pt) return { x: 150, y: 150 };
      
      // Basic scaling: x is inside [-1, 1], y is inside [-1, 1]
      // Map to SVG width 300, height 300
      const x = (pt[0] + 0.8) * 180 + 10;
      const y = (pt[1] + 0.8) * 180 + 10;
      return { x, y };
    };

    // Joint indices mapping:
    // Nose=0, ShoulderL=1, ShoulderR=2, ElbowL=3, ElbowR=4, WristL=5, WristR=6
    // HipL=7, HipR=8, KneeL=9, KneeR=10, AnkleL=11, AnkleR=12
    const n = getCoords(0);
    const sl = getCoords(1);
    const sr = getCoords(2);
    const el = getCoords(3);
    const er = getCoords(4);
    const wl = getCoords(5);
    const wr = getCoords(6);
    const hl = getCoords(7);
    const hr = getCoords(8);
    const kl = getCoords(9);
    const kr = getCoords(10);
    const al = getCoords(11);
    const ar = getCoords(12);

    const isKneeDeviating = Math.abs(simKneeL - 90) > 15 && activePose === "warrior_2";

    return (
      <svg className="skeleton-svg" viewBox="0 0 350 350">
        <defs>
          <radialGradient id="glowGrad" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#10b981" stopOpacity="0.4"/>
            <stop offset="100%" stopColor="#10b981" stopOpacity="0"/>
          </radialGradient>
          <radialGradient id="glowWarn" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#ef4444" stopOpacity="0.5"/>
            <stop offset="100%" stopColor="#ef4444" stopOpacity="0"/>
          </radialGradient>
        </defs>
        
        {/* Ambient glows behind active joints */}
        <circle cx={kl.x} cy={kl.y} r="35" fill={isKneeDeviating ? "url(#glowWarn)" : "url(#glowGrad)"} />
        
        {/* Head */}
        <circle cx={n.x} cy={n.y} r="18" fill="#1e293b" stroke="#3b82f6" strokeWidth="3" />
        
        {/* Torso */}
        <line x1={sl.x} y1={sl.y} x2={sr.x} y2={sr.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={sl.x} y1={sl.y} x2={hl.x} y2={hl.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={sr.x} y1={sr.y} x2={hr.x} y2={hr.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={hl.x} y1={hl.y} x2={hr.x} y2={hr.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        
        {/* Left Arm */}
        <line x1={sl.x} y1={sl.y} x2={el.x} y2={el.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={el.x} y1={el.y} x2={wl.x} y2={wl.y} stroke={occlusionType === "left_elbow" ? "#f59e0b" : "#f8fafc"} strokeWidth="4" strokeLinecap="round" />
        
        {/* Right Arm */}
        <line x1={sr.x} y1={sr.y} x2={er.x} y2={er.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={er.x} y1={er.y} x2={wr.x} y2={wr.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />

        {/* Left Leg */}
        <line x1={hl.x} y1={hl.y} x2={kl.x} y2={kl.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={kl.x} y1={kl.y} x2={al.x} y2={al.y} stroke={isKneeDeviating ? "#ef4444" : "#f8fafc"} strokeWidth="4" strokeLinecap="round" />

        {/* Right Leg */}
        <line x1={hr.x} y1={hr.y} x2={kr.x} y2={kr.y} stroke="#f8fafc" strokeWidth="4" strokeLinecap="round" />
        <line x1={kr.x} y1={kr.y} x2={ar.x} y2={ar.y} stroke={occlusionType === "right_knee" ? "#f59e0b" : "#f8fafc"} strokeWidth="4" strokeLinecap="round" />

        {/* Joint Points */}
        <circle cx={sl.x} cy={sl.y} r="6" fill="#3b82f6" />
        <circle cx={sr.x} cy={sr.y} r="6" fill="#3b82f6" />
        <circle cx={el.x} cy={el.y} r="6" fill={occlusionType === "left_elbow" ? "#f59e0b" : "#3b82f6"} />
        <circle cx={er.x} cy={er.y} r="6" fill="#3b82f6" />
        <circle cx={wl.x} cy={wl.y} r="6" fill="#10b981" />
        <circle cx={wr.x} cy={wr.y} r="6" fill="#10b981" />
        <circle cx={hl.x} cy={hl.y} r="6" fill="#3b82f6" />
        <circle cx={hr.x} cy={hr.y} r="6" fill="#3b82f6" />
        <circle cx={kl.x} cy={kl.y} r="8" fill={isKneeDeviating ? "#ef4444" : "#3b82f6"} />
        <circle cx={kr.x} cy={kr.y} r="6" fill={occlusionType === "right_knee" ? "#f59e0b" : "#3b82f6"} />
        <circle cx={al.x} cy={al.y} r="6" fill="#10b981" />
        <circle cx={ar.x} cy={ar.y} r="6" fill="#10b981" />
      </svg>
    );
  };

  return (
    <div style={{ minHeight: "100vh" }}>
      {/* Navbar Header */}
      <header className="app-header">
        <div className="app-logo">
          <Activity className="status-dot active" style={{ color: "#10b981" }} />
          <span>SmartYoga.AI</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "15px" }}>
          <div className="status-pill">
            <div className={`status-dot ${activePose !== "transition/unknown" ? "active" : ""}`} />
            <span>{activePose === "transition/unknown" ? "Awaiting Input" : `Pose: ${activePose}`}</span>
          </div>
          <button 
            className="btn-toggle" 
            onClick={() => setSpeechEnabled(!speechEnabled)}
            title={speechEnabled ? "Mute audio corrections" : "Unmute audio corrections"}
          >
            {speechEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="dashboard-grid">
        {/* LEFT COLUMN: CONTROL & SIMULATION PANEL */}
        <section style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Preset Selector */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Compass size={20} className="text-primary" style={{ color: "#3b82f6" }} />
              <span>Pose Selector</span>
            </h2>
            <div className="btn-row">
              <button 
                className={`btn-toggle ${activePreset === "warrior_2" ? "active" : ""}`}
                onClick={() => setActivePreset("warrior_2")}
              >
                Warrior II
              </button>
              <button 
                className={`btn-toggle ${activePreset === "plank" ? "active" : ""}`}
                onClick={() => setActivePreset("plank")}
              >
                Plank
              </button>
              <button 
                className={`btn-toggle ${activePreset === "tree_pose" ? "active" : ""}`}
                onClick={() => setActivePreset("tree_pose")}
              >
                Tree Pose
              </button>
            </div>
            <p style={{ marginTop: "12px", fontSize: "0.85rem", color: "var(--text-muted)" }}>
              Current simulated target: <strong>{PRESET_POSES[activePreset].name}</strong>
            </p>
          </div>

          {/* Interactive Joint Slider Controls */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Sliders size={20} style={{ color: "#3b82f6" }} />
              <span>Joint Angle Simulator</span>
            </h2>
            <div className="input-group">
              <label className="input-label">Left Knee Angle (Warrior II Target: 90°)</label>
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
              <label className="input-label">Left Shoulder Angle (Warrior II Target: 90°)</label>
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
              <label className="input-label">Left Elbow Angle (Warrior II Target: 180°)</label>
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

            {/* Simulated Occlusion options */}
            <div className="input-group" style={{ marginTop: "20px" }}>
              <label className="input-label">Simulate joint occlusion</label>
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

          {/* User Digital Twin Calibration Limits */}
          <div className="glass-panel">
            <h2 className="section-title">
              <ShieldCheck size={20} style={{ color: "#10b981" }} />
              <span>Digital Twin Limits Calibration</span>
            </h2>
            <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginBottom: "15px" }}>
              Adjust limits to model physical constraints. Correct angles within limits bypass warnings.
            </p>
            
            <div className="input-group">
              <label className="input-label">Left Knee Min/Max Limits</label>
              <div style={{ display: "flex", gap: "10px", alignItems: "center" }}>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibKneeMin}
                  onChange={(e) => setCalibKneeMin(Number(e.target.value))}
                  placeholder="Min"
                />
                <span className="text-muted">to</span>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibKneeMax}
                  onChange={(e) => setCalibKneeMax(Number(e.target.value))}
                  placeholder="Max"
                />
                <span className="text-muted">degrees</span>
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
                  placeholder="Min"
                />
                <span className="text-muted">to</span>
                <input 
                  type="number" 
                  className="groq-config-input" 
                  style={{ width: "80px", padding: "8px" }}
                  value={calibShoulderMax}
                  onChange={(e) => setCalibShoulderMax(Number(e.target.value))}
                  placeholder="Max"
                />
                <span className="text-muted">degrees</span>
              </div>
            </div>
          </div>

          {/* API and Groq Configuration */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Settings size={20} style={{ color: "#94a3b8" }} />
              <span>API Gateway Configuration</span>
            </h2>
            <div className="input-group">
              <label className="input-label">FastAPI Backend Endpoint</label>
              <input 
                type="text" 
                className="groq-config-input" 
                value={apiURL} 
                onChange={(e) => setApiURL(e.target.value)}
                placeholder="http://localhost:8000/api" 
              />
            </div>
            
            <div className="input-group">
              <label className="input-label">Groq API Key (Optional)</label>
              <input 
                type="password" 
                className="groq-config-input" 
                value={groqKey} 
                onChange={(e) => setGroqKey(e.target.value)}
                placeholder="gsk_..." 
              />
            </div>

            <div className="input-group">
              <label className="input-label">Language Translation</label>
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

        {/* RIGHT COLUMN: VISUAL SKELETON & REALTIME ANALYTICS */}
        <section style={{ display: "flex", flexDirection: "column", gap: "24px" }}>
          
          {/* Skeleton Renderer & Execution */}
          <div className="glass-panel" style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
              <h2 className="section-title" style={{ marginBottom: 0 }}>
                <Activity size={20} style={{ color: "#10b981" }} />
                <span>Anatomical Projection</span>
              </h2>
              <button 
                className="btn-primary" 
                onClick={triggerPipelineStep}
                disabled={isLoading}
                style={{ display: "flex", alignItems: "center", gap: "8px", padding: "8px 18px" }}
              >
                {isLoading ? <RefreshCw className="animate-spin" size={16} /> : <Sparkles size={16} />}
                <span>Evaluate Frame</span>
              </button>
            </div>
            
            {renderSkeleton()}
          </div>

          {/* Real-time metrics dashboard */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Gauge size={20} style={{ color: "#3b82f6" }} />
              <span>Real-Time Feedback Hub</span>
            </h2>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "20px" }}>
              {/* Correctness Score circular gauge */}
              <div className="gauge-container glass-panel" style={{ background: "rgba(0,0,0,0.2)" }}>
                <div className={`gauge-circle ${correctness >= 0.75 ? "success" : "warning"}`}>
                  <span className="gauge-percentage">
                    {Math.round(correctness * 100)}%
                  </span>
                </div>
                <span className="gauge-label">Correctness Score</span>
              </div>

              {/* Status information */}
              <div style={{ display: "flex", flexDirection: "column", gap: "12px", justifyContent: "center" }}>
                <div className="deviation-item" style={{ margin: 0 }}>
                  <span className="text-muted">Detected Pose</span>
                  <span style={{ fontWeight: 600, color: "#60a5fa" }}>
                    {activePose === "transition/unknown" ? "Transition" : activePose.toUpperCase()}
                  </span>
                </div>
                <div className="deviation-item" style={{ margin: 0 }}>
                  <span className="text-muted">Sequence Flow</span>
                  <span style={{ fontWeight: 600 }}>
                    {flowPose === "transition/unknown" ? "Static Check" : flowPose.toUpperCase()}
                  </span>
                </div>
                <div className="deviation-item" style={{ margin: 0 }}>
                  <span className="text-muted">Occlusion Recovery</span>
                  <span style={{ fontWeight: 600, color: recoveredJoints.length > 0 ? "#f59e0b" : "#10b981" }}>
                    {recoveredJoints.length > 0 ? `Fused (${recoveredJoints.length})` : "Inactive"}
                  </span>
                </div>
              </div>
            </div>

            {/* Recycled Joints List */}
            {recoveredJoints.length > 0 && (
              <div style={{ marginTop: "16px", padding: "10px", background: "rgba(245, 158, 11, 0.05)", border: "1px solid rgba(245,158,11,0.2)", borderRadius: "8px" }}>
                <p style={{ fontSize: "0.85rem", color: "#f59e0b", display: "flex", alignItems: "center", gap: "6px" }}>
                  <ShieldAlert size={14} />
                  <span><strong>Occlusion Fused joints:</strong> {recoveredJoints.join(", ")} (Coordinate mirrored from twin joint)</span>
                </p>
              </div>
            )}

            {/* Audio Correction alert text */}
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
                  <span style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", fontWeight: 600 }}>Pose Perfect</span>
                  <span className="guidance-text" style={{ color: "#d1fae5" }}>Pose execution is within biomechanical tolerances. Keep it up!</span>
                </div>
              </div>
            )}
          </div>

          {/* Dynamic joint deviations display list */}
          {activePose !== "transition/unknown" && (
            <div className="glass-panel">
              <h2 className="section-title">
                <HelpCircle size={20} style={{ color: "#ef4444" }} />
                <span>Joint Deviations details</span>
              </h2>
              
              <div className="deviation-item">
                <span className="deviation-name">Left Knee Angle</span>
                <div style={{ display: "flex", alignItems: "center" }}>
                  <div className="deviation-bar-bg">
                    <div 
                      className="deviation-bar-fill" 
                      style={{ 
                        width: `${Math.min(100, Math.abs(simKneeL - 90) * 1.5)}%`,
                        background: Math.abs(simKneeL - 90) > 15 ? "var(--color-error)" : "var(--color-success)"
                      }} 
                    />
                  </div>
                  <span className={`deviation-value ${Math.abs(simKneeL - 90) > 15 ? "error" : "success"}`}>
                    {simKneeL}° (Diff: {Math.round(simKneeL - 90)}°)
                  </span>
                </div>
              </div>

              <div className="deviation-item">
                <span className="deviation-name">Left Shoulder Angle</span>
                <div style={{ display: "flex", alignItems: "center" }}>
                  <div className="deviation-bar-bg">
                    <div 
                      className="deviation-bar-fill" 
                      style={{ 
                        width: `${Math.min(100, Math.abs(simShoulderL - 90) * 1.5)}%`,
                        background: Math.abs(simShoulderL - 90) > 15 ? "var(--color-error)" : "var(--color-success)"
                      }} 
                    />
                  </div>
                  <span className={`deviation-value ${Math.abs(simShoulderL - 90) > 15 ? "error" : "success"}`}>
                    {simShoulderL}° (Diff: {Math.round(simShoulderL - 90)}°)
                  </span>
                </div>
              </div>
              
              <p style={{ fontSize: "0.8rem", color: "var(--text-muted)", marginTop: "12px", textAlign: "right" }}>
                * Target angles calculated relative to standard biomechanical template models.
              </p>
            </div>
          )}

        </section>
      </main>
    </div>
  );
}
