import { useState, useEffect, useRef } from "react";
import Head from "next/head";
import Script from "next/script";
import { 
  Volume2, 
  VolumeX, 
  RefreshCw, 
  ShieldCheck,
  ShieldAlert,
  Activity,
  CheckCircle2, 
  Camera as CameraIcon,
  VideoOff,
  PanelLeftClose,
  PanelLeftOpen,
  ChevronDown,
  Maximize2,
  X,
  Globe,
  LayoutGrid,
  SwitchCamera,
  AlertTriangle,
  Loader2,
  PersonStanding,
  TreePine,
  Mountain,
  Waves,
  Triangle,
  Minus,
  Ruler,
  Sun,
  ScanLine,
  Leaf,
  RotateCw,
  Check,
  Lock
} from "lucide-react";
import { useYogaPipeline } from "../hooks/useYogaPipeline";
import { CalibrationProfile } from "../types/yoga";
import { extractAnglesFromLandmarks } from "../utils/geometry";
import { VisibilityTracker } from "../utils/visibility";
import { pickVoice, splitForSpeech, utteranceLang, voiceMissing as isVoiceMissing } from "../utils/speechText";

/* eslint-disable */
type PresetPoseId = "warrior_2" | "cobra_pose" | "mountain_pose" | "tree_pose" | "plank" | "downward_dog";

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
  cobra_pose: [
    { joint: "neck", label: "Neck Extension", target: 140, tolerance: 20 },
    { joint: "trunk_l", label: "Left Trunk Extension", target: 140, tolerance: 20 }
  ],
  mountain_pose: [
    { joint: "knee_l", label: "Left Knee Extension", target: 180, tolerance: 10 },
    { joint: "knee_r", label: "Right Knee Extension", target: 180, tolerance: 10 },
    { joint: "trunk_l", label: "Left Spine Straightness", target: 180, tolerance: 10 }
  ],
  tree_pose: [
    { joint: "knee_r", label: "Standing Leg Extension", target: 175, tolerance: 15 },
    { joint: "hip_r", label: "Standing Hip Extension", target: 175, tolerance: 10 },
  ],
  plank: [
    { joint: "hip_l", label: "Hip Line Straightness", target: 160, tolerance: 15 },
    { joint: "knee_l", label: "Knee Line Straightness", target: 160, tolerance: 15 },
    { joint: "shoulder_l", label: "Shoulder-Arm Angle", target: 85, tolerance: 20 },
  ],
  downward_dog: [
    { joint: "hip_l", label: "Hip Fold Angle", target: 80, tolerance: 30 },
    { joint: "knee_l", label: "Leg Extension", target: 145, tolerance: 25 },
    { joint: "shoulder_l", label: "Arm-Shoulder Line", target: 137, tolerance: 25 },
  ]
};

// Static alignment cues per pose — displayed in the Pose Guide sidebar section
const POSE_GUIDE: { [key: string]: { cue: string; icon: string }[] } = {
  warrior_2: [
    { icon: "🦵", cue: "Front knee at 90° over ankle" },
    { icon: "💪", cue: "Arms parallel, shoulder height" },
    { icon: "👁️", cue: "Gaze over front fingertips" },
    { icon: "🦴", cue: "Hips open to the long side" },
  ],
  cobra_pose: [
    { icon: "🐍", cue: "Hands under shoulders, elbows close to ribs" },
    { icon: "🦴", cue: "Lift chest with back muscles, not just arms" },
    { icon: "🧘", cue: "Hips and thighs stay grounded" },
    { icon: "👁️", cue: "Gaze forward, neck long and neutral" },
  ],
  mountain_pose: [
    { icon: "🦶", cue: "Feet grounded, weight evenly balanced" },
    { icon: "🦴", cue: "Spine tall, shoulders stacked over hips" },
    { icon: "💪", cue: "Arms relaxed at sides" },
    { icon: "👁️", cue: "Gaze steady, breathing calm" },
  ],
  tree_pose: [
    { icon: "🦶", cue: "Standing leg fully extended, foot rooted" },
    { icon: "🦵", cue: "Lifted foot pressed into inner thigh or calf, never the knee" },
    { icon: "🦴", cue: "Hips level, stacked over the standing foot" },
    { icon: "👁️", cue: "Fix your gaze on one point for balance" },
  ],
  plank: [
    { icon: "🦴", cue: "Straight line from shoulders to heels" },
    { icon: "💪", cue: "Hands under shoulders, arms firm" },
    { icon: "🦵", cue: "Legs active, core engaged" },
    { icon: "👁️", cue: "Gaze slightly forward, neck neutral" },
  ],
  downward_dog: [
    { icon: "🦶", cue: "Hands shoulder-width, feet hip-width" },
    { icon: "🦴", cue: "Hips lifted high, forming an inverted V" },
    { icon: "🦵", cue: "Heels reach toward the floor, knees soft if needed" },
    { icon: "💪", cue: "Arms straight, weight shared between hands and feet" },
  ],
};

const POSE_DIFFICULTY: { [key: string]: { level: string; color: string } } = {
  warrior_2:    { level: "Intermediate", color: "var(--amber)" },
  cobra_pose:   { level: "Beginner",     color: "var(--ok)" },
  mountain_pose:{ level: "Beginner",     color: "var(--ok)" },
  tree_pose:    { level: "Intermediate", color: "var(--amber)" },
  plank:        { level: "Intermediate", color: "var(--amber)" },
  downward_dog: { level: "Beginner",     color: "var(--ok)" },
};

// Poses selectable in the sidebar "Target Pose" grid. Phase B re-validated
// plank, tree_pose, and downward_dog against a fresh 47-image real-world
// test set with new/widened rules (see backend/app/utils/rules_classifier.py)
// and they cleared a real accuracy bar (50%/76.4%/91.7% respectively) --
// added here. chair_pose was re-tested and still measured ~0% (unchanged
// finding, not a threshold miss -- 2D real-world chair_pose photos vary too
// much by camera angle); warrior_1 was newly tested and topped out at 18.3%
// (most sourced photos didn't actually show a clean Warrior I stance). Both
// stay out of the vocabulary, matching backend/app/utils/rules_classifier.py's
// DISABLED_POSES, rather than ship a selectable pose the detector can't
// reliably confirm.
// Icons are thematic only (never a human-figure emoji depicting a specific
// body position) — a wrong body-position emoji invites users to physically
// copy an incorrect pose. warrior_2 used to show 🧘 (a seated meditation
// figure), which is a completely different pose from the standing Warrior II
// lunge; the reference photo below is now the actual "how do I do this"
// source of truth.
// Line icons (consistent weight, inherit colour) instead of emoji; common English names for the sub-line.
const POSE_ICONS: { [id: string]: any } = {
  warrior_2: PersonStanding,
  cobra_pose: Waves,
  mountain_pose: Mountain,
  tree_pose: TreePine,
  plank: Minus,
  downward_dog: Triangle,
};
const POSE_COMMON_NAME: { [id: string]: string } = {
  warrior_2: "Warrior II",
  cobra_pose: "Cobra",
  mountain_pose: "Mountain",
  tree_pose: "Tree",
  plank: "Plank",
  downward_dog: "Downward dog",
};

const POSE_LIBRARY: { id: PresetPoseId; icon: string }[] = [
  { id: "warrior_2", icon: "⚔️" },
  { id: "cobra_pose", icon: "🐍" },
  { id: "mountain_pose", icon: "⛰️" },
  { id: "tree_pose", icon: "🌳" },
  { id: "plank", icon: "📏" },
  { id: "downward_dog", icon: "🐕" },
];

// Real reference photographs for each practice-able pose, shown in the Pose
// Guide panel so users don't have to guess correct form from an emoji.
// All CC-BY / CC-BY-SA licensed from Wikimedia Commons; attribution shown
// inline per license terms.
const POSE_REFERENCE_IMAGES: { [key: string]: { src: string; credit: string } } = {
  warrior_2: { src: "/pose-images/warrior_2.jpg", credit: "lululemon athletica, CC BY 2.0, via Wikimedia Commons" },
  cobra_pose: { src: "/pose-images/cobra_pose.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  mountain_pose: { src: "/pose-images/mountain_pose.jpg", credit: "Witold Fitz-Simon, CC BY-SA 2.5, via Wikimedia Commons" },
  plank: { src: "/pose-images/plank.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  tree_pose: { src: "/pose-images/tree_pose.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  // downward_dog has no local reference image yet -- POSE_REFERENCE_IMAGES
  // lookups are guarded (`POSE_REFERENCE_IMAGES[guidePose] &&`), so this
  // degrades gracefully (no image shown) rather than breaking. Sourcing one
  // is a small, non-heavy download appropriate for a follow-up, not done
  // here per this task's own no-local-download scope.
};


const FEATURE_NAMES_ORDER = [
  "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
  "hip_l", "hip_r", "knee_l", "knee_r",
  "ankle_l", "ankle_r", "trunk_l", "trunk_r",
  "neck", "hip_abduct_l", "hip_abduct_r"
];

/** "knee_l" -> "Left knee". Used by the closed-loop efficacy chip, which
 *  names the joint it measured so the feedback is specific rather than a
 *  generic "better"/"worse". */
function formatJointName(joint: string, lang: "en" | "hi" | "bn"): string {
  const side = joint.endsWith("_l")
    ? { en: "Left", hi: "बायाँ", bn: "বাম" }[lang]
    : joint.endsWith("_r")
    ? { en: "Right", hi: "दायाँ", bn: "ডান" }[lang]
    : "";
  const base = joint.replace(/_(l|r)$/, "").replace("hip_abduct", "hip");
  const names: { [k: string]: { en: string; hi: string; bn: string } } = {
    elbow: { en: "elbow", hi: "कोहनी", bn: "কনুই" },
    shoulder: { en: "shoulder", hi: "कंधा", bn: "কাঁধ" },
    hip: { en: "hip", hi: "कूल्हा", bn: "নিতম্ব" },
    knee: { en: "knee", hi: "घुटना", bn: "হাঁটু" },
    ankle: { en: "ankle", hi: "टखना", bn: "গোড়ালি" },
    trunk: { en: "torso", hi: "धड़", bn: "ধড়" },
    neck: { en: "neck", hi: "गर्दन", bn: "ঘাড়" },
  };
  const n = names[base]?.[lang] ?? base;
  return side ? `${side} ${n}` : n.charAt(0).toUpperCase() + n.slice(1);
}

const SANSKRIT_NAMES: {
  [lang: string]: { [key: string]: string }
} = {
  en: {
    "warrior_2": "Virabhadrasana II",
    "warrior_ii": "Virabhadrasana II",
    "warrior_1": "Virabhadrasana I",
    "warrior_i": "Virabhadrasana I",
    "plank": "Phalakasana",
    "tree": "Vrikshasana",
    "tree_pose": "Vrikshasana",
    "downward_dog": "Adho Mukha Svanasana",
    "downward_facing_dog": "Adho Mukha Svanasana",
    "cobra": "Bhujangasana",
    "cobra_pose": "Bhujangasana",
    "mountain": "Tadasana",
    "mountain_pose": "Tadasana",
    "chair": "Utkatasana",
    "chair_pose": "Utkatasana",
    "chaturanga": "Chaturanga Dandasana",
    "child_pose": "Balasana",
    "corpse": "Savasana",
    "halfway_lift": "Ardha Uttanasana",
    "lunge_pose": "Anjaneyasana",
    "seated_easy_pose": "Sukhasana",
    "seated_forward": "Paschimottanasana",
    "seated_staff": "Dandasana",
    "standing_forward_fold": "Uttanasana",
    "standing_pose": "Tadasana",
    "table_top": "Bharmanasana",
    "triangle": "Trikonasana",
    "upward_dog": "Urdhva Mukha Svanasana",
    "upward_salute": "Urdhva Hastasana",
    "transition/unknown": "Transition/Unknown",
    "unknown": "Transition/Unknown"
  },
  hi: {
    "warrior_2": "वीरभद्रासन २",
    "warrior_ii": "वीरभद्रासन २",
    "warrior_1": "वीरभद्रासन १",
    "warrior_i": "वीरभद्रासन १",
    "plank": "फलकासन",
    "tree": "वृक्षासन",
    "tree_pose": "वृक्षासन",
    "downward_dog": "अधोमुख श्वानासन",
    "downward_facing_dog": "अधोमुख श्वानासन",
    "cobra": "भुजंगासन",
    "cobra_pose": "भुजंगासन",
    "mountain": "ताड़ासन",
    "mountain_pose": "ताड़ासन",
    "chair": "उत्कटासन",
    "chair_pose": "उत्कटासन",
    "chaturanga": "चतुरंग दंडासन",
    "child_pose": "बालासन",
    "corpse": "शवासन",
    "halfway_lift": "अर्ध उत्तानासन",
    "lunge_pose": "अंजनेयासन",
    "seated_easy_pose": "सुखासन",
    "seated_forward": "पश्चिमॉत्तानासन",
    "seated_staff": "दंडासन",
    "standing_forward_fold": "उत्तानासन",
    "standing_pose": "ताड़ासन",
    "table_top": "भरमनासन",
    "triangle": "त्रिकोणासन",
    "upward_dog": "ऊर्ध्वमुख श्वानासन",
    "upward_salute": "ऊर्ध्व हस्तोत्तानासन",
    "transition/unknown": "परिवर्तन/अज्ञात",
    "unknown": "परिवर्तन/अज्ञात"
  },
  bn: {
    "warrior_2": "বীরভদ্রাসন ২",
    "warrior_ii": "বীরভদ্রাসন ২",
    "warrior_1": "বীরভদ্রাসন ১",
    "warrior_i": "বীরভদ্রাসন ১",
    "plank": "ফলকাসন",
    "tree": "বৃক্ষাসন",
    "tree_pose": "বৃক্ষাসন",
    "downward_dog": "অধোমুখ শ্বানাসন",
    "downward_facing_dog": "অধোমুখ শ্বানাসন",
    "cobra": "ভুজঙ্গাসন",
    "cobra_pose": "ভুজঙ্গাসন",
    "mountain": "তাড়াসন",
    "mountain_pose": "তাড়াসন",
    "chair": "উত্কটাসন",
    "chair_pose": "উত্কটাসন",
    "chaturanga": "চতুরঙ্গ দণ্ডাসন",
    "child_pose": "বালাসন",
    "corpse": "শবাসন",
    "halfway_lift": "অর্ধ উত্তানাসন",
    "lunge_pose": "অঞ্জনীয়াসন",
    "seated_easy_pose": "সুখাসন",
    "seated_forward": "পশ্চিমোত্তানাসন",
    "seated_staff": "দণ্ডাসন",
    "standing_forward_fold": "উত্তানাসন",
    "standing_pose": "তাড়াসন",
    "table_top": "ভার্মানাসন",
    "triangle": "ত্রিকোণাসন",
    "upward_dog": "ঊর্ধ্বমুখ শ্বানাসন",
    "upward_salute": "উর্ধ্ব হস্তোত্তানাসন",
    "transition/unknown": "পরিবর্তন/অজানা",
    "unknown": "পরিবর্তন/অজানা"
  }
};

const getSanskritName = (poseId: string, lang: "en" | "hi" | "bn" = "en") => {
  if (!poseId) return lang === "hi" ? "परिवर्तन/अज्ञात" : lang === "bn" ? "পরিবর্তন/অজানা" : "Transition/Unknown";
  const cleanId = poseId.toLowerCase().split('/').pop() || "";
  const key = cleanId.replace("_pose", "").replace("pose_", "").trim();
  
  const dict = SANSKRIT_NAMES[lang] || SANSKRIT_NAMES.en;
  if (dict[cleanId]) return dict[cleanId];
  if (dict[key]) return dict[key];
  
  return cleanId.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
};

// The pose engine (MediaPipe) needs a WebGL context. When a browser refuses one (graphics acceleration off or blocked,
// battery saver, too many contexts open, some in-app browsers) MediaPipe shows a raw alert() on EVERY frame.
// We intercept that alert (see the effect that wraps window.alert) and show our own themed message instead.
// MediaPipe initialises itself on the first frame: never start or call its initialize() in parallel with frame processing.
const GRAPHICS_ERR = "__graphics__";
// A short, local-only summary of what the browser offers, shown under "Details" when the engine cannot start.
const graphicsReport = (): string => {
  const ua = navigator.userAgent || "";
  const probe = (type: string) => {
    try {
      const gl: any = document.createElement("canvas").getContext(type);
      if (!gl) return { ok: false, gpu: "" };
      const ext = gl.getExtension("WEBGL_debug_renderer_info");
      const gpu = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
      gl.getExtension("WEBGL_lose_context")?.loseContext();
      return { ok: true, gpu };
    } catch { return { ok: false, gpu: "" }; }
  };
  const w2 = probe("webgl2"), w1 = probe("webgl");
  const inApp = /FBAN|FBAV|Instagram|Line\/|MicroMessenger|Snapchat|WhatsApp|Telegram|; wv\)/i.test(ua);
  const browser = /SamsungBrowser/i.test(ua) ? "Samsung Internet" : /EdgA|Edg\//.test(ua) ? "Edge" : /OPR|Opera/i.test(ua) ? "Opera" : /Firefox|FxiOS/i.test(ua) ? "Firefox"
    : /CriOS|Chrome/i.test(ua) ? "Chrome" : /Safari/i.test(ua) ? "Safari" : "unknown";
  return [
    `WebGL2: ${w2.ok ? "yes" : "no"} | WebGL: ${w1.ok ? "yes" : "no"}`,
    (w2.gpu || w1.gpu) ? `GPU: ${w2.gpu || w1.gpu}` : "GPU: not reported",
    `OffscreenCanvas: ${typeof OffscreenCanvas !== "undefined" ? "yes" : "no"}`,
    `Browser: ${browser}${inApp ? " (in-app browser: open the page in Chrome or Safari)" : ""}`,
    `Device: ${(ua.match(/\(([^)]*)\)/) || [])[1] || "unknown"}`,
    `Screen: ${window.innerWidth}x${window.innerHeight} @${window.devicePixelRatio}x`,
  ].join("\n");
};

const TRANSLATIONS: {
  [lang: string]: { [key: string]: string }
} = {
  en: {
    tapForGuide: "Tap any pose to preview its guide.",
    previewing: "Previewing",
    followMe: "Follow my pose",
    followingYou: "Following the pose you're doing",
    tagGuide: "Guide",
    readyTitle: "Ready when you are",
    readyBody: "Stand about two metres back so your whole body is in view, in good light.",
    startCamera: "Start camera",
    stopCamera: "Stop",
    retry: "Retry",
    poseEngineLoading: "Loading pose engine…",
    poseEngineSlow: "Pose engine is slow to load. Check your connection.",
    graphicsTitle: "Can't start the pose engine",
    graphicsDetails: "Details",
    graphicsBody: "Your browser couldn't turn on graphics acceleration, which the pose engine needs. Close other tabs and apps, turn off battery saver, make sure hardware acceleration is on, or open this page in Chrome or Safari, then tap Retry.",
    practice: "Practice",
    poseGuide: "Pose guide",
    sessionOverview: "Session",
    cameraOff: "Camera off",
    cameraLive: "Camera live",
    coaching: "Coaching",
    voice: "Voice",
    flip: "Flip",
    poses: "Poses",
    tipDistance: "Stand 2 m back",
    tipLight: "Face the light",
    tipFrame: "Full body in frame",
    measuredJoints: "Joints measured",
    tagNow: "Now",
    lookingForPose: "Looking for a pose…",
    installTitle: "Install AsanaAI",
    installBody: "Open it like an app: full screen, works offline.",
    install: "Install",
    offline: "You're offline — check your connection.",
    jointAlignment: "Joint alignment",
    livePanel: "Live feedback",
    appTitle: "AsanaAI — Smart Yoga Coach",
    newSession: "New Session",
    recognisedAsanas: "Recognised Asanas",
    modeFree: "Free mode",
    modeGuided: "Guided mode",
    modeFreeHint: "Do any pose. I'll recognise it and score that pose.",
    modeGuidedHint: "Pick a pose, then perform it. I'll check you against it.",
    targetPose: "Target Pose",
    wrongPoseDetected: "This looks like {detected}, not your selected {target}. Adjust into {target} to get scored.",
    wrongPoseBadge: "Wrong Pose",
    voiceMissing: "No voice for this language was found on this device, so spoken guidance may be silent or in English. Captions still work. Add a voice in your device's speech settings.",
    detectedPose: "Detected Pose",
    stateHolding: "Holding",
    stateTransitioning: "In Transition",
    stateUnrecognized: "Unrecognised Posture",
    universalScore: "General form",
    personalScore: "For your body",
    digitalTwinProfile: "Your range of motion",
    activeProfile: "✓ Range of motion saved",
    uncalibratedTwin: "When the camera starts, hold a comfortable standing pose for 15 seconds. AsanaAI learns how far your joints naturally move, so feedback fits your body.",
    calibratingTwin: "Learning your range of motion…",
    apiConfig: "API Configuration",
    backendApiUrl: "Backend API URL",
    apiUrlHelper: "Point to localhost or your Hugging Face Space URL.",
    cameraStream: "Camera Stream",
    swapCamera: "Swap Camera",
    focusMode: "Focus Mode",
    stopVideo: "Stop Video",
    startVideo: "Start Video",
    calibratingProgress: "Learning your range of motion",
    stayInView: "Stand comfortably in view and breathe.",
    cameraInactive: "Start the camera and move into any posture. AsanaAI recognises what you are doing.",
    alignBody: "Align your body with the camera...",
    exit: "Exit",
    postureScore: "Posture Score",
    onTarget: "✓ On target",
    needsAdjustment: "Needs adjustment",
    fusing: "Fusing",
    occlusionActive: "Occlusion active",
    allVisible: "All visible",
    staticMode: "Steady",
    flowMode: "Flow mode",
    feedbackHub: "Live feedback",
    correctnessScore: "Correctness Score",
    sequenceFlow: "Flow check",
    occlusionFusing: "Hidden joints",
    active: "Estimating",
    inactive: "Inactive",
    fusingOccluded: "I can't see these, so I'm not checking them:",
    mirroredCoordinates: "",
    systemStatus: "Status",
    detectingPose: "Looking for your pose. Step back until your whole body is in view.",
    safetyCorrection: "Coaching cue",
    alignmentCorrect: "Pose Alignment Correct",
    alignmentCorrectDesc: "Joint angle alignment is correct. Keep breathing steadily.",
    angleDetails: "Joint alignment",
    assumePosePrompt: "Assume a target yoga pose to view real-time joint angle alignments and corrections.",
    targetFor: "Target: {target}° for {pose}",
    diff: "Diff:",
    privacyNote: "Your video stays on your device. Only body-position data is analysed.",
    skip: "Skip",
    relearn: "Learn again",
    rangeSet: "Saved",
    rangeLearning: "Learning…",
    rangeNotSet: "Not set",
    stepIntoView: "Step into view",
    stepBack: "Step back so your feet are in view",
    cantSeeYou: "I can't see you yet. Step back until your whole body is in view.",
    coachWaking: "Waking up your coach… this can take a minute the first time.",
    coachDown: "Can't reach your coach. Check your connection; I'll keep trying.",
    summaryTitle: "Session summary",
    summaryTime: "Time",
    summaryAvg: "Average score",
    summaryBest: "Best pose",
    summaryPoses: "Poses practised",
    summaryNone: "No poses were recognised this time. Next time, step back until your whole body is in view.",
    close: "Close",
    again: "Practise again",
    basicMode: "Basic mode · on your device",
    coachOffline: "You're offline. Basic recognition and tips run on your device; the full coach returns when you're back online.",
    poseEngineOffline: "Open the app once with internet so it can download the pose engine.",
    offlineShort: "Offline",
    appearance: "Appearance",
    themeAuto: "Auto",
    themeLight: "Light",
    themeDark: "Dark",
    notChecking: "Not checking:",
    hiddenWord: "Hidden",
    updateReady: "A new version is ready.",
    updateLater: "It will refresh when you finish this session.",
  },
  hi: {
    tapForGuide: "गाइड देखने के लिए किसी भी आसन पर टैप करें।",
    previewing: "पूर्वावलोकन",
    followMe: "मेरे आसन को फ़ॉलो करें",
    followingYou: "आप जो आसन कर रहे हैं, गाइड वही दिखा रहा है",
    tagGuide: "गाइड",
    readyTitle: "जब आप तैयार हों",
    readyBody: "लगभग दो मीटर पीछे खड़े हों ताकि पूरा शरीर फ्रेम में दिखे, और रोशनी अच्छी हो।",
    startCamera: "कैमरा शुरू करें",
    stopCamera: "रोकें",
    retry: "फिर कोशिश करें",
    poseEngineLoading: "पोज़ इंजन लोड हो रहा है…",
    poseEngineSlow: "पोज़ इंजन देर से लोड हो रहा है। इंटरनेट जाँचें।",
    graphicsTitle: "पोज़ इंजन शुरू नहीं हो सका",
    graphicsDetails: "विवरण",
    graphicsBody: "आपका ब्राउज़र ग्राफ़िक्स एक्सेलेरेशन चालू नहीं कर सका, जो पोज़ इंजन के लिए ज़रूरी है। दूसरे टैब और ऐप बंद करें, बैटरी सेवर बंद करें, हार्डवेयर एक्सेलेरेशन चालू रखें, या इस पेज को Chrome या Safari में खोलें, फिर \"फिर कोशिश करें\" दबाएँ।",
    practice: "अभ्यास",
    poseGuide: "आसन गाइड",
    sessionOverview: "सत्र",
    cameraOff: "कैमरा बंद",
    cameraLive: "कैमरा चालू",
    coaching: "मार्गदर्शन",
    voice: "आवाज़",
    flip: "पलटें",
    poses: "आसन",
    tipDistance: "2 मीटर दूर खड़े हों",
    tipLight: "रोशनी की ओर मुँह रखें",
    tipFrame: "पूरा शरीर फ्रेम में",
    measuredJoints: "मापे गए जोड़",
    tagNow: "अभी",
    lookingForPose: "आसन खोज रहे हैं…",
    installTitle: "AsanaAI इंस्टॉल करें",
    installBody: "ऐप की तरह खोलें: फ़ुल स्क्रीन, ऑफ़लाइन भी।",
    install: "इंस्टॉल",
    offline: "आप ऑफ़लाइन हैं — इंटरनेट जाँचें।",
    jointAlignment: "जोड़ों का संरेखण",
    livePanel: "लाइव फ़ीडबैक",
    appTitle: "असनएआई — स्मार्ट योग कोच",
    newSession: "नया सत्र",
    recognisedAsanas: "पहचानी जाने वाली मुद्राएँ",
    modeFree: "मुक्त मोड",
    modeGuided: "निर्देशित मोड",
    modeFreeHint: "कोई भी मुद्रा करें। उसे पहचानकर उसी का स्कोर दिया जाएगा।",
    modeGuidedHint: "एक मुद्रा चुनें, फिर उसे करें। आपकी जाँच उसी मुद्रा से होगी।",
    targetPose: "लक्ष्य मुद्रा",
    wrongPoseDetected: "यह {detected} जैसा लग रहा है, आपकी चुनी हुई {target} नहीं। स्कोर पाने के लिए {target} में आएं।",
    wrongPoseBadge: "गलत मुद्रा",
    voiceMissing: "इस डिवाइस में इस भाषा की आवाज़ नहीं मिली, इसलिए बोला गया मार्गदर्शन शांत या अंग्रेज़ी में हो सकता है। कैप्शन फिर भी दिखेंगे। डिवाइस की स्पीच सेटिंग में आवाज़ जोड़ें।",
    detectedPose: "पहचानी गई मुद्रा",
    stateHolding: "स्थिर",
    stateTransitioning: "संक्रमण में",
    stateUnrecognized: "अपरिचित मुद्रा",
    universalScore: "सामान्य मुद्रा",
    personalScore: "आपके शरीर के अनुसार",
    digitalTwinProfile: "आपकी गति सीमा",
    activeProfile: "✓ गति सीमा सहेजी गई",
    uncalibratedTwin: "कैमरा शुरू होने पर 15 सेकंड आरामदायक मुद्रा में खड़े रहें। AsanaAI सीखता है कि आपके जोड़ स्वाभाविक रूप से कितना हिलते हैं, ताकि सुझाव आपके शरीर के अनुसार हों।",
    calibratingTwin: "आपकी गति सीमा सीखी जा रही है…",
    apiConfig: "एपीआई कॉन्फ़िगरेशन",
    backendApiUrl: "बैकएंड एपीआई यूआरएल",
    apiUrlHelper: "लोकलहोस्ट या अपने हगिंग फेस स्पेस यूआरएल को इंगित करें।",
    cameraStream: "कैमरा स्ट्रीम",
    swapCamera: "कैमरा बदलें",
    focusMode: "फ़ोकस मोड",
    stopVideo: "वीडियो रोकें",
    startVideo: "वीडियो शुरू करें",
    calibratingProgress: "आपकी गति सीमा सीखी जा रही है",
    stayInView: "आराम से कैमरे के सामने खड़े रहें और साँस लें।",
    cameraInactive: "कैमरा शुरू करें और कोई भी मुद्रा करें। AsanaAI स्वयं पहचान लेगा — कुछ चुनने की ज़रूरत नहीं।",
    alignBody: "कैमरे के साथ अपने शरीर को संरेखित करें...",
    exit: "बाहर निकलें",
    postureScore: "मुद्रा स्कोर",
    onTarget: "✓ सही संरेखण",
    needsAdjustment: "समायोजन की आवश्यकता है",
    fusing: "फ्यूज़िंग",
    occlusionActive: "अस्पष्टता सक्रिय",
    allVisible: "सभी दृश्यमान",
    staticMode: "स्थिर",
    flowMode: "प्रवाह मोड",
    feedbackHub: "रीयल-टाइम फीडबैक हब",
    correctnessScore: "सटीकता स्कोर",
    sequenceFlow: "प्रवाह जाँच",
    occlusionFusing: "छिपे जोड़",
    active: "अनुमान",
    inactive: "निष्क्रिय",
    fusingOccluded: "ये मुझे दिख नहीं रहे, इसलिए इनकी जाँच नहीं कर रहा:",
    mirroredCoordinates: "",
    systemStatus: "सिस्टम की स्थिति",
    detectingPose: "मुद्रा खोजी जा रही है... अपने शरीर को कैमरे के साथ संरेखित करें।",
    safetyCorrection: "सुरक्षा सुधार वॉयस गाइडेंस",
    alignmentCorrect: "आसन संरेखण सही है",
    alignmentCorrectDesc: "जोड़ों का संरेखण बिल्कुल सही है। लगातार सांस लेते रहें।",
    angleDetails: "कोण संरेखण विवरण",
    assumePosePrompt: "वास्तविक समय में जोड़ों के संरेखण और सुधार देखने के लिए एक लक्ष्य योग मुद्रा धारण करें।",
    targetFor: "लक्ष्य: {pose} के लिए {target}°",
    diff: "अंतर:",
    privacyNote: "आपका वीडियो आपके डिवाइस पर ही रहता है। केवल शरीर की स्थिति का डेटा जाँचा जाता है।",
    skip: "छोड़ें",
    relearn: "फिर से सीखें",
    rangeSet: "सहेजी गई",
    rangeLearning: "सीख रहे हैं…",
    rangeNotSet: "तय नहीं",
    stepIntoView: "कैमरे के सामने आएँ",
    stepBack: "थोड़ा पीछे हटें ताकि पैर दिखें",
    cantSeeYou: "आप अभी दिख नहीं रहे। तब तक पीछे हटें जब तक पूरा शरीर दिखे।",
    coachWaking: "आपका कोच जाग रहा है… पहली बार में एक मिनट लग सकता है।",
    coachDown: "कोच से जुड़ नहीं पा रहे। इंटरनेट जाँचें; मैं कोशिश करता रहूँगा।",
    summaryTitle: "सत्र का सारांश",
    summaryTime: "समय",
    summaryAvg: "औसत स्कोर",
    summaryBest: "सबसे अच्छा आसन",
    summaryPoses: "किए गए आसन",
    summaryNone: "इस बार कोई आसन नहीं पहचाना गया। अगली बार पीछे हटें ताकि पूरा शरीर दिखे।",
    close: "बंद करें",
    again: "फिर अभ्यास करें",
    basicMode: "बेसिक मोड · आपके डिवाइस पर",
    coachOffline: "आप ऑफ़लाइन हैं। बेसिक पहचान और सुझाव आपके डिवाइस पर चल रहे हैं; इंटरनेट आने पर पूरा कोच लौट आएगा।",
    poseEngineOffline: "ऐप को एक बार इंटरनेट के साथ खोलें ताकि वह पोज़ इंजन डाउनलोड कर सके।",
    offlineShort: "ऑफ़लाइन",
    appearance: "दिखावट",
    themeAuto: "ऑटो",
    themeLight: "लाइट",
    themeDark: "डार्क",
    notChecking: "जाँच नहीं:",
    hiddenWord: "छिपा",
    updateReady: "नया संस्करण तैयार है।",
    updateLater: "यह सत्र खत्म होने पर अपने आप रीफ़्रेश हो जाएगा।",
  },
  bn: {
    tapForGuide: "গাইড দেখতে যেকোনো আসনে ট্যাপ করুন।",
    previewing: "প্রিভিউ",
    followMe: "আমার আসন অনুসরণ করুন",
    followingYou: "আপনি যে আসন করছেন গাইড সেটাই দেখাচ্ছে",
    tagGuide: "গাইড",
    readyTitle: "আপনি প্রস্তুত হলেই",
    readyBody: "প্রায় দুই মিটার পিছনে দাঁড়ান যাতে পুরো শরীর ফ্রেমে থাকে, আর আলো ভালো হয়।",
    startCamera: "ক্যামেরা চালু করুন",
    stopCamera: "থামান",
    retry: "আবার চেষ্টা করুন",
    poseEngineLoading: "পোজ ইঞ্জিন লোড হচ্ছে…",
    poseEngineSlow: "পোজ ইঞ্জিন লোড হতে দেরি হচ্ছে। ইন্টারনেট দেখুন।",
    graphicsTitle: "পোজ ইঞ্জিন চালু করা যায়নি",
    graphicsDetails: "বিস্তারিত",
    graphicsBody: "আপনার ব্রাউজার গ্রাফিক্স অ্যাক্সিলারেশন চালু করতে পারেনি, যা পোজ ইঞ্জিনের জন্য দরকার। অন্য ট্যাব ও অ্যাপ বন্ধ করুন, ব্যাটারি সেভার বন্ধ করুন, হার্ডওয়্যার অ্যাক্সিলারেশন চালু রাখুন, অথবা পেজটি Chrome বা Safari-তে খুলুন, তারপর \"আবার চেষ্টা করুন\" চাপুন।",
    practice: "অনুশীলন",
    poseGuide: "আসন গাইড",
    sessionOverview: "সেশন",
    cameraOff: "ক্যামেরা বন্ধ",
    cameraLive: "ক্যামেরা চালু",
    coaching: "নির্দেশনা",
    voice: "ভয়েস",
    flip: "ঘোরান",
    poses: "আসন",
    tipDistance: "২ মিটার দূরে দাঁড়ান",
    tipLight: "আলোর দিকে মুখ করুন",
    tipFrame: "পুরো শরীর ফ্রেমে",
    measuredJoints: "পরিমাপ করা জোড়",
    tagNow: "এখন",
    lookingForPose: "আসন খোঁজা হচ্ছে…",
    installTitle: "AsanaAI ইনস্টল করুন",
    installBody: "অ্যাপের মতো খুলুন: ফুল স্ক্রিন, অফলাইনেও চলে।",
    install: "ইনস্টল",
    offline: "আপনি অফলাইনে — ইন্টারনেট দেখুন।",
    jointAlignment: "জোড়ের সারিবদ্ধতা",
    livePanel: "লাইভ ফিডব্যাক",
    appTitle: "আসনএআই — স্মার্ট যোগ কোচ",
    newSession: "নতুন সেশন",
    recognisedAsanas: "চেনা আসনসমূহ",
    modeFree: "মুক্ত মোড",
    modeGuided: "নির্দেশিত মোড",
    modeFreeHint: "যেকোনো আসন করুন। আসনটি শনাক্ত করে সেটির স্কোর দেওয়া হবে।",
    modeGuidedHint: "একটি আসন বেছে নিন, তারপর করুন। সেই আসনের সাথে মিলিয়ে দেখা হবে।",
    targetPose: "লক্ষ্য আসন",
    wrongPoseDetected: "এটি {detected} বলে মনে হচ্ছে, আপনার নির্বাচিত {target} নয়। স্কোর পেতে {target} ভঙ্গিতে আসুন।",
    wrongPoseBadge: "ভুল আসন",
    voiceMissing: "এই ডিভাইসে এই ভাষার কণ্ঠস্বর পাওয়া যায়নি, তাই বলা নির্দেশনা নীরব বা ইংরেজিতে হতে পারে। ক্যাপশন তবুও দেখানো হবে। ডিভাইসের স্পিচ সেটিংসে কণ্ঠস্বর যোগ করুন।",
    detectedPose: "শনাক্ত আসন",
    stateHolding: "স্থির",
    stateTransitioning: "পরিবর্তনে",
    stateUnrecognized: "অচেনা ভঙ্গি",
    universalScore: "সাধারণ ভঙ্গি",
    personalScore: "আপনার শরীর অনুযায়ী",
    digitalTwinProfile: "আপনার গতির সীমা",
    activeProfile: "✓ গতির সীমা সংরক্ষিত",
    uncalibratedTwin: "ক্যামেরা চালু হলে ১৫ সেকেন্ড আরামদায়ক ভঙ্গিতে দাঁড়ান। AsanaAI শেখে আপনার জোড়গুলো স্বাভাবিকভাবে কতটা নড়ে, যাতে পরামর্শ আপনার শরীরের সঙ্গে মেলে।",
    calibratingTwin: "আপনার গতির সীমা শেখা হচ্ছে…",
    apiConfig: "এপিআই কনফিগারেশন",
    backendApiUrl: "ব্যাকএন্ড এপিআই ইউআরএল",
    apiUrlHelper: "লোকালহোস্ট বা আপনার হাগিং ফেস স্পেস ইউআরএল নির্দেশ করুন।",
    cameraStream: "ক্যামেরা স্ট্রীম",
    swapCamera: "ক্যামেরা পরিবর্তন",
    focusMode: "ফোকাস মোড",
    stopVideo: "ভিডিও বন্ধ করুন",
    startVideo: "ভিডিও চালু করুন",
    calibratingProgress: "আপনার গতির সীমা শেখা হচ্ছে",
    stayInView: "আরামে ক্যামেরার সামনে দাঁড়ান এবং শ্বাস নিন।",
    cameraInactive: "ক্যামেরা চালু করে যেকোনো আসন করুন। AsanaAI নিজেই চিনে নেবে — কিছু নির্বাচন করার দরকার নেই।",
    alignBody: "ক্যামেরার সাথে আপনার শরীর সারিবদ্ধ করুন...",
    exit: "প্রস্থান",
    postureScore: "আসন স্কোর",
    onTarget: "✓ সঠিক সারিবদ্ধকরণ",
    needsAdjustment: "সংশোধন প্রয়োজন",
    fusing: "ফিউজিং",
    occlusionActive: "অস্পষ্টতা সক্রিয়",
    allVisible: "সব দৃশ্যমান",
    staticMode: "স্থির",
    flowMode: "প্রবাহ মোড",
    feedbackHub: "রিয়েল-টাইম ফিডব্যাক হাব",
    correctnessScore: "সঠিকতা স্কোর",
    sequenceFlow: "প্রবাহ পরীক্ষা",
    occlusionFusing: "লুকানো জোড়",
    active: "আন্দাজ",
    inactive: "নিষ্ক্রিয়",
    fusingOccluded: "এগুলো দেখা যাচ্ছে না, তাই পরীক্ষা করছি না:",
    mirroredCoordinates: "",
    systemStatus: "সিস্টেমের অবস্থা",
    detectingPose: "আসন শনাক্ত করা হচ্ছে... ক্যামেরার সাথে আপনার শরীর সারিবদ্ধ করুন।",
    safetyCorrection: "সুরক্ষা সংশোধন ভয়েস গাইডেন্স",
    alignmentCorrect: "আসন ভঙ্গি সঠিক আছে",
    alignmentCorrectDesc: "জয়েন্ট অ্যালাইনমেন্ট সঠিক আছে। স্বাভাবিক শ্বাস-প্রশ্বাস বজায় রাখুন।",
    angleDetails: "কোণ সারিবদ্ধকরণ বিবরণ",
    assumePosePrompt: "রিয়েল-টাইম জয়েন্ট অ্যালাইনমেন্ট এবং সংশোধন দেখতে একটি লক্ষ্য যোগাসন অনুশীলন করুন।",
    targetFor: "{pose} এর জন্য লক্ষ্য কোণ {target}°",
    diff: "পার্থক্য:",
    privacyNote: "আপনার ভিডিও আপনার ডিভাইসেই থাকে। শুধু শরীরের অবস্থানের তথ্য বিশ্লেষণ করা হয়।",
    skip: "এড়িয়ে যান",
    relearn: "আবার শিখুন",
    rangeSet: "সংরক্ষিত",
    rangeLearning: "শেখা হচ্ছে…",
    rangeNotSet: "ঠিক করা নেই",
    stepIntoView: "ক্যামেরার সামনে আসুন",
    stepBack: "একটু পিছিয়ে যান যাতে পা দেখা যায়",
    cantSeeYou: "আপনাকে এখনও দেখা যাচ্ছে না। পুরো শরীর দেখা না যাওয়া পর্যন্ত পিছিয়ে যান।",
    coachWaking: "আপনার কোচ জেগে উঠছে… প্রথমবার এক মিনিট লাগতে পারে।",
    coachDown: "কোচের সঙ্গে সংযোগ হচ্ছে না। ইন্টারনেট দেখুন; আমি চেষ্টা চালিয়ে যাব।",
    summaryTitle: "সেশনের সারাংশ",
    summaryTime: "সময়",
    summaryAvg: "গড় স্কোর",
    summaryBest: "সেরা আসন",
    summaryPoses: "অনুশীলন করা আসন",
    summaryNone: "এবার কোনো আসন চেনা যায়নি। পরেরবার পুরো শরীর দেখা যাওয়া পর্যন্ত পিছিয়ে দাঁড়ান।",
    close: "বন্ধ করুন",
    again: "আবার অনুশীলন করুন",
    basicMode: "বেসিক মোড · আপনার ডিভাইসে",
    coachOffline: "আপনি অফলাইনে। বেসিক শনাক্তকরণ ও পরামর্শ আপনার ডিভাইসেই চলছে; ইন্টারনেট ফিরলে পূর্ণ কোচ ফিরে আসবে।",
    poseEngineOffline: "অ্যাপটি একবার ইন্টারনেট সহ খুলুন, যাতে পোজ ইঞ্জিন ডাউনলোড হতে পারে।",
    offlineShort: "অফলাইন",
    appearance: "চেহারা",
    themeAuto: "অটো",
    themeLight: "লাইট",
    themeDark: "ডার্ক",
    notChecking: "পরীক্ষা হচ্ছে না:",
    hiddenWord: "লুকানো",
    updateReady: "নতুন সংস্করণ প্রস্তুত।",
    updateLater: "এই সেশন শেষ হলে নিজে থেকেই রিফ্রেশ হবে।",
  }
};

const PART_LABELS: { [lang: string]: { [feature: string]: string } } = {
  en: { elbow_l: "left elbow", elbow_r: "right elbow", shoulder_l: "left shoulder", shoulder_r: "right shoulder", hip_l: "left hip", hip_r: "right hip", knee_l: "left knee", knee_r: "right knee", ankle_l: "left ankle", ankle_r: "right ankle", trunk_l: "torso", trunk_r: "torso", neck: "neck", hip_abduct_l: "left hip", hip_abduct_r: "right hip" },
  hi: { elbow_l: "बायाँ कोहनी", elbow_r: "दायाँ कोहनी", shoulder_l: "बायाँ कंधा", shoulder_r: "दायाँ कंधा", hip_l: "बायाँ कूल्हा", hip_r: "दायाँ कूल्हा", knee_l: "बायाँ घुटना", knee_r: "दायाँ घुटना", ankle_l: "बायाँ टखना", ankle_r: "दायाँ टखना", trunk_l: "धड़", trunk_r: "धड़", neck: "गर्दन", hip_abduct_l: "बायाँ कूल्हा", hip_abduct_r: "दायाँ कूल्हा" },
  bn: { elbow_l: "বাম কনুই", elbow_r: "ডান কনুই", shoulder_l: "বাম কাঁধ", shoulder_r: "ডান কাঁধ", hip_l: "বাম কোমর", hip_r: "ডান কোমর", knee_l: "বাম হাঁটু", knee_r: "ডান হাঁটু", ankle_l: "বাম গোড়ালি", ankle_r: "ডান গোড়ালি", trunk_l: "ধড়", trunk_r: "ধড়", neck: "ঘাড়", hip_abduct_l: "বাম কোমর", hip_abduct_r: "ডান কোমর" },
};

const JOINT_TRANSLATIONS: {
  [lang: string]: { [joint: string]: string }
} = {
  en: {
    "knee_l": "Left Knee Angle",
    "shoulder_l": "Left Shoulder Angle",
    "knee_r": "Right Knee Angle",
    "shoulder_r": "Right Shoulder Angle",
    "hip_l": "Left Hip Bend",
    "hip_r": "Right Hip Bend",
    "neck": "Neck Extension",
    "trunk_l": "Left Spine Straightness",
    "trunk_r": "Right Spine Straightness",
    "elbow_l": "Left Elbow Extension",
    "elbow_r": "Right Elbow Extension",
    "hip_abduct_l": "Left Hip Abduction",
    "hip_abduct_r": "Right Hip Abduction",
    "ankle_l": "Left Ankle Angle",
    "ankle_r": "Right Ankle Angle"
  },
  hi: {
    "knee_l": "बायां घुटना कोण",
    "shoulder_l": "बायां कंधा कोण",
    "knee_r": "दायां घुटना कोण",
    "shoulder_r": "दायां कंधा कोण",
    "hip_l": "बायां कूल्हा झुकाव",
    "hip_r": "दायां कूल्हा झुकाव",
    "neck": "गर्दन विस्तार",
    "trunk_l": "बायां रीढ़ संरेखण",
    "trunk_r": "दायां रीढ़ संरेखण",
    "elbow_l": "बायां कोहनी विस्तार",
    "elbow_r": "दायां कोहनी विस्तार",
    "hip_abduct_l": "बायां कूल्हा अपवर्तन",
    "hip_abduct_r": "दायां कूल्हा अपवर्तन",
    "ankle_l": "बायां टखना कोण",
    "ankle_r": "दायां टखना कोण"
  },
  bn: {
    "knee_l": "বাম হাঁটু কোণ",
    "shoulder_l": "বাম কাঁধ কোণ",
    "knee_r": "ডান হাঁটু কোণ",
    "shoulder_r": "ডান কাঁধ কোণ",
    "hip_l": "বাম হিপ বাঁক",
    "hip_r": "ডান হিপ বাঁক",
    "neck": "ঘাড়ের প্রসারণ",
    "trunk_l": "বাম মেরুদণ্ড সোজা",
    "trunk_r": "ডান মেরুদণ্ড সোজা",
    "elbow_l": "বাম কনুই প্রসারণ",
    "elbow_r": "ডান কনুই প্রসারণ",
    "hip_abduct_l": "বাম হিপ অপহরণ",
    "hip_abduct_r": "ডান হিপ অপহরণ",
    "ankle_l": "বাম গোড়ালি কোণ",
    "ankle_r": "ডান গোড়ালি কোণ"
  }
};

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


/** The session clock owns its own state, so ticking every second never re-renders the dashboard. */
function SessionTimer({ running }: { running: boolean }) {
  const [sec, setSec] = useState(0);
  useEffect(() => {
    if (!running) return;
    setSec(0);
    const id = setInterval(() => setSec((v) => v + 1), 1000);
    return () => clearInterval(id);
  }, [running]);
  const m = Math.floor(sec / 60).toString().padStart(2, "0");
  const r = (sec % 60).toString().padStart(2, "0");
  return <span className="num">{m}:{r}</span>;
}

/** Collapsible sidebar section: animated, and shows a one-line summary on the header while collapsed. */
function Accordion({ title, open, onToggle, status, children }: {
  title: string; open: boolean; onToggle: () => void; status?: string; children: React.ReactNode;
}) {
  return (
    <div className={`ap-acc ${open ? "" : "closed"}`}>
      <button className="ap-acc-head" aria-expanded={open} onClick={onToggle}>
        <span>{title}</span>
        {!open && status ? <em className="ap-acc-status">{status}</em> : null}
        <ChevronDown size={17} />
      </button>
      <div className="ap-acc-wrap" aria-hidden={!open}>
        <div className="ap-acc-inner">
          <div className="ap-acc-body">{children}</div>
        </div>
      </div>
    </div>
  );
}

interface ScoreRingProps {
  value: number;
  tone: "ok" | "mid" | "low" | "none";
  active: boolean;
}

function ScoreRing({ value, tone, active }: ScoreRingProps) {
  const [shown, setShown] = useState(value);

  useEffect(() => {
    const start = shown;
    if (start === value) return;
    const t0 = performance.now();
    let raf = 0;
    const step = (now: number) => {
      const p = Math.min((now - t0) / 400, 1);
      setShown(Math.round(start + (value - start) * (p * (2 - p))));
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  const r = 44;
  const c = 2 * Math.PI * r;
  const offset = active ? c - (Math.min(100, Math.max(0, value)) / 100) * c : c;
  const color = tone === "ok" ? "var(--ok)" : tone === "mid" ? "var(--amber)" : tone === "low" ? "var(--clay)" : "var(--line-2)";

  return (
    <div className="ap-ring" role="img" aria-label={active ? `Posture score ${value} percent` : "No score yet"}>
      <svg viewBox="0 0 100 100" aria-hidden="true">
        <circle className="track" cx="50" cy="50" r={r} fill="none" strokeWidth="7" />
        <circle className="fill" cx="50" cy="50" r={r} fill="none" strokeWidth="7" stroke={color}
          strokeDasharray={c} strokeDashoffset={offset} strokeLinecap="round" />
      </svg>
      <div className="ap-ring-num">
        {active ? <span className="num">{shown}<small>%</small></span> : <span style={{ color: "var(--ink-3)" }}>—</span>}
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [apiURL, setApiURL] = useState("http://localhost:8000/api");
  const [lang, setLang] = useState<"en" | "hi" | "bn">("en");
  // Two ways to practise. FREE: do any pose, the app recognises it and scores THAT pose (the default).
  // GUIDED: pick a pose first; the app checks you against it and says so when you hold a different one.
  const [practiceMode, setPracticeMode] = useState<"free" | "guided">("free");
  const [targetPose, setTargetPose] = useState<PresetPoseId>("warrior_2");
  useEffect(() => {
    try {
      const m = window.localStorage.getItem("asana.practiceMode");
      if (m === "free" || m === "guided") setPracticeMode(m);
      const t = window.localStorage.getItem("asana.targetPose");
      if (t && POSE_LIBRARY.some((p) => p.id === t)) setTargetPose(t as PresetPoseId);
    } catch { /* storage unavailable (private mode, blocked): the defaults are fine */ }
  }, []);
  useEffect(() => {
    try {
      window.localStorage.setItem("asana.practiceMode", practiceMode);
      window.localStorage.setItem("asana.targetPose", targetPose);
    } catch { /* ignore */ }
  }, [practiceMode, targetPose]);
  useEffect(() => {
    try {
      const saved = window.localStorage.getItem("asana.lang");
      if (saved === "en" || saved === "hi" || saved === "bn") setLang(saved);
      else {
        const nav = (navigator.language || "en").toLowerCase();
        if (nav.startsWith("hi")) setLang("hi");
        else if (nav.startsWith("bn")) setLang("bn");
      }
      const v = window.localStorage.getItem("asana.voice");
      if (v === "off") setSpeechEnabled(false);
    } catch { /* storage unavailable: defaults */ }
  }, []);
  useEffect(() => {
    try { window.localStorage.setItem("asana.lang", lang); } catch { /* ignore */ }
    if (typeof document !== "undefined") document.documentElement.lang = lang;
  }, [lang]);
  // Appearance: follow the system unless the user picked Light/Dark (the choice is applied before first paint in _document).
  const [theme, setTheme] = useState<"auto" | "light" | "dark">("auto");
  useEffect(() => {
    try {
      const t = window.localStorage.getItem("asana.theme");
      if (t === "light" || t === "dark") setTheme(t);
    } catch { /* ignore */ }
  }, []);
  const chooseTheme = (t: "auto" | "light" | "dark") => {
    setTheme(t);
    try {
      if (t === "auto") { window.localStorage.removeItem("asana.theme"); document.documentElement.removeAttribute("data-theme"); }
      else { window.localStorage.setItem("asana.theme", t); document.documentElement.setAttribute("data-theme", t); }
    } catch { /* ignore */ }
  };
  const [langDropOpen, setLangDropOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [speechEnabled, setSpeechEnabled] = useState(true);
  useEffect(() => {
    try { window.localStorage.setItem("asana.voice", speechEnabled ? "on" : "off"); } catch { /* ignore */ }
  }, [speechEnabled]);
  // Installed speech voices. Chrome fills this list asynchronously, so listen for `voiceschanged`.
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>([]);
  useEffect(() => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;
    const synth = window.speechSynthesis;
    const load = () => setVoices(synth.getVoices ? synth.getVoices() : []);
    load();
    synth.addEventListener?.("voiceschanged", load);
    return () => synth.removeEventListener?.("voiceschanged", load);
  }, []);
  const speechTokenRef = useRef(0); // bumped on every announcement; a stale chunk chain stops when it no longer matches
  // True only when the voice list HAS loaded and nothing in it speaks the chosen language (never on an empty, still-loading list).
  const voiceMissingForLang = speechEnabled && isVoiceMissing(voices, lang);
  // Digital Twin Calibration States
  const [calibrationState, setCalibrationState] = useState<"idle" | "calibrating" | "complete">("idle");
  const [calibrationCountdown, setCalibrationCountdown] = useState(5);
  const [calibratedProfile, setCalibratedProfile] = useState<CalibrationProfile | null>(null);
  const calibrationDataRef = useRef<number[][]>([]);
  const calibrationTimerRef = useRef<NodeJS.Timeout | null>(null);

  // Camera Facing Mode States
  const [facingMode, setFacingMode] = useState<"user" | "environment">("user");
  const [hasMultipleCameras, setHasMultipleCameras] = useState(false);
  const videoDevicesRef = useRef<MediaDeviceInfo[]>([]);      // every camera the browser lists (re-read after permission)
  const activeDeviceIdRef = useRef<string | null>(null);      // the camera currently in use
  const facingKnownRef = useRef(false);                       // does the current camera say which way it faces? (phones/tablets yes, most webcams no)
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameIdRef = useRef<number | null>(null);

  // ── Zoom feature ─────────────────────────────────────────
  const [zoomLevel, setZoomLevel] = useState(1);         // current zoom (0.5 – 10)
  const [showZoomBar, setShowZoomBar] = useState(false); // show slider on demand
  const zoomHideTimerRef = useRef<NodeJS.Timeout | null>(null);
  // Pinch gesture tracking
  const pinchStartDistRef = useRef<number | null>(null);
  const pinchStartZoomRef = useRef<number>(1);

  // Dynamic metrics computed in real-time
  const [allCurrentAngles, setAllCurrentAngles] = useState<number[]>(new Array(15).fill(180));

  // MediaPipe state
  const [cameraActive, setCameraActive] = useState(false);
  const [isInitializingCamera, setIsInitializingCamera] = useState(false);
  // Pose-engine lifecycle, so a slow or failed model load is VISIBLE instead of a black screen.
  const [engineState, setEngineState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const engineErrorsRef = useRef(0);
  const graphicsFailedRef = useRef(false);
  const [gfxInfo, setGfxInfo] = useState("");
  const onGraphicsFailureRef = useRef<() => void>(() => {});
  const lastResultAtRef = useRef(0);
  const stageRef = useRef<HTMLDivElement>(null);
  // Per-frame data lives in refs: React state is for what the screen shows, not for 30 Hz inputs.
  const anglesRef = useRef<number[]>(new Array(15).fill(180));
  const lastAnglePublishRef = useRef(0);
  const canvasCssRef = useRef({ w: 0, h: 0, dpr: 1 });
  const videoFrameCbRef = useRef<number | null>(null);
  // Framing: is a body actually in view? (the screen must not keep showing the last pose after someone walks away)
  const [framing, setFraming] = useState<"ok" | "partial" | "none">("ok");
  const framingRef = useRef<"ok" | "partial" | "none">("ok");
  const lastBodySeenRef = useRef(0);
  const framingCandRef = useRef<{ v: "ok" | "partial"; since: number }>({ v: "ok", since: 0 });
  const lastSendRef = useRef(0);
  // Joints the camera cannot actually see (stable, with hysteresis). They are not scored or coached, and are named to the user.
  const visRef = useRef(new VisibilityTracker());
  const hiddenRef = useRef<string[]>([]);
  const hiddenKeyRef = useRef("");
  const [hiddenParts, setHiddenParts] = useState<string[]>([]);
  // Session recap, kept on the device
  const statsRef = useRef<{ poses: { [id: string]: { sec: number; sum: number } }; total: number; startedAt: number }>({ poses: {}, total: 0, startedAt: 0 });
  const liveRef = useRef({ pose: "", score: 0, active: false });
  const [updateReady, setUpdateReady] = useState(false);
  const [summary, setSummary] = useState<null | { total: number; durationSec: number; avg: number; best: string | null; rows: { pose: string; sec: number; avg: number }[] }>(null);

  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fullscreenContainerRef = useRef<HTMLDivElement>(null);
  const poseRef = useRef<any>(null);
  const onPoseResultsRef = useRef<any>(null);
  const inferenceTimesRef = useRef<number[]>([]);
  const hasDowngradedModelRef = useRef(false);
  const lastApiCallTime = useRef<number>(0);
  // Measured real round-trip on the free CPU HF Space is ~1.2-2.2s even for a
  // trivial route (network + HF's own reverse-proxy hop, not our compute).
  // 500ms used to abort every still-in-flight cycle before it could finish,
  // so the ST-GCN sequence buffer never received a completed frame.
  const API_THROTTLE_MS = 500;
  const abortControllerRef = useRef<AbortController | null>(null);
  const isProcessingRef = useRef(false);
  const speechUnlockedRef = useRef(false);
  
  // Dynamic metrics computed in real-time
  const [currentKneeAngle, setCurrentKneeAngle] = useState(180);
  const [currentShoulderAngle, setCurrentShoulderAngle] = useState(0);

  // App Shell Sidebar Drawer Toggle (Mobile)
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Collapsible Groups states
  const [openGroupTwin, setOpenGroupTwin] = useState(true);
  const [openGroupPoseGuide, setOpenGroupPoseGuide] = useState(true);
  const [openGroupSession, setOpenGroupSession] = useState(true);

  // ── Apply zoom (hw first, CSS transform fallback) ─────────
  const applyZoom = (level: number) => {
    const clamped = Math.min(10, Math.max(0.5, level));
    setZoomLevel(clamped);

    // --- Hardware zoom (Chrome on Android / iOS Safari 17.4+) ---
    if (streamRef.current) {
      const track = streamRef.current.getVideoTracks()[0];
      if (track) {
        const caps = track.getCapabilities?.() as any;
        if (caps?.zoom) {
          const hwMin = caps.zoom.min ?? 1;
          const hwMax = caps.zoom.max ?? 1;
          const hwZoom = Math.min(hwMax, Math.max(hwMin, clamped));
          track.applyConstraints({ advanced: [{ zoom: hwZoom } as any] }).catch(() => {});
          return; // hw zoom applied — no CSS transform needed
        }
      }
    }

    // --- CSS canvas transform fallback ---
    // Mirror (scaleX(-1)) only applies to the front camera -- the rear
    // camera must never be flipped, it was being mirrored unconditionally
    // here regardless of which camera was active.
    if (stageRef.current) {
      const mirror = facingMode === "user" ? "scaleX(-1) " : "";
      stageRef.current.style.transform = clamped === 1
        ? `${mirror}scale(1)`.trim()
        : `${mirror}scale(${clamped})`.trim();
    }
  };

  // Auto-detect zoom capability when stream starts
  useEffect(() => {
    if (!cameraActive || !streamRef.current) return;
    // Reset zoom to 1x on new camera start
    setZoomLevel(1);
    if (stageRef.current) {
      stageRef.current.style.transform = facingMode === "user" ? "scaleX(-1)" : "scale(1)";
    }
  }, [cameraActive, facingMode]);

  // Auto-hide zoom bar after 3s of inactivity
  const showZoomBarBriefly = () => {
    setShowZoomBar(true);
    if (zoomHideTimerRef.current) clearTimeout(zoomHideTimerRef.current);
    zoomHideTimerRef.current = setTimeout(() => setShowZoomBar(false), 3000);
  };

  // Pinch-to-zoom handlers
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      pinchStartDistRef.current = Math.hypot(dx, dy);
      pinchStartZoomRef.current = zoomLevel;
      showZoomBarBriefly();
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2 && pinchStartDistRef.current !== null) {
      e.preventDefault();
      const dx = e.touches[0].clientX - e.touches[1].clientX;
      const dy = e.touches[0].clientY - e.touches[1].clientY;
      const dist = Math.hypot(dx, dy);
      const ratio = dist / pinchStartDistRef.current;
      const newZoom = Math.min(10, Math.max(0.5, pinchStartZoomRef.current * ratio));
      applyZoom(newZoom);
      showZoomBarBriefly();
    }
  };

  const handleTouchEnd = () => {
    pinchStartDistRef.current = null;
  };

  // Update slider CSS variable for teal fill progress
  useEffect(() => {
    const slider = document.querySelector('.zoom-slider-vertical') as HTMLInputElement | null;
    if (slider) {
      const min = parseFloat(slider.min) || 0.5;
      const max = parseFloat(slider.max) || 10;
      const percent = ((zoomLevel - min) / (max - min)) * 100;
      slider.style.setProperty('--zoom-fill', `${percent}%`);
    }
  }, [zoomLevel, showZoomBar]);


  // Session timer hook

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
    let timer: any;
    const handler = (e: any) => {
      e.preventDefault();
      setInstallPrompt(e);
      try {
        const dismissed = Number(window.localStorage.getItem("asana.installDismissed") || 0);
        if (Date.now() - dismissed < 7 * 24 * 3600 * 1000) return;
      } catch { /* ignore */ }
      timer = setTimeout(() => setShowInstallBanner(true), 25000); // let people try the app before asking
    };
    window.addEventListener('beforeinstallprompt', handler);
    return () => { window.removeEventListener('beforeinstallprompt', handler); clearTimeout(timer); };
  }, []);

  // Unlock speech on first user gesture (required for iOS Safari)
  useEffect(() => {
    const unlockSpeech = () => {
      if (speechUnlockedRef.current) return;
      if (typeof window !== 'undefined' && window.speechSynthesis) {
        // Speak a silent utterance to unlock the API
        const unlock = new SpeechSynthesisUtterance('');
        unlock.volume = 0;
        window.speechSynthesis.speak(unlock);
        speechUnlockedRef.current = true;
      }
    };
    window.addEventListener('touchstart', unlockSpeech, { once: true });
    window.addEventListener('click', unlockSpeech, { once: true });
    return () => {
      window.removeEventListener('touchstart', unlockSpeech);
      window.removeEventListener('click', unlockSpeech);
    };
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

  // MediaPipe reports a missing WebGL context with window.alert(). Route exactly that message to our themed panel;
  // every other alert is left alone.
  useEffect(() => {
    const original = window.alert;
    window.alert = function (msg?: any) {
      if (/webgl/i.test(String(msg ?? ""))) { onGraphicsFailureRef.current(); return; }
      return original.call(window, msg);
    };
    return () => { window.alert = original; };
  }, []);

  // Count the cameras. Browsers hide the list until camera permission is granted (a phone reports ONE unnamed camera before
  // that), so this runs again right after the camera starts and whenever one is plugged in or removed.
  const refreshCameras = async () => {
    try {
      if (!navigator.mediaDevices?.enumerateDevices) return;
      const cams = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
      videoDevicesRef.current = cams;
      setHasMultipleCameras(cams.length > 1);
    } catch (err) {
      console.error("Enumerate devices failed:", err);
    }
  };
  useEffect(() => {
    if (typeof window === "undefined" || !navigator.mediaDevices) return;
    refreshCameras();
    navigator.mediaDevices.addEventListener?.("devicechange", refreshCameras);
    return () => navigator.mediaDevices.removeEventListener?.("devicechange", refreshCameras);
  }, []);

  // Listen to native fullscreen change event (handles Esc key automatically)
  useEffect(() => {
    const onFSChange = () => {
      if (!document.fullscreenElement && !(document as any).webkitFullscreenElement) {
        setIsFullscreen(false);
      }
    };
    document.addEventListener('fullscreenchange', onFSChange);
    document.addEventListener('webkitfullscreenchange', onFSChange);
    return () => {
      document.removeEventListener('fullscreenchange', onFSChange);
      document.removeEventListener('webkitfullscreenchange', onFSChange);
    };
  }, []);

  const calibrationStateRef = useRef<"idle" | "calibrating" | "complete">("idle");

  const updateCalibrationState = (state: "idle" | "calibrating" | "complete") => {
    setCalibrationState(state);
    calibrationStateRef.current = state;
  };

  const announceTTS = (text: string) => {
    if (!speechEnabled || typeof window === 'undefined' || !window.speechSynthesis) return;
    
    // Unpause / reset speech synthesis on desktop
    window.speechSynthesis.cancel();
    if (window.speechSynthesis.paused) {
      window.speechSynthesis.resume();
    }

    // Small delay to allow cancel/resume to register in browser engine
    const token = ++speechTokenRef.current;
    setTimeout(() => {
      const synth = window.speechSynthesis;
      const available = voices.length ? voices : (synth.getVoices ? synth.getVoices() : []);
      const voice = pickVoice(available, lang);
      // Chrome silently stops a long utterance after ~15 s, so chain short ones. The splitter understands the Hindi/Bengali
      // danda (। ॥) as well as . ! ?, and falls back to word boundaries, so no language gets a single un-split block.
      const chunks = splitForSpeech(text, 180);
      let idx = 0;
      const speakNext = () => {
        if (token !== speechTokenRef.current || idx >= chunks.length) return; // a newer message took over, or done
        const u = new SpeechSynthesisUtterance(chunks[idx]);
        u.lang = utteranceLang(lang);
        if (voice) u.voice = voice;
        u.rate = 0.92;   // Slightly slower for yoga instruction clarity
        u.pitch = 1.0;
        u.volume = 1.0;
        u.onend = () => { idx++; speakNext(); };
        synth.speak(u);
      };
      speakNext();
    }, 50);
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
      ? "आइए आपकी गति की सीमा सीखें। आराम से कैमरे के सामने खड़े हों और गहरी साँस लें।"
      : lang === "bn"
      ? "আসুন আপনার গতির সীমা শিখি। আরামে ক্যামেরার সামনে দাঁড়ান এবং গভীর শ্বাস নিন।"
      : "Let's learn your range of motion. Stand comfortably in view and take a deep breath.";
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
    const data = calibrationDataRef.current;
    // If the person was not in view, there is nothing to learn from: do not pretend we calibrated.
    if (data.length < 20) {
      updateCalibrationState("idle");
      return;
    }
    updateCalibrationState("complete");
    
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
    try { window.localStorage.setItem("asana.range.v1", JSON.stringify({ profile, savedAt: Date.now() })); } catch { /* ignore */ }

    const completeText = lang === "hi"
      ? "आपकी गति की सीमा सहेज ली गई है। अब अभ्यास शुरू करें।"
      : lang === "bn"
      ? "আপনার গতির সীমা সংরক্ষিত হয়েছে। এবার অনুশীলন শুরু করুন।"
      : "All set. Your range of motion is saved. You can begin practising.";
    announceTTS(completeText);
  };

  const skipCalibration = () => {
    if (calibrationTimerRef.current) {
      clearInterval(calibrationTimerRef.current);
      calibrationTimerRef.current = null;
    }
    updateCalibrationState("idle");
  };

  const loadSavedRange = (): CalibrationProfile | null => {
    try {
      const raw = window.localStorage.getItem("asana.range.v1");
      if (!raw) return null;
      const parsed = JSON.parse(raw);
      const prof = parsed?.profile;
      if (prof && FEATURE_NAMES_ORDER.every((j) => prof[j] && typeof prof[j].min === "number")) return prof as CalibrationProfile;
    } catch { /* corrupt or unavailable: learn again */ }
    return null;
  };

  const relearnRange = () => {
    try { window.localStorage.removeItem("asana.range.v1"); } catch { /* ignore */ }
    setCalibratedProfile(null);
    if (cameraActive) startCalibration();
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
    correctionText,
    correctionIsSafe,
    lastEfficacy,
    motionState,
    poseMismatch,
    personalCorrectness,
    deviations,
    predictionTimestamp,
    processFrame,
    pushSequenceFrame,
    coachSource,
    coachReason,
    resetPipeline
  } = useYogaPipeline({
    language: lang,
    groqApiKey: undefined,
    calibrationProfile: calibratedProfile || undefined,
    correctnessThreshold: 0.75,
    targetPose: practiceMode === "guided" ? targetPose : null,
  });

  const setFramingSafe = (v: "ok" | "partial" | "none") => {
    if (framingRef.current === v) return;
    framingRef.current = v;
    setFraming(v);
    if (v === "none") {
      resetPipeline(); // nobody there: clear the stale pose, score and tips
      visRef.current.reset(); hiddenRef.current = []; hiddenKeyRef.current = ""; setHiddenParts([]);
    }
  };
  const noteBody = (lm: any[]) => {
    const now = performance.now();
    const vis = (i: number) => lm[i]?.visibility ?? 0;
    const core = (vis(11) + vis(12) + vis(23) + vis(24)) / 4;
    if (core < 0.4) return; // landmarks exist but the body is barely visible: treat as not seen
    lastBodySeenRef.current = now;
    const feet = (vis(27) + vis(28)) / 2;
    const cand: "ok" | "partial" = feet < 0.4 ? "partial" : "ok";
    if (framingRef.current === "none") { framingCandRef.current = { v: cand, since: now }; setFramingSafe(cand); return; }
    if (framingCandRef.current.v !== cand) framingCandRef.current = { v: cand, since: now };
    else if (now - framingCandRef.current.since >= 1200) setFramingSafe(cand);
  };
  useEffect(() => {
    if (!cameraActive) { framingRef.current = "ok"; setFraming("ok"); return; }
    lastBodySeenRef.current = performance.now() + 1500; // grace while the person steps into view
    const id = setInterval(() => {
      if (performance.now() - lastBodySeenRef.current > 1500) setFramingSafe("none");
    }, 500);
    return () => clearInterval(id);
  }, [cameraActive]);

  // A new version was installed in the background: reload when nothing is in progress, otherwise offer a refresh button.
  useEffect(() => {
    const onUpdate = () => {
      if (!cameraActive && !summary) window.setTimeout(() => window.location.reload(), 600);
      else setUpdateReady(true);
    };
    window.addEventListener("asana-update-ready", onUpdate);
    return () => window.removeEventListener("asana-update-ready", onUpdate);
  }, [cameraActive, summary]);
  // ...and if it was deferred, refresh as soon as the session ends.
  useEffect(() => {
    if (updateReady && !cameraActive && !summary) window.setTimeout(() => window.location.reload(), 600);
  }, [updateReady, cameraActive, summary]);

  // Keep the screen awake while practising: a phone that dims or locks mid-pose ends the session.
  useEffect(() => {
    if (!cameraActive) return;
    let lock: any = null;
    let stopped = false;
    const acquire = async () => {
      try {
        if (!stopped && "wakeLock" in navigator && document.visibilityState === "visible") {
          lock = await (navigator as any).wakeLock.request("screen");
        }
      } catch { /* not supported or refused: nothing to do */ }
    };
    acquire();
    const onVis = () => { if (document.visibilityState === "visible") acquire(); };
    document.addEventListener("visibilitychange", onVis);
    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", onVis);
      lock?.release?.().catch?.(() => {});
    };
  }, [cameraActive]);

  // Session recap: one sample per second while a pose is recognised and scored.
  useEffect(() => {
    if (!cameraActive) return;
    statsRef.current = { poses: {}, total: 0, startedAt: Date.now() };
    const id = setInterval(() => {
      const l = liveRef.current;
      if (!l.active || !l.pose) return;
      const e = statsRef.current.poses[l.pose] || (statsRef.current.poses[l.pose] = { sec: 0, sum: 0 });
      e.sec += 1; e.sum += l.score; statsRef.current.total += 1;
    }, 1000);
    return () => clearInterval(id);
  }, [cameraActive]);

  // Free-form practice: there is no selected target pose any more. The app
  // detects whatever the user is actually doing and analyses THAT, so the
  // old target-vs-detected mismatch guard is gone entirely.
  //
  // `guidePose` drives the reference panel (photo, cues, target angles). It
  // follows the DETECTED pose, and deliberately latches onto the last
  // recognised pose while the user is mid-transition or briefly
  // unrecognised -- otherwise the whole panel would blank out every time
  // they moved between postures, which is visually jarring.
  const [guidePose, setGuidePose] = useState<string>("mountain_pose");
  // FREE mode: the user can pin the guide to any pose to browse it; null = follow the pose being detected.
  const [browsePose, setBrowsePose] = useState<string | null>(null);
  useEffect(() => {
    if (practiceMode === "guided") {
      setGuidePose(targetPose);
      return;
    }
    if (browsePose && POSE_GUIDE[browsePose]) {
      setGuidePose(browsePose);
      return;
    }
    if (activePose && activePose !== "transition/unknown" && POSE_GUIDE[activePose]) {
      setGuidePose(activePose);
    }
  }, [activePose, practiceMode, targetPose, browsePose]);

  const isTransitioning = motionState === "transitioning";
  const isUnrecognized = motionState === "unrecognized" || activePose === "transition/unknown";
  // GUIDED only: a recognised pose that is not the chosen one. The form score describes the DETECTED pose, so showing it
  // (or coaching it) as if it were the target would be misleading. Re-checked live: the hook's flag is only recomputed once
  // per ~10 s prediction cycle, but the target or the detection can change sooner.
  const showMismatch = practiceMode === "guided" && poseMismatch && activePose !== targetPose && !isTransitioning;
  const displayCorrectionText = showMismatch
    ? TRANSLATIONS[lang].wrongPoseDetected
        .replace("{detected}", getSanskritName(activePose, lang))
        .replace(/{target}/g, getSanskritName(targetPose, lang))
    : correctionText;
  // While moving, a form score is meaningless (it describes a held shape), so
  // don't show a number that will swing wildly mid-flow.
  const effectiveCorrectness = isTransitioning || showMismatch ? 0 : correctness;
  // Personalised score, when the user has calibrated their Digital Twin.
  const effectivePersonalCorrectness = isTransitioning ? null : personalCorrectness;

  // Audio speech synthesis loop — tied to the pipeline's own ~10s prediction
  // cadence (predictionTimestamp only changes once per real classification
  // cycle) so spoken feedback never gets ahead of what's actually been re-analyzed.
  useEffect(() => {
    if (speechEnabled && displayCorrectionText && predictionTimestamp > 0) {
      announceTTS(displayCorrectionText);
    }
  }, [predictionTimestamp, speechEnabled, lang]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && window.innerWidth < 768) setSidebarOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && window.speechSynthesis?.paused) {
        window.speechSynthesis.resume();
      }
      // iOS/Android pause the video when the page is backgrounded; bring the picture back.
      const v = videoRef.current;
      if (document.visibilityState === 'visible' && v && v.srcObject && v.paused) v.play().catch(() => {});
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);


  // Sidebar starts open on wide screens, as a drawer on phones
  useEffect(() => {
    if (typeof window !== "undefined") {
      if (window.innerWidth >= 768) {
        setSidebarOpen(true);
      } else {
        setSidebarOpen(false);
      }
    }
  }, []);

  // Initialize MediaPipe Pose Model. Idempotent and independent of React state timing: it checks the
  // global directly, so calling it from the frame loop picks the engine up the moment its script arrives
  // (previously, pressing Start before the script finished loading left the engine uninitialised for the
  // whole session and the camera view stayed black).
  const initMediaPipe = (): boolean => {
    if (poseRef.current) return true;
    if (graphicsFailedRef.current) return false;
    const PoseClass = typeof window !== "undefined" ? (window as any).Pose : undefined;
    if (!PoseClass) return false;
    try {
      const pose = new PoseClass({
        locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`
      });
      // Phones start on the Lite model: the first frames load 3-4x faster and 15+ fps is what matters.
      const coarse = typeof window !== "undefined" && window.matchMedia?.("(pointer: coarse)").matches;
      pose.setOptions({
        modelComplexity: coarse ? 0 : 1,
        smoothLandmarks: true,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
        // MediaPipe's own selfieMode default mirrors the image AND landmark
        // x-coordinates internally, independent of and inconsistent with the
        // CSS-level mirror this app already applies conditionally for the
        // front camera only. Forcing this off makes the CSS mirror the single
        // source of truth for display, and keeps landmark coordinates (and
        // therefore angle-based pose classification) always true-to-camera
        // regardless of which camera is active.
        selfieMode: false
      });
      pose.onResults((results: any) => {
        lastResultAtRef.current = performance.now();
        engineErrorsRef.current = 0;
        setEngineState((s) => (s === "ready" ? s : "ready"));
        onPoseResultsRef.current?.(results);
      });
      poseRef.current = pose;
      return true;
    } catch (e) {
      console.error("MediaPipe init failed:", e);
      return false;
    }
  };

  // Plain-language explanation for each way getUserMedia can fail (no alert() dialogs).
  const describeCameraError = (err: any): string => {
    const name = err?.name || "";
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia)
      return "This browser can't access the camera here. Open the site over HTTPS in Chrome or Safari.";
    if (name === "NotAllowedError" || name === "SecurityError")
      return "Camera permission was blocked. Allow camera access for this site in your browser settings, then tap Retry.";
    if (name === "NotFoundError" || name === "OverconstrainedError")
      return "No camera was found on this device.";
    if (name === "NotReadableError" || name === "AbortError")
      return "The camera is busy in another app or tab. Close it and tap Retry.";
    return "The camera could not be started. Tap Retry, or reload the page.";
  };

  // Start Live Webcam Video Loop (Custom Stream Implementation for swap/facingMode support)
  const startCamera = async (mode = facingMode, deviceId?: string) => {
    setCameraError(null);
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
      setCameraError(describeCameraError(null));
      return;
    }

    graphicsFailedRef.current = false;

    // Start loading the pose engine now; if its script has not arrived yet the frame loop keeps retrying,
    // so the user always SEES the camera and a clear "loading" state instead of a black box.
    initMediaPipe();
    setEngineState("loading");
    lastResultAtRef.current = 0;
    engineErrorsRef.current = 0;

    setIsInitializingCamera(true);
    try {
      // Clean up previous streams and request frames
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (animationFrameIdRef.current) {
        cancelAnimationFrame(animationFrameIdRef.current);
      }
      if (videoFrameCbRef.current != null) {
        (videoRef.current as any)?.cancelVideoFrameCallback?.(videoFrameCbRef.current);
        videoFrameCbRef.current = null;
      }

      // Phones: a smaller stream is decoded, copied and analysed faster, and MediaPipe resizes to ~256px anyway.
      const coarse = !!window.matchMedia?.("(pointer: coarse)").matches;
      const wanted: MediaStreamConstraints = {
        audio: false,
        video: {
          ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: { ideal: mode } }),
          width: { ideal: coarse ? 960 : 1280 },
          height: { ideal: coarse ? 720 : 720 },
          frameRate: { ideal: 30, max: 30 }
        }
      };
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia(wanted);
      } catch (e: any) {
        // Some devices reject the preferred size/frame rate; any working camera beats none.
        if (e?.name === "OverconstrainedError" || e?.name === "TypeError") {
          stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        } else {
          throw e;
        }
      }
      streamRef.current = stream;

      const videoElement = videoRef.current;
      if (!videoElement) {
        stream.getTracks().forEach(t => t.stop());
        return;
      }

      videoElement.srcObject = stream;
      videoElement.muted = true;
      videoElement.playsInline = true;
      videoElement.setAttribute("playsinline", "true");
      try {
        await videoElement.play();
      } catch (e: any) {
        // A play() interrupted by a newer load (e.g. camera swap) is harmless.
        if (e?.name !== "AbortError") throw e;
      }

      setCameraActive(true);

      // Learn which camera we really got (and which way it faces) so Flip can pick the right next one.
      const track = stream.getVideoTracks()[0];
      const got = track?.getSettings?.() ?? {};
      const labelFacing = /front|user|facetime|selfie/i.test(track?.label || "") ? "user" : /back|rear|environment/i.test(track?.label || "") ? "environment" : undefined;
      const facing = (got.facingMode as "user" | "environment" | undefined) || labelFacing;
      activeDeviceIdRef.current = got.deviceId || deviceId || null;
      facingKnownRef.current = !!facing;
      setFacingMode(facing || (deviceId ? "user" : mode));
      refreshCameras();   // permission is granted now: the real camera list (with names) is available

      // Reset performance tracking for the new camera session
      inferenceTimesRef.current = [];
      hasDowngradedModelRef.current = false;

      // Start custom rendering loop tick to feed MediaPipe. This self-paces
      // to however fast the device can actually run inference (it only
      // schedules the next frame once the previous one finishes), so weak
      // devices never queue up a growing backlog of frames. Measure the first
      // ~20 frames after camera start and drop to the lighter "Lite" model
      // automatically if the device can't keep up.
      // The picture itself is the <video> element, so the user sees themselves
      // even while the engine is still loading or has failed.
      const scheduleNext = () => {
        const v: any = videoRef.current;
        if (!v || !stream.active) return;
        if (typeof v.requestVideoFrameCallback === "function") {
          videoFrameCbRef.current = v.requestVideoFrameCallback(() => { tick(); });
        } else {
          animationFrameIdRef.current = requestAnimationFrame(tick);
        }
      };
      const tick = async () => {
        if (!stream.active || !videoRef.current) return;
        const v = videoRef.current;
        const minGap = coarse ? 45 : 0; // phones: at most ~22 analyses/s, plenty for yoga and far kinder to the battery
        if (v.readyState >= 2 && v.videoWidth > 0 && (poseRef.current || initMediaPipe()) && performance.now() - lastSendRef.current >= minGap) {
          lastSendRef.current = performance.now();
          try {
            const t0 = performance.now();
            await poseRef.current.send({ image: v });
            const elapsed = performance.now() - t0;

            if (!hasDowngradedModelRef.current) {
              inferenceTimesRef.current.push(elapsed);
              if (inferenceTimesRef.current.length >= 20) {
                const avg = inferenceTimesRef.current.reduce((a, b) => a + b, 0) / inferenceTimesRef.current.length;
                if (avg > 120) {
                  // Device is struggling (< ~8fps) — switch to the Lite model
                  poseRef.current.setOptions({
                    modelComplexity: 0,
                    smoothLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                    selfieMode: false
                  });
                }
                // NOTE: an auto-upgrade to modelComplexity 2 ("Heavy") for fast
                // devices was tried here and reverted -- it was a one-time,
                // irreversible decision based on only the first 20 frames at
                // startup, with no way back down if "Heavy" turned out too slow
                // under sustained real-world load. Not worth the lag risk for
                // a speculative accuracy gain.
                hasDowngradedModelRef.current = true;
              }
            }
          } catch (e) {
            // A throw here used to kill the loop silently (black screen forever). Back off and keep trying.
            engineErrorsRef.current += 1;
            console.error("Pose engine error:", e);
            if (engineErrorsRef.current >= 3) setEngineState("error");
            await new Promise((r) => setTimeout(r, 300 * Math.min(engineErrorsRef.current, 6)));
          }
        }
        scheduleNext();
      };
      scheduleNext();

      // The camera can vanish mid-session (unplugged, taken over by another app): say so instead of freezing.
      stream.getVideoTracks().forEach((t) => {
        t.onended = () => {
          if (streamRef.current !== stream) return;
          stopCamera();
          setCameraError("The camera was disconnected or is being used by another app. Tap Retry to resume.");
        };
      });

      // If no result ever arrives, say so (slow network, blocked CDN) rather than leave the user guessing.
      window.setTimeout(() => {
        if (stream.active && lastResultAtRef.current === 0) setEngineState("error");
      }, 30000);

      // Learn the user's range of motion only once; later sessions reuse it (they can re-learn from the sidebar).
      const saved = loadSavedRange();
      if (saved) {
        setCalibratedProfile(saved);
        updateCalibrationState("complete");
      } else {
        startCalibration();
      }

    } catch (err) {
      console.error("Camera access failed:", err);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(t => t.stop());
        streamRef.current = null;
      }
      setCameraActive(false);
      setEngineState("idle");
      setCameraError(describeCameraError(err));
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
    if (videoFrameCbRef.current != null) {
      (videoRef.current as any)?.cancelVideoFrameCallback?.(videoFrameCbRef.current);
      videoFrameCbRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (calibrationTimerRef.current) {
      clearInterval(calibrationTimerRef.current);
      calibrationTimerRef.current = null;
    }
    const c = canvasRef.current;
    c?.getContext("2d")?.clearRect(0, 0, c.width, c.height);
    setCameraActive(false);
    setEngineState("idle");
    setCameraError(null);
    updateCalibrationState("idle");
    setCalibratedProfile(null);
    resetPipeline();
  };

  // Release the engine (and its WebGL context), stop the camera and show the themed message with a Retry.
  const releasePose = () => {
    try { Promise.resolve(poseRef.current?.close?.()).catch(() => {}); } catch { /* already gone */ }
    poseRef.current = null;
  };
  const failGraphics = () => {
    if (graphicsFailedRef.current) return;
    graphicsFailedRef.current = true;
    releasePose();
    stopCamera();
    setGfxInfo(graphicsReport());
    setCameraError(GRAPHICS_ERR);
  };
  onGraphicsFailureRef.current = failGraphics;

  // The user pressed Stop: show a short recap if they actually practised something.
  const endSession = () => {
    const st = statsRef.current;
    const durationSec = Math.round((Date.now() - st.startedAt) / 1000);
    const rows = Object.entries(st.poses).map(([pose, e]) => ({ pose, sec: e.sec, avg: e.sum / Math.max(1, e.sec) })).sort((a, b) => b.sec - a.sec);
    const avg = st.total > 0 ? rows.reduce((t, r) => t + r.avg * r.sec, 0) / st.total : 0;
    const best = rows.filter((r) => r.sec >= 5).sort((a, b) => b.avg - a.avg)[0]?.pose ?? null;
    stopCamera();
    if (durationSec >= 20) setSummary({ total: st.total, durationSec, avg, best, rows });
  };

  // Flip: phones/tablets switch front <-> back; laptops and desktops (whose cameras do not report a direction) step to the next camera.
  const toggleCamera = async () => {
    const cams = videoDevicesRef.current;
    const before = activeDeviceIdRef.current;
    const stepToNext = () => {
      const i = cams.findIndex((d) => d.deviceId === activeDeviceIdRef.current);
      const next = cams[(i + 1) % cams.length];
      if (next?.deviceId) return startCamera("user", next.deviceId);
    };
    if (facingKnownRef.current) {
      await startCamera(facingMode === "user" ? "environment" : "user");
      // A camera that claims a direction but ignored the request (some laptops): fall back to stepping through the list.
      if (cams.length > 1 && activeDeviceIdRef.current && activeDeviceIdRef.current === before) await stepToNext();
    } else if (cams.length > 1) {
      await stepToNext();
    }
  };

  const enterFullscreen = async () => {
    const el = fullscreenContainerRef.current;
    if (!el) return;
    try {
      if (el.requestFullscreen) await el.requestFullscreen();
      else if ((el as any).webkitRequestFullscreen) await (el as any).webkitRequestFullscreen();
      else if ((el as any).mozRequestFullScreen) await (el as any).mozRequestFullScreen();
      setIsFullscreen(true);
      // Lock to landscape on mobile if supported
      if (screen.orientation && (screen.orientation as any).lock) {
        try { await (screen.orientation as any).lock('landscape'); } catch(_) {}
      }
    } catch (err) {
      // Fallback to CSS fullscreen if API fails (e.g. iOS Safari)
      setIsFullscreen(true);
    }
  };

  const exitFullscreen = async () => {
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
      } else if ((document as any).webkitFullscreenElement) {
        await (document as any).webkitExitFullscreen();
      }
      if (screen.orientation && (screen.orientation as any).unlock) {
        try { (screen.orientation as any).unlock(); } catch(_) {}
      }
    } catch(_) {}
    setIsFullscreen(false);
  };

  // Cleanup camera on unmount
  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  // Keep the overlay canvas matched to its box at the device's pixel density (crisp skeleton on phones).
  useEffect(() => {
    const resizeCanvas = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 3);
      const w = canvas.clientWidth, h = canvas.clientHeight;
      if (w > 0 && h > 0) {
        canvas.width = Math.round(w * dpr);
        canvas.height = Math.round(h * dpr);
        canvasCssRef.current = { w, h, dpr: canvas.width / w };
      }
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    window.addEventListener('orientationchange', resizeCanvas);
    const ro = typeof ResizeObserver !== "undefined" && canvasRef.current ? new ResizeObserver(resizeCanvas) : null;
    if (ro && canvasRef.current) ro.observe(canvasRef.current);
    return () => {
      window.removeEventListener('resize', resizeCanvas);
      window.removeEventListener('orientationchange', resizeCanvas);
      ro?.disconnect();
    };
  }, []);

  // MediaPipe Result processing callback
  const onPoseResults = (results: any) => {
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    const canvasCtx = canvasElement.getContext("2d");
    if (!canvasCtx) return;

    // The <video> element is the picture; this canvas only carries the skeleton overlay. Geometry mirrors
    // the video's object-fit: contain letterbox (native aspect ratio preserved, never stretched) so the
    // skeleton lands exactly on the body. Drawn in CSS pixels, scaled up to the device pixel density.
    const geo = canvasCssRef.current;
    const dpr = geo.w > 0 ? geo.dpr : 1;
    const cw = geo.w || canvasElement.width;
    const ch = geo.h || canvasElement.height;
    canvasCtx.save();
    canvasCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    canvasCtx.clearRect(0, 0, cw, ch);
    const vid = videoRef.current;
    const srcW = vid?.videoWidth || results.image?.videoWidth || results.image?.width || cw;
    const srcH = vid?.videoHeight || results.image?.videoHeight || results.image?.height || ch;
    const fitScale = Math.min(cw / srcW, ch / srcH);
    const fitW = srcW * fitScale;
    const fitH = srcH * fitScale;
    const fitX = (cw - fitW) / 2;
    const fitY = (ch - fitH) / 2;

    if (results.poseLandmarks) {
      noteBody(results.poseLandmarks);
      // 1. Draw joints skeleton overlay (with clinical palette aesthetics),
      // mapped through the same letterbox transform as the background frame
      drawSkeletonOverlay(canvasCtx, results.poseLandmarks, fitX, fitY, fitW, fitH);

      // 2. Format landmarks for API pipeline: [33 joints, [x, y, z, visibility]]
      const rawLandmarks = results.poseLandmarks.map((pt: any) => [
        pt.x, 
        pt.y, 
        pt.z, 
        pt.visibility || 0.0
      ]);

      // 2b. ST-GCN input: every camera result, at the camera's own rate (the API loop below is throttled to ~0.5-2 fps)
      pushSequenceFrame(rawLandmarks);

      // 2c. Which joints can the camera really see? (hidden ones are not scored or coached)
      const hiddenNow = visRef.current.update(rawLandmarks);
      hiddenRef.current = hiddenNow;
      const hiddenKey = hiddenNow.join(",");
      if (hiddenKey !== hiddenKeyRef.current) {
        hiddenKeyRef.current = hiddenKey;
        setHiddenParts(hiddenNow);
      }

      // 3. Compute client-side angles for feature list (15 biomechanical features)
      const points = results.poseLandmarks.map((pt: any) => ({
        x: pt.x,
        y: pt.y,
        z: pt.z
      }));
      const angles = extractAnglesFromLandmarks(points);

      // 3b. Second independent signal: MediaPipe's poseWorldLandmarks
      // (metric-scale, separately calibrated 3D coordinates, distinct from
      // the image-normalized poseLandmarks above). Kept as genuine depth
      // (zeroZ=false) and sent alongside the 2D angles so the backend's
      // hybrid_classify can use it as a 3-way tiebreak vote. Always present
      // whenever poseLandmarks is (same detection pass), but guarded anyway.
      let worldAngles: number[] | undefined;
      if (results.poseWorldLandmarks) {
        const worldPoints = results.poseWorldLandmarks.map((pt: any) => ({
          x: pt.x,
          y: pt.y,
          z: pt.z,
        }));
        worldAngles = extractAnglesFromLandmarks(worldPoints, false);
      }

      // The overlay reads the ref every frame; the numbers on screen are published at most 4x/second
      // (re-rendering the whole dashboard 30x/second is what made the app feel laggy).
      anglesRef.current = angles;
      const tNow = performance.now();
      if (tNow - lastAnglePublishRef.current > 250) {
        lastAnglePublishRef.current = tNow;
        setAllCurrentAngles(angles);
        setCurrentKneeAngle(Math.round(angles[6])); // Left Knee
        setCurrentShoulderAngle(Math.round(angles[2])); // Left Shoulder
      }

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
      // Skip firing a new cycle while the previous one is still in flight,
      // rather than aborting it — real round-trip latency on the free CPU
      // Space (~1.2-2.2s) is well over this 500ms tick, so aborting on every
      // tick was cancelling every occlusion-recovery call before it could
      // complete, which meant the ST-GCN sequence buffer never filled.
      if (isProcessingRef.current) {
        canvasCtx.restore();
        return;
      }
      lastApiCallTime.current = now;
      isProcessingRef.current = true;

      if (!abortControllerRef.current) {
        abortControllerRef.current = new AbortController();
      }

      processFrame(rawLandmarks, angles, worldAngles, hiddenRef.current).finally(() => {
        isProcessingRef.current = false;
      });
    } else {
      // Clear angles when body is out of frame
      setCurrentKneeAngle(180);
      setCurrentShoulderAngle(0);
    }
    canvasCtx.restore();
  };

  useEffect(() => {
    onPoseResultsRef.current = onPoseResults;
  }, [onPoseResults]);

  // Drawing method for Canvas Overlay
  const drawSkeletonOverlay = (
    ctx: CanvasRenderingContext2D,
    landmarks: any[],
    fitX: number,
    fitY: number,
    fitW: number,
    fitH: number
  ) => {
    const toX = (nx: number) => fitX + nx * fitW;
    const toY = (ny: number) => fitY + ny * fitH;

    const FEATURE_NAMES_ORDER = [
      "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
      "hip_l", "hip_r", "knee_l", "knee_r",
      "ankle_l", "ankle_r", "trunk_l", "trunk_r",
      "neck", "hip_abduct_l", "hip_abduct_r"
    ];

    const getJointStatus = (joint: string) => {
      // Nothing meaningful to show mid-transition: the body is moving, so
      // every joint would flicker red/green frame to frame.
      if (motionState === "transitioning") return "neutral";
      if (activePose === "transition/unknown") return "neutral";

      // Prefer the model's own per-joint deviation (degrees off the pose's
      // expected band, already adjusted for the user's calibrated range when
      // a Digital Twin profile exists). This covers all 15 joints, whereas
      // POSE_TARGET_ANGLES only defines a handful per pose -- so the skeleton
      // now shows the user exactly which joints are off, not just the 3 we
      // happened to hand-author cues for.
      const dev = deviations[joint];
      if (dev !== undefined) {
        if (dev > 15) return "deviating";
        if (dev > 7) return "warning";
        return "correct";
      }

      // Fallback: the hand-authored target-angle heuristic.
      const targets = POSE_TARGET_ANGLES[activePose] || [];
      const targetObj = targets.find(t => t.joint === joint);
      if (!targetObj) return "neutral";

      const idx = FEATURE_NAMES_ORDER.indexOf(joint);
      if (idx === -1) return "neutral";

      const current = anglesRef.current[idx];
      const diff = Math.abs(current - targetObj.target);

      if (diff > targetObj.tolerance) {
        return "deviating";
      } else if (diff > targetObj.tolerance - 7) {
        return "warning";
      }
      return "correct";
    };

    const getNodeJointNames = (nodeIdx: number): string[] => {
      if (nodeIdx === 11) return ["shoulder_l", "trunk_l"];
      if (nodeIdx === 12) return ["shoulder_r", "trunk_r"];
      if (nodeIdx === 13) return ["elbow_l"];
      if (nodeIdx === 14) return ["elbow_r"];
      if (nodeIdx === 23) return ["hip_l", "hip_abduct_l"];
      if (nodeIdx === 24) return ["hip_r", "hip_abduct_r"];
      if (nodeIdx === 25) return ["knee_l"];
      if (nodeIdx === 26) return ["knee_r"];
      if (nodeIdx === 27) return ["ankle_l"];
      if (nodeIdx === 28) return ["ankle_r"];
      return [];
    };

    const getNodeStatus = (nodeIdx: number) => {
      const joints = getNodeJointNames(nodeIdx);
      if (joints.length === 0) return "neutral";
      
      let hasDeviation = false;
      let hasWarning = false;
      let hasCorrect = false;
      
      for (const joint of joints) {
        const status = getJointStatus(joint);
        if (status === "deviating") hasDeviation = true;
        if (status === "warning") hasWarning = true;
        if (status === "correct") hasCorrect = true;
      }
      
      if (hasDeviation) return "deviating";
      if (hasWarning) return "warning";
      if (hasCorrect) return "correct";
      return "neutral";
    };

    const getLineColor = (idx1: number, idx2: number) => {
      const status1 = getNodeStatus(idx1);
      const status2 = getNodeStatus(idx2);
      
      if (status1 === "deviating" || status2 === "deviating") return "#f0715f"; // coral
      if (status1 === "warning" || status2 === "warning") return "#f2b84b"; // amber
      if (status1 === "correct" || status2 === "correct") return "#7fd1a8"; // sage
      
      // Neutral colors based on body side
      const leftNodes = [11, 13, 15, 23, 25, 27, 29, 31];
      const rightNodes = [12, 14, 16, 24, 26, 28, 30, 32];
      
      if (leftNodes.includes(idx1) && leftNodes.includes(idx2)) {
        return "rgba(255, 255, 255, 0.92)"; // left side
      }
      if (rightNodes.includes(idx1) && rightNodes.includes(idx2)) {
        return "rgba(236, 226, 205, 0.92)"; // right side (warm white)
      }
      return "rgba(255, 255, 255, 0.8)"; // white for cross-connections (shoulders, hips)
    };

    const drawLine = (idx1: number, idx2: number, width = 4) => {
      const pt1 = landmarks[idx1];
      const pt2 = landmarks[idx2];
      if (pt1 && pt2) {
        const color = getLineColor(idx1, idx2);
        ctx.globalAlpha = visRef.current.isHidden(idx1) || visRef.current.isHidden(idx2) ? 0.25 : 1;

        ctx.beginPath();
        ctx.moveTo(toX(pt1.x), toY(pt1.y));
        ctx.lineTo(toX(pt2.x), toY(pt2.y));
        ctx.lineCap = "round";
        // dark under-stroke keeps the line readable on any background (cheaper than a blurred shadow)
        ctx.strokeStyle = "rgba(20, 19, 16, 0.42)";
        ctx.lineWidth = width * 0.8 + 3;
        ctx.stroke();
        ctx.strokeStyle = color;
        ctx.lineWidth = width * 0.8;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    };

    // 0. Torso Shading (Digital Twin Hull)
    const pShoulderL = landmarks[11];
    const pShoulderR = landmarks[12];
    const pHipR = landmarks[24];
    const pHipL = landmarks[23];
    
    if (pShoulderL && pShoulderR && pHipR && pHipL) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(toX(pShoulderL.x), toY(pShoulderL.y));
      ctx.lineTo(toX(pShoulderR.x), toY(pShoulderR.y));
      ctx.lineTo(toX(pHipR.x), toY(pHipR.y));
      ctx.lineTo(toX(pHipL.x), toY(pHipL.y));
      ctx.closePath();
      
      // Determine overall pose correctness color for torso fill
      let torsoColor = "rgba(255, 255, 255, 0.07)"; // neutral fill
      if (activePose !== "transition/unknown") {
        if (effectiveCorrectness >= 0.70) {
          torsoColor = "rgba(127, 209, 168, 0.12)"; // sage fill
        } else {
          torsoColor = "rgba(240, 113, 95, 0.12)"; // coral fill
        }
      }
      
      ctx.fillStyle = torsoColor;
      ctx.fill();
      ctx.restore();
    }

    // Connections:
    // Shoulders
    drawLine(11, 12, 5);
    // Left Arm
    drawLine(11, 13, 4);
    drawLine(13, 15, 4);
    // Right Arm
    drawLine(12, 14, 4);
    drawLine(14, 16, 4);
    // Hips & Spine
    drawLine(23, 24, 5);
    drawLine(11, 23, 4);
    drawLine(12, 24, 4);
    // Left Leg
    drawLine(23, 25, 4);
    drawLine(25, 27, 4);
    // Right Leg
    drawLine(24, 26, 4);
    drawLine(26, 28, 4);

    // Draw joints with glowing indicators and white inner core
    landmarks.forEach((pt: any, i: number) => {
      if (pt.visibility > 0.5 && [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28].includes(i)) {
        const status = getNodeStatus(i);
        const x = toX(pt.x);
        const y = toY(pt.y);
        
        let color = "#ffffff";
        let r = 4.5;

        if (status === "deviating") {
          color = "#f0715f";
          r = 6;
        } else if (status === "warning") {
          color = "#f2b84b";
          r = 5.5;
        } else if (status === "correct") {
          color = "#7fd1a8";
          r = 5.5;
        } else {
          color = "#ffffff";
        }

        ctx.save();

        // Soft ring, then the joint itself: clean, no glow
        ctx.beginPath();
        ctx.arc(x, y, r + 2.5, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(20, 19, 16, 0.35)";
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.fill();

        ctx.restore();
      }
    });
  };

  const T = TRANSLATIONS[lang];
  const pct = Math.round(effectiveCorrectness * 100);
  const bodyMissing = cameraActive && framing === "none";
  const scoreActive = !bodyMissing && !showMismatch && !isTransitioning && !isUnrecognized;
  const scoreTone: "ok" | "mid" | "low" | "none" = !scoreActive ? "none" : effectiveCorrectness >= 0.75 ? "ok" : effectiveCorrectness >= 0.5 ? "mid" : "low";
  const verdict = !cameraActive
    ? T.cameraOff
    : bodyMissing
    ? T.stepIntoView
    : showMismatch
    ? T.wrongPoseBadge
    : isTransitioning
    ? T.stateTransitioning
    : isUnrecognized
    ? T.lookingForPose
    : effectiveCorrectness >= 0.75
    ? T.onTarget.replace("✓ ", "")
    : T.needsAdjustment;
  const hiddenLabels = Array.from(new Set(hiddenParts.map((f) => PART_LABELS[lang][f]).filter(Boolean)));
  const hiddenShort = hiddenLabels.slice(0, 2).join(", ") + (hiddenLabels.length > 2 ? "…" : "");
  const poseName = bodyMissing || activePose === "transition/unknown" ? "" : getSanskritName(activePose, lang);
  liveRef.current = { pose: scoreActive ? activePose : "", score: effectiveCorrectness, active: scoreActive };
  const retryEngine = () => {
    releasePose();
    graphicsFailedRef.current = false;
    engineErrorsRef.current = 0;
    setEngineState("loading");
    initMediaPipe();
  };

  // What the coaching card says right now, and in which tone.
  const coach: { tone: string; icon: any; label: React.ReactNode; text: string } =
    !cameraActive
      ? { tone: "", icon: <CameraIcon size={18} />, label: T.systemStatus, text: T.cameraInactive }
      : bodyMissing
      ? { tone: "warn", icon: <ScanLine size={18} />, label: T.stepIntoView, text: T.cantSeeYou }
      : activePose === "transition/unknown"
      ? { tone: "", icon: <Activity size={18} />, label: T.systemStatus, text: T.detectingPose }
      : showMismatch
      ? { tone: "bad", icon: <ShieldAlert size={18} />, label: T.wrongPoseBadge, text: displayCorrectionText }
      : isTransitioning
      ? { tone: "warn", icon: <Activity size={18} />, label: T.stateTransitioning, text: displayCorrectionText }
      : correctionText
      ? {
          tone: "warn",
          icon: <Volume2 size={18} />,
          label: (
            <>
              {T.safetyCorrection}
              {correctionIsSafe ? (
                <ShieldCheck size={13} aria-label="Checked for safety" />
              ) : (
                <ShieldAlert size={13} aria-label="Fell back to a reviewed template" />
              )}
            </>
          ),
          text: correctionText,
        }
      : { tone: "ok", icon: <CheckCircle2 size={18} />, label: T.alignmentCorrect, text: T.alignmentCorrectDesc };

  const StartStopLabel = isInitializingCamera ? (
    <>
      <Loader2 className="spin" size={18} />
      <span>{T.startCamera}</span>
    </>
  ) : cameraActive ? (
    <>
      <VideoOff size={18} />
      <span>{T.stopCamera}</span>
    </>
  ) : (
    <>
      <CameraIcon size={18} />
      <span>{T.startCamera}</span>
    </>
  );

  return (
    <>
      <Head>
        <title>AsanaAI — Smart Yoga Coach</title>
      </Head>

      {/* Pose engine (MediaPipe). Loaded early; the camera view never depends on it to show the picture. */}
      <Script
        src="https://cdn.jsdelivr.net/npm/@mediapipe/pose/pose.js"
        strategy="afterInteractive"
      />

      <div className={`ap ${sidebarOpen ? "side-open" : ""} ${cameraActive ? "cam-on" : ""}`}>
        {/* ─────────── Header ─────────── */}
        <header className={`ap-header ${isOnline ? "" : "is-offline"}`}>
          <button
            className="ap-iconbtn"
            onClick={() => setSidebarOpen(!sidebarOpen)}
            aria-label={sidebarOpen ? "Close sidebar" : "Open sidebar"}
            aria-expanded={sidebarOpen}
          >
            {sidebarOpen ? <PanelLeftClose size={19} /> : <PanelLeftOpen size={19} />}
          </button>
          <div className="ap-brand">
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3c1.6 2.6 2.4 5 2.4 7.2 0 3.2-1.6 6.5-2.4 10.8-.8-4.3-2.4-7.6-2.4-10.8C9.6 8 10.4 5.6 12 3Z" />
              <path d="M12 13.2c2.4-3.4 6.2-4.4 8.6-3.7-.5 3.1-3.9 6.7-8.6 8.2" />
              <path d="M12 13.2C9.6 9.8 5.8 8.8 3.4 9.5c.5 3.1 3.9 6.7 8.6 8.2" />
            </svg>
            <span>AsanaAI</span>
          </div>

          <div className="ap-spacer" />

          <div className="ap-header-actions">
            {!isOnline && (
              <span className="ap-offpill" role="status" title={T.offline}>
                <i />{T.offlineShort}
              </span>
            )}
            <div className={`ap-timer hide-s ${cameraActive ? "on" : ""}`} aria-label="Session time">
              <i />
              <SessionTimer running={cameraActive} />
            </div>

            <div
              className="ap-menu"
              onBlur={(e) => {
                if (!e.currentTarget.contains(e.relatedTarget)) setLangDropOpen(false);
              }}
            >
              <button
                className="ap-menu-trigger"
                onClick={() => setLangDropOpen(!langDropOpen)}
                aria-expanded={langDropOpen}
                aria-haspopup="listbox"
                aria-label="Select language"
              >
                <Globe size={15} />
                <span>{lang.toUpperCase()}</span>
                <ChevronDown size={14} />
              </button>
              {langDropOpen && (
                <div className="ap-menu-list" role="listbox">
                  {(["en", "hi", "bn"] as const).map((code) => (
                    <button
                      key={code}
                      role="option"
                      aria-selected={lang === code}
                      className={`ap-menu-item ${lang === code ? "active" : ""}`}
                      onClick={() => {
                        setLang(code);
                        setLangDropOpen(false);
                      }}
                    >
                      {code === "en" ? "English" : code === "hi" ? "हिन्दी" : "বাংলা"}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              className="ap-iconbtn hide-m"
              onClick={() => setSpeechEnabled(!speechEnabled)}
              aria-pressed={speechEnabled}
              aria-label={speechEnabled ? "Mute voice guidance" : "Enable voice guidance"}
              title={speechEnabled ? "Mute voice guidance" : "Enable voice guidance"}
            >
              {speechEnabled ? <Volume2 size={18} /> : <VolumeX size={18} />}
            </button>

            <button className="ap-btn ghost sm hide-m" onClick={resetPipeline} title={T.newSession}>
              <RotateCw size={15} />
              <span>{T.newSession}</span>
            </button>
          </div>
        </header>

        <div className="ap-scrim" onClick={() => setSidebarOpen(false)} />

        {/* ─────────── Sidebar (drawer on phones) ─────────── */}
        <aside className="ap-side" aria-label={T.practice}>
          <section className="ap-sec">
            <div className="ap-sec-head">
              <span className="ap-eyebrow">{practiceMode === "guided" ? T.targetPose : T.recognisedAsanas}</span>
            </div>
            <div className="ap-seg" role="group" aria-label="Practice mode">
              <button type="button" aria-pressed={practiceMode === "free"} onClick={() => setPracticeMode("free")}>
                {T.modeFree}
              </button>
              <button type="button" aria-pressed={practiceMode === "guided"} onClick={() => setPracticeMode("guided")}>
                {T.modeGuided}
              </button>
            </div>
            <p className="ap-hint">{practiceMode === "guided" ? T.modeGuidedHint : `${T.modeFreeHint} ${T.tapForGuide}`}</p>
            <div className="ap-poses">
              {POSE_LIBRARY.map(({ id }) => {
                const Icon = POSE_ICONS[id] || Leaf;
                const guided = practiceMode === "guided";
                const isOn = guided ? targetPose === id : activePose === id;
                const isPreview = !guided && browsePose === id;
                const choose = () => {
                  if (guided) { setTargetPose(id); setSidebarOpen(false); }
                  else setBrowsePose(browsePose === id ? null : id);
                };
                return (
                  <div
                    key={id}
                    role="button"
                    tabIndex={0}
                    aria-pressed={guided ? targetPose === id : isPreview}
                    className={`ap-pose pick ${isOn ? "active" : ""} ${isPreview ? "preview" : ""}`}
                    onClick={choose}
                    onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(); } }}
                  >
                    <span className="ap-pose-ic"><Icon size={19} strokeWidth={1.8} /></span>
                    <span>
                      <div className="ap-pose-name">{getSanskritName(id, lang)}</div>
                      <div className="ap-pose-sub">{POSE_COMMON_NAME[id]}</div>
                    </span>
                    {(isOn || isPreview) && (
                      <span className="ap-pose-tag"><i />{guided ? T.targetPose : isOn ? T.tagNow : T.tagGuide}</span>
                    )}
                  </div>
                );
              })}
            </div>
          </section>

          {(() => {
            const cues = POSE_GUIDE[guidePose];
            const diff = POSE_DIFFICULTY[guidePose];
            const joints = (POSE_TARGET_ANGLES[guidePose] || []).slice(0, 3);
            return cues ? (
              <Accordion title={T.poseGuide} open={openGroupPoseGuide} onToggle={() => setOpenGroupPoseGuide(!openGroupPoseGuide)} status={getSanskritName(guidePose, lang)}>
                <>
                  {practiceMode === "free" && (
                    <div className="ap-follow">
                      {browsePose ? (
                        <>
                          <span>{T.previewing}: <b>{getSanskritName(browsePose, lang)}</b></span>
                          <button onClick={() => setBrowsePose(null)}>{T.followMe}</button>
                        </>
                      ) : (
                        <span>{T.followingYou}</span>
                      )}
                    </div>
                  )}
                  {!POSE_REFERENCE_IMAGES[guidePose] && (
                    <div className="ap-ref ph" aria-hidden="true">
                      <PersonStanding size={34} strokeWidth={1.4} />
                      <span>Reference photo coming soon</span>
                    </div>
                  )}
                  {POSE_REFERENCE_IMAGES[guidePose] && (
                    <figure className="ap-ref">
                      <img
                        src={POSE_REFERENCE_IMAGES[guidePose].src}
                        alt={`Correct form for ${getSanskritName(guidePose, "en")}`}
                        loading="lazy"
                      />
                      <figcaption>{POSE_REFERENCE_IMAGES[guidePose].credit}</figcaption>
                    </figure>
                  )}
                  {diff && (
                    <div className="ap-level">
                      <i style={{ background: diff.color }} />
                      <span style={{ color: diff.color }}>{diff.level}</span>
                      <span style={{ color: "var(--ink-3)", fontWeight: 500 }}>· {getSanskritName(guidePose, lang)}</span>
                    </div>
                  )}
                  <div className="ap-cues">
                    {cues.map((c, i) => (
                      <div key={i} className="ap-cue">
                        <b><Check size={15} strokeWidth={2.4} /></b>
                        <span>{c.cue}</span>
                      </div>
                    ))}
                  </div>
                  {joints.length > 0 && (
                    <div>
                      <div className="ap-eyebrow" style={{ marginBottom: 8 }}>{T.measuredJoints}</div>
                      <div className="ap-chips">
                        {joints.map((j, i) => (
                          <span key={i} className="ap-chip">{j.label.replace(" Angle", "").replace(" Extension", "")}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              </Accordion>
            ) : null;
          })()}

          <Accordion title={T.digitalTwinProfile} open={openGroupTwin} onToggle={() => setOpenGroupTwin(!openGroupTwin)}
            status={calibratedProfile ? T.rangeSet : calibrationState === "calibrating" ? T.rangeLearning : T.rangeNotSet}>
            <>
              {calibratedProfile ? (
                <>
                  <div className="ap-note" style={{ color: "var(--ok)", fontWeight: 600 }}>{T.activeProfile}</div>
                  <button className="ap-btn ghost sm" style={{ alignSelf: "flex-start" }} onClick={relearnRange}>
                    <RotateCw size={14} /><span>{T.relearn}</span>
                  </button>
                  <div className="ap-joint-list">
                    {Object.keys(calibratedProfile).map((joint) => (
                      <div key={joint} className="ap-joint-row">
                        <span>{JOINT_TRANSLATIONS[lang][joint] || joint.replace("_", " ")}</span>
                        <strong className="num">{calibratedProfile[joint].min}° – {calibratedProfile[joint].max}°</strong>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p className={`ap-note ${calibrationState === "calibrating" ? "warn" : ""}`}>
                  {calibrationState === "calibrating" ? T.calibratingTwin : T.uncalibratedTwin}
                </p>
              )}
            </>
          </Accordion>

          <Accordion title={T.sessionOverview} open={openGroupSession} onToggle={() => setOpenGroupSession(!openGroupSession)}
            status={cameraActive ? "Camera on" : "Camera off"}>
            <>
              <div className="ap-kv">
                <div className="ap-kv-row"><i className={cameraActive ? "on" : ""} /><span>Camera</span><strong>{cameraActive ? "Active" : "Off"}</strong></div>
                <div className="ap-kv-row">
                  <i className={calibratedProfile ? "on" : calibrationState === "calibrating" ? "warn" : ""} />
                  <span>{T.digitalTwinProfile}</span>
                  <strong>{calibratedProfile ? T.rangeSet : calibrationState === "calibrating" ? T.rangeLearning : T.rangeNotSet}</strong>
                </div>
                <div className="ap-kv-row"><i className={speechEnabled ? "on" : ""} /><span>{T.voice}</span><strong>{speechEnabled ? "On" : "Muted"}</strong></div>
                <div className="ap-kv-row"><i className="on" /><span>{T.targetPose}</span><strong>{getSanskritName(guidePose, lang)}</strong></div>
              </div>
              {voiceMissingForLang && <p className="ap-note warn" role="status">{T.voiceMissing}</p>}
            </>
          </Accordion>
          <section className="ap-sec">
            <div className="ap-sec-head"><span className="ap-eyebrow">{T.appearance}</span></div>
            <div className="ap-seg three" role="group" aria-label={T.appearance}>
              {(["auto", "light", "dark"] as const).map((k) => (
                <button key={k} type="button" aria-pressed={theme === k} onClick={() => chooseTheme(k)}>
                  {k === "auto" ? T.themeAuto : k === "light" ? T.themeLight : T.themeDark}
                </button>
              ))}
            </div>
          </section>

          <button className="ap-btn ghost hide-d" onClick={() => { resetPipeline(); setSidebarOpen(false); }}>
            <RotateCw size={16} />
            <span>{T.newSession}</span>
          </button>
        </aside>

        {/* ─────────── Main: pinned camera + its own scroll region ─────────── */}
        <main className="ap-main">
          <section className="ap-stage-col" aria-label="Camera">
            <div className="ap-cam">
              <div ref={fullscreenContainerRef} className={`ap-cam-wrap ${isFullscreen ? "yoga-fullscreen" : ""}`}>
                <div
                  className={`ap-frame ${cameraActive && calibrationState !== "calibrating" ? "has-chip" : ""}`}
                  onTouchStart={cameraActive ? handleTouchStart : undefined}
                  onTouchMove={cameraActive ? handleTouchMove : undefined}
                  onTouchEnd={cameraActive ? handleTouchEnd : undefined}
                  onClick={cameraActive ? showZoomBarBriefly : undefined}
                >
                  {/* The video is the picture (always visible, so a slow pose engine never means a black screen);
                      the canvas on top carries only the skeleton. Both share one mirror/zoom transform. */}
                  <div ref={stageRef} className={`ap-stage ${facingMode === "user" ? "mirrored" : ""}`}>
                    <video ref={videoRef} className="ap-video" playsInline muted autoPlay />
                    <canvas ref={canvasRef} className="ap-canvas" />
                  </div>

                  {!cameraActive && !cameraError && (
                    <div className="ap-idle">
                      <div className="ap-idle-mark"><ScanLine size={26} strokeWidth={1.6} /></div>
                      <h2>{T.readyTitle}</h2>
                      <p>{T.readyBody}</p>
                      <button className="ap-btn" onClick={() => startCamera()} disabled={isInitializingCamera}>
                        {StartStopLabel}
                      </button>
                      <div className="ap-tips">
                        <span><Ruler size={14} />{T.tipDistance}</span>
                        <span><Sun size={14} />{T.tipLight}</span>
                        <span><PersonStanding size={14} />{T.tipFrame}</span>
                      </div>
                      <p className="ap-privacy"><Lock size={13} />{T.privacyNote}</p>
                    </div>
                  )}

                  {cameraError && (
                    <div className="ap-idle err" role="alert">
                      <div className="ap-idle-mark"><AlertTriangle size={26} strokeWidth={1.6} /></div>
                      <h2>{cameraError === GRAPHICS_ERR ? T.graphicsTitle : "Camera unavailable"}</h2>
                      <p>{cameraError === GRAPHICS_ERR ? T.graphicsBody : cameraError}</p>
                      <button className="ap-btn" onClick={() => startCamera()} disabled={isInitializingCamera}>
                        <RefreshCw size={17} className={isInitializingCamera ? "spin" : ""} />
                        <span>{T.retry}</span>
                      </button>
                      {cameraError === GRAPHICS_ERR && gfxInfo && (
                        <details className="ap-gfx"><summary>{T.graphicsDetails}</summary><pre>{gfxInfo}</pre></details>
                      )}
                    </div>
                  )}

                  {cameraActive && calibrationState === "calibrating" && (
                    <div className="ap-calib">
                      <div>
                        <h3>{T.calibratingProgress}</h3>
                        <p>{T.stayInView}</p>
                        <strong className="num">{calibrationCountdown}s</strong>
                        <button className="ap-btn ghost sm" style={{ marginTop: 14 }} onClick={skipCalibration}>{T.skip}</button>
                      </div>
                    </div>
                  )}

                  {cameraActive && (
                    <>
                      <div className="ap-hud tl">
                        <span className="ap-live"><i />{lang === "hi" ? "लाइव" : lang === "bn" ? "লাইভ" : "LIVE"}</span>
                        {coachSource === "device" && (
                          <span className="ap-engine basic" role="status" title={coachReason === "offline" ? T.coachOffline : coachReason === "waking" ? T.coachWaking : T.coachDown}>
                            <ScanLine size={14} />{T.basicMode}
                          </span>
                        )}
                        {framing === "partial" && !bodyMissing ? (
                          <span className="ap-engine" role="status"><PersonStanding size={14} />{T.stepBack}</span>
                        ) : hiddenLabels.length > 0 && !bodyMissing ? (
                          <span className="ap-engine" role="status"><ScanLine size={14} />{T.notChecking} {hiddenShort}</span>
                        ) : null}
                        {engineState === "loading" && (
                          <span className="ap-engine" role="status"><Loader2 size={14} className="spin" />{T.poseEngineLoading}</span>
                        )}
                        {engineState === "error" && (
                          <span className="ap-engine err" role="alert">
                            <AlertTriangle size={14} />{isOnline ? T.poseEngineSlow : T.poseEngineOffline}
                            <button onClick={retryEngine}>{T.retry}</button>
                          </span>
                        )}
                      </div>

                      <div className="ap-hud tr">
                        {hasMultipleCameras && (
                          <button className="ap-hudbtn" onClick={toggleCamera} disabled={isInitializingCamera} aria-label={T.swapCamera} title={T.swapCamera}>
                            <SwitchCamera size={18} />
                          </button>
                        )}
                        <button className="ap-hudbtn" onClick={enterFullscreen} aria-label={T.focusMode} title={T.focusMode}>
                          <Maximize2 size={18} />
                        </button>
                      </div>

                      {calibrationState !== "calibrating" && (
                        <div className="ap-hud bl">
                          <span className="ap-posechip" aria-live="polite">
                            <b>{bodyMissing ? T.stepIntoView : showMismatch ? T.wrongPoseBadge : isTransitioning ? T.stateTransitioning : poseName || T.lookingForPose}</b>
                            {scoreActive && <em className={scoreTone}>{pct}%</em>}
                          </span>
                        </div>
                      )}

                      <div className={`ap-zoom ${showZoomBar || zoomLevel !== 1 ? "show" : ""}`} onClick={(e) => e.stopPropagation()}>
                        <button onClick={() => applyZoom(zoomLevel + 0.5)} aria-label="Zoom in">+</button>
                        <input
                          type="range"
                          min={0.5}
                          max={10}
                          step={0.1}
                          value={zoomLevel}
                          onChange={(e) => applyZoom(parseFloat(e.target.value))}
                          aria-label="Zoom"
                        />
                        <button onClick={() => applyZoom(zoomLevel - 0.5)} aria-label="Zoom out">−</button>
                        <small>{zoomLevel.toFixed(1)}×</small>
                      </div>

                      {calibrationState !== "calibrating" && (
                        <div className="ap-caption" key={displayCorrectionText} aria-live="polite">
                          {displayCorrectionText || T.alignBody}
                        </div>
                      )}
                    </>
                  )}
                </div>

                {isFullscreen && (
                  <>
                    <button className="ap-fs-exit" onClick={exitFullscreen} aria-label="Exit fullscreen">
                      <X size={16} />
                      <span>{T.exit}</span>
                    </button>
                    <div className={`ap-fs-score ${scoreTone === "ok" ? "good" : scoreTone === "none" ? "" : "warn"}`}>
                      {bodyMissing ? "—" : showMismatch ? T.wrongPoseBadge : isTransitioning ? T.stateTransitioning : isUnrecognized ? "—" : `${pct}%`}
                    </div>
                  </>
                )}
              </div>

              {/* tablet / desktop: status + main action under the picture (phones use the dock) */}
              <div className="ap-cam-bar">
                <div className={`ap-cam-status ${cameraActive ? "on" : ""}`}>
                  <i />
                  <span>{cameraActive ? T.cameraLive : T.cameraOff}</span>
                </div>
                <div className="ap-spacer" />
                {cameraActive && hasMultipleCameras && (
                  <button className="ap-btn ghost sm" onClick={toggleCamera} disabled={isInitializingCamera}>
                    <SwitchCamera size={16} /><span>{T.swapCamera}</span>
                  </button>
                )}
                <button
                  className={`ap-btn ${cameraActive ? "dark" : ""}`}
                  onClick={cameraActive ? endSession : () => startCamera()}
                  disabled={isInitializingCamera}
                >
                  {StartStopLabel}
                </button>
              </div>
            </div>
          </section>

          <section className="ap-live-col" aria-label={T.livePanel}>
            {/* Score */}
            <div className="ap-card ap-score" aria-live="polite">
              <ScoreRing value={pct} tone={scoreTone} active={scoreActive} />
              <div>
                <div className="ap-card-title">{T.postureScore}</div>
                <div className="ap-verdict">{verdict}</div>
                {cameraActive && !isUnrecognized && (
                  <span className={`ap-state ${isTransitioning ? "moving" : "holding"}`}>
                    <i />
                    {isTransitioning ? T.stateTransitioning : T.stateHolding}
                  </span>
                )}
                {effectivePersonalCorrectness !== null && effectivePersonalCorrectness !== undefined && !isUnrecognized && (
                  <div className="ap-personal">
                    {T.personalScore}: <b className="num">{Math.round(effectivePersonalCorrectness * 100)}%</b>
                  </div>
                )}
              </div>
            </div>

            {/* Coaching */}
            <div className={`ap-coach ${coach.tone}`} key={coach.text}>
              <div className="ap-coach-ic">{coach.icon}</div>
              <div style={{ minWidth: 0 }}>
                <div className="ap-coach-label">{coach.label}</div>
                <div className="ap-coach-text">{coach.text}</div>
                {cameraActive && coachSource === "device" && (
                  <div className="ap-coach-note">{coachReason === "offline" ? T.coachOffline : coachReason === "waking" ? T.coachWaking : T.coachDown}</div>
                )}
                {coach.tone === "warn" && correctionText && !isTransitioning && lastEfficacy && (
                  <span className={`ap-efficacy ${lastEfficacy.worked ? "good" : ""}`} aria-live="polite">
                    {lastEfficacy.worked
                      ? `${formatJointName(lastEfficacy.joint, lang)} ${lang === "hi" ? "में" : lang === "bn" ? "" : "improved"} ${Math.round(lastEfficacy.improvementDeg)}°${lang === "hi" ? " सुधार" : lang === "bn" ? " উন্নত" : ""}`
                      : lang === "hi"
                      ? "कोई बदलाव नहीं — अलग तरीके से बताता हूँ"
                      : lang === "bn"
                      ? "পরিবর্তন হয়নি — অন্যভাবে বলছি"
                      : "No change yet — I'll try a different cue"}
                  </span>
                )}
              </div>
            </div>

            {/* Pose */}
            <div className="ap-card">
              <div className="ap-card-title">{T.detectedPose}</div>
              <div className={`ap-pose-now ${poseName ? "" : "dim"}`} style={{ marginTop: 6 }}>{poseName || "—"}</div>
              <div className="ap-rows">
                <div className="ap-row">
                  <span>{T.sequenceFlow}</span>
                  <strong>{flowPose === "transition/unknown" ? T.staticMode : getSanskritName(flowPose, lang)}</strong>
                </div>
                <div className="ap-row">
                  <span>{T.occlusionFusing}</span>
                  <strong className={hiddenLabels.length > 0 ? "warn" : "ok"}>
                    {hiddenLabels.length > 0 ? hiddenShort : T.allVisible}
                  </strong>
                </div>
              </div>
              {cameraActive && hiddenLabels.length > 0 && (
                <div className="ap-occl">
                  <ShieldAlert size={15} style={{ flex: "none", marginTop: 2 }} />
                  <span><strong>{T.fusingOccluded}</strong> {hiddenLabels.join(", ")}</span>
                </div>
              )}
            </div>

            {/* Joint alignment */}
            <div className="ap-card">
              <div className="ap-card-head"><span className="ap-card-title">{T.angleDetails}</span></div>
              {activePose === "transition/unknown" ? (
                <p className="ap-empty">{T.assumePosePrompt}</p>
              ) : isTransitioning ? (
                <p className="ap-empty">{displayCorrectionText}</p>
              ) : (
                getPoseJoints(activePose).map(({ joint, label, target, tolerance }) => {
                  const jointIdx = FEATURE_NAMES_ORDER.indexOf(joint);
                  const currentAngle =
                    allCurrentAngles[jointIdx] !== undefined
                      ? Math.round(allCurrentAngles[jointIdx])
                      : joint === "knee_l" ? currentKneeAngle : joint === "shoulder_l" ? currentShoulderAngle : 180;
                  const diff = currentAngle - target;
                  const off = Math.abs(diff) > tolerance;
                  if (hiddenParts.indexOf(joint) >= 0) {
                    return (
                      <div className="ap-joint" key={joint}>
                        <div className="ap-joint-top">
                          <span className="ap-joint-name">{JOINT_TRANSLATIONS[lang][joint] || label}</span>
                          <span className="ap-joint-val hidden">{T.hiddenWord}</span>
                        </div>
                        <div className="ap-joint-sub">{T.fusingOccluded.replace(/:$/, "")}</div>
                      </div>
                    );
                  }
                  return (
                    <div className="ap-joint" key={joint}>
                      <div className="ap-joint-top">
                        <span className="ap-joint-name">{JOINT_TRANSLATIONS[lang][joint] || label}</span>
                        <span className={`ap-joint-val ${off ? "off" : ""}`}>{currentAngle}° ({diff > 0 ? "+" : ""}{diff}°)</span>
                      </div>
                      <div className="ap-joint-sub">
                        {T.targetFor.replace("{target}", target.toString()).replace("{pose}", getSanskritName(activePose, lang))}
                      </div>
                      <div className="ap-bar"><i className={off ? "off" : ""} style={{ width: `${Math.min(100, Math.abs(diff) * 1.5)}%` }} /></div>
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </main>

        {/* ─────────── Phone dock ─────────── */}
        <nav className="ap-dock" aria-label="Controls">
          <button className="ap-dock-btn" onClick={() => setSidebarOpen(!sidebarOpen)} aria-pressed={sidebarOpen}>
            <LayoutGrid size={21} /><span>{T.poses}</span>
          </button>
          <button
            className={`ap-dock-main ${cameraActive ? "stop" : ""}`}
            onClick={cameraActive ? endSession : () => startCamera()}
            disabled={isInitializingCamera}
          >
            {StartStopLabel}
          </button>
          <button className="ap-dock-btn" onClick={toggleCamera} disabled={!cameraActive || !hasMultipleCameras || isInitializingCamera}>
            <SwitchCamera size={21} /><span>{T.flip}</span>
          </button>
          <button className="ap-dock-btn" onClick={() => setSpeechEnabled(!speechEnabled)} aria-pressed={speechEnabled}>
            {speechEnabled ? <Volume2 size={21} /> : <VolumeX size={21} />}<span>{T.voice}</span>
          </button>
        </nav>
      </div>

      {summary && (
        <div className="ap-sheet-wrap" role="dialog" aria-modal="true" aria-label={T.summaryTitle} onClick={() => setSummary(null)}>
          <div className="ap-sheet" onClick={(e) => e.stopPropagation()}>
            <h2>{T.summaryTitle}</h2>
            {summary.rows.length === 0 ? (
              <p className="ap-sheet-none">{T.summaryNone}</p>
            ) : (
              <>
                <div className="ap-sum-stats">
                  <div><b className="num">{Math.floor(summary.durationSec / 60)}:{String(summary.durationSec % 60).padStart(2, "0")}</b><span>{T.summaryTime}</span></div>
                  <div><b className="num">{Math.round(summary.avg * 100)}%</b><span>{T.summaryAvg}</span></div>
                  <div className="wide"><b>{summary.best ? getSanskritName(summary.best, lang) : "—"}</b><span>{T.summaryBest}</span></div>
                </div>
                <div className="ap-eyebrow" style={{ margin: "18px 0 8px" }}>{T.summaryPoses}</div>
                <div className="ap-sum-rows">
                  {summary.rows.map((r) => (
                    <div key={r.pose} className="ap-sum-row">
                      <span>{getSanskritName(r.pose, lang)}</span>
                      <span className="num">{Math.floor(r.sec / 60)}:{String(r.sec % 60).padStart(2, "0")}</span>
                      <strong className="num">{Math.round(r.avg * 100)}%</strong>
                    </div>
                  ))}
                </div>
              </>
            )}
            <div className="ap-sheet-actions">
              <button className="ap-btn ghost" onClick={() => setSummary(null)}>{T.close}</button>
              <button className="ap-btn" onClick={() => { setSummary(null); startCamera(); }}>{T.again}</button>
            </div>
          </div>
        </div>
      )}


      {updateReady && cameraActive && (
        <div className="ap-toast" role="status">
          <div className="grow"><strong>{T.updateReady}</strong><small>{T.updateLater}</small></div>
        </div>
      )}

      {showInstallBanner && !cameraActive && (
        <div className="ap-toast" role="dialog" aria-label={T.installTitle}>
          <div className="grow">
            <strong>{T.installTitle}</strong>
            <small>{T.installBody}</small>
          </div>
          <button className="act" onClick={handleInstall}>{T.install}</button>
          <button className="x" onClick={() => { setShowInstallBanner(false); try { window.localStorage.setItem("asana.installDismissed", String(Date.now())); } catch { /* ignore */ } }} aria-label="Dismiss"><X size={16} /></button>
        </div>
      )}
    </>
  );
}
