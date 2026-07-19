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
  Activity,
  CheckCircle2, 
  HelpCircle,
  Sparkles,
  Camera as CameraIcon,
  VideoOff,
  PanelLeftClose,
  PanelLeftOpen,
  ChevronDown,
  Maximize2,
  X,
  Globe,
  Gauge
} from "lucide-react";
import { useYogaPipeline } from "../hooks/useYogaPipeline";
import { CalibrationProfile } from "../types/yoga";
import { extractAnglesFromLandmarks } from "../utils/geometry";

/* eslint-disable */
type PresetPoseId = "warrior_2" | "plank" | "tree_pose" | "chair_pose" | "cobra_pose" | "mountain_pose";

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

// Static alignment cues per pose — displayed in the Pose Guide sidebar section
const POSE_GUIDE: { [key: string]: { cue: string; icon: string }[] } = {
  warrior_2: [
    { icon: "🦵", cue: "Front knee at 90° over ankle" },
    { icon: "💪", cue: "Arms parallel, shoulder height" },
    { icon: "👁️", cue: "Gaze over front fingertips" },
    { icon: "🦴", cue: "Hips open to the long side" },
  ],
  plank: [
    { icon: "💪", cue: "Wrists under shoulders, arms straight" },
    { icon: "🦴", cue: "Body forms one straight line" },
    { icon: "🧘", cue: "Core + glutes fully engaged" },
    { icon: "👁️", cue: "Neck neutral, gaze at floor" },
  ],
  tree_pose: [
    { icon: "🦶", cue: "Root foot flat, press into ground" },
    { icon: "🦵", cue: "Bent knee open to the side" },
    { icon: "👁️", cue: "Fix gaze on a still point (Drishti)" },
    { icon: "💪", cue: "Hands in Anjali or raised overhead" },
  ],
  chair_pose: [
    { icon: "🪑", cue: "Sit hips back like lowering into a chair" },
    { icon: "🦵", cue: "Knees tracking over ankles, not past toes" },
    { icon: "💪", cue: "Arms reach forward or overhead" },
    { icon: "🦴", cue: "Weight rooted through the heels" },
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
};

const POSE_DIFFICULTY: { [key: string]: { level: string; color: string } } = {
  warrior_2:    { level: "Intermediate", color: "var(--color-warning)" },
  plank:        { level: "Beginner",     color: "var(--color-success)" },
  tree_pose:    { level: "Intermediate", color: "var(--color-warning)" },
  chair_pose:   { level: "Beginner",     color: "var(--color-success)" },
  cobra_pose:   { level: "Beginner",     color: "var(--color-success)" },
  mountain_pose:{ level: "Beginner",     color: "var(--color-success)" },
};

// Poses selectable in the sidebar "Target Pose" grid — all 6 have backend
// correction templates + target-angle rules, so every one is fully live.
// Icons are thematic only (never a human-figure emoji depicting a specific
// body position) — a wrong body-position emoji invites users to physically
// copy an incorrect pose. warrior_2 used to show 🧘 (a seated meditation
// figure), which is a completely different pose from the standing Warrior II
// lunge; the reference photo below is now the actual "how do I do this"
// source of truth.
const POSE_SELECTOR_OPTIONS: { id: PresetPoseId; icon: string }[] = [
  { id: "warrior_2", icon: "⚔️" },
  { id: "plank", icon: "🏋️" },
  { id: "tree_pose", icon: "🌲" },
  { id: "chair_pose", icon: "🪑" },
  { id: "cobra_pose", icon: "🐍" },
  { id: "mountain_pose", icon: "⛰️" },
];

// Real reference photographs for each practice-able pose, shown in the Pose
// Guide panel so users don't have to guess correct form from an emoji.
// All CC-BY / CC-BY-SA licensed from Wikimedia Commons; attribution shown
// inline per license terms.
const POSE_REFERENCE_IMAGES: { [key: string]: { src: string; credit: string } } = {
  warrior_2: { src: "/pose-images/warrior_2.jpg", credit: "lululemon athletica, CC BY 2.0, via Wikimedia Commons" },
  plank: { src: "/pose-images/plank.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  tree_pose: { src: "/pose-images/tree_pose.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  chair_pose: { src: "/pose-images/chair_pose.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  cobra_pose: { src: "/pose-images/cobra_pose.jpg", credit: "Kennguru, CC BY 3.0, via Wikimedia Commons" },
  mountain_pose: { src: "/pose-images/mountain_pose.jpg", credit: "Witold Fitz-Simon, CC BY-SA 2.5, via Wikimedia Commons" },
};


const FEATURE_NAMES_ORDER = [
  "elbow_l", "elbow_r", "shoulder_l", "shoulder_r",
  "hip_l", "hip_r", "knee_l", "knee_r",
  "ankle_l", "ankle_r", "trunk_l", "trunk_r",
  "neck", "hip_abduct_l", "hip_abduct_r"
];

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

const TRANSLATIONS: {
  [lang: string]: { [key: string]: string }
} = {
  en: {
    appTitle: "AsanaAI — Smart Yoga Coach",
    newSession: "New Session",
    targetPose: "Target Pose",
    digitalTwinProfile: "Digital Twin Profile",
    activeProfile: "✓ Active Calibration Profile",
    uncalibratedTwin: "Digital Twin is uncalibrated. Select a pose and start the camera stream to calibrate your joint ranges.",
    calibratingTwin: "Calibrating... recording joint ranges.",
    apiConfig: "API Configuration",
    backendApiUrl: "Backend API URL",
    apiUrlHelper: "Point to localhost or your Hugging Face Space URL.",
    cameraStream: "Camera Stream",
    swapCamera: "Swap Camera",
    focusMode: "Focus Mode",
    stopVideo: "Stop Video",
    startVideo: "Start Video",
    calibratingProgress: "Calibrating Digital Twin",
    stayInView: "Stay in camera view...",
    cameraInactive: "Camera is inactive. Click 'Start Video' to begin.",
    alignBody: "Align your body with the camera...",
    exit: "Exit",
    postureScore: "Posture Score",
    onTarget: "✓ On target",
    needsAdjustment: "Needs adjustment",
    detectedPose: "Detected Pose",
    fusing: "Fusing",
    occlusionActive: "Occlusion active",
    allVisible: "All visible",
    staticMode: "Static Check",
    flowMode: "Flow mode",
    feedbackHub: "Real-Time Feedback Hub",
    correctnessScore: "Correctness Score",
    sequenceFlow: "Sequence Flow",
    occlusionFusing: "Occlusion Fusing",
    active: "Active",
    inactive: "Inactive",
    fusingOccluded: "Fusing Occluded landmarks:",
    mirroredCoordinates: "(Coordinate mirrored dynamically from twin joint)",
    systemStatus: "System Status",
    detectingPose: "Detecting pose... Align your body with the camera.",
    safetyCorrection: "Safety Correction Voice Guidance",
    alignmentCorrect: "Pose Alignment Correct",
    alignmentCorrectDesc: "Joint angle alignment is correct. Keep breathing steadily.",
    angleDetails: "Angle Alignment Details",
    assumePosePrompt: "Assume a target yoga pose to view real-time joint angle alignments and corrections.",
    targetFor: "Target: {target}° for {pose}",
    diff: "Diff:",
    wrongPoseDetected: "This looks like {detected}, not your selected {target}. Adjust into {target} to get scored.",
    wrongPoseBadge: "Wrong Pose"
  },
  hi: {
    appTitle: "असनएआई — स्मार्ट योग कोच",
    newSession: "नया सत्र",
    targetPose: "लक्ष्य मुद्रा",
    digitalTwinProfile: "डिजिटल ट्विन प्रोफ़ाइल",
    activeProfile: "✓ सक्रिय अंशांकन प्रोफ़ाइल",
    uncalibratedTwin: "डिजिटल ट्विन अंशांकित नहीं है। एक मुद्रा चुनें और अपनी संयुक्त सीमाओं को अंशांकित करने के लिए कैमरा स्ट्रीम शुरू करें।",
    calibratingTwin: "अंशांकन हो रहा है... संयुक्त सीमाओं को रिकॉर्ड किया जा रहा है।",
    apiConfig: "एपीआई कॉन्फ़िगरेशन",
    backendApiUrl: "बैकएंड एपीआई यूआरएल",
    apiUrlHelper: "लोकलहोस्ट या अपने हगिंग फेस स्पेस यूआरएल को इंगित करें।",
    cameraStream: "कैमरा स्ट्रीम",
    swapCamera: "कैमरा बदलें",
    focusMode: "फ़ोकस मोड",
    stopVideo: "वीडियो रोकें",
    startVideo: "वीडियो शुरू करें",
    calibratingProgress: "डिजिटल ट्विन का अंशांकन",
    stayInView: "कैमरे के सामने बने रहें...",
    cameraInactive: "कैमरा निष्क्रिय है। शुरू करने के लिए 'वीडियो शुरू करें' पर क्लिक करें।",
    alignBody: "कैमरे के साथ अपने शरीर को संरेखित करें...",
    exit: "बाहर निकलें",
    postureScore: "मुद्रा स्कोर",
    onTarget: "✓ सही संरेखण",
    needsAdjustment: "समायोजन की आवश्यकता है",
    detectedPose: "पहचानी गई मुद्रा",
    fusing: "फ्यूज़िंग",
    occlusionActive: "अस्पष्टता सक्रिय",
    allVisible: "सभी दृश्यमान",
    staticMode: "स्थिर जाँच",
    flowMode: "प्रवाह मोड",
    feedbackHub: "रीयल-टाइम फीडबैक हब",
    correctnessScore: "सटीकता स्कोर",
    sequenceFlow: "अनुक्रम प्रवाह",
    occlusionFusing: "अस्पष्टता फ्यूज़िंग",
    active: "सक्रिय",
    inactive: "निष्क्रिय",
    fusingOccluded: "अस्पष्ट अंगों को फ्यूज करना:",
    mirroredCoordinates: "(ट्विन जोड़ से गतिशील रूप से प्रतिबिंबित समन्वयक)",
    systemStatus: "सिस्टम की स्थिति",
    detectingPose: "मुद्रा खोजी जा रही है... अपने शरीर को कैमरे के साथ संरेखित करें।",
    safetyCorrection: "सुरक्षा सुधार वॉयस गाइडेंस",
    alignmentCorrect: "आसन संरेखण सही है",
    alignmentCorrectDesc: "जोड़ों का संरेखण बिल्कुल सही है। लगातार सांस लेते रहें।",
    angleDetails: "कोण संरेखण विवरण",
    assumePosePrompt: "वास्तविक समय में जोड़ों के संरेखण और सुधार देखने के लिए एक लक्ष्य योग मुद्रा धारण करें।",
    targetFor: "लक्ष्य: {pose} के लिए {target}°",
    diff: "अंतर:",
    wrongPoseDetected: "यह {detected} जैसा लग रहा है, आपकी चुनी हुई {target} नहीं। स्कोर पाने के लिए {target} में आएं।",
    wrongPoseBadge: "गलत मुद्रा"
  },
  bn: {
    appTitle: "আসনএআই — স্মার্ট যোগ কোচ",
    newSession: "নতুন সেশন",
    targetPose: "লক্ষ্য আসন",
    digitalTwinProfile: "ডিজিটাল টুইন প্রোফাইল",
    activeProfile: "✓ সক্রিয় ক্যালিব্রেশন প্রোফাইল",
    uncalibratedTwin: "ডিজিটাল টুইন ক্যালিব্রেট করা নেই। একটি আসন নির্বাচন করুন এবং আপনার জয়েন্ট সীমা ক্যালিব্রেট করতে ক্যামেরা স্ট্রীম শুরু করুন।",
    calibratingTwin: "ক্যালিব্রেট করা হচ্ছে... জয়েন্ট রেঞ্জ রেকর্ড করা হচ্ছে।",
    apiConfig: "এপিআই কনফিগারেশন",
    backendApiUrl: "ব্যাকএন্ড এপিআই ইউআরএল",
    apiUrlHelper: "লোকালহোস্ট বা আপনার হাগিং ফেস স্পেস ইউআরএল নির্দেশ করুন।",
    cameraStream: "ক্যামেরা স্ট্রীম",
    swapCamera: "ক্যামেরা পরিবর্তন",
    focusMode: "ফোকাস মোড",
    stopVideo: "ভিডিও বন্ধ করুন",
    startVideo: "ভিডিও চালু করুন",
    calibratingProgress: "ডিজিটাল টুইন ক্যালিব্রেট হচ্ছে",
    stayInView: "ক্যামেরার সামনে থাকুন...",
    cameraInactive: "ক্যামেরা নিষ্ক্রিয় আছে। শুরু করতে 'ভিডিও চালু করুন' ক্লিক করুন।",
    alignBody: "ক্যামেরার সাথে আপনার শরীর সারিবদ্ধ করুন...",
    exit: "প্রস্থান",
    postureScore: "আসন স্কোর",
    onTarget: "✓ সঠিক সারিবদ্ধকরণ",
    needsAdjustment: "সংশোধন প্রয়োজন",
    detectedPose: "শনাক্ত আসন",
    fusing: "ফিউজিং",
    occlusionActive: "অস্পষ্টতা সক্রিয়",
    allVisible: "সব দৃশ্যমান",
    staticMode: "স্থির পরীক্ষা",
    flowMode: "প্রবাহ মোড",
    feedbackHub: "রিয়েল-টাইম ফিডব্যাক হাব",
    correctnessScore: "সঠিকতা স্কোর",
    sequenceFlow: "সিকোয়েন্স ফ্লো",
    occlusionFusing: "অস্পষ্টতা ফিউজিং",
    active: "সক্রিয়",
    inactive: "নিষ্ক্রিয়",
    fusingOccluded: "অস্পষ্ট জয়েন্ট ফিউজ করা হচ্ছে:",
    mirroredCoordinates: "(টুইন জয়েন্ট থেকে গতিশীলভাবে মিরর করা স্থানাঙ্ক)",
    systemStatus: "সিস্টেমের অবস্থা",
    detectingPose: "আসন শনাক্ত করা হচ্ছে... ক্যামেরার সাথে আপনার শরীর সারিবদ্ধ করুন।",
    safetyCorrection: "সুরক্ষা সংশোধন ভয়েস গাইডেন্স",
    alignmentCorrect: "আসন ভঙ্গি সঠিক আছে",
    alignmentCorrectDesc: "জয়েন্ট অ্যালাইনমেন্ট সঠিক আছে। স্বাভাবিক শ্বাস-প্রশ্বাস বজায় রাখুন।",
    angleDetails: "কোণ সারিবদ্ধকরণ বিবরণ",
    assumePosePrompt: "রিয়েল-টাইম জয়েন্ট অ্যালাইনমেন্ট এবং সংশোধন দেখতে একটি লক্ষ্য যোগাসন অনুশীলন করুন।",
    targetFor: "{pose} এর জন্য লক্ষ্য কোণ {target}°",
    diff: "পার্থক্য:",
    wrongPoseDetected: "এটি {detected} বলে মনে হচ্ছে, আপনার নির্বাচিত {target} নয়। স্কোর পেতে {target} ভঙ্গিতে আসুন।",
    wrongPoseBadge: "ভুল আসন"
  }
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
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  return (
    <div className="gauge-circle">
      <svg width="100" height="100" style={{ transform: "rotate(-90deg)" }}>
        {/* Track */}
        <circle
          cx="50"
          cy="50"
          r={radius}
          fill="transparent"
          stroke={isSuccess ? "var(--color-success-muted)" : "var(--color-warning-muted)"}
          strokeWidth="6"
        />
        {/* Fill */}
        <circle
          cx="50"
          cy="50"
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
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activePreset, setActivePreset] = useState<PresetPoseId>("warrior_2");
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

  // ── Zoom feature ─────────────────────────────────────────
  const [zoomLevel, setZoomLevel] = useState(1);         // current zoom (0.5 – 10)
  const [showZoomBar, setShowZoomBar] = useState(false); // show slider on demand
  const [zoomCapable, setZoomCapable] = useState(false); // hw zoom available
  const zoomHideTimerRef = useRef<NodeJS.Timeout | null>(null);
  // Pinch gesture tracking
  const pinchStartDistRef = useRef<number | null>(null);
  const pinchStartZoomRef = useRef<number>(1);

  // Dynamic metrics computed in real-time
  const [allCurrentAngles, setAllCurrentAngles] = useState<number[]>(new Array(15).fill(180));

  // MediaPipe state
  const [mediaPipeLoaded, setMediaPipeLoaded] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [isInitializingCamera, setIsInitializingCamera] = useState(false);

  // References
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fullscreenContainerRef = useRef<HTMLDivElement>(null);
  const cameraRef = useRef<any>(null);
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
  const [openGroupPose, setOpenGroupPose] = useState(true);
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
    if (canvasRef.current) {
      if (clamped === 1) {
        canvasRef.current.style.transform = 'scaleX(-1)'; // preserve mirror
      } else {
        canvasRef.current.style.transform = `scaleX(-1) scale(${clamped})`;
      }
    }
  };

  // Auto-detect zoom capability when stream starts
  useEffect(() => {
    if (!cameraActive || !streamRef.current) { setZoomCapable(false); return; }
    const track = streamRef.current.getVideoTracks()[0];
    if (!track) { setZoomCapable(false); return; }
    const caps = track.getCapabilities?.() as any;
    setZoomCapable(!!(caps?.zoom));
    // Reset zoom to 1x on new camera start
    setZoomLevel(1);
    if (canvasRef.current) canvasRef.current.style.transform = 'scaleX(-1)';
  }, [cameraActive]);

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
    setTimeout(() => {
      const utterance = new SpeechSynthesisUtterance(text);
      
      // Find voice matching the language if possible (improves desktop synthesis quality)
      if (window.speechSynthesis.getVoices) {
        const voices = window.speechSynthesis.getVoices();
        const langCode = lang === 'hi' ? 'hi-IN' : lang === 'bn' ? 'bn-IN' : 'en-US';
        const voice = voices.find(v => v.lang.includes(langCode) || v.lang.startsWith(lang === 'hi' ? 'hi' : lang === 'bn' ? 'bn' : 'en'));
        if (voice) {
          utterance.voice = voice;
        }
      }
      utterance.lang = lang === 'hi' ? 'hi-IN' : lang === 'bn' ? 'bn-IN' : 'en-US';
      utterance.rate = 0.92;   // Slightly slower for yoga instruction clarity
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      
      // Chrome bug: long utterances silently stop after ~15s. Split and chain.
      const MAX_CHUNK = 180;
      if (text.length > MAX_CHUNK) {
        const sentences = text.match(/[^.!?]+[.!?]*/g) || [text];
        let idx = 0;
        const speakNext = () => {
          if (idx >= sentences.length || !speechEnabled) return;
          const u = new SpeechSynthesisUtterance(sentences[idx].trim());
          u.lang = utterance.lang;
          if (utterance.voice) u.voice = utterance.voice;
          u.rate = utterance.rate;
          u.onend = () => { idx++; speakNext(); };
          window.speechSynthesis.speak(u);
        };
        speakNext();
        return;
      }
      
      window.speechSynthesis.speak(utterance);
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
    correctionIsSafe,
    poseMismatch,
    predictionTimestamp,
    recoveredJoints,
    isLoading,
    processFrame,
    resetPipeline
  } = useYogaPipeline({
    language: lang,
    groqApiKey: undefined,
    calibrationProfile: calibratedProfile || undefined,
    correctnessThreshold: 0.75,
    targetPose: activePreset
  });

  // When the detected pose doesn't match what the user selected to practice,
  // the raw correctness score (which only measures form quality for whatever
  // pose was detected) would be misleading to show — override both the
  // displayed guidance text and the effective score in that case.
  const displayCorrectionText = poseMismatch
    ? TRANSLATIONS[lang].wrongPoseDetected
        .replace('{detected}', getSanskritName(activePose, lang))
        .replace(/{target}/g, getSanskritName(activePreset, lang))
    : correctionText;
  const effectiveCorrectness = poseMismatch ? 0 : correctness;

  // Audio speech synthesis loop — tied to the pipeline's own ~10s prediction
  // cadence (predictionTimestamp only changes once per real classification
  // cycle) so spoken feedback never gets ahead of what's actually been re-analyzed.
  useEffect(() => {
    if (speechEnabled && displayCorrectionText && predictionTimestamp > 0) {
      announceTTS(displayCorrectionText);
    }
  }, [predictionTimestamp, speechEnabled, lang]);

  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && window.speechSynthesis?.paused) {
        window.speechSynthesis.resume();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);


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

    pose.onResults((results: any) => {
      onPoseResultsRef.current?.(results);
    });
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
          width: { ideal: 1280, max: 1920 },
          height: { ideal: 720, max: 1080 },
          frameRate: { ideal: 30, max: 60 }
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

      // Reset performance tracking for the new camera session
      inferenceTimesRef.current = [];
      hasDowngradedModelRef.current = false;

      // Start custom rendering loop tick to feed MediaPipe. This self-paces
      // to however fast the device can actually run inference (it only
      // schedules the next frame once the previous one finishes), so weak
      // devices never queue up a growing backlog of frames — but the model
      // itself was always running the heaviest "Full" complexity regardless
      // of device. On slow/low-end hardware that means each frame can take
      // several hundred ms, which reads as a stuck/laggy camera and can make
      // it look like poses aren't being scanned properly. Measure the first
      // ~20 frames after camera start and drop to the lighter "Lite" model
      // automatically if the device can't keep up.
      const tick = async () => {
        if (!stream.active || !videoRef.current) return;
        if (videoRef.current.readyState >= 3) {
          if (poseRef.current) {
            const t0 = performance.now();
            await poseRef.current.send({ image: videoRef.current });
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
                    minTrackingConfidence: 0.5
                  });
                } else if (avg < 40) {
                  // Device has plenty of headroom (> ~25fps at "Full") —
                  // upgrade to MediaPipe's most accurate "Heavy" model for
                  // better landmark/skeleton precision. Purely additive: the
                  // existing downgrade path above is untouched, and this only
                  // triggers for devices that already proved they're fast.
                  poseRef.current.setOptions({
                    modelComplexity: 2,
                    smoothLandmarks: true,
                    minDetectionConfidence: 0.5,
                    minTrackingConfidence: 0.5
                  });
                }
                hasDowngradedModelRef.current = true;
              }
            }
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

  // Resize canvas to match its container dynamically
  useEffect(() => {
    const resizeCanvas = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        canvas.width = rect.width;
        canvas.height = rect.height;
      }
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);
    return () => window.removeEventListener('resize', resizeCanvas);
  }, []);

  // MediaPipe Result processing callback
  const onPoseResults = (results: any) => {
    const canvasElement = canvasRef.current;
    if (!canvasElement) return;
    const canvasCtx = canvasElement.getContext("2d");
    if (!canvasCtx) return;

    // Draw raw camera frame onto canvas, preserving its native aspect ratio
    // (letterboxed to fit) instead of stretching it to the box's fixed
    // aspect-ratio — a plain stretch distorts the whole picture whenever the
    // camera's native resolution doesn't match the CSS box (nearly always,
    // since real cameras are commonly 16:9 or 4:3 while the box is fixed),
    // making it hard to tell if the user's full body is actually in frame.
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    const cw = canvasElement.width;
    const ch = canvasElement.height;
    const srcW = results.image.videoWidth || results.image.width || cw;
    const srcH = results.image.videoHeight || results.image.height || ch;
    const fitScale = Math.min(cw / srcW, ch / srcH);
    const fitW = srcW * fitScale;
    const fitH = srcH * fitScale;
    const fitX = (cw - fitW) / 2;
    const fitY = (ch - fitH) / 2;
    canvasCtx.drawImage(results.image, fitX, fitY, fitW, fitH);

    if (results.poseLandmarks) {
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

      processFrame(rawLandmarks, angles).finally(() => {
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
      if (activePose === "transition/unknown") return "neutral";
      const targets = POSE_TARGET_ANGLES[activePose] || [];
      const targetObj = targets.find(t => t.joint === joint);
      if (!targetObj) return "neutral";
      
      const idx = FEATURE_NAMES_ORDER.indexOf(joint);
      if (idx === -1) return "neutral";
      
      const current = allCurrentAngles[idx];
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
      
      if (status1 === "deviating" || status2 === "deviating") return "#ff2d55"; // neon red
      if (status1 === "warning" || status2 === "warning") return "#ff9500"; // neon orange
      if (status1 === "correct" || status2 === "correct") return "#34c759"; // neon green
      
      // Neutral colors based on body side
      const leftNodes = [11, 13, 15, 23, 25, 27, 29, 31];
      const rightNodes = [12, 14, 16, 24, 26, 28, 30, 32];
      
      if (leftNodes.includes(idx1) && leftNodes.includes(idx2)) {
        return "#00f0ff"; // neon cyan
      }
      if (rightNodes.includes(idx1) && rightNodes.includes(idx2)) {
        return "#ff007f"; // neon pink
      }
      return "rgba(255, 255, 255, 0.8)"; // white for cross-connections (shoulders, hips)
    };

    const drawLine = (idx1: number, idx2: number, width = 4) => {
      const pt1 = landmarks[idx1];
      const pt2 = landmarks[idx2];
      if (pt1 && pt2) {
        const color = getLineColor(idx1, idx2);
        
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(toX(pt1.x), toY(pt1.y));
        ctx.lineTo(toX(pt2.x), toY(pt2.y));
        
        // Premium Neon Glow Effect
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.lineCap = "round";
        ctx.shadowColor = color;
        ctx.shadowBlur = 8;
        ctx.stroke();
        ctx.restore();
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
      let torsoColor = "rgba(0, 240, 255, 0.06)"; // neutral cyan fill
      if (activePose !== "transition/unknown") {
        if (effectiveCorrectness >= 0.70) {
          torsoColor = "rgba(52, 199, 89, 0.06)"; // green fill
        } else {
          torsoColor = "rgba(255, 45, 85, 0.06)"; // red fill
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
        let r = 5;
        
        if (status === "deviating") {
          color = "#ff2d55";
          r = 7;
        } else if (status === "warning") {
          color = "#ff9500";
          r = 6;
        } else if (status === "correct") {
          color = "#34c759";
          r = 6;
        } else {
          // Neutral side colors
          const leftNodes = [11, 13, 15, 23, 25, 27];
          color = leftNodes.includes(i) ? "#00f0ff" : "#ff007f";
        }
        
        ctx.save();
        
        // Draw outer translucent halo
        ctx.beginPath();
        ctx.arc(x, y, r + 3, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(255, 255, 255, 0.15)";
        ctx.fill();
        
        // Draw main colored circle
        ctx.beginPath();
        ctx.arc(x, y, r, 0, 2 * Math.PI);
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.fill();
        
        // Draw inner white core
        ctx.beginPath();
        ctx.arc(x, y, r * 0.4, 0, 2 * Math.PI);
        ctx.fillStyle = "#ffffff";
        ctx.shadowBlur = 0;
        ctx.fill();
        
        ctx.restore();
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

            <button
              className="btn-primary btn-sm new-session-btn"
              onClick={resetPipeline}
              aria-label={TRANSLATIONS[lang].newSession}
              title={TRANSLATIONS[lang].newSession}
            >
              <RefreshCw size={14} className="new-session-icon" />
              <span className="new-session-label">{TRANSLATIONS[lang].newSession}</span>
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
              <span>{TRANSLATIONS[lang].targetPose}</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupPose ? "collapsed" : ""}`} />
            </div>
            <div className={`sidebar-group-body ${!openGroupPose ? "collapsed" : ""}`}>
              <div className="pose-card-grid">
                {POSE_SELECTOR_OPTIONS.map(({ id, icon }) => (
                  <div
                    key={id}
                    className={`pose-card ${activePreset === id ? "active" : ""}`}
                    onClick={() => {
                      setActivePreset(id);
                      setSidebarOpen(false);
                    }}
                  >
                    <span className="pose-card-icon">{icon}</span>
                    <span className="pose-card-label">{getSanskritName(id, lang)}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>


          {/* Group 4: Digital Twin Profile */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupTwin(!openGroupTwin)}>
              <span>{TRANSLATIONS[lang].digitalTwinProfile}</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupTwin ? "collapsed" : ""}`} />
            </div>
            <div className={`sidebar-group-body ${!openGroupTwin ? "collapsed" : ""}`}>
              {calibratedProfile ? (
                <div className="scrollable-panel" style={{ display: "flex", flexDirection: "column", gap: "10px", maxHeight: "220px", overflowY: "auto", padding: "4px 4px 4px 0", margin: "4px 0" }}>
                  <div style={{ fontSize: "11px", color: "var(--color-success)", fontWeight: "500", marginBottom: "4px" }}>
                    {TRANSLATIONS[lang].activeProfile}
                  </div>
                  {Object.keys(calibratedProfile).map((joint) => (
                    <div key={joint} style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", borderBottom: "1px solid rgba(255,255,255,0.05)", paddingBottom: "4px" }}>
                      <span style={{ color: "rgba(255,255,255,0.7)" }}>{(JOINT_TRANSLATIONS[lang][joint] || joint.replace('_', ' ')).toUpperCase()}</span>
                      <span style={{ fontWeight: "600" }}>
                        {calibratedProfile[joint].min}° – {calibratedProfile[joint].max}°
                      </span>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ fontSize: "12px", color: "rgba(255,255,255,0.5)", lineHeight: "1.5" }}>
                  {calibrationState === "calibrating" ? (
                    <span style={{ color: "var(--color-warning)" }}>{TRANSLATIONS[lang].calibratingTwin}</span>
                  ) : (
                    TRANSLATIONS[lang].uncalibratedTwin
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Group 5: Pose Guide — alignment cues for the selected pose */}
          {(() => {
            const cues = POSE_GUIDE[activePreset];
            const diff = POSE_DIFFICULTY[activePreset];
            const joints = (POSE_TARGET_ANGLES[activePreset] || []).slice(0, 3);
            return cues ? (
              <div className="sidebar-group">
                <div className="sidebar-group-header" onClick={() => setOpenGroupPoseGuide(!openGroupPoseGuide)}>
                  <span>Pose Guide</span>
                  <ChevronDown size={16} className={`chevron-icon ${!openGroupPoseGuide ? "collapsed" : ""}`} />
                </div>
                <div className={`sidebar-group-body ${!openGroupPoseGuide ? "collapsed" : ""}`}>
                  {/* Reference photo — the actual source of truth for what this
                      pose should look like, since a single emoji can't show
                      correct body position and (as warrior_2's old 🧘 icon
                      proved) a wrong one can teach the wrong form entirely. */}
                  {POSE_REFERENCE_IMAGES[activePreset] && (
                    <div className="pose-guide-reference">
                      <img
                        src={POSE_REFERENCE_IMAGES[activePreset].src}
                        alt={`Correct form for ${getSanskritName(activePreset, "en")}`}
                        className="pose-guide-reference-img"
                      />
                      <span className="pose-guide-reference-credit">
                        {POSE_REFERENCE_IMAGES[activePreset].credit}
                      </span>
                    </div>
                  )}

                  {/* Difficulty badge */}
                  {diff && (
                    <div className="pose-guide-diff">
                      <span className="pose-guide-diff-dot" style={{ background: diff.color }} />
                      <span className="pose-guide-diff-label" style={{ color: diff.color }}>{diff.level}</span>
                      <span className="pose-guide-diff-name">{getSanskritName(activePreset, lang)}</span>
                    </div>
                  )}

                  {/* Alignment cue list */}
                  <div className="pose-guide-cues">
                    {cues.map((c, i) => (
                      <div key={i} className="pose-guide-cue-row">
                        <span className="pose-guide-cue-icon">{c.icon}</span>
                        <span className="pose-guide-cue-text">{c.cue}</span>
                      </div>
                    ))}
                  </div>

                  {/* Target joints being measured */}
                  {joints.length > 0 && (
                    <div className="pose-guide-joints">
                      <span className="pose-guide-joints-label">AI measures:</span>
                      <div className="pose-guide-joint-chips">
                        {joints.map((j, i) => (
                          <span key={i} className="pose-guide-chip">{j.label.replace(" Angle", "").replace(" Extension", "")}</span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : null;
          })()}

          {/* Group 6: Session Overview — live status strip */}
          <div className="sidebar-group">
            <div className="sidebar-group-header" onClick={() => setOpenGroupSession(!openGroupSession)}>
              <span>Session Overview</span>
              <ChevronDown size={16} className={`chevron-icon ${!openGroupSession ? "collapsed" : ""}`} />
            </div>
            <div className={`sidebar-group-body ${!openGroupSession ? "collapsed" : ""}`}>
              <div className="session-overview-grid">
                {/* Camera */}
                <div className="session-stat-row">
                  <div className={`session-stat-dot ${cameraActive ? "active" : ""}`} />
                  <span className="session-stat-label">Camera</span>
                  <span className={`session-stat-val ${cameraActive ? "on" : "off"}`}>
                    {cameraActive ? "Active" : "Off"}
                  </span>
                </div>
                {/* Calibration */}
                <div className="session-stat-row">
                  <div className={`session-stat-dot ${calibratedProfile ? "calibrated" : calibrationState === "calibrating" ? "calibrating-anim" : ""}`} />
                  <span className="session-stat-label">Digital Twin</span>
                  <span className={`session-stat-val ${
                    calibratedProfile ? "on" : calibrationState === "calibrating" ? "warn" : "off"
                  }`}>
                    {calibratedProfile ? "Calibrated" : calibrationState === "calibrating" ? "Calibrating…" : "Uncalibrated"}
                  </span>
                </div>
                {/* Voice */}
                <div className="session-stat-row">
                  <div className={`session-stat-dot ${speechEnabled ? "active" : ""}`} />
                  <span className="session-stat-label">Voice</span>
                  <span className={`session-stat-val ${speechEnabled ? "on" : "off"}`}>
                    {speechEnabled ? "On" : "Muted"}
                  </span>
                </div>
                {/* Pose */}
                <div className="session-stat-row">
                  <div className="session-stat-dot active" />
                  <span className="session-stat-label">Target Pose</span>
                  <span className="session-stat-val on" style={{ fontFamily: "var(--font-mono)", fontSize: "0.68rem" }}>
                    {getSanskritName(activePreset, "en").split(" ")[0]}
                  </span>
                </div>
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
                  <span>{TRANSLATIONS[lang].cameraStream}</span>
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
                      <span>{TRANSLATIONS[lang].swapCamera}</span>
                    </button>
                  )}

                  {cameraActive && (
                    <button
                      className="btn-primary btn-sm"
                      style={{
                        background: 'var(--color-surface-offset)',
                        border: '1px solid var(--color-border)',
                        color: 'var(--color-text)',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '6px'
                      }}
                      onClick={enterFullscreen}
                      title="Enter fullscreen yoga mode"
                    >
                      <Maximize2 size={16} />
                      <span>{TRANSLATIONS[lang].focusMode}</span>
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
                        <span>{TRANSLATIONS[lang].stopVideo}</span>
                      </>
                    ) : (
                      <>
                        <CameraIcon size={16} />
                        <span>{TRANSLATIONS[lang].startVideo}</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              <div
                ref={fullscreenContainerRef}
                className={`camera-frame-wrapper ${isFullscreen ? 'yoga-fullscreen' : ''}`}
              >
                <div
                  className="camera-frame"
                  style={{ position: "relative" }}
                  onTouchStart={cameraActive ? handleTouchStart : undefined}
                  onTouchMove={cameraActive ? handleTouchMove : undefined}
                  onTouchEnd={cameraActive ? handleTouchEnd : undefined}
                  onClick={cameraActive ? showZoomBarBriefly : undefined}
                >
                  <video 
                    ref={videoRef} 
                    className="camera-video-element"
                    playsInline 
                    muted 
                  />
                  <canvas 
                    ref={canvasRef} 
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
                        <h3 style={{ fontSize: "20px", fontWeight: "600", margin: "0 0 4px 0" }}>{TRANSLATIONS[lang].calibratingProgress}</h3>
                        <p style={{ fontSize: "13px", color: "rgba(255,255,255,0.7)", margin: "0 0 16px 0" }}>{TRANSLATIONS[lang].stayInView}</p>
                        <div style={{ fontSize: "36px", fontWeight: "bold", color: "var(--color-primary)" }}>{calibrationCountdown}s</div>
                      </div>
                    </div>
                  )}
                  {cameraActive && (
                    <div className="camera-live-badge">
                      <div className="live-dot" />
                      <span>{lang === 'hi' ? "लाइव" : lang === 'bn' ? "লাইভ" : "LIVE"}</span>
                    </div>
                  )}

                  {/* Zoom level badge — top right, always visible while zoomed */}
                  {cameraActive && zoomLevel !== 1 && (
                    <div className="zoom-badge">
                      {zoomLevel < 1 ? `${zoomLevel.toFixed(1)}×` : zoomLevel >= 10 ? '10×' : `${zoomLevel.toFixed(1)}×`}
                    </div>
                  )}

                  {/* Transparent Vertical Zoom Control — always visible */}
                  {cameraActive && (
                    <div className="zoom-vertical-control" onClick={(e) => e.stopPropagation()}>
                      <button 
                        className="zoom-btn"
                        onClick={() => applyZoom(zoomLevel + 0.5)}
                        aria-label="Zoom in"
                      >
                        +
                      </button>
                      
                      <div className="zoom-vertical-slider-container">
                        <input
                          type="range"
                          className="zoom-slider-vertical"
                          min={0.5}
                          max={10}
                          step={0.1}
                          value={zoomLevel}
                          onChange={(e) => applyZoom(parseFloat(e.target.value))}
                        />
                      </div>

                      <button 
                        className="zoom-btn"
                        onClick={() => applyZoom(zoomLevel - 0.5)}
                        aria-label="Zoom out"
                      >
                        −
                      </button>
                      
                      {zoomCapable && (
                        <span className="zoom-hw-dot" title="Hardware Zoom Active" />
                      )}
                    </div>
                  )}

                  {!cameraActive && (
                    <div className="camera-placeholder">
                      <div className="camera-empty-icon">
                        <VideoOff size={32} />
                      </div>
                      <p>{TRANSLATIONS[lang].cameraInactive}</p>
                    </div>
                  )}
                </div>

                {/* Caption bar visible inside the camera frame wrapper unconditionally */}
                {cameraActive && (
                  <div className="fullscreen-caption-bar" key={displayCorrectionText}>
                    <span className={displayCorrectionText ? 'caption-text active' : 'caption-text placeholder'}>
                      {displayCorrectionText || TRANSLATIONS[lang].alignBody}
                    </span>
                  </div>
                )}

                {/* Fullscreen-only overlays */}
                {isFullscreen && (
                  <>
                    {/* Exit button */}
                    <button
                      className="fullscreen-exit-btn"
                      onClick={exitFullscreen}
                      aria-label="Exit fullscreen"
                    >
                      <X size={16} />
                      <span>{TRANSLATIONS[lang].exit}</span>
                    </button>

                    {/* Score badge */}
                    <div className={`fullscreen-score-badge ${effectiveCorrectness >= 0.75 ? 'good' : 'warn'}`}>
                      {poseMismatch ? TRANSLATIONS[lang].wrongPoseBadge : `${Math.round(effectiveCorrectness * 100)}%`}
                    </div>
                  </>
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
                    <span className="kpi-label">{TRANSLATIONS[lang].postureScore}</span>
                    <span className="kpi-value">{poseMismatch ? TRANSLATIONS[lang].wrongPoseBadge : `${Math.round(effectiveCorrectness * 100)}%`}</span>
                    <span className="kpi-sub">{effectiveCorrectness >= 0.75 ? TRANSLATIONS[lang].onTarget : TRANSLATIONS[lang].needsAdjustment}</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">{TRANSLATIONS[lang].detectedPose}</span>
                    <span className="kpi-value">{activePose === "transition/unknown" ? "—" : getSanskritName(activePose, lang)}</span>
                    <span className="kpi-sub">{flowPose === "transition/unknown" ? TRANSLATIONS[lang].staticMode : TRANSLATIONS[lang].flowMode}</span>
                  </div>
                  <div className="kpi-card">
                    <span className="kpi-label">{TRANSLATIONS[lang].fusing}</span>
                    <span className="kpi-value">{recoveredJoints.length > 0 ? recoveredJoints.length : "—"}</span>
                    <span className="kpi-sub">{recoveredJoints.length > 0 ? TRANSLATIONS[lang].occlusionActive : TRANSLATIONS[lang].allVisible}</span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Real-time Feedback & Guidance */}
          <div className="glass-panel">
            <h2 className="section-title">
              <Gauge size={20} />
              <span>{TRANSLATIONS[lang].feedbackHub}</span>
            </h2>

            <div className="feedback-grid">
              <div className="gauge-container glass-panel">
                <ScoreRing correctness={effectiveCorrectness} />
                <span className="gauge-label">{TRANSLATIONS[lang].correctnessScore}</span>
              </div>

              <div className="metrics-list">
                <div className="deviation-item m-0">
                  <span className="deviation-name">{TRANSLATIONS[lang].detectedPose}</span>
                  <span className="metric-value primary">
                    {getSanskritName(activePose, lang)}
                  </span>
                </div>
                <div className="deviation-item m-0">
                  <span className="deviation-name">{TRANSLATIONS[lang].sequenceFlow}</span>
                  <span className="metric-value">
                    {flowPose === "transition/unknown" ? TRANSLATIONS[lang].staticMode : getSanskritName(flowPose, lang)}
                  </span>
                </div>
                <div className="deviation-item m-0">
                  <span className="deviation-name">{TRANSLATIONS[lang].occlusionFusing}</span>
                  <span className={`metric-value ${recoveredJoints.length > 0 ? "warning" : "success"}`}>
                    {recoveredJoints.length > 0 ? `${TRANSLATIONS[lang].active} (${recoveredJoints.length})` : TRANSLATIONS[lang].inactive}
                  </span>
                </div>
              </div>
            </div>

            {/* Occlusion recovery logs */}
            {recoveredJoints.length > 0 && (
              <div className="occlusion-alert">
                <p className="occlusion-alert-text">
                  <ShieldAlert size={14} />
                  <span><strong>{TRANSLATIONS[lang].fusingOccluded}</strong> {recoveredJoints.map(j => JOINT_TRANSLATIONS[lang][j] || j).join(", ")} {TRANSLATIONS[lang].mirroredCoordinates}</span>
                </p>
              </div>
            )}

            {/* Stable height status/guidance box */}
            {activePose === "transition/unknown" ? (
              <div className="guidance-box info" key="waiting-pose">
                <Activity size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">{TRANSLATIONS[lang].systemStatus}</span>
                  <span className="guidance-text">{TRANSLATIONS[lang].detectingPose}</span>
                </div>
              </div>
            ) : poseMismatch ? (
              <div className="guidance-box" key="pose-mismatch">
                <ShieldAlert size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">{TRANSLATIONS[lang].wrongPoseBadge}</span>
                  <span className="guidance-text">{displayCorrectionText}</span>
                </div>
              </div>
            ) : correctionText ? (
              <div className="guidance-box" key={correctionText}>
                <Volume2 size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">
                    {TRANSLATIONS[lang].safetyCorrection}
                    {correctionIsSafe ? (
                      <ShieldCheck size={13} className="safety-verified-icon" aria-label="AI-verified safe" />
                    ) : (
                      <ShieldAlert size={13} className="safety-flagged-icon" aria-label="Fell back to reviewed template" />
                    )}
                  </span>
                  <span className="guidance-text">{correctionText}</span>
                </div>
              </div>
            ) : (
              <div className="guidance-box success" key="alignment-correct">
                <CheckCircle2 size={24} />
                <div className="guidance-content">
                  <span className="guidance-label-text">{TRANSLATIONS[lang].alignmentCorrect}</span>
                  <span className="guidance-text">{TRANSLATIONS[lang].alignmentCorrectDesc}</span>
                </div>
              </div>
            )}
          </div>

          {/* Angle details */}
          <div className="glass-panel">
            <h2 className="section-title">
              <HelpCircle size={20} />
              <span>{TRANSLATIONS[lang].angleDetails}</span>
            </h2>
            
            {activePose === "transition/unknown" ? (
              <div className="angle-placeholder-text">
                <p>{TRANSLATIONS[lang].assumePosePrompt}</p>
              </div>
            ) : poseMismatch ? (
              <div className="angle-placeholder-text">
                <p>{displayCorrectionText}</p>
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
                  const translatedLabel = JOINT_TRANSLATIONS[lang][joint] || label;

                  return (
                    <div className="deviation-item" key={joint}>
                      <span className="deviation-name">{translatedLabel} ({TRANSLATIONS[lang].targetFor.replace('{target}', target.toString()).replace('{pose}', getSanskritName(activePose, lang))})</span>
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
                          {currentAngle}° ({TRANSLATIONS[lang].diff} {diff > 0 ? "+" : ""}{diff}°)
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
