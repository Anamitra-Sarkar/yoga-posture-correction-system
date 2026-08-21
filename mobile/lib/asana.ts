export type PoseTarget = {
  joint: string;
  label: string;
  target: number;
  tolerance: number;
};

export type Pose = {
  id: string;
  name: string;
  Sanskrit: string;
  cue: string;
  level: "Beginner" | "Intermediate";
  accent: "moss" | "terracotta";
  targets: PoseTarget[];
  steps: string[];
};

export const POSES: Pose[] = [
  {
    id: "warrior_2",
    name: "Warrior II",
    Sanskrit: "Virabhadrasana II",
    cue: "Front knee at 90° over ankle; arms parallel at shoulder height.",
    level: "Intermediate",
    accent: "terracotta",
    targets: [
      { joint: "knee_l", label: "Front knee", target: 90, tolerance: 15 },
      { joint: "shoulder_l", label: "Left shoulder", target: 90, tolerance: 15 },
      { joint: "knee_r", label: "Back knee", target: 180, tolerance: 15 },
    ],
    steps: ["Step feet wide and turn the front toes outward.", "Bend the front knee until it tracks above the ankle.", "Extend both arms long at shoulder height and look over front fingers."],
  },
  {
    id: "cobra_pose",
    name: "Cobra",
    Sanskrit: "Bhujangasana",
    cue: "Hands under shoulders; lift the chest while hips remain grounded.",
    level: "Beginner",
    accent: "moss",
    targets: [
      { joint: "neck", label: "Neck extension", target: 140, tolerance: 20 },
      { joint: "trunk_l", label: "Trunk extension", target: 140, tolerance: 20 },
    ],
    steps: ["Lie face down with hands below the shoulders.", "Press lightly through the hands and draw shoulders back.", "Lift the chest while keeping hips and thighs grounded."],
  },
  {
    id: "mountain_pose",
    name: "Mountain",
    Sanskrit: "Tadasana",
    cue: "Stand tall with feet grounded and shoulders stacked over hips.",
    level: "Beginner",
    accent: "moss",
    targets: [
      { joint: "knee_l", label: "Left knee", target: 180, tolerance: 10 },
      { joint: "knee_r", label: "Right knee", target: 180, tolerance: 10 },
      { joint: "trunk_l", label: "Spine", target: 180, tolerance: 10 },
    ],
    steps: ["Stand with both feet grounded and weight even.", "Stack ribs and shoulders over the hips.", "Relax the arms and keep a quiet forward gaze."],
  },
  {
    id: "tree_pose",
    name: "Tree",
    Sanskrit: "Vrikshasana",
    cue: "Root through the standing foot and keep hips level.",
    level: "Intermediate",
    accent: "terracotta",
    targets: [
      { joint: "knee_r", label: "Standing knee", target: 175, tolerance: 15 },
      { joint: "hip_r", label: "Standing hip", target: 175, tolerance: 10 },
    ],
    steps: ["Root through one foot before lifting the other.", "Rest the lifted foot at the calf or inner thigh, never the knee.", "Keep hips level and choose one steady point to gaze at."],
  },
  {
    id: "plank",
    name: "Plank",
    Sanskrit: "Phalakasana",
    cue: "Create one long line from shoulders to heels; keep hands under shoulders.",
    level: "Intermediate",
    accent: "moss",
    targets: [
      { joint: "hip_l", label: "Hip line", target: 160, tolerance: 15 },
      { joint: "knee_l", label: "Knee line", target: 160, tolerance: 15 },
      { joint: "shoulder_l", label: "Shoulder arm", target: 85, tolerance: 20 },
    ],
    steps: ["Place hands directly beneath the shoulders.", "Step the feet back and engage through both legs.", "Keep one long line from shoulders through the heels."],
  },
  {
    id: "downward_dog",
    name: "Downward Dog",
    Sanskrit: "Adho Mukha Svanasana",
    cue: "Lift hips into an inverted V; share weight between hands and feet.",
    level: "Beginner",
    accent: "terracotta",
    targets: [
      { joint: "hip_l", label: "Hip fold", target: 80, tolerance: 30 },
      { joint: "knee_l", label: "Leg extension", target: 145, tolerance: 25 },
      { joint: "shoulder_l", label: "Arm line", target: 137, tolerance: 25 },
    ],
    steps: ["Set hands shoulder-width and feet hip-width apart.", "Lift the hips high to make an inverted V shape.", "Reach heels down gently while lengthening through both arms."],
  },
];

const LEGACY_POSE_IDS: Record<string, string> = {
  cobra: "cobra_pose",
  mountain: "mountain_pose",
  tree: "tree_pose",
};

export const DEFAULT_POSE_ID = "warrior_2";

export function normalizePoseId(id: string | null | undefined) {
  if (!id) return DEFAULT_POSE_ID;
  return LEGACY_POSE_IDS[id] ?? id;
}

export function getPose(id: string | null | undefined) {
  return POSES.find((pose) => pose.id === normalizePoseId(id)) ?? POSES[0];
}

export function displayPoseId(value: string | null | undefined) {
  if (!value || value === "transition/unknown") return "Waiting for a reading";
  const known = POSES.find((pose) => pose.id === normalizePoseId(value));
  if (known) return known.name;
  return value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export type ServiceState = "idle" | "checking" | "available" | "offline" | "permission" | "no-person";

export function serviceCopy(state: ServiceState) {
  const copy: Record<ServiceState, { title: string; detail: string }> = {
    idle: { title: "Ready when you are", detail: "Set your pose, then begin a camera check." },
    checking: { title: "Checking alignment", detail: "Keep your whole body in the framing guide." },
    available: { title: "Coaching is live", detail: "Landmarks and guidance update as you hold the pose." },
    offline: { title: "Coaching unavailable", detail: "Check your connection, then try again." },
    permission: { title: "Camera access is needed", detail: "Allow access to begin a live practice." },
    "no-person": { title: "Step back a little", detail: "Include your head, hands, and feet in the frame." },
  };
  return copy[state];
}
