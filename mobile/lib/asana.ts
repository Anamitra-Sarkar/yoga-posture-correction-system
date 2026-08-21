export type Pose = {
  id: string;
  name: string;
  Sanskrit: string;
  cue: string;
  level: "Grounded" | "Open" | "Steady";
  accent: "moss" | "terracotta";
};

export const LIVE_COACH_URL = "https://yoga-posture-correction-system.vercel.app";

export const POSES: Pose[] = [
  {
    id: "warrior_2",
    name: "Warrior II",
    Sanskrit: "Virabhadrasana II",
    cue: "Front knee above ankle; arms reach long.",
    level: "Steady",
    accent: "terracotta",
  },
  {
    id: "cobra",
    name: "Cobra",
    Sanskrit: "Bhujangasana",
    cue: "Draw shoulders back and lengthen through the chest.",
    level: "Open",
    accent: "moss",
  },
  {
    id: "mountain",
    name: "Mountain",
    Sanskrit: "Tadasana",
    cue: "Stand tall, soften the ribs, feel both feet.",
    level: "Grounded",
    accent: "moss",
  },
  {
    id: "tree",
    name: "Tree",
    Sanskrit: "Vrikshasana",
    cue: "Find one quiet point to look at and breathe.",
    level: "Steady",
    accent: "terracotta",
  },
  {
    id: "plank",
    name: "Plank",
    Sanskrit: "Phalakasana",
    cue: "Press the floor away; make one long line.",
    level: "Steady",
    accent: "moss",
  },
  {
    id: "downward_dog",
    name: "Downward Dog",
    Sanskrit: "Adho Mukha Svanasana",
    cue: "Lift the hips and lengthen both sides of the waist.",
    level: "Open",
    accent: "terracotta",
  },
];

export const DEFAULT_POSE_ID = "warrior_2";

export function getPose(id: string) {
  return POSES.find((pose) => pose.id === id) ?? POSES[0];
}

export function displayPoseId(value: string | null | undefined) {
  if (!value) return "Waiting for a reading";
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export type ServiceState = "idle" | "checking" | "available" | "offline" | "permission";

export function serviceCopy(state: ServiceState) {
  const copy: Record<ServiceState, { title: string; detail: string }> = {
    idle: {
      title: "Ready when you are",
      detail: "Set your pose, then begin with the camera.",
    },
    checking: {
      title: "Checking your setup",
      detail: "Keep your whole body in frame.",
    },
    available: {
      title: "Camera ready",
      detail: "Open the live coach when your stance feels set.",
    },
    offline: {
      title: "Live coach unavailable",
      detail: "Check your connection, then try again.",
    },
    permission: {
      title: "Camera access is needed",
      detail: "Allow access to use the live coach and practice view.",
    },
  };
  return copy[state];
}
