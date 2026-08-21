import Svg, { Circle, Line, Polygon } from "react-native-svg";
import { StyleSheet, View } from "react-native";

import { colors } from "@/components/asana-ui";
import { LANDMARK_CONNECTIONS, SKELETON_NODES, type Landmark } from "@/lib/pose-geometry";

const nodeJoints: Record<number, string[]> = {
  11: ["shoulder_l", "trunk_l"], 12: ["shoulder_r", "trunk_r"], 13: ["elbow_l"], 14: ["elbow_r"],
  23: ["hip_l", "hip_abduct_l"], 24: ["hip_r", "hip_abduct_r"], 25: ["knee_l"], 26: ["knee_r"], 27: ["ankle_l"], 28: ["ankle_r"],
};

function jointColor(index: number, deviations: Record<string, number>) {
  const deviation = (nodeJoints[index] ?? []).reduce((maximum, joint) => Math.max(maximum, deviations[joint] ?? 0), 0);
  if (deviation >= 25) return colors.terracotta;
  if (deviation >= 12) return "#C58A2B";
  return colors.moss;
}

export function PoseSkeleton({ landmarks, deviations, mirror = false }: { landmarks: Landmark[]; deviations: Record<string, number>; mirror?: boolean }) {
  if (landmarks.length !== 33) return null;
  const coordinate = (landmark: Landmark) => ({ x: `${(mirror ? 1 - landmark.x : landmark.x) * 100}%`, y: `${landmark.y * 100}%` });
  const torso = [11, 12, 24, 23].map((index) => coordinate(landmarks[index]));
  return (
    <View pointerEvents="none" style={StyleSheet.absoluteFill}>
      <Svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
        <Polygon points={torso.map((point) => `${Number.parseFloat(point.x)},${Number.parseFloat(point.y)}`).join(" ")} fill="rgba(92,107,78,0.14)" />
        {LANDMARK_CONNECTIONS.map(([from, to]) => {
          const start = landmarks[from]; const end = landmarks[to];
          if (start.visibility < 0.35 || end.visibility < 0.35) return null;
          const a = coordinate(start); const b = coordinate(end);
          return <Line key={`${from}-${to}`} x1={a.x} y1={a.y} x2={b.x} y2={b.y} stroke={jointColor(from, deviations)} strokeWidth="1.15" strokeLinecap="round" />;
        })}
        {SKELETON_NODES.map((index) => {
          const point = landmarks[index];
          if (!point || point.visibility < 0.35) return null;
          const location = coordinate(point);
          return <Circle key={index} cx={location.x} cy={location.y} r={index === 0 ? "1.35" : "1.6"} fill={colors.paper} stroke={jointColor(index, deviations)} strokeWidth="0.8" />;
        })}
      </Svg>
    </View>
  );
}
