import { Camera } from "expo-camera";
import { Platform } from "react-native";

import { analyseFrame } from "@/lib/yoga-api";

export type DeviceReadiness = {
  cameraAvailable: boolean;
  permission: "granted" | "denied" | "undetermined";
  serviceReachable: boolean;
  serviceLatencyMs: number | null;
  error?: string;
};

const PROBE_ANGLES = [180, 180, 90, 90, 170, 170, 175, 175, 160, 160, 170, 170, 180, 90, 90];

export async function checkDeviceReadiness(): Promise<DeviceReadiness> {
  const permission = await Camera.getCameraPermissionsAsync();
  const cameraAvailable = Platform.OS !== "web";
  const startedAt = Date.now();
  try {
    await analyseFrame(PROBE_ANGLES);
    return {
      cameraAvailable,
      permission: permission.granted ? "granted" : permission.canAskAgain ? "undetermined" : "denied",
      serviceReachable: true,
      serviceLatencyMs: Date.now() - startedAt,
    };
  } catch (error) {
    return {
      cameraAvailable,
      permission: permission.granted ? "granted" : permission.canAskAgain ? "undetermined" : "denied",
      serviceReachable: false,
      serviceLatencyMs: null,
      error: error instanceof Error ? error.message : "Unable to reach the posture analysis service.",
    };
  }
}
