import "./scripts/load-env.js";
import type { ExpoConfig } from "expo/config";

const env = {
  appName: "AsanaAI",
  appSlug: "asana-ai-mobile",
  logoUrl: "/manus-storage/asana-ai-icon_f37971ad.png",
  scheme: "asanaai",
  iosBundleId: "space.manus.asana.ai.mobile",
  androidPackage: "space.manus.asana.ai.mobile",
};

const config: ExpoConfig = {
  name: env.appName,
  slug: env.appSlug,
  version: "1.0.0",
  orientation: "portrait",
  icon: "./assets/images/icon.png",
  scheme: env.scheme,
  userInterfaceStyle: "light",
  newArchEnabled: true,
  ios: {
    supportsTablet: true,
    bundleIdentifier: env.iosBundleId,
    infoPlist: { ITSAppUsesNonExemptEncryption: false },
  },
  android: {
    adaptiveIcon: {
      backgroundColor: "#F8F7F2",
      foregroundImage: "./assets/images/android-icon-foreground.png",
      backgroundImage: "./assets/images/android-icon-background.png",
      monochromeImage: "./assets/images/android-icon-monochrome.png",
    },
    edgeToEdgeEnabled: true,
    predictiveBackGestureEnabled: true,
    package: env.androidPackage,
    permissions: ["CAMERA"],
  },
  web: {
    bundler: "metro",
    output: "static",
    favicon: "./assets/images/favicon.png",
  },
  plugins: [
    "expo-router",
    "expo-asset",
    "expo-audio",
    "expo-font",
    "expo-video",
    "expo-web-browser",
    [
      "expo-camera",
      {
        cameraPermission: "Allow AsanaAI to use your camera for yoga practice.",
        microphonePermission: "Allow AsanaAI to use your microphone for optional voice guidance.",
        recordAudioAndroid: false,
      },
    ],
    [
      "expo-splash-screen",
      {
        image: "./assets/images/splash-icon.png",
        imageWidth: 200,
        resizeMode: "contain",
        backgroundColor: "#F8F7F2",
      },
    ],
    [
      "expo-build-properties",
      {
        android: { buildArchs: ["armeabi-v7a", "arm64-v8a"], minSdkVersion: 24 },
      },
    ],
  ],
  experiments: { typedRoutes: true, reactCompiler: true },
  extra: env,
};

export default config;
