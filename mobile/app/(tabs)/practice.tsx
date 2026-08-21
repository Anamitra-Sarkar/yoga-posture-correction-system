import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import * as Haptics from "expo-haptics";
import { CameraView, useCameraPermissions } from "expo-camera";
import { useKeepAwake } from "expo-keep-awake";
import { useFocusEffect } from "expo-router";
import { useCallback, useRef, useState } from "react";
import { Pressable, StyleSheet, Text, View } from "react-native";
import { WebView } from "react-native-webview";

import { AsanaMark, colors, IconButton, PrimaryButton, SectionLabel, StatusDot } from "@/components/asana-ui";
import { PosePicker } from "@/components/pose-picker";
import { ScreenContainer } from "@/components/screen-container";
import { DEFAULT_POSE_ID, getPose, LIVE_COACH_URL, serviceCopy, type ServiceState } from "@/lib/asana";
import { DEFAULT_PREFERENCES, loadPreferences, savePreferences, type Preferences } from "@/lib/preferences";
import { recordPractice } from "@/lib/practice-history";

export default function PracticeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [coachOpen, setCoachOpen] = useState(false);
  const [coachLoadError, setCoachLoadError] = useState(false);
  const [serviceState, setServiceState] = useState<ServiceState>("idle");
  const [cameraReady, setCameraReady] = useState(false);
  const cameraRef = useRef<CameraView>(null);
  useKeepAwake();
  const pose = getPose(preferences.poseId || DEFAULT_POSE_ID);

  useFocusEffect(
    useCallback(() => {
      let mounted = true;
      loadPreferences().then((value) => mounted && setPreferences(value));
      return () => { mounted = false; };
    }, []),
  );

  const getCameraReady = async () => {
    if (!permission?.granted) {
      setServiceState("permission");
      const result = await requestPermission();
      if (!result.granted) return;
    }
    setServiceState("checking");
    setCameraReady(true);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const choosePose = async (nextPose: typeof pose) => {
    const next = { ...preferences, poseId: nextPose.id };
    setPreferences(next);
    setPickerOpen(false);
    await savePreferences(next);
    Haptics.selectionAsync();
  };

  const openCoach = async () => {
    await recordPractice(pose.id);
    setCoachLoadError(false);
    setCoachOpen(true);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const closeCoach = () => {
    setCoachOpen(false);
    setServiceState(cameraReady ? "available" : "idle");
  };

  if (coachOpen) {
    return (
      <View style={styles.coachScreen}>
        <View style={styles.coachHeader}>
          <IconButton label="Close live coach" icon="close" onPress={closeCoach} />
          <View style={{ flex: 1, marginLeft: 12 }}>
            <Text style={styles.coachTitle}>Live coach</Text>
            <Text style={styles.coachSubhead}>AsanaAI web experience</Text>
          </View>
          <View style={styles.livePill}><StatusDot /><Text style={styles.liveText}>LIVE</Text></View>
        </View>
        {coachLoadError ? (
          <View style={styles.coachError}>
            <View style={styles.errorMark}><MaterialIcons name="wifi-off" size={25} color={colors.terracotta} /></View>
            <Text style={styles.errorTitle}>The live coach could not open</Text>
            <Text style={styles.errorDetail}>Check your connection, then try again. Your target pose and local preferences are still here.</Text>
            <PrimaryButton label="Try again" icon="refresh" tone="terracotta" onPress={() => setCoachLoadError(false)} />
          </View>
        ) : (
          <WebView
            source={{ uri: LIVE_COACH_URL }}
            startInLoadingState
            javaScriptEnabled
            domStorageEnabled
            mediaPlaybackRequiresUserAction={false}
            onLoadStart={() => { setCoachLoadError(false); setServiceState("checking"); }}
            onLoadEnd={() => setServiceState("available")}
            onError={() => { setServiceState("offline"); setCoachLoadError(true); }}
            onHttpError={() => { setServiceState("offline"); setCoachLoadError(true); }}
            renderLoading={() => <View style={styles.webLoading}><Text style={styles.webLoadingText}>Opening your practice space…</Text></View>}
          />
        )}
      </View>
    );
  }

  const status = serviceCopy(permission?.granted && cameraReady ? "available" : serviceState);
  return (
    <ScreenContainer className="flex-1" containerClassName="bg-background">
      <View style={styles.topBar}>
        <AsanaMark />
        <Pressable onPress={() => setPickerOpen(true)} style={({ pressed }) => [styles.poseControl, pressed && styles.pressed]} accessibilityRole="button">
          <Text style={styles.poseControlText}>{pose.name}</Text>
          <MaterialIcons name="expand-more" size={18} color={colors.ink} />
        </Pressable>
      </View>

      <View style={styles.content}>
        <View style={styles.context}>
          <SectionLabel>Practice view</SectionLabel>
          <Text style={styles.title}>Set your stance.</Text>
          <Text style={styles.subtitle}>{pose.cue}</Text>
        </View>

        <View style={styles.cameraFrame}>
          {permission?.granted && cameraReady ? (
            <CameraView
              ref={cameraRef}
              style={StyleSheet.absoluteFill}
              facing="front"
              mirror
              onCameraReady={() => setServiceState("available")}
              onMountError={() => setServiceState("offline")}
            />
          ) : (
            <View style={styles.cameraEmpty}>
              <View style={styles.cameraIcon}><MaterialIcons name="videocam-off" size={29} color={colors.moss} /></View>
              <Text style={styles.cameraEmptyTitle}>See your whole body</Text>
              <Text style={styles.cameraEmptyDetail}>Place your phone far enough away to include hands and feet.</Text>
            </View>
          )}
          {permission?.granted && cameraReady ? <View pointerEvents="none" style={styles.frameGuide}><View style={styles.frameGuideInner} /></View> : null}
          <View style={styles.cameraLabel}>
            <StatusDot tone={permission?.granted && cameraReady ? "moss" : "mist"} />
            <Text style={styles.cameraLabelText}>{permission?.granted && cameraReady ? "Camera ready" : "Camera off"}</Text>
          </View>
        </View>

        <View style={styles.statusPanel}>
          <View style={[styles.statusIcon, { backgroundColor: serviceState === "offline" ? colors.clay : colors.sage }]}>
            <MaterialIcons name={serviceState === "offline" ? "wifi-off" : "spa"} size={20} color={serviceState === "offline" ? colors.terracotta : colors.moss} />
          </View>
          <View style={{ flex: 1 }}>
            <Text style={styles.statusTitle}>{status.title}</Text>
            <Text style={styles.statusDetail}>{status.detail}</Text>
          </View>
        </View>

        {!permission?.granted || !cameraReady ? (
          <PrimaryButton label="Enable camera" icon="videocam" onPress={getCameraReady} />
        ) : (
          <PrimaryButton label="Open live coach" icon="open-in-new" tone="terracotta" onPress={openCoach} />
        )}
        <Text style={styles.privacyNote}>The embedded live coach uses the existing AsanaAI service. This native screen does not invent posture scores or store your camera feed.</Text>
      </View>
      <PosePicker visible={pickerOpen} selectedPoseId={pose.id} onSelect={choosePose} onClose={() => setPickerOpen(false)} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  topBar: { paddingHorizontal: 20, paddingTop: 12, flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  poseControl: { flexDirection: "row", alignItems: "center", gap: 3, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, paddingLeft: 12, paddingRight: 8, height: 38, borderRadius: 19 },
  poseControlText: { color: colors.ink, fontSize: 12, fontWeight: "700" },
  content: { flex: 1, paddingHorizontal: 20, paddingTop: 29, paddingBottom: 20, gap: 17 },
  context: { gap: 7 },
  title: { color: colors.ink, fontSize: 31, letterSpacing: -0.8, fontWeight: "700" },
  subtitle: { color: colors.mist, fontSize: 14, lineHeight: 20 },
  cameraFrame: { minHeight: 286, flex: 1, maxHeight: 394, backgroundColor: "#E9ECE5", borderRadius: 26, overflow: "hidden", position: "relative" },
  cameraEmpty: { flex: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: 40, gap: 10 },
  cameraIcon: { width: 58, height: 58, borderRadius: 29, backgroundColor: colors.paper, alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: colors.line, marginBottom: 2 },
  cameraEmptyTitle: { color: colors.ink, fontSize: 17, fontWeight: "700" },
  cameraEmptyDetail: { color: colors.mist, fontSize: 13, lineHeight: 19, textAlign: "center" },
  cameraLabel: { position: "absolute", top: 14, left: 14, flexDirection: "row", gap: 7, alignItems: "center", paddingHorizontal: 10, height: 31, borderRadius: 15.5, backgroundColor: "rgba(248,247,242,0.92)" },
  cameraLabelText: { color: colors.ink, fontSize: 11, fontWeight: "700" },
  frameGuide: { ...StyleSheet.absoluteFillObject, alignItems: "center", justifyContent: "center" },
  frameGuideInner: { height: "78%", width: "62%", borderWidth: 1.5, borderColor: "rgba(248,247,242,0.75)", borderRadius: 86 },
  statusPanel: { flexDirection: "row", gap: 12, padding: 15, borderRadius: 18, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.white, alignItems: "center" },
  statusIcon: { width: 38, height: 38, borderRadius: 19, alignItems: "center", justifyContent: "center" },
  statusTitle: { color: colors.ink, fontSize: 14, fontWeight: "700" },
  statusDetail: { color: colors.mist, fontSize: 12, lineHeight: 17, marginTop: 3 },
  privacyNote: { color: colors.mist, fontSize: 11, lineHeight: 16, textAlign: "center", paddingHorizontal: 13 },
  pressed: { opacity: 0.82, transform: [{ scale: 0.98 }] },
  coachScreen: { flex: 1, backgroundColor: colors.paper },
  coachHeader: { minHeight: 70, paddingHorizontal: 16, paddingTop: 14, paddingBottom: 10, flexDirection: "row", alignItems: "center", borderBottomWidth: 1, borderBottomColor: colors.line },
  coachTitle: { color: colors.ink, fontSize: 15, fontWeight: "700" },
  coachSubhead: { color: colors.mist, fontSize: 11, marginTop: 2 },
  livePill: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: colors.sage, paddingHorizontal: 10, height: 28, borderRadius: 14 },
  liveText: { color: colors.moss, fontSize: 10, fontWeight: "800", letterSpacing: 0.8 },
  webLoading: { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: colors.paper },
  webLoadingText: { color: colors.mist, fontSize: 14 },
  coachError: { flex: 1, alignItems: "center", justifyContent: "center", backgroundColor: colors.paper, paddingHorizontal: 36, gap: 13 },
  errorMark: { width: 54, height: 54, borderRadius: 27, alignItems: "center", justifyContent: "center", backgroundColor: colors.clay },
  errorTitle: { color: colors.ink, fontSize: 20, fontWeight: "700", textAlign: "center", letterSpacing: -0.3 },
  errorDetail: { color: colors.mist, fontSize: 14, textAlign: "center", lineHeight: 20, marginBottom: 4 },
});
