import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import * as Haptics from "expo-haptics";
import * as Speech from "expo-speech";
import { CameraView, useCameraPermissions, type CameraType } from "expo-camera";
import { activateKeepAwakeAsync, deactivateKeepAwake } from "expo-keep-awake";
import { useFocusEffect, useNavigation } from "expo-router";
import { useCallback, useEffect, useRef, useState } from "react";
import { LayoutAnimation, Platform, Pressable, ScrollView, StyleSheet, Text, UIManager, useWindowDimensions, View } from "react-native";

import { AsanaMark, colors, IconButton, PrimaryButton, SectionLabel, StatusDot } from "@/components/asana-ui";
import { PoseDetectorWorker } from "@/components/pose-detector-worker";
import { PosePicker } from "@/components/pose-picker";
import { PoseGuide } from "@/components/pose-guide";
import { PoseSkeleton } from "@/components/pose-skeleton";
import { ScreenContainer } from "@/components/screen-container";
import { DEFAULT_POSE_ID, getPose, serviceCopy, type Pose, type ServiceState } from "@/lib/asana";
import { getCoachingDiagnostic } from "@/lib/coaching-diagnostics";
import { extractAnglesFromLandmarks, landmarksFromRows, type Landmark } from "@/lib/pose-geometry";
import { DEFAULT_PREFERENCES, loadPreferences, savePreferences, type Preferences } from "@/lib/preferences";
import { completePractice, recordPractice } from "@/lib/practice-history";
import { analyseFrame, generateCorrection, recoverOcclusion } from "@/lib/yoga-api";

type DetectorRequest = { id: string; base64: string } | null;
type CoachingResult = { poseId: string; score: number; deviations: Record<string, number>; correction: string; safe: boolean; recovered: string[] };

export default function PracticeScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const [immersive, setImmersive] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [facing, setFacing] = useState<CameraType>("front");
  const [serviceState, setServiceState] = useState<ServiceState>("idle");
  const [detectorReady, setDetectorReady] = useState(Platform.OS === "web");
  const [detectorRequest, setDetectorRequest] = useState<DetectorRequest>(null);
  const [landmarks, setLandmarks] = useState<Landmark[]>([]);
  const [result, setResult] = useState<CoachingResult | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [processing, setProcessing] = useState(false);
  const [lastError, setLastError] = useState("");
  const cameraRef = useRef<CameraView>(null);
  const requestCounter = useRef(0);
  const frameInFlight = useRef(false);
  const lastCorrectionAt = useRef(0);
  const lastSpoken = useRef("");
  const navigation = useNavigation();
  const { height } = useWindowDimensions();
  const pose = getPose(preferences.poseId || DEFAULT_POSE_ID);
  const normalStageHeight = Math.max(218, Math.min(300, Math.round(height * 0.34)));

  useEffect(() => {
    if (Platform.OS === "android") UIManager.setLayoutAnimationEnabledExperimental?.(true);
  }, []);

  useEffect(() => {
    navigation.setOptions({ tabBarStyle: immersive ? { display: "none" } : undefined });
    return () => navigation.setOptions({ tabBarStyle: undefined });
  }, [immersive, navigation]);

  useFocusEffect(useCallback(() => {
    let mounted = true;
    loadPreferences().then((value) => mounted && setPreferences(value));
    return () => { mounted = false; setSessionActive(false); setCameraActive(false); Speech.stop(); };
  }, []));

  useEffect(() => {
    if (!sessionActive) { setElapsedSeconds(0); return; }
    const timer = setInterval(() => setElapsedSeconds((current) => current + 1), 1000);
    return () => clearInterval(timer);
  }, [sessionActive]);

  useEffect(() => {
    if (!sessionActive || Platform.OS === "web") return;
    void activateKeepAwakeAsync("asana-live-practice").catch(() => undefined);
    return () => { void deactivateKeepAwake("asana-live-practice").catch(() => undefined); };
  }, [sessionActive]);

  const resetResult = useCallback(() => {
    setLandmarks([]); setResult(null); setLastError(""); setServiceState("idle"); lastCorrectionAt.current = 0; lastSpoken.current = "";
  }, []);

  const choosePose = async (nextPose: Pose) => {
    const next = { ...preferences, poseId: nextPose.id };
    setPreferences(next); setPickerOpen(false); resetResult(); await savePreferences(next); Haptics.selectionAsync();
  };

  const speakIfNeeded = useCallback(async (text: string) => {
    if (!preferences.voiceEnabled || !text || text === lastSpoken.current) return;
    const speaking = await Speech.isSpeakingAsync();
    if (speaking) return;
    lastSpoken.current = text;
    Speech.speak(text, { language: preferences.language === "hi" ? "hi-IN" : preferences.language === "bn" ? "bn-IN" : "en-US", rate: 0.92 });
  }, [preferences.language, preferences.voiceEnabled]);

  const processLandmarks = useCallback(async (_id: string, detected: Landmark[]) => {
    if (detected.length !== 33) {
      frameInFlight.current = false; setProcessing(false); setLastError("Move far enough back to keep your head, hands, and feet inside the guide."); setServiceState("no-person"); setLandmarks([]); return;
    }
    try {
      const occlusion = await recoverOcclusion(detected);
      const fused = landmarksFromRows(occlusion.fused_landmarks);
      const frame = await analyseFrame(extractAnglesFromLandmarks(fused));
      const diagnostic = getCoachingDiagnostic(pose.name, pose.id, frame.pose_id, frame.correctness_score);
      let correction = diagnostic.correction;
      let safe = diagnostic.safe;
      if (!diagnostic.isMismatch && (frame.correctness_score < 0.7 || Date.now() - lastCorrectionAt.current > 30000)) {
        const generated = await generateCorrection(frame.pose_id, frame.deviations, preferences.language);
        correction = generated.correction_text; safe = generated.is_safe; lastCorrectionAt.current = Date.now();
      }
      setLandmarks(fused);
      setResult({ poseId: diagnostic.detectedPose, score: frame.correctness_score, deviations: frame.deviations, correction, safe, recovered: occlusion.occluded_joints_recovered });
      setServiceState("available");
      void speakIfNeeded(correction);
    } catch (error) {
      setLastError(error instanceof Error ? error.message : "Unable to reach the coaching service.");
      setServiceState("offline");
    } finally {
      frameInFlight.current = false; setProcessing(false);
    }
  }, [pose.id, pose.name, preferences.language, speakIfNeeded]);

  const captureFrame = useCallback(async () => {
    if (Platform.OS === "web" || !sessionActive || !cameraRef.current || frameInFlight.current || !detectorReady) return;
    try {
      frameInFlight.current = true; setProcessing(true); setServiceState("checking");
      const picture = await cameraRef.current.takePictureAsync({ base64: true, quality: 0.34, skipProcessing: true, mirror: facing === "front" });
      if (!picture.base64) throw new Error("Camera did not return an analyzable frame.");
      const id = String(++requestCounter.current);
      setDetectorRequest({ id, base64: picture.base64 });
    } catch (error) {
      frameInFlight.current = false; setProcessing(false); setLastError(error instanceof Error ? error.message : "Unable to capture the camera frame."); setServiceState("offline");
    }
  }, [detectorReady, facing, sessionActive]);

  useEffect(() => {
    if (!sessionActive || Platform.OS === "web") return;
    void captureFrame();
    const interval = setInterval(() => void captureFrame(), 2200);
    return () => clearInterval(interval);
  }, [captureFrame, sessionActive]);

  const beginSession = async () => {
    if (!permission?.granted) {
      setServiceState("permission");
      const next = await requestPermission();
      if (!next.granted) return;
    }
    setCameraActive(true); setSessionActive(true); setServiceState("checking"); setLastError("");
    await recordPractice(pose.id); Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const endSession = async () => {
    if (result) await completePractice({ poseId: pose.id, score: result.score, detectedPoseId: result.poseId, durationSeconds: elapsedSeconds, correction: result.correction });
    setSessionActive(false); setCameraActive(false); setDetectorRequest(null); frameInFlight.current = false; setProcessing(false); Speech.stop(); setServiceState("idle");
  };

  const flipCamera = () => {
    if (processing) return;
    setFacing((current) => current === "front" ? "back" : "front");
    resetResult(); Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const setImmersiveMode = (next: boolean) => {
    LayoutAnimation.configureNext(LayoutAnimation.Presets.easeInEaseOut);
    setImmersive(next);
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };

  const status = serviceCopy(serviceState);
  const score = result ? `${Math.round(result.score * 100)}%` : "—";
  const timer = `${String(Math.floor(elapsedSeconds / 60)).padStart(2, "0")}:${String(elapsedSeconds % 60).padStart(2, "0")}`;

  const renderCameraStage = (fullScreen = false) => (
    <View style={[fullScreen ? styles.fullscreenCameraFrame : styles.cameraFrame, !fullScreen && { height: normalStageHeight }]}>
      {permission?.granted && cameraActive ? <CameraView ref={cameraRef} style={StyleSheet.absoluteFill} facing={facing} mirror={facing === "front"} onCameraReady={() => setServiceState(sessionActive ? "checking" : "idle")} onMountError={() => setServiceState("offline")} /> : <View style={styles.cameraEmpty}><View style={styles.cameraIcon}><MaterialIcons name="center-focus-strong" size={30} color={colors.moss} /></View><Text style={styles.cameraEmptyTitle}>Frame your whole body</Text><Text style={styles.cameraEmptyDetail}>Set the phone far enough away to include head, hands, and feet.</Text></View>}
      {landmarks.length === 33 ? <PoseSkeleton landmarks={landmarks} deviations={result?.deviations ?? {}} mirror={facing === "front"} /> : null}
      {permission?.granted && cameraActive && landmarks.length === 0 ? <View pointerEvents="none" style={styles.frameGuide}><View style={styles.frameGuideInner} /></View> : null}
      {fullScreen ? <>
        <View style={styles.fullscreenTop}><View style={styles.fullscreenStatus}><StatusDot tone={sessionActive ? "moss" : "mist"} /><Text style={styles.fullscreenStatusText}>{processing ? "Checking frame" : sessionActive ? "Camera live" : "Camera off"}</Text></View><View style={styles.fullscreenControls}>{permission?.granted && cameraActive ? <IconButton label="Switch camera" icon="flip-camera-android" onPress={flipCamera} /> : null}<IconButton label="Exit full-screen camera" icon="fullscreen-exit" onPress={() => setImmersiveMode(false)} /></View></View>
        <View style={styles.fullscreenBottom}><View style={styles.fullscreenMetrics}><Text style={styles.fullscreenPose}>{pose.name}</Text><Text style={styles.fullscreenMetric}>{result ? `${score} · ${result.poseId}` : status.title}</Text></View>{result?.correction ? <Text numberOfLines={2} style={styles.fullscreenCue}>{result.correction}</Text> : null}<PrimaryButton label={sessionActive ? "End session" : permission?.granted ? "Start coaching" : "Enable camera"} icon={sessionActive ? "stop-circle" : "videocam"} tone={sessionActive ? "terracotta" : "moss"} onPress={sessionActive ? endSession : beginSession} disabled={Platform.OS === "web"} /></View>
      </> : <>
        <View style={styles.cameraLabel}><StatusDot tone={sessionActive ? "moss" : "mist"} /><Text style={styles.cameraLabelText}>{processing ? "Checking frame" : sessionActive ? "Camera live" : "Camera off"}</Text></View>
        <View style={styles.stageControls}>{permission?.granted && cameraActive ? <IconButton label="Switch camera" icon="flip-camera-android" onPress={flipCamera} /> : null}<IconButton label="Open full-screen camera" icon="fullscreen" onPress={() => setImmersiveMode(true)} /></View>
        {result?.recovered.length ? <View style={styles.recoveryPill}><MaterialIcons name="visibility" size={14} color={colors.paper} /><Text style={styles.recoveryText}>{result.recovered.length} joint{result.recovered.length === 1 ? "" : "s"} restored</Text></View> : null}
      </>}
    </View>
  );

  if (immersive) return (
    <ScreenContainer edges={["top", "bottom", "left", "right"]} className="flex-1" containerClassName="bg-background">
      <View style={styles.immersiveRoot}>{renderCameraStage(true)}</View>
      <PosePicker visible={pickerOpen} selectedPoseId={pose.id} onSelect={choosePose} onClose={() => setPickerOpen(false)} />
      <PoseDetectorWorker request={detectorRequest} onReady={() => setDetectorReady(true)} onResult={processLandmarks} onError={(message) => { frameInFlight.current = false; setProcessing(false); setLastError(message); setServiceState("offline"); }} />
    </ScreenContainer>
  );

  return (
    <ScreenContainer className="flex-1" containerClassName="bg-background">
      <View style={styles.root}>
        <View style={styles.topBar}>
          <AsanaMark />
          <View style={styles.topActions}>
            <View style={styles.timerPill}><StatusDot tone={sessionActive ? "moss" : "mist"} /><Text style={styles.timer}>{timer}</Text></View>
            <Pressable onPress={() => setPickerOpen(true)} style={({ pressed }) => [styles.poseControl, pressed && styles.pressed]} accessibilityRole="button" accessibilityLabel="Change target pose">
              <Text style={styles.poseControlText}>{pose.name}</Text><MaterialIcons name="expand-more" size={18} color={colors.ink} />
            </Pressable>
          </View>
        </View>

        <ScrollView style={styles.scroller} contentContainerStyle={styles.content} showsVerticalScrollIndicator={false} bounces={false}>
          <View style={styles.context}><SectionLabel>Live practice</SectionLabel><Text style={styles.title}>{sessionActive ? "Hold your shape." : "Set your stance."}</Text><Text style={styles.subtitle}>{pose.cue}</Text><Pressable onPress={() => setGuideOpen(true)} accessibilityRole="button" style={({ pressed }) => [styles.guideLink, pressed && styles.pressed]}><MaterialIcons name="menu-book" size={16} color={colors.moss} /><Text style={styles.guideLinkText}>View pose guide</Text></Pressable></View>
          {renderCameraStage()}

          <View style={styles.metricRow}>
            <View style={styles.scoreCard}><Text style={styles.metricLabel}>Posture score</Text><Text style={styles.score}>{score}</Text><Text style={styles.metricHint}>{result ? result.score >= 0.7 ? "On target" : "Needs adjustment" : "Awaiting check"}</Text></View>
            <View style={styles.detectCard}><Text style={styles.metricLabel}>Detected pose</Text><Text numberOfLines={2} style={styles.detected}>{result?.poseId ?? "Waiting"}</Text><Text style={styles.metricHint}>{result ? result.poseId === pose.name ? "Target confirmed" : "Compare with target" : "Camera analysis"}</Text></View>
          </View>

          <View style={[styles.statusPanel, serviceState === "offline" && styles.statusPanelError]}><View style={[styles.statusIcon, { backgroundColor: serviceState === "offline" ? colors.clay : colors.sage }]}><MaterialIcons name={serviceState === "offline" ? "wifi-off" : serviceState === "no-person" ? "person-search" : "self-improvement"} size={20} color={serviceState === "offline" ? colors.terracotta : colors.moss} /></View><View style={{ flex: 1 }}><Text style={styles.statusTitle}>{status.title}</Text><Text style={styles.statusDetail}>{lastError || status.detail}</Text></View></View>

          {result?.correction ? <View style={[styles.correctionPanel, !result.safe && styles.correctionPanelAlert]}><MaterialIcons name={result.safe ? "tips-and-updates" : "warning-amber"} size={20} color={result.safe ? colors.moss : colors.terracotta} /><View style={{ flex: 1 }}><Text style={styles.correctionLabel}>{result.safe ? "Coach cue" : "Pause and adjust"}</Text><Text style={styles.correctionText}>{result.correction}</Text></View></View> : null}

          <View style={styles.actions}>{!sessionActive ? <PrimaryButton label={permission?.granted ? "Start coaching" : "Enable camera"} icon="videocam" onPress={beginSession} disabled={Platform.OS === "web"} /> : <PrimaryButton label="End session" icon="stop-circle" tone="terracotta" onPress={endSession} />}</View>
          <Text style={styles.privacyNote}>Your camera stays off until you start. Landmark and alignment requests are made only while coaching is active.</Text>
        </ScrollView>
      </View>
      <PosePicker visible={pickerOpen} selectedPoseId={pose.id} onSelect={choosePose} onClose={() => setPickerOpen(false)} />
      <PoseGuide pose={pose} visible={guideOpen} onClose={() => setGuideOpen(false)} />
      <PoseDetectorWorker request={detectorRequest} onReady={() => setDetectorReady(true)} onResult={processLandmarks} onError={(message) => { frameInFlight.current = false; setProcessing(false); setLastError(message); setServiceState("offline"); }} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, minHeight: 0 }, immersiveRoot: { flex: 1, backgroundColor: colors.ink }, topBar: { paddingHorizontal: 20, paddingTop: 12, flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 }, topActions: { flexDirection: "row", alignItems: "center", gap: 8 }, timerPill: { flexDirection: "row", alignItems: "center", gap: 6, height: 34, paddingHorizontal: 10, borderRadius: 17, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line }, timer: { color: colors.ink, fontSize: 12, fontWeight: "800", fontVariant: ["tabular-nums"] }, poseControl: { maxWidth: 130, flexDirection: "row", alignItems: "center", backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, paddingLeft: 11, paddingRight: 6, height: 38, borderRadius: 19 }, poseControlText: { color: colors.ink, fontSize: 12, fontWeight: "700", flexShrink: 1 }, scroller: { flex: 1 }, content: { paddingHorizontal: 20, paddingTop: 22, paddingBottom: 26, gap: 14 }, context: { gap: 7 }, title: { color: colors.ink, fontSize: 31, letterSpacing: -0.8, fontWeight: "700" }, subtitle: { color: colors.mist, fontSize: 14, lineHeight: 20 }, guideLink: { alignSelf: "flex-start", flexDirection: "row", alignItems: "center", gap: 6, marginTop: 2, minHeight: 32 }, guideLinkText: { color: colors.moss, fontSize: 13, fontWeight: "800" }, cameraFrame: { backgroundColor: "#E9ECE5", borderRadius: 26, overflow: "hidden", position: "relative" }, fullscreenCameraFrame: { flex: 1, backgroundColor: "#172019", overflow: "hidden", position: "relative" }, cameraEmpty: { flex: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: 40, gap: 10 }, cameraIcon: { width: 58, height: 58, borderRadius: 29, backgroundColor: colors.paper, alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: colors.line, marginBottom: 2 }, cameraEmptyTitle: { color: colors.ink, fontSize: 17, fontWeight: "700" }, cameraEmptyDetail: { color: colors.mist, fontSize: 13, lineHeight: 19, textAlign: "center" }, cameraLabel: { position: "absolute", top: 14, left: 14, flexDirection: "row", gap: 7, alignItems: "center", paddingHorizontal: 10, height: 31, borderRadius: 15.5, backgroundColor: "rgba(248,247,242,0.94)" }, cameraLabelText: { color: colors.ink, fontSize: 11, fontWeight: "700" }, stageControls: { position: "absolute", top: 10, right: 10, flexDirection: "row", gap: 8 }, frameGuide: { ...StyleSheet.absoluteFillObject, alignItems: "center", justifyContent: "center" }, frameGuideInner: { height: "78%", width: "58%", borderWidth: 1.5, borderColor: "rgba(248,247,242,0.78)", borderRadius: 80 }, recoveryPill: { position: "absolute", bottom: 13, left: 13, flexDirection: "row", alignItems: "center", gap: 5, paddingHorizontal: 9, height: 29, borderRadius: 14.5, backgroundColor: "rgba(31,38,33,0.78)" }, recoveryText: { color: colors.paper, fontSize: 10, fontWeight: "700" }, fullscreenTop: { position: "absolute", top: 16, left: 16, right: 16, flexDirection: "row", justifyContent: "space-between", alignItems: "center" }, fullscreenStatus: { flexDirection: "row", alignItems: "center", gap: 7, height: 34, paddingHorizontal: 11, borderRadius: 17, backgroundColor: "rgba(248,247,242,0.94)" }, fullscreenStatusText: { color: colors.ink, fontSize: 11, fontWeight: "800" }, fullscreenControls: { flexDirection: "row", gap: 8 }, fullscreenBottom: { position: "absolute", left: 16, right: 16, bottom: 20, gap: 11, padding: 15, borderRadius: 22, backgroundColor: "rgba(248,247,242,0.94)" }, fullscreenMetrics: { flexDirection: "row", justifyContent: "space-between", alignItems: "baseline", gap: 12 }, fullscreenPose: { flex: 1, color: colors.ink, fontSize: 18, fontWeight: "800" }, fullscreenMetric: { color: colors.moss, fontSize: 12, fontWeight: "800" }, fullscreenCue: { color: colors.ink, fontSize: 13, lineHeight: 18 }, metricRow: { flexDirection: "row", gap: 10 }, scoreCard: { flex: 0.9, minHeight: 100, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, borderRadius: 18, padding: 14, justifyContent: "space-between" }, detectCard: { flex: 1.1, minHeight: 100, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, borderRadius: 18, padding: 14, justifyContent: "space-between" }, metricLabel: { color: colors.mist, fontSize: 10, fontWeight: "800", letterSpacing: 0.8, textTransform: "uppercase" }, score: { color: colors.moss, fontSize: 31, fontWeight: "800", letterSpacing: -1 }, detected: { color: colors.ink, fontSize: 16, lineHeight: 19, fontWeight: "700" }, metricHint: { color: colors.mist, fontSize: 11, marginTop: 2 }, statusPanel: { flexDirection: "row", gap: 12, padding: 14, borderRadius: 18, borderWidth: 1, borderColor: colors.line, backgroundColor: colors.white, alignItems: "center" }, statusPanelError: { borderColor: "#D7AAA0", backgroundColor: "#FFF9F6" }, statusIcon: { width: 38, height: 38, borderRadius: 19, alignItems: "center", justifyContent: "center" }, statusTitle: { color: colors.ink, fontSize: 14, fontWeight: "700" }, statusDetail: { color: colors.mist, fontSize: 12, lineHeight: 17, marginTop: 3 }, correctionPanel: { flexDirection: "row", gap: 11, padding: 15, borderRadius: 18, backgroundColor: colors.sage, borderWidth: 1, borderColor: "#CBD4C4", alignItems: "flex-start" }, correctionPanelAlert: { backgroundColor: "#FFF6F2", borderColor: "#E5BCB1" }, correctionLabel: { color: colors.ink, fontSize: 11, fontWeight: "800", letterSpacing: 0.8, textTransform: "uppercase", marginBottom: 4 }, correctionText: { color: colors.ink, fontSize: 13, lineHeight: 19 }, actions: { marginTop: 1 }, privacyNote: { color: colors.mist, fontSize: 11, lineHeight: 16, textAlign: "center", paddingHorizontal: 13 }, pressed: { opacity: 0.82, transform: [{ scale: 0.98 }] },
});
