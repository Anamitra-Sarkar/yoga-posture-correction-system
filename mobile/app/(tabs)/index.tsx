import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import * as Haptics from "expo-haptics";
import { router, useFocusEffect } from "expo-router";
import { useCallback, useState } from "react";
import { ActivityIndicator, Pressable, ScrollView, StyleSheet, Text, View } from "react-native";

import { AsanaMark, colors, IconButton, PrimaryButton, SectionLabel, StatusDot } from "@/components/asana-ui";
import { PosePicker } from "@/components/pose-picker";
import { ScreenContainer } from "@/components/screen-container";
import { DEFAULT_POSE_ID, getPose, type Pose } from "@/lib/asana";
import { DEFAULT_PREFERENCES, loadPreferences, savePreferences, type Preferences } from "@/lib/preferences";
import { loadLastPractice, type PracticeRecord } from "@/lib/practice-history";

export default function TodayScreen() {
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);
  const [lastPractice, setLastPractice] = useState<PracticeRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [pickerOpen, setPickerOpen] = useState(false);
  const pose = getPose(preferences.poseId || DEFAULT_POSE_ID);

  useFocusEffect(
    useCallback(() => {
      let mounted = true;
      Promise.all([loadPreferences(), loadLastPractice()]).then(([nextPreferences, nextLastPractice]) => {
        if (!mounted) return;
        setPreferences(nextPreferences);
        setLastPractice(nextLastPractice);
        setLoading(false);
      });
      return () => { mounted = false; };
    }, []),
  );

  const choosePose = async (nextPose: Pose) => {
    const next = { ...preferences, poseId: nextPose.id };
    setPreferences(next);
    setPickerOpen(false);
    await savePreferences(next);
    Haptics.selectionAsync();
  };

  const begin = () => {
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
    router.push("./practice");
  };

  return (
    <ScreenContainer className="flex-1" containerClassName="bg-background">
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <AsanaMark />
          <IconButton label="Open settings" icon="tune" onPress={() => router.push("./settings")} />
        </View>

        <View style={styles.intro}>
          <SectionLabel>Today’s practice</SectionLabel>
          <Text style={styles.title}>Make space{`\n`}for your body.</Text>
          <Text style={styles.subtitle}>A short camera check, then focused coaching when you are ready.</Text>
        </View>

        <View style={styles.practiceCard}>
          <View style={styles.cardHeader}>
            <View style={styles.cardBadge}>
              <MaterialIcons name="self-improvement" size={18} color={colors.paper} />
              <Text style={styles.cardBadgeText}>TARGET POSE</Text>
            </View>
            <View style={styles.levelPill}><Text style={styles.levelText}>{pose.level}</Text></View>
          </View>
          {loading ? (
            <View style={styles.loadingPose}><ActivityIndicator color={colors.paper} /></View>
          ) : (
            <>
              <Text style={styles.poseTitle}>{pose.name}</Text>
              <Text style={styles.sanskrit}>{pose.Sanskrit}</Text>
              <View style={styles.cueRow}>
                <MaterialIcons name="tips-and-updates" size={17} color="#E6E9DF" />
                <Text style={styles.cueText}>{pose.cue}</Text>
              </View>
            </>
          )}
          <View style={styles.cardFooter}>
            <Pressable onPress={() => setPickerOpen(true)} style={({ pressed }) => [styles.changePose, pressed && styles.pressed]} accessibilityRole="button">
              <Text style={styles.changePoseText}>Change pose</Text>
              <MaterialIcons name="arrow-forward" size={18} color={colors.paper} />
            </Pressable>
          </View>
        </View>

        <View style={styles.statusRow}>
          <View style={styles.statusTextArea}>
            <View style={styles.statusHeading}><StatusDot /><Text style={styles.statusTitle}>Native camera coaching</Text></View>
            <Text style={styles.statusDetail}>Landmark-based posture checks begin when you start a camera practice.</Text>
          </View>
          <MaterialIcons name="wifi" size={20} color={colors.moss} />
        </View>

        <View style={styles.littleTitleRow}>
          <SectionLabel>{lastPractice ? "Last practice" : "On this device"}</SectionLabel>
          <Text style={styles.littleDetail}>Private by default</Text>
        </View>
        <View style={styles.emptyHistory}>
          <MaterialIcons name={lastPractice ? "self-improvement" : "history"} size={21} color={colors.mist} />
          <View style={{ flex: 1 }}>
            <Text style={styles.emptyTitle}>{lastPractice?.score !== undefined ? `${Math.round(lastPractice.score * 100)}% · ${lastPractice.detectedPoseId ?? getPose(lastPractice.poseId).name}` : lastPractice ? `${getPose(lastPractice.poseId).name} camera practice` : "No practice summary yet"}</Text>
            <Text style={styles.emptyDetail}>{lastPractice?.score !== undefined ? `${lastPractice.durationSeconds ? `${Math.max(1, Math.round(lastPractice.durationSeconds / 60))} min · ` : ""}${lastPractice.correction ?? "Latest real coaching result saved on this device."}` : lastPractice ? "Your last practice was recorded on this device. Start a new check for current landmarks and feedback." : "Your first completed session will appear here on this device."}</Text>
          </View>
        </View>

        <PrimaryButton label="Begin practice" icon="videocam" onPress={begin} />
      </ScrollView>
      <PosePicker visible={pickerOpen} selectedPoseId={pose.id} onSelect={choosePose} onClose={() => setPickerOpen(false)} />
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: 20, paddingTop: 12, paddingBottom: 118, gap: 22 },
  header: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  intro: { marginTop: 15, gap: 9 },
  title: { color: colors.ink, fontSize: 38, lineHeight: 42, letterSpacing: -1.5, fontWeight: "700" },
  subtitle: { color: colors.mist, fontSize: 15, lineHeight: 22, maxWidth: 310 },
  practiceCard: { minHeight: 284, backgroundColor: colors.moss, borderRadius: 26, padding: 22, overflow: "hidden" },
  cardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  cardBadge: { flexDirection: "row", alignItems: "center", gap: 6 },
  cardBadgeText: { color: colors.paper, fontSize: 10, fontWeight: "800", letterSpacing: 1.05 },
  levelPill: { backgroundColor: "rgba(255,255,255,0.13)", borderWidth: 1, borderColor: "rgba(255,255,255,0.18)", paddingHorizontal: 10, paddingVertical: 6, borderRadius: 14 },
  levelText: { color: colors.paper, fontSize: 11, fontWeight: "700" },
  poseTitle: { color: colors.paper, fontSize: 34, lineHeight: 40, letterSpacing: -1, fontWeight: "700", marginTop: 37 },
  sanskrit: { color: "#DDE4D8", fontSize: 15, marginTop: 2, fontStyle: "italic" },
  cueRow: { flexDirection: "row", alignItems: "flex-start", gap: 9, marginTop: 26, paddingRight: 8 },
  cueText: { flex: 1, color: "#EDEFEA", fontSize: 14, lineHeight: 20 },
  cardFooter: { position: "absolute", left: 22, right: 22, bottom: 18, borderTopWidth: 1, borderTopColor: "rgba(255,255,255,0.16)", paddingTop: 13 },
  changePose: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  changePoseText: { color: colors.paper, fontSize: 14, fontWeight: "700" },
  loadingPose: { flex: 1, justifyContent: "center", alignItems: "center" },
  statusRow: { backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, padding: 16, borderRadius: 18, flexDirection: "row", alignItems: "center", gap: 12 },
  statusTextArea: { flex: 1, gap: 5 },
  statusHeading: { flexDirection: "row", alignItems: "center", gap: 8 },
  statusTitle: { color: colors.ink, fontSize: 14, fontWeight: "700" },
  statusDetail: { color: colors.mist, fontSize: 12, lineHeight: 17 },
  littleTitleRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 4 },
  littleDetail: { color: colors.mist, fontSize: 12 },
  emptyHistory: { minHeight: 81, backgroundColor: "#F0F1ED", borderRadius: 16, padding: 16, flexDirection: "row", gap: 12, alignItems: "center" },
  emptyTitle: { color: colors.ink, fontSize: 14, fontWeight: "700" },
  emptyDetail: { color: colors.mist, fontSize: 12, lineHeight: 17, marginTop: 3 },
  pressed: { opacity: 0.82, transform: [{ scale: 0.98 }] },
});
