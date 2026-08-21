import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import * as Haptics from "expo-haptics";
import { useFocusEffect } from "expo-router";
import { useCallback, useState } from "react";
import { Pressable, ScrollView, StyleSheet, Switch, Text, View } from "react-native";

import { AsanaMark, colors, SectionLabel } from "@/components/asana-ui";
import { ScreenContainer } from "@/components/screen-container";
import { DEFAULT_PREFERENCES, loadPreferences, savePreferences, type Preferences } from "@/lib/preferences";

const languageOptions: { id: Preferences["language"]; label: string }[] = [
  { id: "en", label: "English" },
  { id: "hi", label: "हिन्दी" },
  { id: "bn", label: "বাংলা" },
];

export default function SettingsScreen() {
  const [preferences, setPreferences] = useState<Preferences>(DEFAULT_PREFERENCES);

  useFocusEffect(
    useCallback(() => {
      loadPreferences().then(setPreferences);
    }, []),
  );

  const update = async (value: Partial<Preferences>) => {
    const next = { ...preferences, ...value };
    setPreferences(next);
    await savePreferences(next);
    Haptics.selectionAsync();
  };

  return (
    <ScreenContainer className="flex-1" containerClassName="bg-background">
      <ScrollView contentContainerStyle={styles.content} showsVerticalScrollIndicator={false}>
        <AsanaMark />
        <View style={styles.intro}>
          <SectionLabel>Preferences</SectionLabel>
          <Text style={styles.title}>Make it yours.</Text>
          <Text style={styles.subtitle}>These controls are saved on this device and travel with your practice, not your profile.</Text>
        </View>

        <View style={styles.group}>
          <Text style={styles.groupTitle}>Guidance</Text>
          <View style={styles.row}>
            <View style={styles.rowIcon}><MaterialIcons name="volume-up" size={20} color={colors.moss} /></View>
            <View style={styles.rowText}><Text style={styles.rowTitle}>Voice guidance</Text><Text style={styles.rowDetail}>Spoken prompts in the live coach</Text></View>
            <Switch
              accessibilityLabel="Toggle voice guidance"
              value={preferences.voiceEnabled}
              onValueChange={(voiceEnabled) => update({ voiceEnabled })}
              trackColor={{ false: colors.line, true: colors.moss }}
              thumbColor={colors.white}
            />
          </View>
          <View style={[styles.row, styles.dividedRow]}>
            <View style={styles.rowIcon}><MaterialIcons name="language" size={20} color={colors.moss} /></View>
            <View style={styles.rowText}><Text style={styles.rowTitle}>Language</Text><Text style={styles.rowDetail}>Applied where the live coach supports it</Text></View>
          </View>
          <View style={styles.languageRow}>
            {languageOptions.map((option) => {
              const active = preferences.language === option.id;
              return <Pressable key={option.id} accessibilityRole="button" accessibilityState={{ selected: active }} onPress={() => update({ language: option.id })} style={({ pressed }) => [styles.languagePill, active && styles.activeLanguagePill, pressed && styles.pressed]}><Text style={[styles.languageText, active && styles.activeLanguageText]}>{option.label}</Text></Pressable>;
            })}
          </View>
        </View>

        <View style={styles.group}>
          <Text style={styles.groupTitle}>Practice service</Text>
          <View style={styles.row}>
            <View style={styles.rowIcon}><MaterialIcons name="lock-outline" size={20} color={colors.moss} /></View>
            <View style={styles.rowText}><Text style={styles.rowTitle}>Embedded live coach</Text><Text style={styles.rowDetail}>Uses the existing AsanaAI web service inside the app</Text></View>
            <MaterialIcons name="check-circle" size={20} color={colors.moss} />
          </View>
          <View style={[styles.row, styles.dividedRow]}>
            <View style={styles.rowIcon}><MaterialIcons name="privacy-tip" size={20} color={colors.moss} /></View>
            <View style={styles.rowText}><Text style={styles.rowTitle}>Camera handling</Text><Text style={styles.rowDetail}>Camera access stays off until you begin a practice</Text></View>
          </View>
        </View>

        <Text style={styles.footer}>AsanaAI Mobile pairs a thoughtful native practice space with the existing real-time coaching experience. It does not fabricate biomechanical results when a service cannot respond.</Text>
      </ScrollView>
    </ScreenContainer>
  );
}

const styles = StyleSheet.create({
  content: { paddingHorizontal: 20, paddingTop: 12, paddingBottom: 112, gap: 27 },
  intro: { gap: 8, marginTop: 7 },
  title: { color: colors.ink, fontSize: 37, lineHeight: 43, letterSpacing: -1.2, fontWeight: "700" },
  subtitle: { color: colors.mist, fontSize: 14, lineHeight: 20, maxWidth: 326 },
  group: { backgroundColor: colors.white, borderRadius: 20, borderWidth: 1, borderColor: colors.line, overflow: "hidden" },
  groupTitle: { color: colors.mist, fontSize: 11, fontWeight: "800", letterSpacing: 1, textTransform: "uppercase", paddingHorizontal: 16, paddingTop: 15, paddingBottom: 7 },
  row: { minHeight: 70, flexDirection: "row", alignItems: "center", paddingHorizontal: 16, gap: 12 },
  dividedRow: { borderTopWidth: 1, borderTopColor: colors.line },
  rowIcon: { width: 36, height: 36, borderRadius: 18, backgroundColor: colors.sage, alignItems: "center", justifyContent: "center" },
  rowText: { flex: 1 },
  rowTitle: { color: colors.ink, fontSize: 14, fontWeight: "700" },
  rowDetail: { color: colors.mist, fontSize: 11, lineHeight: 16, marginTop: 3 },
  languageRow: { paddingHorizontal: 16, paddingTop: 5, paddingBottom: 15, flexDirection: "row", gap: 8 },
  languagePill: { flex: 1, minHeight: 38, borderRadius: 12, borderWidth: 1, borderColor: colors.line, justifyContent: "center", alignItems: "center", backgroundColor: colors.paper },
  activeLanguagePill: { backgroundColor: colors.moss, borderColor: colors.moss },
  languageText: { color: colors.ink, fontSize: 12, fontWeight: "700" },
  activeLanguageText: { color: colors.paper },
  footer: { color: colors.mist, fontSize: 12, lineHeight: 18, paddingHorizontal: 6 },
  pressed: { opacity: 0.82, transform: [{ scale: 0.98 }] },
});
