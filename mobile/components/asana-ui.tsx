import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import { ReactNode } from "react";
import { ActivityIndicator, Pressable, StyleSheet, Text, View, type StyleProp, type ViewStyle } from "react-native";

export const colors = {
  paper: "#F8F7F2",
  ink: "#1F2621",
  moss: "#5C6B4E",
  terracotta: "#B86145",
  sage: "#DDE2D8",
  clay: "#E9DED4",
  mist: "#6B736D",
  white: "#FFFFFF",
  line: "#D7D9D2",
};

export function AsanaMark({ inverse = false }: { inverse?: boolean }) {
  return (
    <View style={styles.markRow}>
      <View style={[styles.mark, inverse && styles.markInverse]}>
        <MaterialIcons name="self-improvement" size={18} color={inverse ? colors.paper : colors.moss} />
      </View>
      <Text style={[styles.wordmark, inverse && { color: colors.paper }]}>asana</Text>
    </View>
  );
}

export function SectionLabel({ children }: { children: ReactNode }) {
  return <Text style={styles.sectionLabel}>{children}</Text>;
}

export function PrimaryButton({
  label,
  onPress,
  icon,
  disabled = false,
  tone = "moss",
  loading = false,
}: {
  label: string;
  onPress: () => void;
  icon?: keyof typeof MaterialIcons.glyphMap;
  disabled?: boolean;
  tone?: "moss" | "terracotta" | "paper";
  loading?: boolean;
}) {
  const background = tone === "terracotta" ? colors.terracotta : tone === "paper" ? colors.paper : colors.moss;
  const foreground = tone === "paper" ? colors.ink : colors.paper;
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityState={{ disabled, busy: loading }}
      disabled={disabled || loading}
      onPress={onPress}
      style={({ pressed }) => [
        styles.primaryButton,
        { backgroundColor: background },
        tone === "paper" && styles.paperButton,
        (disabled || loading) && styles.disabled,
        pressed && styles.pressed,
      ]}
    >
      {loading ? <ActivityIndicator color={foreground} /> : icon ? <MaterialIcons name={icon} size={20} color={foreground} /> : null}
      <Text style={[styles.primaryLabel, { color: foreground }]}>{label}</Text>
    </Pressable>
  );
}

export function IconButton({
  label,
  icon,
  onPress,
  style,
}: {
  label: string;
  icon: keyof typeof MaterialIcons.glyphMap;
  onPress: () => void;
  style?: StyleProp<ViewStyle>;
}) {
  return (
    <Pressable
      accessibilityRole="button"
      accessibilityLabel={label}
      hitSlop={8}
      onPress={onPress}
      style={({ pressed }) => [styles.iconButton, style, pressed && styles.pressed]}
    >
      <MaterialIcons name={icon} size={21} color={colors.ink} />
    </Pressable>
  );
}

export function StatusDot({ tone = "moss" }: { tone?: "moss" | "terracotta" | "mist" }) {
  return <View style={[styles.statusDot, { backgroundColor: colors[tone] }]} />;
}

const styles = StyleSheet.create({
  markRow: { flexDirection: "row", alignItems: "center", gap: 8 },
  mark: { width: 32, height: 32, borderRadius: 16, backgroundColor: colors.sage, alignItems: "center", justifyContent: "center" },
  markInverse: { backgroundColor: "rgba(255,255,255,0.13)", borderWidth: 1, borderColor: "rgba(255,255,255,0.2)" },
  wordmark: { color: colors.ink, fontSize: 21, fontWeight: "700", letterSpacing: -0.8 },
  sectionLabel: { color: colors.mist, fontSize: 11, fontWeight: "700", letterSpacing: 1.1, textTransform: "uppercase" },
  primaryButton: { minHeight: 52, borderRadius: 15, paddingHorizontal: 18, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 9 },
  primaryLabel: { fontSize: 16, fontWeight: "700", letterSpacing: -0.15 },
  paperButton: { borderWidth: 1, borderColor: colors.line },
  disabled: { opacity: 0.52 },
  pressed: { transform: [{ scale: 0.97 }], opacity: 0.9 },
  iconButton: { width: 42, height: 42, borderRadius: 21, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, alignItems: "center", justifyContent: "center" },
  statusDot: { width: 8, height: 8, borderRadius: 4 },
});
