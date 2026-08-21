import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import { Modal, Pressable, StyleSheet, Text, View } from "react-native";

import { colors, SectionLabel } from "@/components/asana-ui";
import type { Pose } from "@/lib/asana";

export function PoseGuide({ pose, visible, onClose }: { pose: Pose; visible: boolean; onClose: () => void }) {
  return (
    <Modal visible={visible} transparent animationType="slide" onRequestClose={onClose}>
      <View style={styles.backdrop}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} accessibilityLabel="Close pose guide" />
        <View style={styles.sheet}>
          <View style={styles.handle} />
          <View style={styles.header}>
            <View><SectionLabel>Pose guide</SectionLabel><Text style={styles.title}>{pose.name}</Text><Text style={styles.sanskrit}>{pose.Sanskrit}</Text></View>
            <Pressable onPress={onClose} accessibilityLabel="Close pose guide" style={({ pressed }) => [styles.close, pressed && styles.pressed]}><MaterialIcons name="close" size={21} color={colors.ink} /></Pressable>
          </View>
          <View style={styles.cueCard}><MaterialIcons name="tips-and-updates" size={19} color={colors.moss} /><Text style={styles.cue}>{pose.cue}</Text></View>
          <Text style={styles.heading}>Set up in three steps</Text>
          <View style={styles.steps}>{pose.steps.map((step, index) => <View key={step} style={styles.step}><View style={styles.number}><Text style={styles.numberText}>{index + 1}</Text></View><Text style={styles.stepText}>{step}</Text></View>)}</View>
          <Text style={styles.heading}>Measured alignment</Text>
          <View style={styles.targets}>{pose.targets.map((target) => <View key={target.joint} style={styles.target}><Text style={styles.targetLabel}>{target.label}</Text><Text style={styles.targetValue}>{target.target}° ± {target.tolerance}°</Text></View>)}</View>
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, justifyContent: "flex-end", backgroundColor: "rgba(31,38,33,0.34)" }, sheet: { backgroundColor: colors.paper, borderTopLeftRadius: 28, borderTopRightRadius: 28, paddingHorizontal: 20, paddingTop: 10, paddingBottom: 32 }, handle: { width: 42, height: 4, borderRadius: 2, backgroundColor: colors.line, alignSelf: "center", marginBottom: 18 }, header: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" }, title: { color: colors.ink, fontSize: 28, fontWeight: "700", letterSpacing: -0.7, marginTop: 3 }, sanskrit: { color: colors.mist, fontSize: 13, fontStyle: "italic", marginTop: 2 }, close: { width: 40, height: 40, borderRadius: 20, alignItems: "center", justifyContent: "center", backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line }, cueCard: { flexDirection: "row", gap: 10, marginTop: 19, padding: 14, borderRadius: 16, backgroundColor: colors.sage, alignItems: "flex-start" }, cue: { flex: 1, color: colors.ink, fontSize: 13, lineHeight: 19, fontWeight: "600" }, heading: { color: colors.mist, fontSize: 11, fontWeight: "800", letterSpacing: 1, textTransform: "uppercase", marginTop: 22, marginBottom: 9 }, steps: { gap: 10 }, step: { flexDirection: "row", gap: 11, alignItems: "flex-start" }, number: { width: 24, height: 24, borderRadius: 12, backgroundColor: colors.moss, alignItems: "center", justifyContent: "center", marginTop: 1 }, numberText: { color: colors.paper, fontSize: 12, fontWeight: "800" }, stepText: { flex: 1, color: colors.ink, fontSize: 14, lineHeight: 20 }, targets: { gap: 7 }, target: { backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, borderRadius: 12, paddingHorizontal: 12, paddingVertical: 10, flexDirection: "row", justifyContent: "space-between", gap: 12 }, targetLabel: { color: colors.ink, fontSize: 12, fontWeight: "700" }, targetValue: { color: colors.moss, fontSize: 12, fontWeight: "800" }, pressed: { opacity: 0.8, transform: [{ scale: 0.97 }] },
});
