import MaterialIcons from "@expo/vector-icons/MaterialIcons";
import { Modal, Pressable, StyleSheet, Text, View, FlatList } from "react-native";

import { colors, SectionLabel } from "@/components/asana-ui";
import { POSES, type Pose } from "@/lib/asana";

export function PosePicker({
  visible,
  selectedPoseId,
  onSelect,
  onClose,
}: {
  visible: boolean;
  selectedPoseId: string;
  onSelect: (pose: Pose) => void;
  onClose: () => void;
}) {
  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <View style={styles.backdrop}>
        <Pressable style={StyleSheet.absoluteFill} onPress={onClose} accessibilityLabel="Close pose picker" />
        <View style={styles.sheet}>
          <View style={styles.handle} />
          <View style={styles.header}>
            <View>
              <SectionLabel>Practice direction</SectionLabel>
              <Text style={styles.title}>Choose a pose</Text>
            </View>
            <Pressable onPress={onClose} hitSlop={8} accessibilityLabel="Close pose picker" style={({ pressed }) => [styles.close, pressed && styles.pressed]}>
              <MaterialIcons name="close" size={21} color={colors.ink} />
            </Pressable>
          </View>
          <FlatList
            data={POSES}
            keyExtractor={(item) => item.id}
            showsVerticalScrollIndicator={false}
            contentContainerStyle={styles.list}
            renderItem={({ item }) => {
              const active = item.id === selectedPoseId;
              return (
                <Pressable
                  onPress={() => onSelect(item)}
                  accessibilityRole="button"
                  accessibilityState={{ selected: active }}
                  style={({ pressed }) => [styles.row, active && styles.activeRow, pressed && styles.pressed]}
                >
                  <View style={[styles.poseMark, { backgroundColor: item.accent === "terracotta" ? colors.clay : colors.sage }]}>
                    <MaterialIcons name="self-improvement" size={20} color={item.accent === "terracotta" ? colors.terracotta : colors.moss} />
                  </View>
                  <View style={styles.textArea}>
                    <Text style={styles.poseName}>{item.name}</Text>
                    <Text style={styles.poseSanskrit}>{item.Sanskrit}</Text>
                    <Text style={styles.cue}>{item.cue}</Text>
                    <Text style={styles.measurement}>{item.level} · {item.targets.map((target) => target.label).join(" · ")}</Text>
                  </View>
                  {active ? <MaterialIcons name="check-circle" size={22} color={colors.moss} /> : <MaterialIcons name="chevron-right" size={22} color={colors.mist} />}
                </Pressable>
              );
            }}
          />
        </View>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  backdrop: { flex: 1, justifyContent: "flex-end", backgroundColor: "rgba(31,38,33,0.32)" },
  sheet: { maxHeight: "82%", borderTopLeftRadius: 28, borderTopRightRadius: 28, backgroundColor: colors.paper, paddingTop: 10, paddingHorizontal: 20, paddingBottom: 26 },
  handle: { alignSelf: "center", width: 42, height: 4, borderRadius: 2, backgroundColor: colors.line, marginBottom: 17 },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 13 },
  title: { fontSize: 27, lineHeight: 33, fontWeight: "700", color: colors.ink, letterSpacing: -0.65, marginTop: 4 },
  close: { width: 40, height: 40, borderRadius: 20, alignItems: "center", justifyContent: "center", backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line },
  list: { gap: 8, paddingBottom: 12 },
  row: { minHeight: 108, borderRadius: 18, padding: 14, backgroundColor: colors.white, borderWidth: 1, borderColor: colors.line, flexDirection: "row", alignItems: "center", gap: 12 },
  activeRow: { borderColor: colors.moss, backgroundColor: "#F2F5EF" },
  poseMark: { width: 42, height: 42, borderRadius: 21, justifyContent: "center", alignItems: "center", alignSelf: "flex-start", marginTop: 1 },
  textArea: { flex: 1, gap: 1 },
  poseName: { fontSize: 16, fontWeight: "700", color: colors.ink, letterSpacing: -0.2 },
  poseSanskrit: { fontSize: 12, color: colors.mist, fontStyle: "italic" },
  cue: { fontSize: 12, lineHeight: 17, color: colors.ink, marginTop: 5 },
  measurement: { fontSize: 10, lineHeight: 14, color: colors.moss, marginTop: 4, fontWeight: "700" },
  pressed: { opacity: 0.82, transform: [{ scale: 0.98 }] },
});
