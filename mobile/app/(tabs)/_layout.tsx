import { Tabs } from "expo-router";
import { Platform, StyleSheet, View } from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";

import { HapticTab } from "@/components/haptic-tab";
import { IconSymbol } from "@/components/ui/icon-symbol";
import { colors } from "@/components/asana-ui";

export default function TabLayout() {
  const insets = useSafeAreaInsets();
  const bottomPadding = Platform.OS === "web" ? 12 : Math.max(insets.bottom, 10);
  const tabBarHeight = 64 + bottomPadding;

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: colors.moss,
        tabBarInactiveTintColor: colors.mist,
        tabBarButton: HapticTab,
        tabBarStyle: {
          paddingTop: 6,
          paddingBottom: bottomPadding,
          height: tabBarHeight,
          backgroundColor: colors.paper,
          borderTopColor: colors.line,
          borderTopWidth: 1,
        },
        tabBarLabelStyle: { fontSize: 10, fontWeight: "700", marginTop: 1 },
      }}
    >
      <Tabs.Screen name="index" options={{ title: "Today", tabBarIcon: ({ color, focused }) => <TabIcon name="house.fill" color={color} focused={focused} /> }} />
      <Tabs.Screen name="practice" options={{ title: "Practice", tabBarIcon: ({ color, focused }) => <TabIcon name="camera.fill" color={color} focused={focused} /> }} />
      <Tabs.Screen name="settings" options={{ title: "Settings", tabBarIcon: ({ color, focused }) => <TabIcon name="gearshape.fill" color={color} focused={focused} /> }} />
    </Tabs>
  );
}

function TabIcon({ name, color, focused }: { name: "house.fill" | "camera.fill" | "gearshape.fill"; color: string; focused: boolean }) {
  return <View style={[styles.tabIcon, focused && styles.tabIconActive]}><IconSymbol size={20} name={name} color={focused ? colors.paper : color} /></View>;
}

const styles = StyleSheet.create({
  tabIcon: { width: 32, height: 28, borderRadius: 14, justifyContent: "center", alignItems: "center" },
  tabIconActive: { backgroundColor: colors.moss },
});
