import type { CapacitorConfig } from '@capacitor/cli';

// Server mode: the Android WebView loads the live production site directly,
// same as any browser, rather than bundling a static export. This app has no
// Next.js API-route dependency (utils/api.ts always resolves to the external
// HF Space URL) and is fundamentally online-first anyway (camera analysis
// needs a live backend every session) -- so routine JS/CSS/logic fixes just
// need a normal `git push`, no APK rebuild/resubmit cycle. webDir is unused
// in this mode but required by the CapacitorConfig type.
const config: CapacitorConfig = {
  appId: 'com.asanaai.app',
  appName: 'AsanaAI',
  webDir: '.next',
  server: {
    url: 'https://yoga-posture-correction-system.vercel.app',
    cleartext: false
  }
};

export default config;
