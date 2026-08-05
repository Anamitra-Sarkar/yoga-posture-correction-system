import type { AppProps } from 'next/app';
import Head from 'next/head';
import { useEffect } from 'react';
import '../styles/globals.css';

// Native-app-only polish (status bar theming, hardware back-button handling).
// Dynamically imported and no-op outside the Capacitor Android WebView (the
// PWA/browser build never touches these) -- Capacitor.isNativePlatform()
// gates it so this has zero effect on the existing production PWA.
function useNativeAppPolish() {
  useEffect(() => {
    let backListener: { remove: () => void } | undefined;

    (async () => {
      const { Capacitor } = await import('@capacitor/core');
      if (!Capacitor.isNativePlatform()) return;

      const { StatusBar, Style } = await import('@capacitor/status-bar');
      try {
        await StatusBar.setBackgroundColor({ color: '#0f0c29' });
        await StatusBar.setStyle({ style: Style.Dark });
      } catch {
        // Non-fatal on devices/OEM skins that restrict status bar theming.
      }

      const { App } = await import('@capacitor/app');
      const sub = await App.addListener('backButton', ({ canGoBack }) => {
        // Default WebView behavior can otherwise instantly exit the app on
        // the very first back press. Mirror normal Android app conventions:
        // navigate web history if there is any, otherwise minimize (never
        // hard-exit, since users expect the task-switcher/home-button model).
        if (canGoBack) {
          window.history.back();
        } else {
          App.minimizeApp();
        }
      });
      backListener = sub;
    })();

    return () => backListener?.remove();
  }, []);
}

export default function App({ Component, pageProps }: AppProps) {
  useNativeAppPolish();
  return (
    <>
      <Head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover" />
        <meta name="theme-color" content="#0f0c29" media="(prefers-color-scheme: light)" />
        <meta name="theme-color" content="#0f0c29" />
        <meta name="mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="default" />
        <meta name="apple-mobile-web-app-title" content="AsanaAI" />
        <meta name="application-name" content="AsanaAI" />
        <meta name="description" content="AI-powered real-time yoga posture correction. Practice smarter with instant feedback." />
        <meta name="msapplication-TileColor" content="#01696f" />
        <meta name="msapplication-tap-highlight" content="no" />

        {/* Font Preconnect Hints */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />

        {/* PWA manifest */}
        <link rel="manifest" href="/manifest.json" />

        {/* Favicons */}
        <link rel="icon" href="/icons/icon-96.png" />
        <link rel="shortcut icon" href="/icons/icon-96.png" />

        {/* Apple touch icons */}
        <link rel="apple-touch-icon" href="/icons/icon-192.png" />
        <link rel="apple-touch-icon" sizes="152x152" href="/icons/icon-152.png" />
        <link rel="apple-touch-icon" sizes="180x180" href="/icons/icon-192.png" />
        <link rel="apple-touch-icon" sizes="167x167" href="/icons/icon-192.png" />

        {/* OG / Social */}
        <meta property="og:type" content="website" />
        <meta property="og:title" content="AsanaAI — Smart Yoga Coach" />
        <meta property="og:description" content="AI-powered real-time yoga posture correction." />
        <meta property="og:image" content="/icons/icon-512.png" />
        <meta name="twitter:card" content="summary" />
        <meta name="twitter:title" content="AsanaAI" />
        <meta name="twitter:description" content="AI-powered real-time yoga posture correction." />
      </Head>
      <Component {...pageProps} />
    </>
  );
}
