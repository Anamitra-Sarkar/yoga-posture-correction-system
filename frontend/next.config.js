const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === 'development',
  runtimeCaching: [
    {
      urlPattern: /^https:\/\/fonts\.googleapis\.com\/.*/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'google-fonts-cache',
        expiration: { maxEntries: 10, maxAgeSeconds: 365 * 24 * 60 * 60 },
        cacheableResponse: { statuses: [0, 200] }
      }
    },
    {
      urlPattern: /^https:\/\/fonts\.gstatic\.com\/.*/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'gstatic-fonts-cache',
        expiration: { maxEntries: 10, maxAgeSeconds: 365 * 24 * 60 * 60 },
        cacheableResponse: { statuses: [0, 200] }
      }
    },
    // Pose engines (MediaPipe and the CPU compatibility engine): cache-first in ONE cache that the page also fills during the first
    // visit (utils/cpuPose.ts warmEngineCache), so the app can start with no internet later. No expiry plugin: bump the name to refresh.
    {
      urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/npm\/(?:@mediapipe\/pose|@tensorflow|@tensorflow-models)\//i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'asana-engine-v1',
        cacheableResponse: { statuses: [0, 200] }
      }
    },
    {
      urlPattern: /^https:\/\/(?:tfhub\.dev\/mediapipe\/tfjs-model|storage\.googleapis\.com\/kagglesdsdata)\//i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'asana-engine-v1',
        cacheableResponse: { statuses: [0, 200] }
      }
    },
    {
      urlPattern: /^https:\/\/cdn\.jsdelivr\.net\/.*/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'mediapipe-cdn-cache',
        expiration: { maxEntries: 20, maxAgeSeconds: 30 * 24 * 60 * 60 },
        cacheableResponse: { statuses: [0, 200] }
      }
    },
    {
      urlPattern: /^https:\/\/api\.fontshare\.com\/.*/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'fontshare-cache',
        expiration: { maxEntries: 10, maxAgeSeconds: 365 * 24 * 60 * 60 },
        cacheableResponse: { statuses: [0, 200] }
      }
    },
    {
      urlPattern: ({ url }) => url.pathname.startsWith('/api'),
      handler: 'NetworkFirst',
      options: {
        cacheName: 'api-cache',
        expiration: { maxEntries: 32, maxAgeSeconds: 24 * 60 * 60 },
        networkTimeoutSeconds: 10
      }
    },
    {
      urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp|ico)$/i,
      handler: 'CacheFirst',
      options: {
        cacheName: 'image-cache',
        expiration: { maxEntries: 64, maxAgeSeconds: 30 * 24 * 60 * 60 }
      }
    },
    {
      urlPattern: /\.(?:js|css)$/i,
      handler: 'StaleWhileRevalidate',
      options: {
        cacheName: 'static-resources',
        expiration: { maxEntries: 32, maxAgeSeconds: 24 * 60 * 60 }
      }
    }
  ]
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
};

module.exports = withPWA(nextConfig);
