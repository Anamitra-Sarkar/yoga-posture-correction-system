import type { AppProps } from "next/app";
import Head from "next/head";
import "../styles/globals.css";

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>Smart Yoga Posture Correction System</title>
        <meta name="description" content="Production-ready real-time yoga posture feedback and analytics" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      <Component pageProps={pageProps} />
    </>
  );
}
