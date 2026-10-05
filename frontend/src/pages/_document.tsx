import { Html, Head, Main, NextScript } from 'next/document';
import { fontUi, fontDisplay, fontMono } from '../lib/fonts';

export default function Document() {
  return (
    <Html lang="en" className={`${fontUi.variable} ${fontDisplay.variable} ${fontMono.variable}`}>
      <Head />
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
