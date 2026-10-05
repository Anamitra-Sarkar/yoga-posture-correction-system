import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        {/* Apply a stored Light/Dark choice before first paint (otherwise the page flashes the wrong theme). */}
        <script dangerouslySetInnerHTML={{ __html: "try{var t=localStorage.getItem('asana.theme');if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t)}catch(e){}" }} />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
