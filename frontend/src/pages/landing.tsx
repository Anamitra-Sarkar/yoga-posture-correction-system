import { useState } from "react";
import Head from "next/head";
import Link from "next/link";
import {
  Menu,
  X,
  Smartphone,
  PersonStanding,
  Volume2,
  ShieldCheck,
  Languages,
  WifiOff,
  Gauge,
  Lock,
  ChevronDown,
  Camera,
  Ruler,
} from "lucide-react";

/* ── pose figures: simple joint-and-bone drawings in the same style as the app's live overlay ───────────────── */
type Pt = [number, number];
interface Figure { head: Pt; lines: Pt[][]; ground?: number; accent?: Pt }

const FIGURES: { [id: string]: Figure } = {
  mountain: {
    head: [60, 18],
    lines: [[[60, 28], [60, 72]], [[50, 34], [70, 34]], [[54, 72], [66, 72]], [[50, 34], [46, 54], [45, 74]], [[70, 34], [74, 54], [75, 74]], [[54, 72], [54, 102], [54, 132]], [[66, 72], [66, 102], [66, 132]]],
  },
  tree: {
    head: [60, 22],
    lines: [[[60, 30], [60, 72]], [[50, 36], [70, 36]], [[54, 72], [66, 72]], [[50, 36], [44, 20], [58, 8]], [[70, 36], [76, 20], [62, 8]], [[54, 72], [54, 102], [54, 132]], [[66, 72], [84, 90], [58, 98]]],
  },
  warrior: {
    head: [60, 21],
    lines: [[[60, 34], [60, 70]], [[50, 36], [70, 36]], [[52, 72], [68, 72]], [[50, 36], [26, 37], [6, 37]], [[70, 36], [94, 37], [114, 37]], [[52, 72], [28, 96], [24, 130]], [[68, 72], [92, 100], [108, 130]]],
    accent: [28, 96],
  },
  cobra: {
    head: [36, 68], ground: 124,
    lines: [[[44, 84], [70, 112]], [[70, 112], [96, 118], [116, 122]], [[44, 84], [46, 104], [48, 122]]],
  },
  plank: {
    head: [20, 60], ground: 112,
    lines: [[[34, 66], [70, 76], [92, 82], [112, 88]], [[34, 66], [34, 88], [34, 110]]],
  },
  dog: {
    head: [38, 88], ground: 114,
    lines: [[[104, 112], [88, 82], [70, 50]], [[70, 50], [42, 76], [20, 112]]],
  },
};

function PoseFigure({ id, stroke = "var(--ink)", joint = "var(--brand)", ground = "var(--line-2)", width = 3 }: { id: string; stroke?: string; joint?: string; ground?: string; width?: number }) {
  const f = FIGURES[id];
  const dots = new Map<string, Pt>();
  f.lines.forEach((l) => l.forEach((p) => dots.set(p.join(","), p)));
  return (
    <svg viewBox="0 0 120 140" role="img" aria-hidden="true" className="lp-figure">
      {f.ground !== undefined && <line x1="6" y1={f.ground} x2="114" y2={f.ground} stroke={ground} strokeWidth="2" strokeLinecap="round" />}
      <g fill="none" stroke={stroke} strokeWidth={width} strokeLinecap="round" strokeLinejoin="round">
        <circle cx={f.head[0]} cy={f.head[1]} r="8" />
        {f.lines.map((l, i) => <polyline key={i} points={l.map((p) => p.join(",")).join(" ")} />)}
      </g>
      {[...dots.values()].map((p, i) => <circle key={i} cx={p[0]} cy={p[1]} r="3.4" fill={joint} />)}
      {f.accent && <circle cx={f.accent[0]} cy={f.accent[1]} r="5.2" fill="#f2b84b" stroke="rgba(20,19,16,0.4)" strokeWidth="1.5" />}
    </svg>
  );
}

const POSES = [
  { fig: "mountain", sanskrit: "Tadasana", name: "Mountain", line: "Stand tall and find your balance." },
  { fig: "tree", sanskrit: "Vrikshasana", name: "Tree", line: "Steady focus, steady legs." },
  { fig: "warrior", sanskrit: "Virabhadrasana II", name: "Warrior II", line: "Strong legs, open chest." },
  { fig: "cobra", sanskrit: "Bhujangasana", name: "Cobra", line: "A gentle opening for chest and spine." },
  { fig: "plank", sanskrit: "Phalakasana", name: "Plank", line: "Builds core and arm strength." },
  { fig: "dog", sanskrit: "Adho Mukha Svanasana", name: "Downward dog", line: "Lengthens the back and legs." },
];

const FEATURES = [
  { icon: PersonStanding, title: "Knows the pose you're in", text: "Just practise. AsanaAI recognises what you are doing, so there is nothing to select." },
  { icon: Gauge, title: "A score you can watch", text: "See how close your alignment is, live, and which joint to adjust first." },
  { icon: Volume2, title: "Guidance you can hear", text: "Gentle spoken cues in English, हिन्दी or বাংলা, so you never have to look at the screen." },
  { icon: ShieldCheck, title: "Fits your body", text: "It learns how far your joints naturally move, so feedback is about you, not an ideal." },
  { icon: WifiOff, title: "Keeps going without internet", text: "After your first visit, basic recognition and tips still work offline." },
  { icon: Lock, title: "Private by design", text: "Your video stays on your device. Nothing is recorded, and there is no account." },
];

const STEPS = [
  { icon: Smartphone, title: "Prop up your phone", text: "Stand about two metres back, in good light, with your whole body in view." },
  { icon: Camera, title: "Move into a pose", text: "Practise freely, or pick a pose to work on and get checked against it." },
  { icon: Volume2, title: "Adjust with gentle guidance", text: "Watch your score and hear what to change, one small step at a time." },
];

const FAQ = [
  { q: "Do I need an account?", a: "No. Open the app and start. There is nothing to sign up for and nothing to install." },
  { q: "Which devices does it work on?", a: "Any recent phone, tablet or laptop with a camera, in a current browser such as Chrome or Safari." },
  { q: "Does it work without internet?", a: "Open the app once online. After that, basic pose recognition and tips keep working offline, and the full coach returns when you reconnect." },
  { q: "Which languages are supported?", a: "English, हिन्दी and বাংলা, both on screen and spoken aloud (spoken guidance depends on the voices installed on your device)." },
  { q: "Where does my video go?", a: "It doesn't leave your device. AsanaAI analyses the position of your body, not your picture, and nothing is recorded or stored." },
  { q: "Can it replace a yoga teacher?", a: "No. AsanaAI gives general guidance and is not medical advice. Listen to your body, and stop if something hurts. If you have an injury or health condition, check with a professional first." },
];

export default function LandingPage() {
  const [navOpen, setNavOpen] = useState(false);
  const [open, setOpen] = useState<number | null>(0);
  const close = () => setNavOpen(false);

  return (
    <div className="lp">
      <Head>
        <title>AsanaAI — Yoga guidance that watches your form</title>
        <meta name="description" content="Open your camera, move into a pose and get gentle, instant feedback on your alignment, spoken in English, Hindi or Bengali. Free, no sign-up, and your video stays on your device." />
        <meta property="og:title" content="AsanaAI — Yoga guidance that watches your form" />
        <meta property="og:description" content="Gentle, instant feedback on your yoga alignment, in your language. Free, private, nothing to install." />
        <meta name="theme-color" content="#f5f2ec" />
      </Head>

      <header className="lp-nav">
        <div className="lp-wrap lp-nav-in">
          <a href="#top" className="lp-brand" onClick={close}>
            <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3c1.6 2.6 2.4 5 2.4 7.2 0 3.2-1.6 6.5-2.4 10.8-.8-4.3-2.4-7.6-2.4-10.8C9.6 8 10.4 5.6 12 3Z" />
              <path d="M12 13.2c2.4-3.4 6.2-4.4 8.6-3.7-.5 3.1-3.9 6.7-8.6 8.2" />
              <path d="M12 13.2C9.6 9.8 5.8 8.8 3.4 9.5c.5 3.1 3.9 6.7 8.6 8.2" />
            </svg>
            <span>AsanaAI</span>
          </a>
          <nav className={`lp-links ${navOpen ? "open" : ""}`} aria-label="Sections">
            <a href="#how" onClick={close}>How it works</a>
            <a href="#poses" onClick={close}>Poses</a>
            <a href="#private" onClick={close}>Privacy</a>
            <a href="#faq" onClick={close}>FAQ</a>
            <Link href="/" className="lp-btn sm" onClick={close}>Start practising</Link>
          </nav>
          <button className="lp-burger" aria-label={navOpen ? "Close menu" : "Open menu"} aria-expanded={navOpen} onClick={() => setNavOpen(!navOpen)}>
            {navOpen ? <X size={22} /> : <Menu size={22} />}
          </button>
        </div>
      </header>

      <main id="top">
        {/* Hero */}
        <section className="lp-hero">
          <div className="lp-wrap lp-hero-in">
            <div className="lp-hero-copy">
              <p className="lp-eyebrow">Yoga guidance, right on your phone</p>
              <h1>A calm coach that watches your form.</h1>
              <p className="lp-lede">
                Open your camera, move into a pose, and get gentle, instant feedback on your alignment, spoken aloud in English, हिन्दी or বাংলা. No sign-up. Nothing to install.
              </p>
              <div className="lp-cta">
                <Link href="/" className="lp-btn">Start practising</Link>
                <a href="#how" className="lp-btn ghost">See how it works</a>
              </div>
              <ul className="lp-trust">
                <li>Free to use</li>
                <li>Works in your browser</li>
                <li>Your video stays on your device</li>
              </ul>
            </div>

            <div className="lp-device" aria-hidden="true">
              <div className="lp-device-screen">
                <span className="lp-live"><i />LIVE</span>
                <div className="lp-device-fig"><PoseFigure id="warrior" stroke="#ffffff" joint="#7fd1a8" ground="rgba(255,255,255,0.25)" width={3.2} /></div>
                <div className="lp-device-caption">Soften your front knee a little more.</div>
                <div className="lp-device-chip"><b>Virabhadrasana II</b><em>92%</em></div>
              </div>
              <div className="lp-device-card">
                <span className="lp-eyebrow">Posture score</span>
                <strong>92%</strong>
                <small>On target · Holding</small>
              </div>
            </div>
          </div>
        </section>

        {/* How it works */}
        <section id="how" className="lp-sec">
          <div className="lp-wrap">
            <h2>Three steps. That's all.</h2>
            <ol className="lp-steps">
              {STEPS.map((s, i) => (
                <li key={s.title}>
                  <span className="lp-step-no">{i + 1}</span>
                  <s.icon size={22} strokeWidth={1.7} />
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* What you get */}
        <section className="lp-sec lp-alt">
          <div className="lp-wrap">
            <h2>Everything a good practice partner does, nothing it doesn't.</h2>
            <div className="lp-features">
              {FEATURES.map((f) => (
                <article key={f.title}>
                  <span className="lp-ic"><f.icon size={20} strokeWidth={1.8} /></span>
                  <h3>{f.title}</h3>
                  <p>{f.text}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* Poses */}
        <section id="poses" className="lp-sec">
          <div className="lp-wrap">
            <h2>Start with six foundational poses.</h2>
            <p className="lp-sub">Each one has a short guide in the app: what to line up, and what to feel.</p>
            <div className="lp-poses">
              {POSES.map((p) => (
                <article key={p.sanskrit} className="lp-pose">
                  <div className="lp-pose-art"><PoseFigure id={p.fig} /></div>
                  <h3>{p.sanskrit}</h3>
                  <span>{p.name}</span>
                  <p>{p.line}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* Privacy */}
        <section id="private" className="lp-sec lp-dark">
          <div className="lp-wrap lp-priv">
            <div>
              <p className="lp-eyebrow light">Privacy</p>
              <h2>Your camera stays yours.</h2>
              <p className="lp-lede light">
                AsanaAI works from the position of your body, not your picture. Your video is processed on your own device and is never recorded or stored. There is no account and nothing to hand over.
              </p>
            </div>
            <ul className="lp-priv-list">
              <li><Lock size={18} /><span>Video never leaves your device</span></li>
              <li><Ruler size={18} /><span>Only body-position numbers are analysed</span></li>
              <li><Languages size={18} /><span>No sign-up, no profile, no tracking of you</span></li>
            </ul>
          </div>
        </section>

        {/* FAQ */}
        <section id="faq" className="lp-sec">
          <div className="lp-wrap lp-faq-wrap">
            <h2>Good questions</h2>
            <div className="lp-faq">
              {FAQ.map((f, i) => (
                <div key={f.q} className={`lp-q ${open === i ? "open" : ""}`}>
                  <button aria-expanded={open === i} onClick={() => setOpen(open === i ? null : i)}>
                    <span>{f.q}</span>
                    <ChevronDown size={18} />
                  </button>
                  <div className="lp-a"><div><p>{f.a}</p></div></div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Final call */}
        <section className="lp-final">
          <div className="lp-wrap">
            <h2>Ready when you are.</h2>
            <p>Prop up your phone, step back, and begin.</p>
            <Link href="/" className="lp-btn light">Start practising</Link>
          </div>
        </section>
      </main>

      <footer className="lp-foot">
        <div className="lp-wrap lp-foot-in">
          <span className="lp-brand sm">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M12 3c1.6 2.6 2.4 5 2.4 7.2 0 3.2-1.6 6.5-2.4 10.8-.8-4.3-2.4-7.6-2.4-10.8C9.6 8 10.4 5.6 12 3Z" />
              <path d="M12 13.2c2.4-3.4 6.2-4.4 8.6-3.7-.5 3.1-3.9 6.7-8.6 8.2" />
              <path d="M12 13.2C9.6 9.8 5.8 8.8 3.4 9.5c.5 3.1 3.9 6.7 8.6 8.2" />
            </svg>
            AsanaAI
          </span>
          <p>A final-year project from RCC Institute of Information Technology, Kolkata. General guidance only, not medical advice.</p>
        </div>
      </footer>
    </div>
  );
}
