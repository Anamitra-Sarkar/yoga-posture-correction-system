import { useState } from "react";
import Head from "next/head";
import Link from "next/link";
import { 
  Menu, 
  X, 
  Sparkles, 
  Activity, 
  ShieldCheck, 
  Volume2, 
  Cpu, 
  Layers, 
  Globe, 
  Server, 
  Cloud, 
  HelpCircle,
  Smartphone,
  CheckCircle2,
  Sliders
} from "lucide-react";

export default function LandingPage() {
  const [navOpen, setNavOpen] = useState(false);

  return (
    <div className="landing-wrapper">
      <Head>
        <title>AsanaAI — Real-Time AI Yoga Coach</title>
        <meta name="description" content="Practice yoga smarter with AsanaAI. Real-time posture correction, 13-stage AI pipeline, personalized digital twin calibration, and multilingual voice feedback." />
      </Head>

      {/* 1. Sticky Navbar */}
      <nav className="landing-nav">
        <div className="app-logo">
          <svg
            width="24"
            height="24"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-primary)"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="logo-mark"
          >
            <path d="M12 2C12 2 15 7 15 11C15 15 12 22 12 22C12 22 9 15 9 11C9 7 12 2 12 2Z" />
            <path d="M12 11C15 9 20 9 21 11C22 13 19 16 12 22" />
            <path d="M12 11C9 9 4 9 3 11C2 13 5 16 12 22" />
          </svg>
          <span>AsanaAI</span>
        </div>

        <ul className={`landing-nav-links ${navOpen ? "open" : ""}`}>
          <li>
            <a href="#features" onClick={() => setNavOpen(false)}>Features</a>
          </li>
          <li>
            <a href="#how-it-works" onClick={() => setNavOpen(false)}>How It Works</a>
          </li>
          <li>
            <a href="#tech-stack" onClick={() => setNavOpen(false)}>Tech Stack</a>
          </li>
          <li>
            <Link href="/" className="btn-primary btn-sm" onClick={() => setNavOpen(false)}>
              Launch App
            </Link>
          </li>
        </ul>

        <button 
          className="mobile-menu-btn" 
          onClick={() => setNavOpen(!navOpen)}
          aria-label="Toggle navigation menu"
        >
          {navOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </nav>

      {/* 2. Hero Section */}
      <header className="landing-section hero-section">
        <div className="hero-content">
          <div className="landing-section-label">Intelligent Movement Analysis</div>
          <h1 className="hero-title">Real-Time AI Yoga Coach</h1>
          <p className="hero-subtitle">
            Detect your pose. Get instant biomechanical feedback. Practice yoga smarter with a 13-stage AI pipeline.
          </p>
          <div className="hero-actions-row">
            <Link href="/" className="btn-primary">
              Launch Dashboard
            </Link>
            <a 
              href="https://github.com/Anamitra-Sarkar/yoga-posture-correction-system" 
              className="btn-outline"
              target="_blank" 
              rel="noopener noreferrer"
            >
              View on GitHub
            </a>
          </div>
        </div>

        {/* Decorative Background Skeleton Overlay */}
        <div className="hero-skeleton-wrapper" aria-hidden="true">
          <svg viewBox="0 0 400 400" width="100%" height="100%">
            {/* Ground line */}
            <line x1="50" y1="350" x2="350" y2="350" stroke="var(--color-divider)" strokeWidth="2" />
            
            {/* Spine & Head */}
            <line x1="200" y1="200" x2="200" y2="150" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <circle cx="200" cy="120" r="16" fill="none" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            
            {/* Hips & Legs (Warrior II pose representation) */}
            <line x1="200" y1="200" x2="160" y2="200" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <line x1="200" y1="200" x2="240" y2="200" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            
            {/* Back Leg (Right) - Straight */}
            <line x1="160" y1="200" x2="110" y2="270" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <line x1="110" y1="270" x2="70" y2="350" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <circle cx="70" cy="350" r="6" fill="var(--color-primary)" opacity="0.25" />
            
            {/* Front Leg (Left) - Bent at 90 degrees */}
            <line x1="240" y1="200" x2="300" y2="200" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <line x1="300" y1="200" x2="300" y2="350" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <circle cx="300" cy="350" r="6" fill="var(--color-primary)" opacity="0.25" />
            
            {/* Arms - Extended horizontally */}
            <line x1="200" y1="165" x2="110" y2="165" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <circle cx="110" cy="165" r="5" fill="var(--color-primary)" opacity="0.25" />
            
            {/* Front Arm (Right) */}
            <line x1="200" y1="165" x2="300" y2="165" stroke="var(--color-primary)" strokeWidth="4" opacity="0.25" />
            <circle cx="300" cy="165" r="5" fill="var(--color-primary)" opacity="0.25" />

            {/* Joints highlights */}
            <circle cx="200" cy="200" r="7" fill="var(--color-primary)" opacity="0.35" /> {/* Pelvis */}
            <circle cx="200" y1="165" r="7" fill="var(--color-primary)" opacity="0.35" /> {/* Shoulder Center */}
            <circle cx="160" cy="200" r="6" fill="var(--color-primary)" opacity="0.35" /> {/* Hip R */}
            <circle cx="240" cy="200" r="6" fill="var(--color-primary)" opacity="0.35" /> {/* Hip L */}
            <circle cx="300" cy="200" r="7" fill="var(--color-primary)" opacity="0.35" /> {/* Knee L (target angle 90) */}
            <circle cx="110" cy="270" r="6" fill="var(--color-primary)" opacity="0.35" /> {/* Knee R */}
          </svg>
        </div>
      </header>

      {/* 3. Features Grid */}
      <section id="features" className="landing-section features-section">
        <div className="landing-section-label">Capability Matrix</div>
        <h2 className="landing-section-heading">Why AsanaAI</h2>
        
        <div className="landing-feature-grid">
          <div className="landing-feature-card">
            <Activity className="feature-icon" size={20} />
            <span className="landing-feature-category">Detection</span>
            <h3 className="landing-feature-title">Real-Time Pose Detection</h3>
            <p className="landing-feature-desc">
              MediaPipe landmark tracking at 30fps with client-side biomechanical angle extraction.
            </p>
          </div>

          <div className="landing-feature-card">
            <Layers className="feature-icon" size={20} />
            <span className="landing-feature-category">Pipeline</span>
            <h3 className="landing-feature-title">13-Stage AI Pipeline</h3>
            <p className="landing-feature-desc">
              From landmark ingestion to correction voice output, every frame is processed through a typed pipeline.
            </p>
          </div>

          <div className="landing-feature-card">
            <ShieldCheck className="feature-icon" size={20} />
            <span className="landing-feature-category">Personalization</span>
            <h3 className="landing-feature-title">Digital Twin Calibration</h3>
            <p className="landing-feature-desc">
              User-specific joint range limits that personalize the posture feedback thresholds to prevent hyperextensions.
            </p>
          </div>

          <div className="landing-feature-card">
            <HelpCircle className="feature-icon" size={20} />
            <span className="landing-feature-category">Occlusions</span>
            <h3 className="landing-feature-title">Occlusion Recovery</h3>
            <p className="landing-feature-desc">
              When a joint is self-occluded, the system mirrors the symmetric counterpart using anatomical constraints.
            </p>
          </div>

          <div className="landing-feature-card">
            <Volume2 className="feature-icon" size={20} />
            <span className="landing-feature-category">Audio</span>
            <h3 className="landing-feature-title">Multilingual Voice Guidance</h3>
            <p className="landing-feature-desc">
              Natural, safety-bounded text-to-speech correction cues delivered in English, Hindi, and Bengali.
            </p>
          </div>

          <div className="landing-feature-card">
            <Sliders className="feature-icon" size={20} />
            <span className="landing-feature-category">Simulator</span>
            <h3 className="landing-feature-title">Offline Simulation Mode</h3>
            <p className="landing-feature-desc">
              Practice and test target asana coordinates using the slider simulator without camera access or internet.
            </p>
          </div>
        </div>
      </section>

      {/* 4. How It Works */}
      <section id="how-it-works" className="landing-section how-works-section">
        <div className="landing-section-label">Workflow Architecture</div>
        <h2 className="landing-section-heading">From Frame to Feedback</h2>
        
        <div className="steps-flow">
          <div className="step-item">
            <div className="step-number">01</div>
            <div className="step-text">
              <h3 className="step-name">Capture</h3>
              <p className="step-desc">Webcam frame or simulation input coordinates received by client.</p>
            </div>
          </div>

          <div className="step-item">
            <div className="step-number">02</div>
            <div className="step-text">
              <h3 className="step-name">Landmark Detection</h3>
              <p className="step-desc">MediaPipe model parses 33 3D skeleton keypoints in real time.</p>
            </div>
          </div>

          <div className="step-item">
            <div className="step-number">03</div>
            <div className="step-text">
              <h3 className="step-name">Angle Extraction</h3>
              <p className="step-desc">15 biomechanically relevant joint angles are computed client-side.</p>
            </div>
          </div>

          <div className="step-item">
            <div className="step-number">04</div>
            <div className="step-text">
              <h3 className="step-name">Pipeline Processing</h3>
              <p className="step-desc">13-stage backend classifies pose, correctness, and runs Groq LLaMA.</p>
            </div>
          </div>

          <div className="step-item">
            <div className="step-number">05</div>
            <div className="step-text">
              <h3 className="step-name">Feedback</h3>
              <p className="step-desc">Overlay skeleton rendering and voice guidance delivered in under 500ms.</p>
            </div>
          </div>
        </div>
      </section>

      {/* 5. Tech Stack */}
      <section id="tech-stack" className="landing-section tech-stack-section">
        <div className="landing-section-label">Architecture</div>
        <h2 className="landing-section-heading">Built With</h2>
        
        <div className="tech-groups-grid">
          <div className="tech-category">
            <h3 className="tech-category-label">Frontend</h3>
            <div className="tech-pills">
              <span className="tech-pill">Next.js</span>
              <span className="tech-pill">TypeScript</span>
              <span className="tech-pill">MediaPipe.js</span>
              <span className="tech-pill">Lucide Icons</span>
            </div>
          </div>

          <div className="tech-category">
            <h3 className="tech-category-label">Backend</h3>
            <div className="tech-pills">
              <span className="tech-pill">FastAPI</span>
              <span className="tech-pill">Python</span>
              <span className="tech-pill">Groq API</span>
              <span className="tech-pill">Uvicorn</span>
            </div>
          </div>

          <div className="tech-category">
            <h3 className="tech-category-label">Machine Learning</h3>
            <div className="tech-pills">
              <span className="tech-pill">MediaPipe Pose</span>
              <span className="tech-pill">3-Head MLP (PyTorch)</span>
              <span className="tech-pill">ST-GCN temporal classifier</span>
            </div>
          </div>

          <div className="tech-category">
            <h3 className="tech-category-label">Deployment</h3>
            <div className="tech-pills">
              <span className="tech-pill">Vercel (Frontend)</span>
              <span className="tech-pill">HuggingFace Spaces (Backend API)</span>
              <span className="tech-pill">Docker Containers</span>
            </div>
          </div>
        </div>
      </section>

      {/* 6. CTA Banner */}
      <section className="cta-banner">
        <h2 className="cta-banner-title">Start practicing smarter today</h2>
        <p className="cta-banner-sub">
          No account needed. Works in your browser. Free and open source.
        </p>
        <Link href="/" className="btn-cta-white">
          Launch Dashboard
        </Link>
      </section>

      {/* 7. Footer */}
      <footer className="landing-footer">
        <div className="footer-left">
          <svg
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="var(--color-text-muted)"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
            className="logo-mark"
          >
            <path d="M12 2C12 2 15 7 15 11C15 15 12 22 12 22C12 22 9 15 9 11C9 7 12 2 12 2Z" />
            <path d="M12 11C15 9 20 9 21 11C22 13 19 16 12 22" />
            <path d="M12 11C9 9 4 9 3 11C2 13 5 16 12 22" />
          </svg>
          <span className="footer-logo-text">AsanaAI</span>
        </div>
        <div className="footer-center">
          &copy; {new Date().getFullYear()} AsanaAI. All rights reserved.
        </div>
        <div className="footer-right">
          <a 
            href="https://github.com/Anamitra-Sarkar/yoga-posture-correction-system" 
            target="_blank" 
            rel="noopener noreferrer"
            className="footer-link"
          >
            GitHub
          </a>
        </div>
      </footer>

      {/* Landing Specific Styles */}
      <style jsx global>{`
        .landing-wrapper {
          min-height: 100vh;
          background-color: var(--color-bg);
          color: var(--color-text);
          font-family: var(--font-body);
        }

        .landing-nav {
          position: sticky;
          top: 0;
          z-index: 100;
          background: rgba(245, 244, 240, 0.9);
          backdrop-filter: blur(12px);
          -webkit-backdrop-filter: blur(12px);
          border-bottom: 1px solid var(--color-divider);
          padding: 0 5%;
          height: 60px;
          display: flex;
          align-items: center;
          justify-content: space-between;
        }

        .landing-nav-links {
          display: flex;
          align-items: center;
          gap: 2rem;
          list-style: none;
        }

        .landing-nav-links a {
          font-size: var(--text-sm);
          color: var(--color-text-muted);
          text-decoration: none;
          font-weight: 500;
          transition: color var(--transition);
        }

        .landing-nav-links a:hover {
          color: var(--color-text);
        }

        /* On mobile, collapse nav links into hamburger */
        @media (max-width: 767px) {
          .landing-nav-links {
            display: none;
          }
          .landing-nav-links.open {
            display: flex;
            flex-direction: column;
            position: absolute;
            top: 60px;
            left: 0;
            right: 0;
            background: var(--color-surface-2);
            border-bottom: 1px solid var(--color-divider);
            padding: var(--space-4);
            gap: var(--space-4);
          }
        }

        .landing-section {
          padding: clamp(4rem, 8vw, 7rem) 5%;
          position: relative;
          max-width: 1200px;
          margin-inline: auto;
          width: 100%;
        }

        .hero-section {
          display: grid;
          grid-template-columns: 1.2fr 0.8fr;
          gap: var(--space-8);
          align-items: center;
          min-height: 100vh;
          min-height: 100svh;
          padding-top: calc(60px + var(--space-8));
        }

        @media (max-width: 900px) {
          .hero-section {
            grid-template-columns: 1fr;
            text-align: center;
            padding-top: var(--space-12);
          }
          .hero-skeleton-wrapper {
            max-width: 300px;
            margin-inline: auto;
          }
        }

        .hero-content {
          display: flex;
          flex-direction: column;
          align-items: flex-start;
          gap: var(--space-4);
        }

        @media (max-width: 900px) {
          .hero-content {
            align-items: center;
          }
        }

        .hero-title {
          font-size: clamp(2.5rem, 5vw, 4.5rem);
          font-weight: 700;
          color: var(--color-text);
          line-height: 1.1;
          letter-spacing: -0.02em;
        }

        .hero-subtitle {
          font-size: clamp(1rem, 1.5vw, 1.25rem);
          color: var(--color-text-muted);
          line-height: 1.5;
          max-width: 45ch;
        }

        .hero-actions-row {
          display: flex;
          align-items: center;
          gap: var(--space-4);
          margin-top: var(--space-2);
        }

        .hero-skeleton-wrapper {
          opacity: 0.8;
          width: 100%;
          display: flex;
          align-items: center;
          justify-content: center;
        }

        .landing-section-label {
          font-size: var(--text-xs);
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.1em;
          color: var(--color-primary);
          margin-bottom: var(--space-3);
        }

        .landing-section-heading {
          font-size: clamp(1.75rem, 3vw, 2.5rem);
          font-weight: 700;
          color: var(--color-text);
          line-height: 1.2;
          margin-bottom: var(--space-4);
        }

        .landing-feature-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: var(--space-4);
          margin-top: var(--space-8);
        }

        @media (max-width: 900px) {
          .landing-feature-grid {
            grid-template-columns: repeat(2, 1fr);
          }
        }

        @media (max-width: 560px) {
          .landing-feature-grid {
            grid-template-columns: 1fr;
          }
        }

        .landing-feature-card {
          background: var(--color-surface-2);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-lg);
          padding: var(--space-6);
          display: flex;
          flex-direction: column;
          gap: var(--space-2);
          box-shadow: var(--shadow-sm);
          transition: box-shadow var(--transition);
        }

        .landing-feature-card:hover {
          box-shadow: var(--shadow-md);
        }

        .feature-icon {
          color: var(--color-primary);
          margin-bottom: var(--space-1);
        }

        .landing-feature-category {
          font-size: var(--text-xs);
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--color-primary);
        }

        .landing-feature-title {
          font-size: var(--text-base);
          font-weight: 600;
          color: var(--color-text);
        }

        .landing-feature-desc {
          font-size: var(--text-sm);
          color: var(--color-text-muted);
          line-height: 1.6;
        }

        /* How it works - step flow */
        .steps-flow {
          display: flex;
          align-items: flex-start;
          gap: 0;
          margin-top: var(--space-8);
          position: relative;
          width: 100%;
        }

        .step-item {
          flex: 1;
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: 0 var(--space-3);
          position: relative;
        }

        .step-item:not(:last-child)::after {
          content: '';
          position: absolute;
          top: 20px;
          right: calc(-1 * var(--space-3));
          left: 50%;
          height: 1px;
          background: var(--color-divider);
          transform: translateX(50%);
          width: calc(100% - var(--space-3));
        }

        .step-number {
          width: 40px;
          height: 40px;
          border-radius: 50%;
          background: var(--color-surface-offset);
          border: 1px solid var(--color-border);
          display: flex;
          align-items: center;
          justify-content: center;
          font-family: var(--font-mono);
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--color-primary);
          flex-shrink: 0;
          z-index: 1;
          position: relative;
        }

        .step-name {
          font-size: var(--text-sm);
          font-weight: 600;
          color: var(--color-text);
          margin-top: var(--space-3);
        }

        .step-desc {
          font-size: var(--text-xs);
          color: var(--color-text-muted);
          margin-top: var(--space-1);
          line-height: 1.5;
        }

        @media (max-width: 767px) {
          .steps-flow {
            flex-direction: column;
            gap: 0;
          }
          .step-item {
            flex-direction: row;
            text-align: left;
            padding: var(--space-4) 0;
            gap: var(--space-4);
            align-items: flex-start;
          }
          .step-item:not(:last-child)::after {
            top: auto;
            left: 20px;
            right: auto;
            bottom: 0;
            width: 1px;
            height: calc(100% - 40px);
            transform: translateY(-50%) translateX(0);
          }
          .step-name, .step-desc {
            margin: 0;
          }
          .step-text {
            display: flex;
            flex-direction: column;
            gap: var(--space-1);
          }
        }

        /* Tech stack pills */
        .tech-groups-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          gap: var(--space-6) var(--space-10);
          margin-top: var(--space-6);
        }

        @media (max-width: 640px) {
          .tech-groups-grid {
            grid-template-columns: 1fr;
          }
        }

        .tech-category {
          margin-bottom: var(--space-5);
        }

        .tech-category-label {
          font-size: var(--text-xs);
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.08em;
          color: var(--color-text-muted);
          margin-bottom: var(--space-3);
        }

        .tech-pills {
          display: flex;
          flex-wrap: wrap;
          gap: var(--space-2);
        }

        .tech-pill {
          background: var(--color-surface);
          border: 1px solid var(--color-border);
          border-radius: var(--radius-full);
          padding: var(--space-1) var(--space-3);
          font-size: var(--text-xs);
          font-weight: 500;
          color: var(--color-text-muted);
        }

        /* CTA Banner */
        .cta-banner {
          background: var(--color-primary);
          color: white;
          text-align: center;
          padding: clamp(3rem, 6vw, 5rem) 5%;
        }

        .cta-banner-title {
          font-size: clamp(1.5rem, 2.5vw, 2rem);
          font-weight: 700;
          margin-bottom: var(--space-3);
        }

        .cta-banner-sub {
          font-size: var(--text-sm);
          opacity: 0.8;
          margin-bottom: var(--space-6);
          max-width: 42ch;
          margin-inline: auto;
          line-height: 1.6;
        }

        .btn-cta-white {
          background: white;
          color: var(--color-primary);
          border: none;
          border-radius: var(--radius-md);
          padding: var(--space-3) var(--space-8);
          font-size: var(--text-sm);
          font-weight: 600;
          cursor: pointer;
          text-decoration: none;
          display: inline-block;
          transition: opacity 150ms ease;
        }

        .btn-cta-white:hover {
          opacity: 0.9;
        }

        /* Ghost outline button for landing hero */
        .btn-outline {
          background: transparent;
          border: 1.5px solid var(--color-primary-border);
          color: var(--color-primary);
          border-radius: var(--radius-md);
          padding: var(--space-3) var(--space-6);
          font-size: var(--text-sm);
          font-weight: 600;
          cursor: pointer;
          text-decoration: none;
          display: inline-flex;
          align-items: center;
          gap: var(--space-2);
          transition: background var(--transition), border-color var(--transition);
        }

        .btn-outline:hover {
          background: var(--color-primary-muted);
          border-color: var(--color-primary);
        }

        /* Landing Footer */
        .landing-footer {
          border-top: 1px solid var(--color-divider);
          padding: var(--space-6) 5%;
          display: flex;
          align-items: center;
          justify-content: space-between;
          font-size: var(--text-sm);
          color: var(--color-text-muted);
          max-width: 1200px;
          margin-inline: auto;
          width: 100%;
        }

        @media (max-width: 640px) {
          .landing-footer {
            flex-direction: column;
            gap: var(--space-4);
            text-align: center;
          }
        }

        .footer-left {
          display: flex;
          align-items: center;
          gap: var(--space-2);
        }

        .footer-logo-text {
          font-weight: 700;
          color: var(--color-text-muted);
        }

        .footer-link {
          color: var(--color-text-muted);
          text-decoration: none;
          font-weight: 500;
          transition: color var(--transition);
        }

        .footer-link:hover {
          color: var(--color-text);
        }
      `}</style>
    </div>
  );
}
