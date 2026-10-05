import { DM_Sans, DM_Mono, Fraunces } from 'next/font/google';

// Self-hosted at build time (no request to Google at runtime, no render-blocking CSS @import chain).
// The CSS variables are applied on <html> in _document.tsx and consumed by the design tokens.
export const fontUi = DM_Sans({ subsets: ['latin'], display: 'swap', variable: '--nf-ui', axes: ['opsz'] });
export const fontDisplay = Fraunces({ subsets: ['latin'], display: 'swap', variable: '--nf-display', axes: ['opsz'] });
export const fontMono = DM_Mono({ subsets: ['latin'], display: 'swap', weight: ['400', '500'], variable: '--nf-mono' });
