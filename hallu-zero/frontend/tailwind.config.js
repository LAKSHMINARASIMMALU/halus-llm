/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        display: ['"DM Serif Display"', 'Georgia', 'serif'],
        body: ['"IBM Plex Mono"', 'monospace'],
      },
      colors: {
        ink: {
          950: '#0a0a0f',
          900: '#0f0f18',
          800: '#161622',
          700: '#1e1e2e',
          600: '#2a2a3e',
          500: '#3a3a52',
          400: '#5a5a7a',
          300: '#8a8aaa',
          200: '#b0b0cc',
          100: '#d8d8ee',
          50:  '#f0f0f8',
        },
        signal: {
          green:  '#00ff88',
          amber:  '#ffb347',
          red:    '#ff4466',
          blue:   '#4488ff',
          purple: '#aa66ff',
        },
      },
      animation: {
        'fade-in':    'fadeIn 0.4s ease forwards',
        'slide-up':   'slideUp 0.3s ease forwards',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
        'scan':       'scan 2s linear infinite',
      },
      keyframes: {
        fadeIn:  { from: { opacity: 0 }, to: { opacity: 1 } },
        slideUp: { from: { opacity: 0, transform: 'translateY(8px)' }, to: { opacity: 1, transform: 'translateY(0)' } },
        scan:    { from: { backgroundPosition: '0% 0%' }, to: { backgroundPosition: '0% 100%' } },
      },
    },
  },
  plugins: [],
}
