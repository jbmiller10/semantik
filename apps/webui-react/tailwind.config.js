/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Ink: Primary actions and headings (replaces "signal")
        ink: {
          900: '#1a1a2e',
          800: '#2d2d44',
          700: '#3f3f5c',
          600: '#525274',
          500: '#6b6b8c',
        },
        // Paper: Surface colors (replaces "void")
        paper: {
          50: '#fefdfb',
          100: '#f8f6f0',
          200: '#f0ece0',
          300: '#e4dfd0',
          400: '#d4cfc0',
        },
        // Accent: Scholar's gold for highlights
        accent: {
          600: '#b8860b',
          500: '#daa520',
          400: '#f4c430',
        },
        // Semantic status colors (muted, academic)
        success: {
          DEFAULT: '#059669',
          light: '#10b981',
        },
        warning: {
          DEFAULT: '#d97706',
          light: '#f59e0b',
        },
        error: {
          DEFAULT: '#dc2626',
          light: '#ef4444',
        },
        info: {
          DEFAULT: '#2563eb',
          light: '#3b82f6',
        },
      },
      fontFamily: {
        serif: ['"Source Serif 4"', 'Georgia', 'Cambria', 'serif'],
        sans: ['"Source Sans 3"', 'system-ui', '-apple-system', 'sans-serif'],
        mono: ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
      },
      animation: {
        'fade-in': 'fadeIn 0.3s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'spin-slow': 'spin 2s linear infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(10px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      boxShadow: {
        'card': '0 1px 3px 0 rgb(0 0 0 / 0.05), 0 1px 2px -1px rgb(0 0 0 / 0.05)',
        'card-hover': '0 4px 6px -1px rgb(0 0 0 / 0.07), 0 2px 4px -2px rgb(0 0 0 / 0.05)',
        'panel': '0 1px 2px 0 rgb(0 0 0 / 0.03)',
      },
    },
  },
  plugins: [],
}
