/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // The Void: Deep, warm blacks for backgrounds
        void: {
          950: '#0a0a0b', // Main background
          900: '#18181b', // Secondary / Cards
          800: '#27272a', // Borders / Accents
          700: '#3f3f46',
          50: '#fafafa', // Text on dark (inverted logic if needed, but standardizing on surface mainly)
        },
        // The Signal: Electric Violet for primary actions
        signal: {
          400: '#a78bfa',
          500: '#8b5cf6',
          600: '#7c3aed', // Primary Brand
          700: '#6d28d9',
        },
        // Data/Success (Legacy preserved but sharpened)
        data: {
          teal: '#14b8a6',
          cyan: '#06b6d4',
        },
        // Destructive
        alert: {
          DEFAULT: '#f43f5e',
        },
        // Keeping surface for compatibility but pushing towards void
        surface: {
          50: '#f9fafb',
          100: '#f3f4f6',
          200: '#e5e7eb',
          900: '#111827',
        }
      },
      fontFamily: {
        sans: ['Inter', 'Geist Sans', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      backgroundImage: {
        'void-gradient': 'radial-gradient(circle at 50% 0%, rgba(124, 58, 237, 0.08) 0%, rgba(10, 10, 11, 0) 50%)',
      },
      fontFamily: {
        sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.5s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { transform: 'translateY(20px)', opacity: '0' },
          '100%': { transform: 'translateY(0)', opacity: '1' },
        },
      },
      backdropBlur: {
        xs: '2px',
      }
    },
  },
  plugins: [],
}