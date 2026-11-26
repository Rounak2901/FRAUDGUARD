/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        ink: {
          950: "#0b1220",
          900: "#111827",
          800: "#1a2332",
          700: "#243044",
        },
        mint: {
          400: "#34d399",
          500: "#10b981",
          600: "#059669",
        },
        alert: {
          400: "#fb7185",
          500: "#f43f5e",
        },
        warn: {
          400: "#fbbf24",
          500: "#f59e0b",
        },
      },
      fontFamily: {
        display: ['"Fraunces"', "Georgia", "serif"],
        sans: ['"DM Sans"', "system-ui", "sans-serif"],
        mono: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
      },
      boxShadow: {
        glow: "0 0 40px rgba(16, 185, 129, 0.15)",
      },
    },
  },
  plugins: [],
};
