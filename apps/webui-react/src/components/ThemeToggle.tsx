import { Sun, Moon, Monitor } from 'lucide-react';
import { useUIStore } from '../stores/uiStore';
import type { Theme } from '../stores/uiStore';

const themeOptions: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'system', label: 'System', icon: Monitor },
];

function ThemeToggle() {
  const { theme, setTheme } = useUIStore();

  // Cycle through themes on click
  const cycleTheme = () => {
    const currentIndex = themeOptions.findIndex(opt => opt.value === theme);
    const nextIndex = (currentIndex + 1) % themeOptions.length;
    setTheme(themeOptions[nextIndex].value);
  };

  const currentOption = themeOptions.find(opt => opt.value === theme) || themeOptions[0];
  const Icon = currentOption.icon;

  return (
    <button
      onClick={cycleTheme}
      className="p-2 rounded-md transition-colors duration-150 hover:bg-[var(--bg-tertiary)]"
      aria-label={`Theme: ${currentOption.label}. Click to change.`}
      title={`Theme: ${currentOption.label}`}
    >
      <Icon className="w-4 h-4 text-[var(--text-secondary)]" />
    </button>
  );
}

export default ThemeToggle;
