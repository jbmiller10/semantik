import type { ReactNode } from 'react';
import type { LucideIcon } from 'lucide-react';
import { ChevronDown } from 'lucide-react';
import { useSettingsUIStore } from '../../stores/settingsUIStore';

interface CollapsibleSectionProps {
  /** Unique identifier for persistence (used as key in settingsUIStore) */
  name: string;
  /** Display title for the section header */
  title: string;
  /** Lucide icon component to display in header */
  icon?: LucideIcon;
  /** Whether section should be open by default if no persisted state */
  defaultOpen?: boolean;
  /** Section content */
  children: ReactNode;
  /** Optional badge to show in header (e.g., count, status) */
  badge?: ReactNode;
  /** Show loading state - spinner in header, skeleton in body */
  isLoading?: boolean;
  /** Additional className for the container */
  className?: string;
  /** Callback when section is toggled */
  onToggle?: (isOpen: boolean) => void;
}

/**
 * Loading skeleton for section content
 */
function CollapsibleSectionSkeleton() {
  return (
    <div className="animate-pulse space-y-4">
      <div className="h-4 bg-[var(--bg-tertiary)] rounded w-3/4" />
      <div className="h-4 bg-[var(--bg-tertiary)] rounded w-1/2" />
      <div className="h-10 bg-[var(--bg-tertiary)] rounded" />
      <div className="h-4 bg-[var(--bg-tertiary)] rounded w-2/3" />
    </div>
  );
}

/**
 * Collapsible section component for settings pages.
 * Persists open/closed state to localStorage via Zustand store.
 */
export function CollapsibleSection({
  name,
  title,
  icon: Icon,
  defaultOpen = true,
  children,
  badge,
  isLoading = false,
  className = '',
  onToggle,
}: CollapsibleSectionProps) {
  const { isSectionOpen, toggleSection } = useSettingsUIStore();
  const isOpen = isSectionOpen(name, defaultOpen);

  const handleToggle = () => {
    const newState = !isOpen;
    toggleSection(name, defaultOpen);
    onToggle?.(newState);
  };

  return (
    <div className={`bg-[var(--bg-secondary)] shadow rounded-lg overflow-hidden border border-[var(--border)] ${className}`}>
      {/* Header - clickable toggle */}
      <button
        type="button"
        onClick={handleToggle}
        className="w-full flex items-center justify-between p-4 bg-[var(--bg-tertiary)] hover:bg-[var(--bg-primary)] transition-colors"
        aria-expanded={isOpen}
        aria-controls={`section-content-${name}`}
      >
        <div className="flex items-center space-x-3">
          {Icon && (
            <Icon
              className={`w-5 h-5 ${isOpen ? 'text-[var(--accent-primary)]' : 'text-[var(--text-muted)]'}`}
            />
          )}
          <span className="font-medium text-[var(--text-primary)]">{title}</span>
          {badge && <span className="ml-2">{badge}</span>}
          {isLoading && (
            <svg
              className="animate-spin h-4 w-4 text-[var(--text-muted)] ml-2"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
          )}
        </div>
        <ChevronDown
          className={`w-5 h-5 text-[var(--text-muted)] transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {/* Content - conditionally rendered */}
      {isOpen && (
        <div
          id={`section-content-${name}`}
          className="p-4 border-t border-[var(--border)]"
        >
          {isLoading ? <CollapsibleSectionSkeleton /> : children}
        </div>
      )}
    </div>
  );
}

export default CollapsibleSection;
