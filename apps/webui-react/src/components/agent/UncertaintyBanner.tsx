/**
 * Banner component for displaying uncertainties.
 * Groups by severity with appropriate styling.
 */

import type { Uncertainty, UncertaintySeverity } from '../../types/agent';

interface UncertaintyBannerProps {
  uncertainties: Uncertainty[];
  showResolved?: boolean;
}

const severityConfig: Record<
  UncertaintySeverity,
  { bg: string; border: string; text: string; icon: string; label: string }
> = {
  blocking: {
    bg: 'bg-red-500/10',
    border: 'border-red-500/30',
    text: 'text-red-400',
    icon: 'M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z',
    label: 'Blocking',
  },
  notable: {
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    text: 'text-amber-400',
    icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
    label: 'Notable',
  },
  info: {
    bg: 'bg-blue-500/10',
    border: 'border-blue-500/30',
    text: 'text-blue-400',
    icon: 'M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z',
    label: 'Info',
  },
};

export function UncertaintyBanner({
  uncertainties,
  showResolved = false,
}: UncertaintyBannerProps) {
  // Filter out resolved if not showing them
  const visibleUncertainties = showResolved
    ? uncertainties
    : uncertainties.filter((u) => !u.resolved);

  if (visibleUncertainties.length === 0) {
    return null;
  }

  // Group by severity
  const grouped = visibleUncertainties.reduce(
    (acc, u) => {
      acc[u.severity] = acc[u.severity] || [];
      acc[u.severity].push(u);
      return acc;
    },
    {} as Record<UncertaintySeverity, Uncertainty[]>
  );

  // Sort by severity priority
  const severityOrder: UncertaintySeverity[] = ['blocking', 'notable', 'info'];

  return (
    <div className="space-y-2">
      {severityOrder.map((severity) => {
        const items = grouped[severity];
        if (!items || items.length === 0) return null;

        const config = severityConfig[severity];

        return (
          <div
            key={severity}
            className={`${config.bg} ${config.border} border rounded-lg p-3`}
          >
            <div className="flex items-start gap-2">
              <svg
                className={`w-5 h-5 ${config.text} flex-shrink-0 mt-0.5`}
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d={config.icon}
                />
              </svg>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-xs font-medium ${config.text} uppercase tracking-wide`}>
                    {config.label}
                  </span>
                  <span className={`text-xs ${config.text} opacity-70`}>
                    ({items.length})
                  </span>
                </div>
                <ul className="space-y-1">
                  {items.map((uncertainty) => (
                    <li
                      key={uncertainty.id}
                      className={`text-sm ${config.text} ${uncertainty.resolved ? 'line-through opacity-60' : ''}`}
                    >
                      {uncertainty.message}
                      {uncertainty.resolved && (
                        <span className="ml-2 text-xs text-green-400">
                          (resolved)
                        </span>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

export default UncertaintyBanner;
