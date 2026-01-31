// apps/webui-react/src/components/pipeline/ActivityLog.tsx
import { formatDistanceToNow } from 'date-fns';

export interface ActivityLogProps {
  activities: Array<{ message: string; timestamp: string }>;
  maxHeight?: string;
}

export function ActivityLog({ activities, maxHeight = '150px' }: ActivityLogProps) {
  if (activities.length === 0) {
    return (
      <div className="text-sm text-[var(--text-muted)] py-2">
        No activity yet
      </div>
    );
  }

  return (
    <div
      className="space-y-1 overflow-y-auto"
      style={{ maxHeight }}
    >
      {activities.map((activity, index) => (
        <div
          key={`${activity.timestamp}-${index}`}
          className="flex items-start gap-2 text-sm"
        >
          <span className="text-[var(--text-muted)] shrink-0">•</span>
          <span className="text-[var(--text-secondary)] flex-1">
            {activity.message}
          </span>
          <time
            dateTime={activity.timestamp}
            className="text-[var(--text-muted)] text-xs shrink-0"
            title={new Date(activity.timestamp).toLocaleString()}
          >
            {formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}
          </time>
        </div>
      ))}
    </div>
  );
}
