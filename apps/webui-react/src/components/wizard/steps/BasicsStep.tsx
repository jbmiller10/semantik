// apps/webui-react/src/components/wizard/steps/BasicsStep.tsx
import { ConnectorTypeSelector, ConnectorForm } from '../../connectors';
import { useConnectorCatalog } from '../../../hooks/useConnectors';
import type { SyncMode } from '../../../types/collection';

interface BasicsStepProps {
  name: string;
  description: string;
  connectorType: string;
  configValues: Record<string, unknown>;
  secrets: Record<string, string>;
  syncMode: SyncMode;
  syncIntervalMinutes?: number;
  onNameChange: (name: string) => void;
  onDescriptionChange: (description: string) => void;
  onConnectorTypeChange: (type: string) => void;
  onConfigChange: (config: Record<string, unknown>) => void;
  onSecretsChange: (secrets: Record<string, string>) => void;
  onSyncModeChange: (mode: SyncMode) => void;
  onSyncIntervalChange?: (minutes: number) => void;
  errors: Record<string, string>;
}

export function BasicsStep({
  name,
  description,
  connectorType,
  configValues,
  secrets,
  syncMode,
  syncIntervalMinutes = 60,
  onNameChange,
  onDescriptionChange,
  onConnectorTypeChange,
  onConfigChange,
  onSecretsChange,
  onSyncModeChange,
  onSyncIntervalChange,
  errors,
}: BasicsStepProps) {
  const { data: catalog, isLoading: catalogLoading } = useConnectorCatalog();

  return (
    <div className="space-y-6">
      {/* Collection Name */}
      <div>
        <label htmlFor="collection-name" className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
          Collection Name <span className="text-red-500">*</span>
        </label>
        <input
          id="collection-name"
          type="text"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          aria-label="Collection name"
          className={`
            w-full px-3 py-2 rounded-lg border bg-[var(--bg-primary)] text-[var(--text-primary)]
            ${errors.name ? 'border-red-500' : 'border-[var(--border)]'}
          `}
          placeholder="My Collection"
        />
        {errors.name && <p className="mt-1 text-sm text-red-500">{errors.name}</p>}
      </div>

      {/* Description */}
      <div>
        <label htmlFor="collection-description" className="block text-sm font-medium text-[var(--text-secondary)] mb-1">
          Description (optional)
        </label>
        <textarea
          id="collection-description"
          value={description}
          onChange={(e) => onDescriptionChange(e.target.value)}
          aria-label="Description"
          className="w-full px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] text-[var(--text-primary)] resize-none"
          rows={2}
          placeholder="Describe what this collection contains..."
        />
        {errors.description && <p className="mt-1 text-sm text-red-500">{errors.description}</p>}
      </div>

      {/* Data Source */}
      <div>
        {catalogLoading ? (
          <div className="animate-pulse">
            <div className="h-4 bg-[var(--bg-tertiary)] rounded w-24 mb-3" />
            <div className="grid grid-cols-4 gap-3">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-24 bg-[var(--bg-tertiary)] rounded-lg" />
              ))}
            </div>
          </div>
        ) : catalog ? (
          <>
            <ConnectorTypeSelector
              catalog={catalog}
              selectedType={connectorType}
              onSelect={(type) => {
                onConnectorTypeChange(type);
                onConfigChange({});
                onSecretsChange({});
              }}
              showNoneOption={true}
            />

            {connectorType !== 'none' && (
              <ConnectorForm
                catalog={catalog}
                connectorType={connectorType}
                values={configValues}
                secrets={secrets}
                onValuesChange={onConfigChange}
                onSecretsChange={onSecretsChange}
                errors={errors}
              />
            )}
          </>
        ) : null}
      </div>

      {/* Sync Mode (only show when source is configured) */}
      {connectorType !== 'none' && (
        <div>
          <h3 className="text-sm font-medium text-[var(--text-secondary)] mb-3">Sync Mode</h3>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="syncMode"
                value="one_time"
                checked={syncMode === 'one_time'}
                onChange={() => onSyncModeChange('one_time')}
                className="accent-gray-600 dark:accent-white"
              />
              <span className="text-sm text-[var(--text-primary)]">One-time import</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="radio"
                name="syncMode"
                value="continuous"
                checked={syncMode === 'continuous'}
                onChange={() => onSyncModeChange('continuous')}
                className="accent-gray-600 dark:accent-white"
              />
              <span className="text-sm text-[var(--text-primary)]">Continuous sync</span>
            </label>
          </div>

          {syncMode === 'continuous' && onSyncIntervalChange && (
            <div className="mt-3">
              <label className="block text-sm text-[var(--text-muted)] mb-1">
                Sync interval (minutes)
              </label>
              <input
                type="number"
                min={15}
                value={syncIntervalMinutes}
                onChange={(e) => onSyncIntervalChange(parseInt(e.target.value, 10))}
                className="w-32 px-3 py-2 rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] text-[var(--text-primary)]"
              />
              {errors.sync_interval_minutes && (
                <p className="mt-1 text-sm text-red-500">{errors.sync_interval_minutes}</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
