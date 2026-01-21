/**
 * AdminTab displays admin-only settings and operations.
 * This tab is only visible to superusers.
 *
 * Contains collapsible sections for:
 * - Resource Limits (per-user quotas)
 * - Performance Tuning (cache, model timeouts)
 * - GPU & Memory (memory management)
 * - Search & Reranking (search tuning)
 * - Danger Zone (destructive operations)
 */
import { Shield, Database, Zap, Cpu, Search, AlertTriangle } from 'lucide-react';
import { CollapsibleSection } from './CollapsibleSection';
import SectionErrorBoundary from './SectionErrorBoundary';
import ResourceLimitsSettings from './ResourceLimitsSettings';
import PerformanceSettings from './PerformanceSettings';
import GpuMemorySettings from './GpuMemorySettings';
import SearchRerankSettings from './SearchRerankSettings';
import DangerZoneSettings from './DangerZoneSettings';

export default function AdminTab() {
  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h3 className="text-lg font-medium text-[var(--text-primary)] flex items-center gap-2">
          <Shield className="h-5 w-5 text-amber-500" />
          Admin Settings
        </h3>
        <p className="mt-1 text-sm text-[var(--text-secondary)]">
          Administrative operations and system configuration. These settings affect all users.
        </p>
      </div>

      {/* Resource Limits Section */}
      <SectionErrorBoundary sectionName="Resource Limits">
        <CollapsibleSection
          name="admin-resource-limits"
          title="Resource Limits"
          icon={Database}
          defaultOpen={true}
        >
          <ResourceLimitsSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>

      {/* Performance Section */}
      <SectionErrorBoundary sectionName="Performance">
        <CollapsibleSection
          name="admin-performance"
          title="Performance Tuning"
          icon={Zap}
          defaultOpen={false}
        >
          <PerformanceSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>

      {/* GPU & Memory Section */}
      <SectionErrorBoundary sectionName="GPU & Memory">
        <CollapsibleSection
          name="admin-gpu-memory"
          title="GPU & Memory"
          icon={Cpu}
          defaultOpen={false}
        >
          <GpuMemorySettings />
        </CollapsibleSection>
      </SectionErrorBoundary>

      {/* Search & Reranking Section */}
      <SectionErrorBoundary sectionName="Search & Reranking">
        <CollapsibleSection
          name="admin-search-rerank"
          title="Search & Reranking"
          icon={Search}
          defaultOpen={false}
        >
          <SearchRerankSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>

      {/* Danger Zone Section */}
      <SectionErrorBoundary sectionName="Danger Zone">
        <CollapsibleSection
          name="admin-danger-zone"
          title="Danger Zone"
          icon={AlertTriangle}
          defaultOpen={false}
        >
          <DangerZoneSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>
    </div>
  );
}
