import { Search, FileText, Sparkles } from 'lucide-react';
import { CollapsibleSection } from './CollapsibleSection';
import SectionErrorBoundary from './SectionErrorBoundary';
import SearchPreferencesSettings from './SearchPreferencesSettings';
import CollectionDefaultsSettings from './CollectionDefaultsSettings';
import LLMSettings from './LLMSettings';

/**
 * PreferencesTab displays user-configurable settings in collapsible sections.
 * This tab is visible to all users.
 */
export default function PreferencesTab() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-medium text-gray-900">Preferences</h3>
        <p className="mt-1 text-sm text-gray-500">
          Configure your search, collection defaults, and AI settings.
        </p>
      </div>

      <SectionErrorBoundary sectionName="Search Preferences">
        <CollapsibleSection
          name="preferences-search"
          title="Search Preferences"
          icon={Search}
          defaultOpen={true}
        >
          <SearchPreferencesSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>

      <SectionErrorBoundary sectionName="Collection Defaults">
        <CollapsibleSection
          name="preferences-collection-defaults"
          title="Collection Defaults"
          icon={FileText}
          defaultOpen={false}
        >
          <CollectionDefaultsSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>

      <SectionErrorBoundary sectionName="LLM Configuration">
        <CollapsibleSection
          name="preferences-llm"
          title="LLM Configuration"
          icon={Sparkles}
          defaultOpen={false}
        >
          <LLMSettings />
        </CollapsibleSection>
      </SectionErrorBoundary>
    </div>
  );
}
