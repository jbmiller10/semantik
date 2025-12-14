import { Folder, GitBranch, Mail, Ban } from 'lucide-react';
import type { ConnectorCatalog } from '../../types/connector';

interface ConnectorTypeSelectorProps {
  catalog: ConnectorCatalog;
  selectedType: string;
  onSelect: (type: string) => void;
  disabled?: boolean;
  /** Show a "None" option for cases where source is optional */
  showNoneOption?: boolean;
}

/**
 * Icon mapping for connector types
 */
const connectorIcons: Record<string, React.ReactNode> = {
  directory: <Folder className="h-6 w-6" />,
  git: <GitBranch className="h-6 w-6" />,
  imap: <Mail className="h-6 w-6" />,
};

/**
 * Display order for connector types
 */
const displayOrder = ['directory', 'git', 'imap'];

/**
 * Card-based connector type selector
 * Displays available connectors with icons and descriptions
 */
export function ConnectorTypeSelector({
  catalog,
  selectedType,
  onSelect,
  disabled = false,
  showNoneOption = false,
}: ConnectorTypeSelectorProps) {
  // Sort connector types by display order
  const sortedTypes = Object.keys(catalog).sort((a, b) => {
    const aIndex = displayOrder.indexOf(a);
    const bIndex = displayOrder.indexOf(b);
    // Put unknown types at the end
    if (aIndex === -1) return 1;
    if (bIndex === -1) return -1;
    return aIndex - bIndex;
  });

  // Determine grid columns based on whether we show the none option
  const gridCols = showNoneOption ? 'grid-cols-4' : 'grid-cols-3';

  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        Select Source Type
      </label>
      <div className={`grid ${gridCols} gap-3`}>
        {/* None option - only shown when showNoneOption is true */}
        {showNoneOption && (
          <button
            type="button"
            onClick={() => onSelect('none')}
            disabled={disabled}
            className={`
              relative flex flex-col items-center p-4 rounded-lg border-2 transition-all
              ${
                selectedType === 'none'
                  ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                  : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
              }
              ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
            `}
          >
            {selectedType === 'none' && (
              <div className="absolute top-2 right-2">
                <div className="h-2 w-2 rounded-full bg-blue-500" />
              </div>
            )}
            <div className={`mb-2 ${selectedType === 'none' ? 'text-blue-600' : 'text-gray-500'}`}>
              <Ban className="h-6 w-6" />
            </div>
            <span className={`text-sm font-medium ${selectedType === 'none' ? 'text-blue-900' : 'text-gray-900'}`}>
              None
            </span>
            <span className={`text-xs mt-1 text-center ${selectedType === 'none' ? 'text-blue-700' : 'text-gray-500'}`}>
              Add later
            </span>
          </button>
        )}

        {sortedTypes.map((type) => {
          const definition = catalog[type];
          const isSelected = type === selectedType;
          const icon = connectorIcons[type] || <Folder className="h-6 w-6" />;

          return (
            <button
              key={type}
              type="button"
              onClick={() => onSelect(type)}
              disabled={disabled}
              className={`
                relative flex flex-col items-center p-4 rounded-lg border-2 transition-all
                ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 ring-2 ring-blue-200'
                    : 'border-gray-200 bg-white hover:border-gray-300 hover:bg-gray-50'
                }
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              {/* Selection indicator */}
              {isSelected && (
                <div className="absolute top-2 right-2">
                  <div className="h-2 w-2 rounded-full bg-blue-500" />
                </div>
              )}

              {/* Icon */}
              <div
                className={`mb-2 ${isSelected ? 'text-blue-600' : 'text-gray-500'}`}
              >
                {icon}
              </div>

              {/* Name */}
              <span
                className={`text-sm font-medium ${
                  isSelected ? 'text-blue-900' : 'text-gray-900'
                }`}
              >
                {definition.name}
              </span>

              {/* Short description */}
              <span
                className={`text-xs mt-1 text-center ${
                  isSelected ? 'text-blue-700' : 'text-gray-500'
                }`}
              >
                {getShortDescription(type)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Get a short description for each connector type
 */
function getShortDescription(type: string): string {
  switch (type) {
    case 'directory':
      return 'Local files';
    case 'git':
      return 'Repository';
    case 'imap':
      return 'Email';
    default:
      return '';
  }
}

export default ConnectorTypeSelector;
