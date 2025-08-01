import React, { useState, useRef, useEffect } from 'react';
import { Check, ChevronDown, ChevronUp, Search } from 'lucide-react';
import type { Collection } from '@/types/collection';

interface CollectionMultiSelectProps {
  collections: Collection[];
  selectedCollections: string[];
  onChange: (selected: string[]) => void;
  disabled?: boolean;
  placeholder?: string;
}

export const CollectionMultiSelect: React.FC<CollectionMultiSelectProps> = ({
  collections,
  selectedCollections,
  onChange,
  disabled = false,
  placeholder = 'Select collections...',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Filter collections based on search term and ready status
  const readyCollections = collections.filter(
    (col) => col.status === 'ready' && col.vector_count > 0
  );

  const filteredCollections = readyCollections.filter((col) =>
    col.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const handleToggle = (collectionId: string) => {
    if (selectedCollections.includes(collectionId)) {
      onChange(selectedCollections.filter((id) => id !== collectionId));
    } else {
      onChange([...selectedCollections, collectionId]);
    }
  };

  const handleSelectAll = () => {
    onChange(filteredCollections.map((col) => col.id));
  };

  const handleClearAll = () => {
    onChange([]);
  };

  const selectedCount = selectedCollections.length;
  const selectedNames = collections
    .filter((col) => selectedCollections.includes(col.id))
    .map((col) => col.name)
    .join(', ');

  return (
    <div className="relative" ref={dropdownRef}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        disabled={disabled || readyCollections.length === 0}
        className={`
          w-full px-3 py-2 text-left border rounded-lg flex items-center justify-between
          ${disabled || readyCollections.length === 0
            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
            : 'bg-white hover:bg-gray-50 cursor-pointer'
          }
          ${isOpen ? 'ring-2 ring-blue-500 border-blue-500' : 'border-gray-300'}
        `}
        aria-label="Select collections"
        aria-expanded={isOpen}
        aria-haspopup="listbox"
        aria-describedby="selected-collections-count"
      >
        <span className="truncate" id="selected-collections-count">
          {selectedCount === 0
            ? placeholder
            : selectedCount === 1
            ? selectedNames
            : `${selectedCount} collections selected`}
        </span>
        {isOpen ? (
          <ChevronUp className="w-4 h-4 text-gray-400" />
        ) : (
          <ChevronDown className="w-4 h-4 text-gray-400" />
        )}
      </button>

      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg" role="listbox" aria-label="Available collections">
          {/* Search input */}
          <div className="p-2 border-b border-gray-200">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                placeholder="Search collections..."
                className="w-full pl-8 pr-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                onClick={(e) => e.stopPropagation()}
                aria-label="Search through collections"
              />
            </div>
          </div>

          {/* Select/Clear all buttons */}
          <div className="flex gap-2 px-2 py-2 border-b border-gray-200">
            <button
              type="button"
              onClick={handleSelectAll}
              className="px-3 py-1 text-xs font-medium text-blue-600 hover:bg-blue-50 rounded"
              aria-label="Select all filtered collections"
            >
              Select All
            </button>
            <button
              type="button"
              onClick={handleClearAll}
              className="px-3 py-1 text-xs font-medium text-gray-600 hover:bg-gray-50 rounded"
              aria-label="Clear all selections"
            >
              Clear All
            </button>
          </div>

          {/* Collections list */}
          <div className="max-h-64 overflow-y-auto" role="group" aria-label="Collection list">
            {filteredCollections.length === 0 ? (
              <div className="px-3 py-4 text-sm text-gray-500 text-center">
                {searchTerm
                  ? 'No collections found'
                  : readyCollections.length === 0
                  ? 'No collections with indexed vectors'
                  : 'No collections available'}
              </div>
            ) : (
              filteredCollections.map((collection) => (
                <label
                  key={collection.id}
                  className="flex items-center px-3 py-2 hover:bg-gray-50 cursor-pointer"
                  role="option"
                  aria-selected={selectedCollections.includes(collection.id)}
                >
                  <input
                    type="checkbox"
                    checked={selectedCollections.includes(collection.id)}
                    onChange={() => handleToggle(collection.id)}
                    className="mr-3 text-blue-600 rounded focus:ring-blue-500"
                    aria-label={`Select ${collection.name}`}
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium">{collection.name}</div>
                    <div className="text-xs text-gray-500">
                      {collection.document_count} documents • {collection.vector_count} vectors
                      {collection.embedding_model && (
                        <span className="ml-2">• {collection.embedding_model.split('/').pop()}</span>
                      )}
                    </div>
                  </div>
                  {selectedCollections.includes(collection.id) && (
                    <Check className="w-4 h-4 text-blue-600 ml-2" />
                  )}
                </label>
              ))
            )}
          </div>

          {/* Info about non-ready collections */}
          {collections.some((col) => col.status !== 'ready' || col.vector_count === 0) && (
            <div className="px-3 py-2 text-xs text-gray-500 bg-gray-50 border-t border-gray-200" role="note">
              Only showing collections that are ready with indexed vectors
            </div>
          )}
        </div>
      )}
    </div>
  );
};