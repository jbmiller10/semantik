# Frontend Components Architecture

> **Location:** `apps/webui-react/src/components/`

## Overview

React 19 component library with TypeScript, styled with TailwindCSS. Components follow functional patterns with hooks for state and effects.

## Component Hierarchy

```
App
├── ErrorBoundary
│   └── QueryClientProvider
│       └── Router
│           ├── LoginPage
│           └── ProtectedRoute
│               └── Layout
│                   ├── Header (tabs)
│                   ├── Toast
│                   ├── DocumentViewerModal
│                   ├── CollectionDetailsModal
│                   └── Outlet
│                       └── HomePage
│                           ├── CollectionsDashboard
│                           │   ├── CollectionCard[]
│                           │   └── CreateCollectionModal
│                           ├── SearchInterface
│                           │   ├── SearchForm
│                           │   └── SearchResults
│                           └── ActiveOperationsTab
```

## Core Components

### Layout.tsx
Main application shell with navigation and tabs.

```typescript
function Layout() {
  const { activeTab, setActiveTab } = useUIStore();

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white shadow">
        {/* Navigation tabs */}
      </header>
      <main className="container mx-auto py-6">
        <Outlet />
      </main>
      <Toast />
    </div>
  );
}
```

### CollectionsDashboard.tsx
Grid display of collections with search and filtering.

**Features:**
- Debounced search input
- Status filter dropdown
- Responsive grid (1/2/3 columns)
- Empty state with CTA

```typescript
function CollectionsDashboard() {
  const { data: collections, isLoading } = useCollections();
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState<CollectionStatus | 'all'>('all');

  const filtered = useMemo(() =>
    collections?.filter(c =>
      c.name.toLowerCase().includes(searchQuery.toLowerCase()) &&
      (statusFilter === 'all' || c.status === statusFilter)
    ), [collections, searchQuery, statusFilter]
  );

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {filtered?.map(collection => (
        <CollectionCard key={collection.id} collection={collection} />
      ))}
    </div>
  );
}
```

### CollectionCard.tsx
Individual collection card with status and actions.

**Visual Indicators:**
- Border color by status (blue=processing, green=ready, red=error)
- Progress bar for active operations
- Document/vector counts

### SearchInterface.tsx
Container for search functionality.

```typescript
function SearchInterface() {
  const { data: collections } = useCollections();

  return (
    <div className="space-y-6">
      <SearchForm collections={collections || []} />
      <SearchResults />
    </div>
  );
}
```

### SearchForm.tsx
Complex search input with validation and options.

**Features:**
- Query input with real-time validation
- Collection multi-select
- Search type selector (semantic/hybrid/question/code)
- Hybrid config (alpha slider, fusion mode)
- Advanced options drawer
- Cancel button during search

```typescript
function SearchForm({ collections }: Props) {
  const { searchParams, validateAndUpdateSearchParams, hasValidationErrors } = useSearchStore();
  const [abortController, setAbortController] = useState<AbortController | null>(null);

  const handleSearch = async () => {
    if (hasValidationErrors()) return;

    const controller = new AbortController();
    setAbortController(controller);

    try {
      await searchV2Api.search(searchParams, controller.signal);
    } catch (e) {
      if (e.name !== 'AbortError') throw e;
    }
  };

  return (
    <form onSubmit={handleSearch}>
      <input
        value={searchParams.query}
        onChange={e => validateAndUpdateSearchParams({ query: e.target.value })}
      />
      {/* Collection selector, search type, options */}
      <button type="submit">Search</button>
    </form>
  );
}
```

### SearchResults.tsx
Hierarchical display of search results.

**Structure:**
- Grouped by Collection → Document → Chunk
- Expandable sections
- Score display with color coding
- Reranking badges

### CreateCollectionModal.tsx
Full-featured collection creation (largest component ~760 lines).

**Sections:**
1. Collection metadata (name, description)
2. Initial data source (optional connector)
3. Embedding model selection
4. Chunking strategy
5. Sync configuration
6. Advanced settings

**Multi-step Process:**
1. Validate form
2. Create collection via API
3. Wait for INDEX operation if source provided
4. Add data source
5. Show progress via WebSocket

### Connector Components

**ConnectorTypeSelector.tsx**
Card-based picker for data source type.

```typescript
const connectorIcons = {
  directory: Folder,
  git: GitBranch,
  imap: Mail,
};

function ConnectorTypeSelector({ catalog, selectedType, onSelect }: Props) {
  return (
    <div className="grid grid-cols-3 gap-4">
      {Object.entries(catalog).map(([type, def]) => (
        <button
          key={type}
          onClick={() => onSelect(type)}
          className={selectedType === type ? 'ring-2 ring-blue-500' : ''}
        >
          <Icon icon={connectorIcons[type]} />
          <span>{def.name}</span>
        </button>
      ))}
    </div>
  );
}
```

**ConnectorForm.tsx**
Dynamic field rendering based on connector definition.

```typescript
function ConnectorForm({ catalog, connectorType, values, onValuesChange }: Props) {
  const definition = catalog[connectorType];

  return (
    <div className="space-y-4">
      {definition.fields
        .filter(field => shouldShowField(field, values))
        .map(field => (
          <DynamicField
            key={field.name}
            field={field}
            value={values[field.name]}
            onChange={val => onValuesChange({ ...values, [field.name]: val })}
          />
        ))}
    </div>
  );
}
```

### ActiveOperationsTab.tsx
Real-time operation monitoring.

**Features:**
- WebSocket updates via `useOperationsSocket`
- Progress bars with ETA
- Status badges (processing/pending/completed/failed)
- Collection navigation

## Component Patterns

### Modal Pattern
```typescript
const [showModal, setShowModal] = useState(false);

return (
  <>
    <button onClick={() => setShowModal(true)}>Open</button>
    {showModal && (
      <Modal onClose={() => setShowModal(false)} onSuccess={handleSuccess} />
    )}
  </>
);
```

### Form Validation Pattern
```typescript
const [errors, setErrors] = useState<Record<string, string>>({});

const validateForm = (): boolean => {
  const newErrors: Record<string, string> = {};
  if (!formData.name.trim()) newErrors.name = 'Required';
  setErrors(newErrors);
  return Object.keys(newErrors).length === 0;
};

const handleSubmit = async (e: FormEvent) => {
  e.preventDefault();
  if (!validateForm()) return;
  // Submit
};
```

### Loading/Error Pattern
```typescript
const { data, isLoading, error } = useQuery(...);

if (isLoading) return <LoadingSpinner />;
if (error) return <ErrorMessage error={error} />;
if (!data?.length) return <EmptyState />;
return <DataDisplay data={data} />;
```

## Styling Conventions

**TailwindCSS Utilities:**
```typescript
// Button variants
const buttonPrimary = "px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700";
const buttonSecondary = "px-4 py-2 bg-white border border-gray-300 rounded-md hover:bg-gray-50";

// Card
const card = "bg-white rounded-lg border border-gray-200 shadow-sm p-6";

// Modal overlay
const overlay = "fixed inset-0 bg-gray-500 bg-opacity-75 flex items-center justify-center z-50";
```

**Responsive Grid:**
```typescript
className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6"
```

## Extension Points

### Adding a New Page
1. Create component in `pages/`
2. Add route in `App.tsx`
3. Add navigation link in `Layout.tsx`
4. Wrap with ErrorBoundary

### Adding a New Modal
1. Create modal component
2. Add state for visibility (local or Zustand)
3. Wire up trigger and close handlers
4. Add success/error toast notifications

### Adding a New Connector
1. Add icon to `connectorIcons` in ConnectorTypeSelector
2. Add to `displayOrder` array
3. Add short description in `getShortDescription()`
4. Add preview handler if connector supports it
