# Extension Cookbook: UI Features

<!-- doc-lint-ignore-file -->

> **Audience:** Software architects implementing new UI functionality in Semantik
> **Prerequisites:** Understanding of React 19, TypeScript, Zustand, TanStack Query

---

> [!NOTE]
> **Example Paths:** The file paths in this cookbook (e.g., `apps/webui-react/src/pages/AnalyticsPage.tsx`) are **illustrative examples** demonstrating recommended patterns and naming conventions. They do not reference existing files in the codebase. When implementing features, adapt these patterns to the actual project structure:
>
> | Cookbook Example | Actual Project Pattern |
> |------------------|------------------------|
> | `packages/webui/api/routers/*.py` | `packages/webui/api/v2/*.py` |
> | `packages/shared/database/models/*.py` | `packages/shared/database/models.py` |
> | `packages/webui/repositories/*.py` | `packages/shared/database/repositories/*.py` |
> | `apps/webui-react/src/components/modals/*.tsx` | `apps/webui-react/src/components/*.tsx` |
> | `apps/webui-react/src/components/collections/*.tsx` | `apps/webui-react/src/components/*.tsx` |
>
> Always check existing files for current conventions before creating new ones.

---

## Table of Contents

1. [Add a New Page](#1-add-a-new-page)
2. [Add a New API Endpoint (Full Stack)](#2-add-a-new-api-endpoint-full-stack)
3. [Add a Modal Dialog](#3-add-a-modal-dialog)
4. [Add Real-Time Updates](#4-add-real-time-updates)
5. [Add a New Collection Feature](#5-add-a-new-collection-feature)
6. [Add Form with Validation](#6-add-form-with-validation)
7. [Add Data Table with Filtering](#7-add-data-table-with-filtering)

---

## 1. Add a New Page

### Overview

Pages in Semantik are React components that represent distinct views in the application. This guide shows how to add a new page (example: Analytics Dashboard).

### Step 1: Create the Page Component

**Example file:** `apps/webui-react/src/pages/AnalyticsPage.tsx`

```typescript
import { Suspense } from 'react';
import { useSearchParams } from 'react-router-dom';
import { Layout } from '@/components/Layout';
import { PageHeader } from '@/components/ui/PageHeader';
import { AnalyticsDashboard } from '@/components/analytics/AnalyticsDashboard';
import { DateRangePicker } from '@/components/ui/DateRangePicker';
import { Skeleton } from '@/components/ui/skeleton';

export function AnalyticsPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const dateRange = {
    from: searchParams.get('from') || getDefaultStartDate(),
    to: searchParams.get('to') || new Date().toISOString()
  };

  const handleDateChange = (range: DateRange) => {
    setSearchParams({
      from: range.from.toISOString(),
      to: range.to.toISOString()
    });
  };

  return (
    <Layout>
      <PageHeader
        title="Analytics"
        description="Search usage and performance metrics"
        actions={
          <DateRangePicker
            value={dateRange}
            onChange={handleDateChange}
          />
        }
      />

      <Suspense fallback={<AnalyticsSkeleton />}>
        <AnalyticsDashboard dateRange={dateRange} />
      </Suspense>
    </Layout>
  );
}

function AnalyticsSkeleton() {
  return (
    <div className="grid grid-cols-4 gap-4">
      {[1, 2, 3, 4].map(i => (
        <Skeleton key={i} className="h-32" />
      ))}
      <Skeleton className="col-span-4 h-64" />
    </div>
  );
}
```

### Step 2: Add Route Configuration

**Existing file:** `apps/webui-react/src/App.tsx`

```typescript
import { AnalyticsPage } from '@/pages/AnalyticsPage';

const routes = [
  // ... existing routes
  {
    path: '/analytics',
    element: <ProtectedRoute><AnalyticsPage /></ProtectedRoute>,
    // Optional: require specific permission
    meta: { requiredPermission: 'view_analytics' }
  }
];
```

### Step 3: Add Navigation Link

**Existing file:** `apps/webui-react/src/components/Layout.tsx` *(add navigation items)*

```typescript
import { BarChart3 } from 'lucide-react';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard', icon: Home },
  { path: '/collections', label: 'Collections', icon: Database },
  { path: '/search', label: 'Search', icon: Search },
  { path: '/analytics', label: 'Analytics', icon: BarChart3 },  // Add
  { path: '/settings', label: 'Settings', icon: Settings },
];
```

### Step 4: Create Supporting Components

**Example file:** `apps/webui-react/src/components/AnalyticsDashboard.tsx`

```typescript
import { useAnalytics } from '@/hooks/useAnalytics';
import { MetricCard } from './MetricCard';
import { SearchTrendsChart } from './SearchTrendsChart';
import { PopularQueriesTable } from './PopularQueriesTable';

interface AnalyticsDashboardProps {
  dateRange: { from: string; to: string };
}

export function AnalyticsDashboard({ dateRange }: AnalyticsDashboardProps) {
  const { data, isLoading, error } = useAnalytics(dateRange);

  if (error) {
    return <ErrorState error={error} />;
  }

  return (
    <div className="space-y-6">
      {/* Summary Metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard
          title="Total Searches"
          value={data.totalSearches}
          trend={data.searchTrend}
          icon={Search}
        />
        <MetricCard
          title="Unique Users"
          value={data.uniqueUsers}
          trend={data.userTrend}
          icon={Users}
        />
        <MetricCard
          title="Avg. Response Time"
          value={`${data.avgResponseTime}ms`}
          trend={data.responseTrend}
          icon={Clock}
        />
        <MetricCard
          title="Click-through Rate"
          value={`${data.ctr}%`}
          trend={data.ctrTrend}
          icon={MousePointerClick}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-3 gap-4">
        <div className="col-span-2">
          <Card>
            <CardHeader>
              <CardTitle>Search Volume Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <SearchTrendsChart data={data.dailySearches} />
            </CardContent>
          </Card>
        </div>

        <Card>
          <CardHeader>
            <CardTitle>Popular Queries</CardTitle>
          </CardHeader>
          <CardContent>
            <PopularQueriesTable queries={data.popularQueries} />
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
```

### Step 5: Create Data Hook

**Example file:** `apps/webui-react/src/hooks/useAnalytics.ts`

```typescript
import { useQuery } from '@tanstack/react-query';
import { apiClient } from '@/services/api';

interface AnalyticsData {
  totalSearches: number;
  uniqueUsers: number;
  avgResponseTime: number;
  ctr: number;
  // ... trends
  dailySearches: { date: string; count: number }[];
  popularQueries: { query: string; count: number }[];
}

export function useAnalytics(dateRange: { from: string; to: string }) {
  return useQuery({
    queryKey: ['analytics', dateRange],
    queryFn: () => apiClient.get<AnalyticsData>('/api/analytics', {
      params: dateRange
    }),
    staleTime: 5 * 60 * 1000,  // 5 minutes
    refetchOnWindowFocus: false
  });
}
```

### Checklist for New Pages

- [ ] Create page component with Layout wrapper
- [ ] Add route in App.tsx
- [ ] Add navigation link in Sidebar
- [ ] Create supporting components
- [ ] Create data hooks with React Query
- [ ] Add loading skeletons
- [ ] Add error boundary/state
- [ ] Add page-level tests
- [ ] Update e2e tests

---

## 2. Add a New API Endpoint (Full Stack)

### Overview

This guide covers adding a new API endpoint from database to frontend, following the three-layer architecture.

### Example: User Preferences Endpoint

### Step 1: Database Model

**Existing file:** `packages/shared/database/models.py` *(add new model class)*

```python
from sqlalchemy import String, JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .base import Base, TimestampMixin

class UserPreferences(Base, TimestampMixin):
    """User preferences storage."""

    __tablename__ = "user_preferences"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True
    )

    # Preferences as JSON
    theme: Mapped[str] = mapped_column(String(20), default="system")
    default_search_mode: Mapped[str] = mapped_column(String(20), default="semantic")
    results_per_page: Mapped[int] = mapped_column(default=10)
    notifications_enabled: Mapped[bool] = mapped_column(default=True)
    custom_settings: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="preferences")
```

### Step 2: Alembic Migration

**Example file:** `alembic/versions/<revision>_add_user_preferences.py`

```python
"""add user preferences table

Revision ID: xxx
Revises: yyy
Create Date: 2025-01-15 10:00:00
"""

from alembic import op
import sqlalchemy as sa

revision = 'xxx'
down_revision = 'yyy'

def upgrade():
    op.create_table(
        'user_preferences',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True),
        sa.Column('theme', sa.String(20), default='system'),
        sa.Column('default_search_mode', sa.String(20), default='semantic'),
        sa.Column('results_per_page', sa.Integer(), default=10),
        sa.Column('notifications_enabled', sa.Boolean(), default=True),
        sa.Column('custom_settings', sa.JSON(), default={}),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), onupdate=sa.func.now())
    )

def downgrade():
    op.drop_table('user_preferences')
```

### Step 3: Repository

**Example file:** `packages/shared/database/repositories/user_preferences_repository.py`

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from packages.shared.database.models import UserPreferences

class UserPreferencesRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_by_user_id(self, user_id: int) -> UserPreferences | None:
        result = await self.db.execute(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        )
        return result.scalar_one_or_none()

    async def create(self, user_id: int, **kwargs) -> UserPreferences:
        prefs = UserPreferences(user_id=user_id, **kwargs)
        self.db.add(prefs)
        await self.db.commit()
        await self.db.refresh(prefs)
        return prefs

    async def update(
        self,
        user_id: int,
        **updates
    ) -> UserPreferences:
        prefs = await self.get_by_user_id(user_id)
        if not prefs:
            # Create with defaults if doesn't exist
            return await self.create(user_id, **updates)

        for key, value in updates.items():
            if hasattr(prefs, key):
                setattr(prefs, key, value)

        await self.db.commit()
        await self.db.refresh(prefs)
        return prefs
```

### Step 4: Service

**Example file:** `packages/webui/services/preferences_service.py`

```python
from sqlalchemy.ext.asyncio import AsyncSession
from packages.webui.repositories import UserPreferencesRepository
from packages.shared.contracts.preferences import (
    UserPreferencesResponse,
    UpdatePreferencesRequest
)

class PreferencesService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.repo = UserPreferencesRepository(db)

    async def get_preferences(self, user_id: int) -> UserPreferencesResponse:
        """Get user preferences, returning defaults if none exist."""
        prefs = await self.repo.get_by_user_id(user_id)

        if not prefs:
            # Return defaults
            return UserPreferencesResponse(
                theme="system",
                default_search_mode="semantic",
                results_per_page=10,
                notifications_enabled=True,
                custom_settings={}
            )

        return UserPreferencesResponse.from_orm(prefs)

    async def update_preferences(
        self,
        user_id: int,
        request: UpdatePreferencesRequest
    ) -> UserPreferencesResponse:
        """Update user preferences."""
        # Validate search mode
        if request.default_search_mode:
            valid_modes = ["semantic", "keyword", "hybrid", "mmr"]
            if request.default_search_mode not in valid_modes:
                raise ValueError(f"Invalid search mode: {request.default_search_mode}")

        prefs = await self.repo.update(
            user_id=user_id,
            **request.dict(exclude_unset=True)
        )

        return UserPreferencesResponse.from_orm(prefs)
```

### Step 5: API Router

**Example file:** `packages/webui/api/v2/preferences.py`

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from packages.webui.api.dependencies import get_db, get_current_user
from packages.webui.services import PreferencesService
from packages.shared.contracts.preferences import (
    UserPreferencesResponse,
    UpdatePreferencesRequest
)

router = APIRouter(prefix="/preferences", tags=["preferences"])

@router.get("", response_model=UserPreferencesResponse)
async def get_preferences(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's preferences."""
    service = PreferencesService(db)
    return await service.get_preferences(user.id)


@router.patch("", response_model=UserPreferencesResponse)
async def update_preferences(
    request: UpdatePreferencesRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user's preferences."""
    service = PreferencesService(db)
    return await service.update_preferences(user.id, request)
```

### Step 6: Register Router

**Existing file:** `packages/webui/main.py` *(add router import and registration)*

```python
from packages.webui.api.v2 import preferences

app.include_router(preferences.router, prefix="/api/v2")
```

### Step 7: Frontend API Client

**Example file:** `apps/webui-react/src/services/api/preferences.ts`

```typescript
import { apiClient } from './client';

export interface UserPreferences {
  theme: 'light' | 'dark' | 'system';
  defaultSearchMode: string;
  resultsPerPage: number;
  notificationsEnabled: boolean;
  customSettings: Record<string, unknown>;
}

export const preferencesApi = {
  get: () =>
    apiClient.get<UserPreferences>('/api/preferences'),

  update: (updates: Partial<UserPreferences>) =>
    apiClient.patch<UserPreferences>('/api/preferences', updates),
};
```

### Step 8: React Query Hook

**Example file:** `apps/webui-react/src/hooks/usePreferences.ts`

```typescript
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { preferencesApi, UserPreferences } from '@/services/api/preferences';

export function usePreferences() {
  return useQuery({
    queryKey: ['preferences'],
    queryFn: preferencesApi.get,
    staleTime: Infinity,  // Preferences rarely change
  });
}

export function useUpdatePreferences() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: preferencesApi.update,
    onSuccess: (newPrefs) => {
      queryClient.setQueryData(['preferences'], newPrefs);
    },
    // Optimistic update
    onMutate: async (updates) => {
      await queryClient.cancelQueries({ queryKey: ['preferences'] });

      const previous = queryClient.getQueryData<UserPreferences>(['preferences']);

      queryClient.setQueryData(['preferences'], (old: UserPreferences) => ({
        ...old,
        ...updates
      }));

      return { previous };
    },
    onError: (err, variables, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['preferences'], context.previous);
      }
    }
  });
}
```

### Step 9: Use in Component

**Example file:** `apps/webui-react/src/components/PreferencesForm.tsx`

```typescript
import { usePreferences, useUpdatePreferences } from '@/hooks/usePreferences';

export function PreferencesForm() {
  const { data: preferences, isLoading } = usePreferences();
  const { mutate: updatePreferences, isPending } = useUpdatePreferences();

  if (isLoading) return <Skeleton />;

  return (
    <form className="space-y-4">
      <div>
        <Label>Theme</Label>
        <Select
          value={preferences.theme}
          onValueChange={(theme) => updatePreferences({ theme })}
          disabled={isPending}
        >
          <SelectItem value="light">Light</SelectItem>
          <SelectItem value="dark">Dark</SelectItem>
          <SelectItem value="system">System</SelectItem>
        </Select>
      </div>

      <div>
        <Label>Default Search Mode</Label>
        <Select
          value={preferences.defaultSearchMode}
          onValueChange={(mode) => updatePreferences({ defaultSearchMode: mode })}
          disabled={isPending}
        >
          <SelectItem value="semantic">Semantic</SelectItem>
          <SelectItem value="keyword">Keyword</SelectItem>
          <SelectItem value="hybrid">Hybrid</SelectItem>
        </Select>
      </div>

      {/* ... more fields */}
    </form>
  );
}
```

### Checklist for Full-Stack Endpoint

- [ ] Create/update database model
- [ ] Create Alembic migration
- [ ] Implement repository with CRUD operations
- [ ] Implement service with business logic
- [ ] Create API router with endpoints
- [ ] Register router in main app
- [ ] Add request/response contracts
- [ ] Create frontend API client function
- [ ] Create React Query hook
- [ ] Use in component
- [ ] Write backend tests (repository, service, API)
- [ ] Write frontend tests (hook, component)
- [ ] Update OpenAPI documentation

---

## 3. Add a Modal Dialog

### Overview

Modal dialogs are used for focused user interactions. This guide shows the pattern used in Semantik.

### Step 1: Create Modal Component

**Example file:** `apps/webui-react/src/components/ConfirmDeleteModal.tsx`

```typescript
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { AlertTriangle } from 'lucide-react';

interface ConfirmDeleteModalProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  description: string;
  itemName: string;
  onConfirm: () => void;
  isPending?: boolean;
}

export function ConfirmDeleteModal({
  open,
  onOpenChange,
  title,
  description,
  itemName,
  onConfirm,
  isPending = false
}: ConfirmDeleteModalProps) {
  const handleConfirm = () => {
    onConfirm();
    // Note: Don't close here - let parent close on success
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <div className="flex items-center gap-3">
            <div className="p-2 bg-destructive/10 rounded-full">
              <AlertTriangle className="h-5 w-5 text-destructive" />
            </div>
            <DialogTitle>{title}</DialogTitle>
          </div>
          <DialogDescription className="pt-2">
            {description}
          </DialogDescription>
        </DialogHeader>

        <div className="py-4">
          <p className="text-sm">
            You are about to delete{' '}
            <span className="font-semibold">{itemName}</span>.
            This action cannot be undone.
          </p>
        </div>

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleConfirm}
            disabled={isPending}
          >
            {isPending ? 'Deleting...' : 'Delete'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

### Step 2: Create Modal State Hook (Optional Pattern)

**Example file:** `apps/webui-react/src/hooks/useModal.ts`

```typescript
import { useState, useCallback } from 'react';

interface UseModalReturn<T = undefined> {
  isOpen: boolean;
  data: T | undefined;
  open: (data?: T) => void;
  close: () => void;
  setOpen: (open: boolean) => void;
}

export function useModal<T = undefined>(): UseModalReturn<T> {
  const [isOpen, setIsOpen] = useState(false);
  const [data, setData] = useState<T | undefined>(undefined);

  const open = useCallback((newData?: T) => {
    setData(newData);
    setIsOpen(true);
  }, []);

  const close = useCallback(() => {
    setIsOpen(false);
    // Clear data after animation
    setTimeout(() => setData(undefined), 200);
  }, []);

  return { isOpen, data, open, close, setOpen: setIsOpen };
}
```

### Step 3: Use Modal in Parent Component

**Existing file:** `apps/webui-react/src/components/CollectionCard.tsx` *(example usage)*

```typescript
import { useModal } from '@/hooks/useModal';
import { useDeleteCollection } from '@/hooks/useCollections';
import { ConfirmDeleteModal } from '@/components/modals/ConfirmDeleteModal';

export function CollectionCard({ collection }: { collection: Collection }) {
  const deleteModal = useModal();
  const { mutate: deleteCollection, isPending } = useDeleteCollection();

  const handleDelete = () => {
    deleteCollection(collection.id, {
      onSuccess: () => {
        deleteModal.close();
        toast.success('Collection deleted');
      },
      onError: (error) => {
        toast.error(`Failed to delete: ${error.message}`);
      }
    });
  };

  return (
    <>
      <Card>
        {/* Card content */}
        <DropdownMenu>
          <DropdownMenuItem onClick={() => deleteModal.open()}>
            Delete
          </DropdownMenuItem>
        </DropdownMenu>
      </Card>

      <ConfirmDeleteModal
        open={deleteModal.isOpen}
        onOpenChange={deleteModal.setOpen}
        title="Delete Collection"
        description="This will permanently delete the collection and all its documents."
        itemName={collection.name}
        onConfirm={handleDelete}
        isPending={isPending}
      />
    </>
  );
}
```

### Step 4: Complex Modal with Form

**Example file:** `apps/webui-react/src/components/EditCollectionModal.tsx`

```typescript
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const editCollectionSchema = z.object({
  name: z.string().min(1, 'Name is required').max(100),
  description: z.string().max(500).optional(),
});

type EditCollectionForm = z.infer<typeof editCollectionSchema>;

interface EditCollectionModalProps {
  collection: Collection;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function EditCollectionModal({
  collection,
  open,
  onOpenChange
}: EditCollectionModalProps) {
  const { mutate: updateCollection, isPending } = useUpdateCollection();

  const form = useForm<EditCollectionForm>({
    resolver: zodResolver(editCollectionSchema),
    defaultValues: {
      name: collection.name,
      description: collection.description || '',
    },
  });

  const onSubmit = (data: EditCollectionForm) => {
    updateCollection(
      { id: collection.id, ...data },
      {
        onSuccess: () => {
          onOpenChange(false);
          toast.success('Collection updated');
        },
        onError: (error) => {
          form.setError('root', { message: error.message });
        }
      }
    );
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Edit Collection</DialogTitle>
        </DialogHeader>

        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
          <div>
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              {...form.register('name')}
              error={form.formState.errors.name?.message}
            />
          </div>

          <div>
            <Label htmlFor="description">Description</Label>
            <Textarea
              id="description"
              {...form.register('description')}
              error={form.formState.errors.description?.message}
            />
          </div>

          {form.formState.errors.root && (
            <Alert variant="destructive">
              {form.formState.errors.root.message}
            </Alert>
          )}

          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isPending}>
              {isPending ? 'Saving...' : 'Save Changes'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}
```

---

## 4. Add Real-Time Updates

### Overview

Real-time updates use WebSocket connections to push progress and status changes to the UI.

### Step 1: Create WebSocket Hook

**Example file:** `apps/webui-react/src/hooks/useOperationProgress.ts`

```typescript
import { useEffect, useCallback, useRef } from 'react';
import { useQueryClient } from '@tanstack/react-query';

interface OperationProgress {
  operationId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  message?: string;
}

export function useOperationProgress(operationId: string | null) {
  const queryClient = useQueryClient();
  const wsRef = useRef<WebSocket | null>(null);
  const [progress, setProgress] = useState<OperationProgress | null>(null);

  useEffect(() => {
    if (!operationId) return;

    const ws = new WebSocket(
      `${WS_BASE_URL}/ws/operations/${operationId}`
    );

    ws.onopen = () => {
      console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data) as OperationProgress;
      setProgress(data);

      // Update related queries when operation completes
      if (data.status === 'completed') {
        queryClient.invalidateQueries({
          queryKey: ['collections']
        });
        queryClient.invalidateQueries({
          queryKey: ['operations']
        });
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    ws.onclose = () => {
      console.log('WebSocket disconnected');
    };

    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [operationId, queryClient]);

  return progress;
}
```

### Step 2: Create Progress Component

**Example file:** `apps/webui-react/src/components/OperationProgress.tsx`

```typescript
import { useOperationProgress } from '@/hooks/useOperationProgress';
import { Progress } from '@/components/ui/progress';
import { CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface OperationProgressProps {
  operationId: string;
  onComplete?: () => void;
  onError?: (error: string) => void;
}

export function OperationProgress({
  operationId,
  onComplete,
  onError
}: OperationProgressProps) {
  const progress = useOperationProgress(operationId);

  useEffect(() => {
    if (progress?.status === 'completed') {
      onComplete?.();
    } else if (progress?.status === 'failed') {
      onError?.(progress.message || 'Operation failed');
    }
  }, [progress?.status, onComplete, onError]);

  if (!progress) {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Loader2 className="h-4 w-4 animate-spin" />
        Connecting...
      </div>
    );
  }

  const statusIcons = {
    pending: <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />,
    processing: <Loader2 className="h-4 w-4 animate-spin text-primary" />,
    completed: <CheckCircle className="h-4 w-4 text-green-500" />,
    failed: <XCircle className="h-4 w-4 text-destructive" />,
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {statusIcons[progress.status]}
          <span className="text-sm font-medium capitalize">
            {progress.status}
          </span>
        </div>
        <span className="text-sm text-muted-foreground">
          {progress.progress}%
        </span>
      </div>

      <Progress value={progress.progress} />

      {progress.message && (
        <p className="text-sm text-muted-foreground">
          {progress.message}
        </p>
      )}
    </div>
  );
}
```

### Step 3: Use in Modal

**Example file:** `apps/webui-react/src/components/IndexingProgressModal.tsx`

```typescript
export function IndexingProgressModal({
  operationId,
  open,
  onOpenChange
}: Props) {
  const [status, setStatus] = useState<'progress' | 'success' | 'error'>('progress');

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Indexing Documents</DialogTitle>
        </DialogHeader>

        {status === 'progress' && (
          <OperationProgress
            operationId={operationId}
            onComplete={() => setStatus('success')}
            onError={() => setStatus('error')}
          />
        )}

        {status === 'success' && (
          <div className="text-center py-6">
            <CheckCircle className="h-12 w-12 text-green-500 mx-auto mb-4" />
            <p>Indexing completed successfully!</p>
          </div>
        )}

        {status === 'error' && (
          <div className="text-center py-6">
            <XCircle className="h-12 w-12 text-destructive mx-auto mb-4" />
            <p>Indexing failed. Please try again.</p>
          </div>
        )}

        <DialogFooter>
          <Button onClick={() => onOpenChange(false)}>
            {status === 'progress' ? 'Cancel' : 'Close'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
```

---

## 5. Add a New Collection Feature

### Overview

This guide shows how to add a new feature to collections (example: collection tags).

### Step 1: Update Database Model

**Existing file:** `packages/shared/database/models.py` *(update Collection class)*

```python
class Collection(Base, TimestampMixin):
    # ... existing fields

    # Add tags as JSON array
    tags: Mapped[list[str]] = mapped_column(
        ARRAY(String),
        default=list,
        server_default="{}"
    )
```

### Step 2: Update Contracts

**Example file:** `packages/webui/api/v2/schemas.py` *(add/update schema classes)*

```python
class CollectionResponse(BaseModel):
    # ... existing fields
    tags: list[str] = []

class UpdateCollectionRequest(BaseModel):
    name: str | None = None
    description: str | None = None
    tags: list[str] | None = None

    @validator("tags")
    def validate_tags(cls, v):
        if v is not None:
            if len(v) > 10:
                raise ValueError("Maximum 10 tags allowed")
            for tag in v:
                if len(tag) > 30:
                    raise ValueError("Tag must be 30 characters or less")
        return v
```

### Step 3: Update Service

**Existing file:** `packages/webui/services/collection_service.py` *(update service methods)*

```python
class CollectionService:
    async def update(
        self,
        collection_id: str,
        user_id: int,
        request: UpdateCollectionRequest
    ) -> Collection:
        collection = await self.repo.get_by_id(collection_id, user_id)
        if not collection:
            raise ResourceNotFoundError("Collection not found")

        if request.tags is not None:
            collection.tags = request.tags

        # ... other updates

        await self.db.commit()
        return collection

    async def search_by_tag(
        self,
        user_id: int,
        tag: str
    ) -> list[Collection]:
        """Find collections with a specific tag."""
        return await self.repo.find_by_tag(user_id, tag)
```

### Step 4: Frontend Component

**Example file:** `apps/webui-react/src/components/CollectionTags.tsx`

```typescript
import { useState } from 'react';
import { X, Plus } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';

interface CollectionTagsProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  editable?: boolean;
  maxTags?: number;
}

export function CollectionTags({
  tags,
  onChange,
  editable = true,
  maxTags = 10
}: CollectionTagsProps) {
  const [inputValue, setInputValue] = useState('');
  const [isAdding, setIsAdding] = useState(false);

  const addTag = () => {
    const newTag = inputValue.trim().toLowerCase();
    if (newTag && !tags.includes(newTag) && tags.length < maxTags) {
      onChange([...tags, newTag]);
      setInputValue('');
      setIsAdding(false);
    }
  };

  const removeTag = (tagToRemove: string) => {
    onChange(tags.filter(t => t !== tagToRemove));
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      addTag();
    } else if (e.key === 'Escape') {
      setIsAdding(false);
      setInputValue('');
    }
  };

  return (
    <div className="flex flex-wrap gap-2 items-center">
      {tags.map(tag => (
        <Badge key={tag} variant="secondary" className="gap-1">
          {tag}
          {editable && (
            <button
              onClick={() => removeTag(tag)}
              className="ml-1 hover:text-destructive"
            >
              <X className="h-3 w-3" />
            </button>
          )}
        </Badge>
      ))}

      {editable && tags.length < maxTags && (
        isAdding ? (
          <div className="flex gap-1">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleKeyDown}
              onBlur={() => {
                if (!inputValue) setIsAdding(false);
              }}
              placeholder="Tag name"
              className="h-6 w-24 text-xs"
              autoFocus
              maxLength={30}
            />
            <Button size="sm" variant="ghost" onClick={addTag}>
              Add
            </Button>
          </div>
        ) : (
          <Button
            size="sm"
            variant="ghost"
            className="h-6 text-xs"
            onClick={() => setIsAdding(true)}
          >
            <Plus className="h-3 w-3 mr-1" />
            Add Tag
          </Button>
        )
      )}
    </div>
  );
}
```

---

## 6. Add Form with Validation

### Overview

Forms in Semantik use react-hook-form with Zod validation.

### Complete Form Example

**Example file:** `apps/webui-react/src/components/CreateSourceForm.tsx`

```typescript
import { useForm, Controller } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

// Schema definition
const createSourceSchema = z.object({
  type: z.enum(['directory', 'git', 'imap']),
  name: z.string().min(1, 'Name is required').max(100),

  // Directory fields
  path: z.string().optional(),
  recursive: z.boolean().default(true),

  // Git fields
  url: z.string().url().optional(),
  branch: z.string().default('main'),

  // Shared
  fileExtensions: z.array(z.string()).default(['.md', '.txt']),
}).refine((data) => {
  if (data.type === 'directory' && !data.path) {
    return false;
  }
  if (data.type === 'git' && !data.url) {
    return false;
  }
  return true;
}, {
  message: 'Required fields missing for selected source type',
  path: ['type']
});

type CreateSourceFormData = z.infer<typeof createSourceSchema>;

interface CreateSourceFormProps {
  onSubmit: (data: CreateSourceFormData) => void;
  isPending?: boolean;
}

export function CreateSourceForm({
  onSubmit,
  isPending = false
}: CreateSourceFormProps) {
  const form = useForm<CreateSourceFormData>({
    resolver: zodResolver(createSourceSchema),
    defaultValues: {
      type: 'directory',
      recursive: true,
      branch: 'main',
      fileExtensions: ['.md', '.txt'],
    },
    mode: 'onChange',  // Validate on change
  });

  const sourceType = form.watch('type');

  return (
    <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
      {/* Source Type Selection */}
      <div className="space-y-2">
        <Label>Source Type</Label>
        <Controller
          name="type"
          control={form.control}
          render={({ field }) => (
            <RadioGroup
              value={field.value}
              onValueChange={field.onChange}
              className="flex gap-4"
            >
              <RadioGroupItem value="directory" label="Directory" />
              <RadioGroupItem value="git" label="Git Repository" />
              <RadioGroupItem value="imap" label="Email (IMAP)" />
            </RadioGroup>
          )}
        />
        {form.formState.errors.type && (
          <p className="text-sm text-destructive">
            {form.formState.errors.type.message}
          </p>
        )}
      </div>

      {/* Source Name */}
      <div className="space-y-2">
        <Label htmlFor="name">Source Name</Label>
        <Input
          id="name"
          {...form.register('name')}
          placeholder="My Documents"
        />
        {form.formState.errors.name && (
          <p className="text-sm text-destructive">
            {form.formState.errors.name.message}
          </p>
        )}
      </div>

      {/* Conditional Fields */}
      {sourceType === 'directory' && (
        <>
          <div className="space-y-2">
            <Label htmlFor="path">Directory Path</Label>
            <Input
              id="path"
              {...form.register('path')}
              placeholder="/path/to/documents"
            />
            {form.formState.errors.path && (
              <p className="text-sm text-destructive">
                {form.formState.errors.path.message}
              </p>
            )}
          </div>

          <div className="flex items-center gap-2">
            <Controller
              name="recursive"
              control={form.control}
              render={({ field }) => (
                <Checkbox
                  id="recursive"
                  checked={field.value}
                  onCheckedChange={field.onChange}
                />
              )}
            />
            <Label htmlFor="recursive">Include subdirectories</Label>
          </div>
        </>
      )}

      {sourceType === 'git' && (
        <>
          <div className="space-y-2">
            <Label htmlFor="url">Repository URL</Label>
            <Input
              id="url"
              {...form.register('url')}
              placeholder="https://github.com/user/repo.git"
            />
            {form.formState.errors.url && (
              <p className="text-sm text-destructive">
                {form.formState.errors.url.message}
              </p>
            )}
          </div>

          <div className="space-y-2">
            <Label htmlFor="branch">Branch</Label>
            <Input
              id="branch"
              {...form.register('branch')}
              placeholder="main"
            />
          </div>
        </>
      )}

      {/* File Extensions (shared) */}
      <div className="space-y-2">
        <Label>File Extensions</Label>
        <Controller
          name="fileExtensions"
          control={form.control}
          render={({ field }) => (
            <MultiSelect
              options={['.md', '.txt', '.pdf', '.docx', '.html', '.py', '.js']}
              value={field.value}
              onChange={field.onChange}
            />
          )}
        />
      </div>

      {/* Form Actions */}
      <div className="flex justify-end gap-2">
        <Button type="button" variant="outline" onClick={() => form.reset()}>
          Reset
        </Button>
        <Button
          type="submit"
          disabled={isPending || !form.formState.isValid}
        >
          {isPending ? 'Creating...' : 'Create Source'}
        </Button>
      </div>

      {/* Debug (development only) */}
      {process.env.NODE_ENV === 'development' && (
        <pre className="text-xs bg-muted p-2 rounded">
          {JSON.stringify(form.formState.errors, null, 2)}
        </pre>
      )}
    </form>
  );
}
```

---

## 7. Add Data Table with Filtering

### Overview

Data tables in Semantik use TanStack Table with server-side pagination.

**Example file:** `apps/webui-react/src/components/DocumentsTable.tsx`

```typescript
import { useState, useMemo } from 'react';
import {
  useReactTable,
  getCoreRowModel,
  getSortedRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  flexRender,
  ColumnDef,
  SortingState,
  ColumnFiltersState,
} from '@tanstack/react-table';
import { useDocuments } from '@/hooks/useDocuments';

interface Document {
  id: string;
  name: string;
  path: string;
  size: number;
  createdAt: string;
  chunkCount: number;
}

export function DocumentsTable({ collectionId }: { collectionId: string }) {
  const [sorting, setSorting] = useState<SortingState>([]);
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [pagination, setPagination] = useState({
    pageIndex: 0,
    pageSize: 20,
  });

  // Server-side data fetching
  const { data, isLoading } = useDocuments({
    collectionId,
    page: pagination.pageIndex,
    pageSize: pagination.pageSize,
    sortBy: sorting[0]?.id,
    sortOrder: sorting[0]?.desc ? 'desc' : 'asc',
    filters: columnFilters.reduce((acc, f) => ({
      ...acc,
      [f.id]: f.value
    }), {}),
  });

  const columns = useMemo<ColumnDef<Document>[]>(() => [
    {
      accessorKey: 'name',
      header: 'Name',
      cell: ({ row }) => (
        <div className="flex items-center gap-2">
          <FileText className="h-4 w-4 text-muted-foreground" />
          <span className="font-medium">{row.original.name}</span>
        </div>
      ),
    },
    {
      accessorKey: 'path',
      header: 'Path',
      cell: ({ row }) => (
        <span className="text-sm text-muted-foreground truncate max-w-xs">
          {row.original.path}
        </span>
      ),
    },
    {
      accessorKey: 'size',
      header: 'Size',
      cell: ({ row }) => formatBytes(row.original.size),
    },
    {
      accessorKey: 'chunkCount',
      header: 'Chunks',
    },
    {
      accessorKey: 'createdAt',
      header: 'Added',
      cell: ({ row }) => formatDate(row.original.createdAt),
    },
    {
      id: 'actions',
      cell: ({ row }) => (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon">
              <MoreHorizontal className="h-4 w-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem onClick={() => handleView(row.original)}>
              View Details
            </DropdownMenuItem>
            <DropdownMenuItem onClick={() => handleDelete(row.original)}>
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      ),
    },
  ], []);

  const table = useReactTable({
    data: data?.documents ?? [],
    columns,
    pageCount: data?.totalPages ?? -1,
    state: {
      sorting,
      columnFilters,
      pagination,
    },
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
    manualSorting: true,
    manualFiltering: true,
  });

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex gap-4">
        <Input
          placeholder="Filter by name..."
          value={(table.getColumn('name')?.getFilterValue() as string) ?? ''}
          onChange={(e) =>
            table.getColumn('name')?.setFilterValue(e.target.value)
          }
          className="max-w-xs"
        />
      </div>

      {/* Table */}
      <div className="border rounded-md">
        <Table>
          <TableHeader>
            {table.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHead
                    key={header.id}
                    onClick={header.column.getToggleSortingHandler()}
                    className={header.column.getCanSort() ? 'cursor-pointer' : ''}
                  >
                    <div className="flex items-center gap-2">
                      {flexRender(
                        header.column.columnDef.header,
                        header.getContext()
                      )}
                      {header.column.getIsSorted() && (
                        header.column.getIsSorted() === 'desc'
                          ? <ChevronDown className="h-4 w-4" />
                          : <ChevronUp className="h-4 w-4" />
                      )}
                    </div>
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {isLoading ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="text-center">
                  <Loader2 className="h-6 w-6 animate-spin mx-auto" />
                </TableCell>
              </TableRow>
            ) : table.getRowModel().rows.length === 0 ? (
              <TableRow>
                <TableCell colSpan={columns.length} className="text-center">
                  No documents found
                </TableCell>
              </TableRow>
            ) : (
              table.getRowModel().rows.map((row) => (
                <TableRow key={row.id}>
                  {row.getVisibleCells().map((cell) => (
                    <TableCell key={cell.id}>
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext()
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">
          Showing {pagination.pageIndex * pagination.pageSize + 1} to{' '}
          {Math.min(
            (pagination.pageIndex + 1) * pagination.pageSize,
            data?.totalCount ?? 0
          )}{' '}
          of {data?.totalCount ?? 0}
        </p>

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.previousPage()}
            disabled={!table.getCanPreviousPage()}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => table.nextPage()}
            disabled={!table.getCanNextPage()}
          >
            Next
          </Button>
        </div>
      </div>
    </div>
  );
}
```

---

## Quick Reference Checklists

### New Page Checklist
- [ ] Create page component with Layout
- [ ] Add route in App.tsx
- [ ] Add navigation link
- [ ] Create data hooks
- [ ] Add loading states
- [ ] Add error handling
- [ ] Write tests

### New API Endpoint Checklist
- [ ] Database model (if needed)
- [ ] Alembic migration (if needed)
- [ ] Repository methods
- [ ] Service with business logic
- [ ] API router + endpoints
- [ ] Request/Response contracts
- [ ] Frontend API client
- [ ] React Query hook
- [ ] Component integration
- [ ] Tests (backend + frontend)

### New Modal Checklist
- [ ] Modal component with Dialog
- [ ] Form validation (if form)
- [ ] Loading states
- [ ] Error handling
- [ ] Parent integration with useModal hook
- [ ] Tests

### New Real-Time Feature Checklist
- [ ] WebSocket endpoint (backend)
- [ ] Redis stream publishing
- [ ] useWebSocket hook
- [ ] Progress/status component
- [ ] Cleanup on unmount
- [ ] Reconnection logic
- [ ] Tests
