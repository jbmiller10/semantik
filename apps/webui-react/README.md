# Semantik WebUI React Frontend

This is the React-based frontend for the Semantik Document Embedding System, providing a modern user interface for managing embedding jobs and searching documents.

## Architecture Overview

The frontend is built with:
- **React 19** with TypeScript for type safety
- **Vite** for fast development and optimized builds
- **Tailwind CSS** for utility-first styling
- **Zustand** for state management
- **React Query** for server state management
- **Axios** for API communication
- **React Router** for client-side routing

## Project Structure

```
src/
├── components/         # Reusable UI components
│   ├── CreateJobForm.tsx
│   ├── DocumentViewer.tsx
│   ├── JobCard.tsx
│   ├── JobList.tsx
│   ├── SearchInterface.tsx
│   └── ...
├── pages/             # Route-based page components
│   ├── HomePage.tsx
│   ├── LoginPage.tsx
│   ├── SettingsPage.tsx
│   └── VerificationPage.tsx
├── stores/            # Zustand state stores
│   ├── authStore.ts
│   ├── jobsStore.ts
│   ├── searchStore.ts
│   └── uiStore.ts
├── services/          # API service layer
│   └── api.ts
├── hooks/             # Custom React hooks
│   ├── useWebSocket.ts
│   ├── useJobProgress.ts
│   └── ...
├── utils/             # Utility functions
└── types/             # TypeScript type definitions
```

## Development

### Prerequisites

- Node.js 18+ and npm
- The backend services running (Search API and WebUI)

### Getting Started

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview
```

### Environment Configuration

The frontend dynamically detects the backend URL based on the current location. No environment variables are required for basic development.

## Key Features

### 1. Job Management
- Create embedding jobs with directory selection
- Real-time progress tracking via WebSocket
- Job status monitoring and management
- Support for multiple embedding models and quantization options

### 2. Search Interface
- Vector similarity search
- Hybrid search (vector + keyword)
- Search result visualization
- Document preview with multiple format support, including inline PDF rendering via pdf.js with loading/error fallback
- GPU memory guidance when reranking exceeds resources, with quick model switching or reranker disable flows

### 3. Authentication
- JWT-based authentication
- Protected routes
- Automatic token refresh
- Logout on 401 responses

### 4. Real-time Updates
- WebSocket connections for job progress
- Directory scanning progress
- Automatic reconnection on disconnect

## State Management

The application uses Zustand for global state management with the following stores:

- **authStore**: User authentication state
- **jobsStore**: Embedding jobs and their status
- **searchStore**: Search results and parameters
- **uiStore**: UI state (modals, toasts, etc.)

## API Integration

All API calls are centralized in `services/api.ts` with:
- Axios interceptors for authentication
- Automatic token injection
- Error handling and retry logic
- Request/response logging in development
