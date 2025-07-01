# Document Embedding Frontend (SvelteKit)

This is the modern SvelteKit-based frontend for the Document Embedding System, replacing the vanilla JavaScript implementation.

## Features

- **Component-based architecture** using Svelte
- **Type safety** with TypeScript
- **Modern routing** with SvelteKit
- **Reactive state management** using Svelte stores
- **Real-time updates** via WebSocket connections
- **Optimized builds** with Vite

## Development

### Prerequisites

- Node.js 18+ 
- npm or yarn

### Setup

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Or using Make from project root
make frontend-dev
```

The development server will start at http://localhost:5173 with hot module replacement enabled.

### Building for Production

```bash
# Build the frontend
npm run build

# Or using Make from project root
make frontend-build
```

The build output will be in the `build/` directory and will be automatically copied to `webui/static/` by the Make command.

## Project Structure

```
src/
├── lib/
│   ├── stores/          # Svelte stores for state management
│   │   ├── auth.ts      # Authentication state
│   │   ├── jobs.ts      # Jobs state
│   │   └── websocket.ts # WebSocket connections
│   ├── components/      # Reusable Svelte components
│   │   ├── CreateJobForm.svelte
│   │   ├── JobList.svelte
│   │   ├── JobCard.svelte
│   │   ├── SearchInterface.svelte
│   │   ├── DocumentViewer.svelte
│   │   └── ui/          # Generic UI components
│   └── api/            # API client utilities
│       └── client.ts
├── routes/             # SvelteKit routes
│   ├── +layout.svelte  # Root layout with auth
│   ├── +page.svelte    # Main dashboard
│   ├── login/          # Login page
│   └── settings/       # Settings page
└── app.css            # Global styles with Tailwind
```

## Key Components

### Authentication (`lib/stores/auth.ts`)
- JWT token-based authentication
- Automatic token refresh
- Protected routes with auth guards

### Job Management
- Real-time job progress via WebSocket
- Create, monitor, and manage embedding jobs
- Live metrics and status updates

### Search Interface
- Vector and hybrid search modes
- Result grouping and highlighting
- Document preview integration

### Document Viewer
- Multi-format support (PDF, DOCX, TXT, MD, HTML, PPTX, EML)
- Search term highlighting
- Lazy loading of format-specific libraries

## Migration from Vanilla JS

The new SvelteKit frontend maintains feature parity with the original implementation while providing:

1. **Better code organization** - Components instead of monolithic files
2. **Type safety** - Full TypeScript support
3. **Improved performance** - Optimized builds and code splitting
4. **Better developer experience** - Hot reload, better debugging
5. **Modern tooling** - Vite, PostCSS, ESLint, Prettier

## Configuration

### API Proxy

In development, the Vite server proxies API requests to the backend:

```js
// vite.config.ts
proxy: {
  '/api': {
    target: 'http://localhost:8080',
    changeOrigin: true
  },
  '/ws': {
    target: 'ws://localhost:8080',
    ws: true,
    changeOrigin: true
  }
}
```

### Environment Variables

Create a `.env` file for environment-specific configuration:

```env
PUBLIC_API_URL=http://localhost:8080
```

## Deployment

The frontend is built as a static SPA and served by the FastAPI backend. The build process:

1. Compiles TypeScript and Svelte components
2. Bundles and optimizes with Vite
3. Outputs static files to `build/`
4. Copies to `webui/static/` for serving

## Troubleshooting

### Common Issues

1. **WebSocket connection failed**
   - Ensure the backend is running on port 8080
   - Check proxy configuration in `vite.config.ts`

2. **Authentication redirect loop**
   - Clear localStorage
   - Ensure backend auth endpoints are working

3. **Build failures**
   - Delete `node_modules` and `.svelte-kit`
   - Run `npm install` again

### Debug Mode

Enable debug logging in the browser console:
```js
localStorage.setItem('debug', 'true')
```