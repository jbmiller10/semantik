import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@/tests/utils/test-utils';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { server } from '@/tests/mocks/server';
import AvailablePluginsTab from '../AvailablePluginsTab';
import type { AvailablePlugin, PluginType, AvailablePluginsListResponse } from '@/types/plugin';

// Mock available plugins data
const mockAvailablePlugins: AvailablePlugin[] = [
  {
    id: 'openai-embeddings',
    type: 'embedding' as PluginType,
    name: 'OpenAI Embeddings',
    description: 'text-embedding-3-small/large via OpenAI API',
    author: 'semantik',
    repository: 'https://github.com/semantik-plugins/openai-embeddings',
    pypi: 'semantik-plugin-openai',
    verified: true,
    min_semantik_version: '0.7.5',
    tags: ['api', 'openai', 'cloud'],
    is_compatible: true,
    compatibility_message: null,
    is_installed: false,
    pending_restart: false,
    install_command: 'pip install semantik-plugin-openai',
  },
  {
    id: 'cohere-reranker',
    type: 'reranker' as PluginType,
    name: 'Cohere Reranker',
    description: 'rerank-english-v3, rerank-multilingual-v3',
    author: 'semantik',
    repository: 'https://github.com/semantik-plugins/cohere-reranker',
    pypi: 'semantik-plugin-cohere-reranker',
    verified: true,
    min_semantik_version: '0.7.5',
    tags: ['api', 'cohere', 'reranking'],
    is_compatible: true,
    compatibility_message: null,
    is_installed: false,
    pending_restart: false,
    install_command: 'pip install semantik-plugin-cohere-reranker',
  },
  {
    id: 'community-extractor',
    type: 'extractor' as PluginType,
    name: 'Community Extractor',
    description: 'A community-built extractor',
    author: 'community-dev',
    repository: 'https://github.com/community/extractor',
    pypi: 'semantik-community-extractor',
    verified: false,
    min_semantik_version: null,
    tags: ['community'],
    is_compatible: true,
    compatibility_message: null,
    is_installed: false,
    pending_restart: false,
    install_command: 'pip install semantik-community-extractor',
  },
];

const mockResponse: AvailablePluginsListResponse = {
  plugins: mockAvailablePlugins,
  registry_version: '1.0',
  last_updated: '2026-01-01T00:00:00Z',
  registry_source: 'bundled',
  semantik_version: '0.7.7',
};

describe('AvailablePluginsTab', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Setup default handlers for available plugins
    server.use(
      http.get('/api/v2/plugins/available', () => {
        return HttpResponse.json(mockResponse);
      }),
      http.post('/api/v2/plugins/available/refresh', () => {
        return HttpResponse.json(mockResponse);
      })
    );
  });

  afterEach(() => {
    cleanup();
    server.resetHandlers();
  });

  describe('loading state', () => {
    it('shows loading spinner initially', () => {
      render(<AvailablePluginsTab />);

      expect(screen.getByText('Loading available plugins...')).toBeInTheDocument();
    });
  });

  describe('successful data loading', () => {
    it('displays plugins after loading', async () => {
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      expect(screen.getByText('Cohere Reranker')).toBeInTheDocument();
      expect(screen.getByText('Community Extractor')).toBeInTheDocument();
    });

    it('displays registry metadata', async () => {
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('Registry v1.0')).toBeInTheDocument();
      });

      expect(screen.getByText('Semantik v0.7.7')).toBeInTheDocument();
      expect(screen.getByText('(bundled)')).toBeInTheDocument();
    });

    it('groups plugins by type', async () => {
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      // Check for group headers - use getAllByText since type may appear in dropdown too
      const embeddingHeaders = screen.getAllByText(/Embedding Providers/);
      expect(embeddingHeaders.length).toBeGreaterThan(0);

      const rerankerHeaders = screen.getAllByText(/Rerankers/);
      expect(rerankerHeaders.length).toBeGreaterThan(0);

      const extractorHeaders = screen.getAllByText(/Extractors/);
      expect(extractorHeaders.length).toBeGreaterThan(0);
    });
  });

  describe('search functionality', () => {
    it('filters plugins by search query', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search plugins...');
      await user.type(searchInput, 'cohere');

      expect(screen.queryByText('OpenAI Embeddings')).not.toBeInTheDocument();
      expect(screen.getByText('Cohere Reranker')).toBeInTheDocument();
    });

    it('filters by author', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search plugins...');
      await user.type(searchInput, 'community-dev');

      expect(screen.queryByText('OpenAI Embeddings')).not.toBeInTheDocument();
      expect(screen.getByText('Community Extractor')).toBeInTheDocument();
    });

    it('filters by tag', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search plugins...');
      await user.type(searchInput, 'reranking');

      expect(screen.queryByText('OpenAI Embeddings')).not.toBeInTheDocument();
      expect(screen.getByText('Cohere Reranker')).toBeInTheDocument();
    });

    it('shows empty state when no results match', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search plugins...');
      await user.type(searchInput, 'nonexistent');

      expect(screen.getByText('No plugins found')).toBeInTheDocument();
      expect(screen.getByText('Try a different search term.')).toBeInTheDocument();
    });
  });

  describe('type filter', () => {
    it('filters plugins by type when selecting from dropdown', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const typeSelect = screen.getByRole('combobox');
      await user.selectOptions(typeSelect, 'embedding');

      expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      expect(screen.queryByText('Cohere Reranker')).not.toBeInTheDocument();
      expect(screen.queryByText('Community Extractor')).not.toBeInTheDocument();
    });

    it('shows all types when "All Types" is selected', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const typeSelect = screen.getByRole('combobox');
      await user.selectOptions(typeSelect, 'embedding');
      await user.selectOptions(typeSelect, 'all');

      expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      expect(screen.getByText('Cohere Reranker')).toBeInTheDocument();
    });
  });

  describe('verified only filter', () => {
    it('filters to verified plugins only when checkbox is checked', async () => {
      const user = userEvent.setup();
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const verifiedCheckbox = screen.getByRole('checkbox', {
        name: /verified only/i,
      });
      await user.click(verifiedCheckbox);

      expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      expect(screen.getByText('Cohere Reranker')).toBeInTheDocument();
      expect(screen.queryByText('Community Extractor')).not.toBeInTheDocument();
    });
  });

  describe('refresh button', () => {
    it('calls refresh API when clicked', async () => {
      const user = userEvent.setup();
      let refreshCalled = false;

      server.use(
        http.post('/api/v2/plugins/available/refresh', () => {
          refreshCalled = true;
          return HttpResponse.json(mockResponse);
        })
      );

      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      await user.click(refreshButton);

      await waitFor(() => {
        expect(refreshCalled).toBe(true);
      });
    });
  });

  describe('restart banner', () => {
    it('shows restart banner when a plugin has pending_restart', async () => {
      const pluginsWithPending = mockAvailablePlugins.map((p, i) => ({
        ...p,
        pending_restart: i === 0, // First plugin has pending restart
      }));

      server.use(
        http.get('/api/v2/plugins/available', () => {
          return HttpResponse.json({
            ...mockResponse,
            plugins: pluginsWithPending,
          });
        })
      );

      render(<AvailablePluginsTab />);

      // Wait for data to load
      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      // Check for the restart banner specifically (in the orange banner div)
      expect(
        screen.getByText(
          /One or more plugins have been installed or modified/
        )
      ).toBeInTheDocument();
      expect(
        screen.getByText('docker compose restart webui worker vecpipe')
      ).toBeInTheDocument();
    });

    it('does not show restart banner when no plugins have pending_restart', async () => {
      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('OpenAI Embeddings')).toBeInTheDocument();
      });

      expect(screen.queryByText('Restart Required')).not.toBeInTheDocument();
    });
  });

  describe('error state', () => {
    it('displays error message on API failure', async () => {
      server.use(
        http.get('/api/v2/plugins/available', () => {
          return HttpResponse.json(
            { detail: 'Network error' },
            { status: 500 }
          );
        })
      );

      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(
          screen.getByText('Failed to load plugin registry')
        ).toBeInTheDocument();
      });
    });

    it('shows retry button on error', async () => {
      server.use(
        http.get('/api/v2/plugins/available', () => {
          return HttpResponse.json(
            { detail: 'Network error' },
            { status: 500 }
          );
        })
      );

      render(<AvailablePluginsTab />);

      await waitFor(() => {
        expect(screen.getByText('Try again')).toBeInTheDocument();
      });
    });
  });
});
