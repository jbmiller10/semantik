import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { AgentChat } from '../AgentChat'

// Mock the hooks
vi.mock('../../../hooks/useAgentConversation', () => ({
  useAgentConversation: vi.fn(),
  useApplyPipeline: vi.fn(),
  useAbandonConversation: vi.fn(),
  useUpdateConversationCache: vi.fn(),
}))

vi.mock('../../../hooks/useAgentStream', () => ({
  useAgentStream: vi.fn(),
}))

import {
  useAgentConversation,
  useApplyPipeline,
  useAbandonConversation,
  useUpdateConversationCache,
} from '../../../hooks/useAgentConversation'
import { useAgentStream } from '../../../hooks/useAgentStream'

// Helper to create a test wrapper with providers
function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>{children}</BrowserRouter>
      </QueryClientProvider>
    )
  }
}

// Cast mocked hooks for easier use
const mockedUseAgentConversation = vi.mocked(useAgentConversation)
const mockedUseAgentStream = vi.mocked(useAgentStream)
const mockedUseApplyPipeline = vi.mocked(useApplyPipeline)
const mockedUseAbandonConversation = vi.mocked(useAbandonConversation)
const mockedUseUpdateConversationCache = vi.mocked(useUpdateConversationCache)

const mockConversation = {
  id: 'conv-test-123',
  status: 'active' as const,
  source_id: 42,
  collection_id: null,
  current_pipeline: {
    embedding_model: 'Qwen/Qwen3-Embedding-0.6B',
    quantization: 'float16',
    chunking_strategy: 'semantic',
    chunking_config: { max_tokens: 512 },
  },
  source_analysis: {
    total_files: 247,
    total_size_bytes: 47185920,
    file_types: { '.pdf': 150, '.txt': 97 },
    sample_files: [],
    warnings: [],
  },
  uncertainties: [],
  messages: [
    {
      role: 'user' as const,
      content: 'Help me set up a pipeline',
      timestamp: '2025-01-20T10:00:00Z',
    },
    {
      role: 'assistant' as const,
      content: 'I can help you configure the best settings for your documents.',
      timestamp: '2025-01-20T10:00:05Z',
    },
  ],
  summary: null,
  created_at: '2025-01-20T10:00:00Z',
  updated_at: '2025-01-20T10:00:05Z',
}

describe('AgentChat', () => {
  const mockOnClose = vi.fn()
  const mockOnApplySuccess = vi.fn()
  const mockSendMessage = vi.fn()
  const mockCancel = vi.fn()
  const mockReset = vi.fn()
  const mockApplyMutate = vi.fn()
  const mockAbandonMutate = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    // Setup useAgentConversation mock
    mockedUseAgentConversation.mockReturnValue({
      conversation: mockConversation,
      messages: mockConversation.messages,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      addOptimisticUserMessage: vi.fn(),
      addOptimisticAssistantMessage: vi.fn(),
      updateLastAssistantMessage: vi.fn(),
      clearOptimisticMessages: vi.fn(),
      syncWithServer: vi.fn(),
    })

    // Setup useAgentStream mock
    mockedUseAgentStream.mockReturnValue({
      isStreaming: false,
      error: null,
      currentContent: '',
      toolCalls: [],
      subagents: [],
      uncertainties: [],
      pipeline: null,
      sendMessage: mockSendMessage,
      cancel: mockCancel,
      reset: mockReset,
    })

    // Setup useApplyPipeline mock - cast to any to avoid complex type issues
    mockedUseApplyPipeline.mockReturnValue({
      mutate: mockApplyMutate,
      isPending: false,
    } as unknown as ReturnType<typeof useApplyPipeline>)

    // Setup useAbandonConversation mock
    mockedUseAbandonConversation.mockReturnValue({
      mutate: mockAbandonMutate,
      isPending: false,
    } as unknown as ReturnType<typeof useAbandonConversation>)

    // Setup useUpdateConversationCache mock
    mockedUseUpdateConversationCache.mockReturnValue({
      updatePipeline: vi.fn(),
      addUncertainty: vi.fn(),
    })
  })

  it('renders the chat interface with messages', () => {
    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    // Check header
    expect(screen.getByText('Pipeline Setup')).toBeInTheDocument()
    expect(screen.getByText('Source #42')).toBeInTheDocument()

    // Check messages are rendered
    expect(screen.getByText('Help me set up a pipeline')).toBeInTheDocument()
    expect(
      screen.getByText('I can help you configure the best settings for your documents.')
    ).toBeInTheDocument()

    // Check pipeline preview
    expect(screen.getByText('Pipeline Preview')).toBeInTheDocument()
    expect(screen.getByText('Qwen/Qwen3-Embedding-0.6B')).toBeInTheDocument()
    expect(screen.getByText('semantic')).toBeInTheDocument()
  })

  it('shows loading state while fetching conversation', () => {
    mockedUseAgentConversation.mockReturnValue({
      conversation: undefined,
      messages: [],
      isLoading: true,
      error: null,
      refetch: vi.fn(),
      addOptimisticUserMessage: vi.fn(),
      addOptimisticAssistantMessage: vi.fn(),
      updateLastAssistantMessage: vi.fn(),
      clearOptimisticMessages: vi.fn(),
      syncWithServer: vi.fn(),
    })

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText('Loading conversation...')).toBeInTheDocument()
  })

  it('handles sending a message', async () => {
    const user = userEvent.setup()

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    const input = screen.getByPlaceholderText('Type a message...')
    await user.type(input, 'Use semantic chunking')

    // Find send button - it has a specific SVG path
    const buttons = screen.getAllByRole('button')
    const sendBtn = buttons.find((btn) => btn.querySelector('svg path[d*="M12 19l9 2"]'))

    if (sendBtn) {
      await user.click(sendBtn)
    }

    await waitFor(() => {
      expect(mockSendMessage).toHaveBeenCalledWith('Use semantic chunking')
    })
  })

  it('disables input while streaming', () => {
    mockedUseAgentStream.mockReturnValue({
      isStreaming: true,
      error: null,
      currentContent: 'Processing your request...',
      toolCalls: [],
      subagents: [],
      uncertainties: [],
      pipeline: null,
      sendMessage: mockSendMessage,
      cancel: mockCancel,
      reset: mockReset,
    })

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    const input = screen.getByPlaceholderText('Type a message...')
    expect(input).toBeDisabled()
  })

  it('shows streaming content', () => {
    mockedUseAgentStream.mockReturnValue({
      isStreaming: true,
      error: null,
      currentContent: 'Analyzing your documents...',
      toolCalls: [],
      subagents: [],
      uncertainties: [],
      pipeline: null,
      sendMessage: mockSendMessage,
      cancel: mockCancel,
      reset: mockReset,
    })

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText('Analyzing your documents...')).toBeInTheDocument()
  })

  it('shows active tool calls indicator', () => {
    mockedUseAgentStream.mockReturnValue({
      isStreaming: true,
      error: null,
      currentContent: '',
      toolCalls: [
        {
          id: 'tc-1',
          tool: 'list_plugins',
          arguments: { type: 'embedding' },
          status: 'running',
        },
      ],
      subagents: [],
      uncertainties: [],
      pipeline: null,
      sendMessage: mockSendMessage,
      cancel: mockCancel,
      reset: mockReset,
    })

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText(/Running.*list_plugins/)).toBeInTheDocument()
  })

  it('calls onClose when back button is clicked', async () => {
    const user = userEvent.setup()

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    const backButton = screen.getByLabelText('Go back')
    await user.click(backButton)

    expect(mockOnClose).toHaveBeenCalled()
  })

  it('shows applied state after conversation is applied', () => {
    mockedUseAgentConversation.mockReturnValue({
      conversation: { ...mockConversation, status: 'applied' as const },
      messages: mockConversation.messages,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      addOptimisticUserMessage: vi.fn(),
      addOptimisticAssistantMessage: vi.fn(),
      updateLastAssistantMessage: vi.fn(),
      clearOptimisticMessages: vi.fn(),
      syncWithServer: vi.fn(),
    })

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText('Pipeline Applied')).toBeInTheDocument()
    expect(screen.getByText('Your collection is being created.')).toBeInTheDocument()
  })

  it('renders source analysis in pipeline preview', () => {
    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    // Check source analysis is rendered
    expect(screen.getByText('Source Analysis')).toBeInTheDocument()
    expect(screen.getByText('247')).toBeInTheDocument() // total files
  })

  it('shows stream error when present', () => {
    mockedUseAgentStream.mockReturnValue({
      isStreaming: false,
      error: 'Connection failed',
      currentContent: '',
      toolCalls: [],
      subagents: [],
      uncertainties: [],
      pipeline: null,
      sendMessage: mockSendMessage,
      cancel: mockCancel,
      reset: mockReset,
    })

    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByText('Connection failed')).toBeInTheDocument()
  })
})

describe('AgentChat Pipeline Preview', () => {
  const mockOnClose = vi.fn()
  const mockOnApplySuccess = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()

    mockedUseAgentConversation.mockReturnValue({
      conversation: mockConversation,
      messages: mockConversation.messages,
      isLoading: false,
      error: null,
      refetch: vi.fn(),
      addOptimisticUserMessage: vi.fn(),
      addOptimisticAssistantMessage: vi.fn(),
      updateLastAssistantMessage: vi.fn(),
      clearOptimisticMessages: vi.fn(),
      syncWithServer: vi.fn(),
    })

    mockedUseAgentStream.mockReturnValue({
      isStreaming: false,
      error: null,
      currentContent: '',
      toolCalls: [],
      subagents: [],
      uncertainties: [],
      pipeline: null,
      sendMessage: vi.fn(),
      cancel: vi.fn(),
      reset: vi.fn(),
    })

    mockedUseApplyPipeline.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    } as unknown as ReturnType<typeof useApplyPipeline>)

    mockedUseAbandonConversation.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
    } as unknown as ReturnType<typeof useAbandonConversation>)

    mockedUseUpdateConversationCache.mockReturnValue({
      updatePipeline: vi.fn(),
      addUncertainty: vi.fn(),
    })
  })

  it('renders pipeline configuration details', () => {
    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    // Check configuration section
    expect(screen.getByText('Configuration')).toBeInTheDocument()
    expect(screen.getByText('Qwen/Qwen3-Embedding-0.6B')).toBeInTheDocument()
    expect(screen.getByText('semantic')).toBeInTheDocument()
  })

  it('shows apply button with collection name input', () => {
    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    expect(screen.getByLabelText('Collection Name')).toBeInTheDocument()
    expect(screen.getByText('Apply Pipeline')).toBeInTheDocument()
  })

  it('disables apply button when collection name is empty', () => {
    render(
      <AgentChat
        conversationId="conv-test-123"
        onClose={mockOnClose}
        onApplySuccess={mockOnApplySuccess}
      />,
      { wrapper: createWrapper() }
    )

    const applyButton = screen.getByText('Apply Pipeline').closest('button')
    expect(applyButton).toBeDisabled()
  })
})
