import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent } from '@/tests/utils/test-utils'
import userEvent from '@testing-library/user-event'
import { ChunkingAnalyticsDashboard } from '../ChunkingAnalyticsDashboard'
import { useChunkingStore } from '@/stores/chunkingStore'
import type { ChunkingAnalytics, ChunkingRecommendation } from '@/types/chunking'

// Mock the chunking store
vi.mock('@/stores/chunkingStore', () => ({
  useChunkingStore: vi.fn(),
}))

// Mock URL.createObjectURL and URL.revokeObjectURL
global.URL.createObjectURL = vi.fn(() => 'blob:mock-url')
global.URL.revokeObjectURL = vi.fn()

// Mock document.createElement for download testing
const mockClick = vi.fn()
const originalCreateElement = document.createElement.bind(document)
document.createElement = vi.fn((tagName: string) => {
  const element = originalCreateElement(tagName)
  if (tagName === 'a') {
    element.click = mockClick
  }
  return element
})

const mockAnalyticsData: ChunkingAnalytics = {
  performance: {
    avgChunkSize: 512,
    avgOverlapSize: 50,
    totalChunks: 1500,
    avgProcessingTime: 0.25,
    trend: 'up' as const,
  },
  quality: {
    coherenceScore: 0.85,
    completenessScore: 0.92,
    redundancyScore: 0.15,
    trend: 'stable' as const,
  },
  recommendations: [
    {
      type: 'performance',
      priority: 'high',
      title: 'Optimize chunk size',
      description: 'Current chunk size may be too large for optimal retrieval',
      impact: 'Reduce retrieval latency by 30%',
      action: {
        type: 'adjust_parameters',
        configuration: {
          strategy: 'recursive',
          parameters: {
            chunk_size: 256,
            chunk_overlap: 25,
          },
        },
      },
    },
    {
      type: 'quality',
      priority: 'medium',
      title: 'Increase chunk overlap',
      description: 'More overlap could improve context preservation',
      impact: 'Improve coherence score by 10%',
      action: {
        type: 'adjust_parameters',
        configuration: {
          strategy: 'recursive',
          parameters: {
            chunk_size: 512,
            chunk_overlap: 100,
          },
        },
      },
    },
    {
      type: 'cost',
      priority: 'low',
      title: 'Consider semantic chunking',
      description: 'Semantic chunking may provide better results for your content',
      impact: 'Improve quality scores by 15%',
      action: {
        type: 'change_strategy',
        configuration: {
          strategy: 'semantic',
          parameters: {
            embedding_batch_size: 32,
            similarity_threshold: 0.7,
          },
        },
      },
    },
  ],
  strategyPerformance: {
    recursive: {
      usage: 0.6,
      avgQuality: 0.85,
      avgSpeed: 0.25,
    },
    fixed: {
      usage: 0.2,
      avgQuality: 0.75,
      avgSpeed: 0.15,
    },
    semantic: {
      usage: 0.15,
      avgQuality: 0.92,
      avgSpeed: 0.45,
    },
    markdown: {
      usage: 0.05,
      avgQuality: 0.88,
      avgSpeed: 0.30,
    },
  },
  historicalTrends: [
    {
      date: '2025-01-01',
      avgChunkSize: 500,
      avgQuality: 0.82,
      totalOperations: 45,
    },
    {
      date: '2025-01-08',
      avgChunkSize: 510,
      avgQuality: 0.84,
      totalOperations: 52,
    },
    {
      date: '2025-01-15',
      avgChunkSize: 512,
      avgQuality: 0.85,
      totalOperations: 48,
    },
  ],
}

describe('ChunkingAnalyticsDashboard', () => {
  const mockLoadAnalytics = vi.fn()
  const mockSetStrategy = vi.fn()
  const mockUpdateConfiguration = vi.fn()
  const mockOnApplyRecommendation = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: mockAnalyticsData,
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  it('renders analytics data correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    // Check performance metrics
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
    expect(screen.getByText('512')).toBeInTheDocument() // avg chunk size
    expect(screen.getByText('50')).toBeInTheDocument() // avg overlap size
    expect(screen.getByText('1,500')).toBeInTheDocument() // total chunks
    expect(screen.getByText('0.25s')).toBeInTheDocument() // avg processing time

    // Check quality metrics
    expect(screen.getByText('Quality Metrics')).toBeInTheDocument()
    expect(screen.getByText('85%')).toBeInTheDocument() // coherence score
    expect(screen.getByText('92%')).toBeInTheDocument() // completeness score
    expect(screen.getByText('15%')).toBeInTheDocument() // redundancy score
  })

  it('shows loading state when analytics are loading', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: null,
      analyticsLoading: true,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('Loading analytics...')).toBeInTheDocument()
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })

  it('shows empty state when no analytics data', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: null,
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('No Analytics Available')).toBeInTheDocument()
    expect(screen.getByText(/No chunking analytics data/)).toBeInTheDocument()
  })

  it('loads analytics on mount', () => {
    render(<ChunkingAnalyticsDashboard />)
    expect(mockLoadAnalytics).toHaveBeenCalledTimes(1)
  })

  it('handles refresh button click', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    const refreshButton = screen.getByRole('button', { name: /refresh/i })
    await user.click(refreshButton)

    expect(mockLoadAnalytics).toHaveBeenCalledTimes(2) // Once on mount, once on refresh
  })

  it('handles export analytics button click', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    const exportButton = screen.getByRole('button', { name: /export/i })
    await user.click(exportButton)

    expect(global.URL.createObjectURL).toHaveBeenCalled()
    expect(mockClick).toHaveBeenCalled()
    expect(global.URL.revokeObjectURL).toHaveBeenCalled()
  })

  it('renders recommendations correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('Recommendations')).toBeInTheDocument()
    expect(screen.getByText('Optimize chunk size')).toBeInTheDocument()
    expect(screen.getByText('Current chunk size may be too large for optimal retrieval')).toBeInTheDocument()
    expect(screen.getByText('Reduce retrieval latency by 30%')).toBeInTheDocument()
  })

  it('handles recommendation priority colors correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    const highPriorityCard = screen.getByText('Optimize chunk size').closest('.border')
    expect(highPriorityCard).toHaveClass('border-red-200')

    const mediumPriorityCard = screen.getByText('Increase chunk overlap').closest('.border')
    expect(mediumPriorityCard).toHaveClass('border-yellow-200')

    const lowPriorityCard = screen.getByText('Consider semantic chunking').closest('.border')
    expect(lowPriorityCard).toHaveClass('border-blue-200')
  })

  it('expands and collapses recommendation details', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    // Initially details should not be visible
    expect(screen.queryByText('Current chunk size may be too large for optimal retrieval')).not.toBeVisible()

    // Click to expand
    const expandButton = screen.getAllByRole('button', { name: /chevron/i })[0]
    await user.click(expandButton)

    // Details should now be visible
    await waitFor(() => {
      expect(screen.getByText('Current chunk size may be too large for optimal retrieval')).toBeVisible()
    })

    // Click to collapse
    await user.click(expandButton)

    // Details should be hidden again
    await waitFor(() => {
      expect(screen.queryByText('Current chunk size may be too large for optimal retrieval')).not.toBeVisible()
    })
  })

  it('applies recommendation when apply button is clicked', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard onApplyRecommendation={mockOnApplyRecommendation} />)

    // Expand first recommendation
    const expandButton = screen.getAllByRole('button', { name: /chevron/i })[0]
    await user.click(expandButton)

    // Click apply button
    const applyButton = screen.getByRole('button', { name: /apply recommendation/i })
    await user.click(applyButton)

    expect(mockSetStrategy).toHaveBeenCalledWith('recursive')
    expect(mockUpdateConfiguration).toHaveBeenCalledWith({
      chunk_size: 256,
      chunk_overlap: 25,
    })
    expect(mockOnApplyRecommendation).toHaveBeenCalledWith(mockAnalyticsData.recommendations[0])
  })

  it('changes time range selection', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    // Check initial state
    expect(screen.getByRole('button', { name: /30d/i })).toHaveClass('bg-blue-600')

    // Click 7d button
    const sevenDayButton = screen.getByRole('button', { name: /7d/i })
    await user.click(sevenDayButton)

    expect(sevenDayButton).toHaveClass('bg-blue-600')
    expect(screen.getByRole('button', { name: /30d/i })).not.toHaveClass('bg-blue-600')
  })

  it('renders strategy performance correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('Strategy Performance')).toBeInTheDocument()
    
    // Check recursive strategy stats
    expect(screen.getByText('Recursive')).toBeInTheDocument()
    expect(screen.getByText('60%')).toBeInTheDocument() // usage
    expect(screen.getByText('0.85')).toBeInTheDocument() // quality
    expect(screen.getByText('0.25s')).toBeInTheDocument() // speed
  })

  it('renders historical trends', () => {
    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('Historical Trends')).toBeInTheDocument()
    
    // Check if chart container is rendered
    const chartContainer = screen.getByTestId('historical-trends-chart')
    expect(chartContainer).toBeInTheDocument()
  })

  it('handles trend icons correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    // Check for up trend icon
    const upTrendIcon = screen.getByTestId('trend-up-icon')
    expect(upTrendIcon).toHaveClass('text-green-600')

    // Check for stable trend icon
    const stableTrendIcon = screen.getByTestId('trend-stable-icon')
    expect(stableTrendIcon).toHaveClass('text-gray-600')
  })

  it('handles empty recommendations gracefully', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: {
        ...mockAnalyticsData,
        recommendations: [],
      },
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('No recommendations available')).toBeInTheDocument()
  })

  it('handles missing action in recommendation', async () => {
    const recommendationWithoutAction: ChunkingRecommendation = {
      type: 'performance',
      priority: 'high',
      title: 'Test recommendation',
      description: 'Test description',
      impact: 'Test impact',
    }

    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: {
        ...mockAnalyticsData,
        recommendations: [recommendationWithoutAction],
      },
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard onApplyRecommendation={mockOnApplyRecommendation} />)

    // Expand recommendation
    const expandButton = screen.getByRole('button', { name: /chevron/i })
    await user.click(expandButton)

    // Apply button should not be rendered
    expect(screen.queryByRole('button', { name: /apply recommendation/i })).not.toBeInTheDocument()
  })

  it('does not export when no analytics data', async () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: null,
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    // Export button should be disabled or not present
    const exportButton = screen.queryByRole('button', { name: /export/i })
    if (exportButton) {
      await user.click(exportButton)
      expect(global.URL.createObjectURL).not.toHaveBeenCalled()
    }
  })

  it('formats large numbers correctly', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: {
        ...mockAnalyticsData,
        performance: {
          ...mockAnalyticsData.performance,
          totalChunks: 1234567,
        },
      },
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('1,234,567')).toBeInTheDocument()
  })

  it('handles loading state with existing data', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: mockAnalyticsData,
      analyticsLoading: true,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    // Should still show data while loading
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
    
    // Should show loading indicator
    expect(screen.getByRole('progressbar')).toBeInTheDocument()
  })
})