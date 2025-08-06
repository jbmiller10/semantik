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
  strategyUsage: [
    {
      strategy: 'recursive',
      count: 600,
      percentage: 60,
      trend: 'up',
    },
    {
      strategy: 'semantic',
      count: 200,
      percentage: 20,
      trend: 'stable',
    },
    {
      strategy: 'markdown',
      count: 150,
      percentage: 15,
      trend: 'down',
    },
    {
      strategy: 'character',
      count: 50,
      percentage: 5,
      trend: 'stable',
    },
  ],
  performanceMetrics: [
    {
      strategy: 'recursive',
      avgProcessingTimeMs: 250,
      avgChunksPerDocument: 10,
      successRate: 95.5,
    },
    {
      strategy: 'semantic',
      avgProcessingTimeMs: 450,
      avgChunksPerDocument: 8,
      successRate: 98.2,
    },
    {
      strategy: 'markdown',
      avgProcessingTimeMs: 300,
      avgChunksPerDocument: 12,
      successRate: 96.8,
    },
    {
      strategy: 'character',
      avgProcessingTimeMs: 150,
      avgChunksPerDocument: 15,
      successRate: 92.0,
    },
  ],
  fileTypeDistribution: [
    {
      fileType: 'pdf',
      count: 450,
      preferredStrategy: 'recursive',
    },
    {
      fileType: 'md',
      count: 250,
      preferredStrategy: 'markdown',
    },
    {
      fileType: 'txt',
      count: 150,
      preferredStrategy: 'recursive',
    },
    {
      fileType: 'docx',
      count: 100,
      preferredStrategy: 'recursive',
    },
    {
      fileType: 'html',
      count: 50,
      preferredStrategy: 'semantic',
    },
  ],
  recommendations: [
    {
      id: 'rec-1',
      type: 'strategy',
      priority: 'high',
      title: 'Optimize chunk size',
      description: 'Current chunk size may be too large for optimal retrieval',
      action: {
        label: 'Apply recommendation',
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
      id: 'rec-2',
      type: 'parameter',
      priority: 'medium',
      title: 'Increase chunk overlap',
      description: 'More overlap could improve context preservation',
      action: {
        label: 'Apply settings',
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
      id: 'rec-3',
      type: 'general',
      priority: 'low',
      title: 'Consider semantic chunking',
      description: 'Semantic chunking may provide better results for your content',
      action: {
        label: 'Switch strategy',
        configuration: {
          strategy: 'semantic',
          parameters: {
            breakpoint_percentile_threshold: 90,
            max_chunk_size: 1000,
          },
        },
      },
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

    // Check headers
    expect(screen.getByText('Chunking Analytics')).toBeInTheDocument()
    expect(screen.getByText('Strategy Usage')).toBeInTheDocument()
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
    expect(screen.getByText('File Type Distribution')).toBeInTheDocument()

    // Check strategy usage data - use getAllByText since "Recursive" appears multiple times
    const recursiveElements = screen.getAllByText('Recursive')
    expect(recursiveElements.length).toBeGreaterThan(0)
    expect(screen.getByText('600 (60%)')).toBeInTheDocument()
    
    // Check performance metrics
    expect(screen.getByText('250ms')).toBeInTheDocument() // processing time
    expect(screen.getByText('95.5%')).toBeInTheDocument() // success rate
    
    // Check file type distribution
    expect(screen.getByText('.pdf')).toBeInTheDocument()
    expect(screen.getByText('450 files')).toBeInTheDocument()
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
    // The spinner is an SVG with animation, not a progressbar role
    const spinner = document.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
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

    expect(screen.getByText('No analytics data available')).toBeInTheDocument()
    expect(screen.getByText('Load Analytics')).toBeInTheDocument()
  })

  it('loads analytics on mount', () => {
    render(<ChunkingAnalyticsDashboard />)
    expect(mockLoadAnalytics).toHaveBeenCalledTimes(1)
  })

  it('handles refresh button click', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    const refreshButton = screen.getByTitle('Refresh analytics')
    await user.click(refreshButton)

    expect(mockLoadAnalytics).toHaveBeenCalledTimes(2) // Once on mount, once on refresh
  })

  it('handles export analytics button click', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard />)

    const exportButton = screen.getByTitle('Export analytics')
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
    // Check that all three recommendations are rendered
    expect(screen.getByText('Increase chunk overlap')).toBeInTheDocument()
    expect(screen.getByText('Consider semantic chunking')).toBeInTheDocument()
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

    // Initially, the configuration JSON should not be visible
    expect(screen.queryByText('Suggested Configuration:')).not.toBeInTheDocument()

    // Click to expand
    const detailsButton = screen.getAllByText('Details')[0]
    await user.click(detailsButton)

    // Configuration should now be visible
    await waitFor(() => {
      expect(screen.getByText('Suggested Configuration:')).toBeInTheDocument()
    })

    // The button text should change to 'Hide'
    expect(screen.getByText('Hide')).toBeInTheDocument()

    // Click to collapse
    await user.click(screen.getByText('Hide'))

    // Configuration should be hidden again
    await waitFor(() => {
      expect(screen.queryByText('Suggested Configuration:')).not.toBeInTheDocument()
    })
  })

  it('applies recommendation when apply button is clicked', async () => {
    const user = userEvent.setup()
    render(<ChunkingAnalyticsDashboard onApplyRecommendation={mockOnApplyRecommendation} />)

    // Click apply button
    const applyButtons = screen.getAllByText('Apply')
    await user.click(applyButtons[0])

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

    // Find the select element
    const timeRangeSelect = screen.getByRole('combobox')
    
    // Check initial value
    expect(timeRangeSelect).toHaveValue('30d')

    // Change to 7 days
    await user.selectOptions(timeRangeSelect, '7d')

    expect(timeRangeSelect).toHaveValue('7d')
  })

  it('renders strategy performance correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
    
    // Check table headers
    expect(screen.getByText('Strategy')).toBeInTheDocument()
    expect(screen.getByText('Avg Time')).toBeInTheDocument()
    expect(screen.getByText('Avg Chunks')).toBeInTheDocument()
    expect(screen.getByText('Success Rate')).toBeInTheDocument()
    
    // Check performance metrics values
    expect(screen.getByText('250ms')).toBeInTheDocument()
    expect(screen.getByText('10')).toBeInTheDocument() // avg chunks per document
    expect(screen.getByText('95.5%')).toBeInTheDocument()
    
    // Check that Semantic strategy appears (multiple times is OK)
    const semanticElements = screen.getAllByText('Semantic')
    expect(semanticElements.length).toBeGreaterThan(0)
    expect(screen.getByText('450ms')).toBeInTheDocument()
    expect(screen.getByText('98.2%')).toBeInTheDocument()
  })

  it('renders historical trends', () => {
    render(<ChunkingAnalyticsDashboard />)

    // The component doesn't render historical trends anymore,
    // it renders summary stats instead
    expect(screen.getByText('Total Documents Processed')).toBeInTheDocument()
    expect(screen.getByText('Most Used Strategy')).toBeInTheDocument()
    expect(screen.getByText('Average Success Rate')).toBeInTheDocument()
    
    // Check the calculated values
    expect(screen.getByText('1,000')).toBeInTheDocument() // total documents (600+200+150+50)
    // Average success rate calculation: (95.5 + 98.2 + 96.8 + 92.0) / 4 = 95.625, rounds to 95.6%
    expect(screen.getByText('95.6%')).toBeInTheDocument()
  })

  it('handles trend icons correctly', () => {
    render(<ChunkingAnalyticsDashboard />)

    // The component uses lucide icons without test-ids
    // Check that the icons are rendered in the strategy usage section
    const strategyUsageSection = screen.getByText('Strategy Usage').closest('.bg-white')
    
    // Check for trend icons - they are SVG elements with specific classes
    const trendIcons = strategyUsageSection?.querySelectorAll('svg.h-4.w-4')
    expect(trendIcons?.length).toBeGreaterThan(0)
    
    // At least one should have green color for 'up' trend
    const upTrendIcon = strategyUsageSection?.querySelector('svg.text-green-600')
    expect(upTrendIcon).toBeInTheDocument()
    
    // At least one should have gray color for 'stable' trend
    const stableTrendIcon = strategyUsageSection?.querySelector('svg.text-gray-600')
    expect(stableTrendIcon).toBeInTheDocument()
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

    // When there are no recommendations, the recommendations section is not rendered at all
    expect(screen.queryByText('Recommendations')).not.toBeInTheDocument()
  })

  it('handles missing action in recommendation', async () => {
    const recommendationWithoutAction: ChunkingRecommendation = {
      id: 'rec-no-action',
      type: 'general',
      priority: 'high',
      title: 'Test recommendation',
      description: 'Test description',
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

    render(<ChunkingAnalyticsDashboard onApplyRecommendation={mockOnApplyRecommendation} />)

    // When there's no action, neither Details nor Apply buttons should be rendered
    expect(screen.queryByText('Details')).not.toBeInTheDocument()
    expect(screen.queryByText('Apply')).not.toBeInTheDocument()
    
    // But the recommendation itself should still be shown
    expect(screen.getByText('Test recommendation')).toBeInTheDocument()
    expect(screen.getByText('Test description')).toBeInTheDocument()
  })

  it('does not export when no analytics data', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: null,
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    // When no analytics data, the export button should not be present
    // (the whole analytics dashboard shows an empty state)
    const exportButton = screen.queryByTitle('Export analytics')
    expect(exportButton).not.toBeInTheDocument()
    
    // Verify empty state is shown instead
    expect(screen.getByText('No analytics data available')).toBeInTheDocument()
  })

  it('formats large numbers correctly', () => {
    ;(useChunkingStore as any).mockReturnValue({
      analyticsData: {
        ...mockAnalyticsData,
        strategyUsage: [
          {
            strategy: 'recursive',
            count: 1234567,
            percentage: 100,
            trend: 'up',
          },
        ],
      },
      analyticsLoading: false,
      loadAnalytics: mockLoadAnalytics,
      setStrategy: mockSetStrategy,
      updateConfiguration: mockUpdateConfiguration,
    })

    render(<ChunkingAnalyticsDashboard />)

    // The total documents processed should be formatted with commas
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

    // Should still show data while loading (component doesn't hide data when loading with existing data)
    expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
    expect(screen.getByText('Strategy Usage')).toBeInTheDocument()
    
    // Should show loading indicator in refresh button
    const refreshButton = screen.getByTitle('Refresh analytics')
    const spinner = refreshButton.querySelector('.animate-spin')
    expect(spinner).toBeInTheDocument()
  })
})