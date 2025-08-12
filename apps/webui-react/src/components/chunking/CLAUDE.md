<component>
  <name>Chunking UI Components</name>
  <purpose>Interactive UI for document chunking configuration and preview</purpose>
  <location>apps/webui-react/src/components/chunking/</location>
</component>

<components>
  <component name="ChunkingStrategySelector">
    <purpose>Main strategy selection interface</purpose>
    <strategies>
      - CHARACTER: Fixed-size splitting
      - RECURSIVE: Intelligent boundary detection
      - MARKDOWN: Structure-preserving
      - SEMANTIC: AI-powered segmentation
      - HIERARCHICAL: Parent-child relationships
      - HYBRID: Auto-selection based on content
    </strategies>
  </component>
  
  <component name="SimplifiedChunkingStrategySelector">
    <purpose>Streamlined selector for basic use cases</purpose>
    <usage>Used in CreateCollectionModal</usage>
  </component>
  
  <component name="ChunkingParameterTuner">
    <purpose>Advanced parameter configuration</purpose>
    <parameters>
      - chunk_size: 100-10000 characters
      - chunk_overlap: 0-500 characters
      - Strategy-specific params
    </parameters>
  </component>
  
  <component name="ChunkingPreviewPanel">
    <purpose>Real-time preview with WebSocket updates</purpose>
    <features>
      - Split-view: original + chunks
      - Chunk boundary highlighting
      - Synchronized scrolling
    </features>
  </component>
  
  <component name="ChunkingComparisonView">
    <purpose>Compare up to 3 strategies side-by-side</purpose>
    <metrics>
      - Chunk count
      - Size distribution
      - Processing time
      - Quality score
    </metrics>
  </component>
  
  <component name="ChunkingAnalyticsDashboard">
    <purpose>Performance metrics and recommendations</purpose>
  </component>
</components>

<state-management>
  <store>stores/chunkingStore.ts</store>
  <state>
    - selectedStrategy
    - parameters
    - previewResults
    - comparisonResults
  </state>
</state-management>

<api-integration>
  <preview>POST /api/v2/chunking/preview</preview>
  <compare>POST /api/v2/chunking/compare</compare>
  <caching>15-minute TTL for preview results</caching>
</api-integration>

<ux-patterns>
  <debouncing>500ms delay on parameter changes</debouncing>
  <loading>Skeleton screens during processing</loading>
  <errors>Inline validation messages</errors>
  <tooltips>Strategy descriptions and best practices</tooltips>
</ux-patterns>

<testing>
  <location>__tests__/</location>
  <coverage>
    - User interactions
    - WebSocket updates
    - Error states
    - Parameter validation
  </coverage>
</testing>