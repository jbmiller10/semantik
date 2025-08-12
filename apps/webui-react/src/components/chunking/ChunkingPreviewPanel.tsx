import { useState, useMemo, useRef, useEffect } from 'react';
import { 
  FileText, 
  Hash, 
  BarChart3,
  ZoomIn,
  ZoomOut,
  ChevronLeft,
  ChevronRight,
  Copy,
  Check,
  Info,
  Wifi,
  WifiOff,
  RefreshCw
} from 'lucide-react';
import { useChunkingStore } from '../../stores/chunkingStore';
import { useChunkingWebSocket } from '../../hooks/useChunkingWebSocket';
import type { ChunkPreview } from '../../types/chunking';

interface ChunkingPreviewPanelProps {
  document?: { id?: string; content?: string; name?: string };
  onDocumentSelect?: () => void;
  height?: string;
}

export function ChunkingPreviewPanel({ 
  document: providedDocument,
  onDocumentSelect,
  height = '600px'
}: ChunkingPreviewPanelProps) {
  const {
    previewDocument,
    previewChunks,
    previewStatistics,
    previewLoading,
    previewError,
    setPreviewDocument,
    loadPreview,
    selectedStrategy,
    strategyConfig
  } = useChunkingStore();

  // WebSocket integration for real-time updates
  const {
    connectionStatus,
    connect: connectWebSocket,
    isConnected,
    chunks: wsChunks,
    progress: wsProgress,
    statistics: wsStatistics,
    error: wsError,
    startPreview: startWebSocketPreview,
    clearData: clearWebSocketData
  } = useChunkingWebSocket({
    autoConnect: true,
    onChunkReceived: (_chunk, index, total) => {
      console.log(`Received chunk ${index + 1}/${total}`);
    },
    onComplete: (statistics) => {
      console.log('Preview complete with statistics:', statistics);
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    }
  });

  const [selectedChunkIndex, setSelectedChunkIndex] = useState(0);
  const [highlightedChunkId, setHighlightedChunkId] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'split' | 'chunks' | 'original'>('split');
  const [fontSize, setFontSize] = useState(14);
  const [copiedChunkId, setCopiedChunkId] = useState<string | null>(null);
  const [useWebSocket, setUseWebSocket] = useState(true);
  const leftPanelRef = useRef<HTMLDivElement>(null);
  const rightPanelRef = useRef<HTMLDivElement>(null);

  // Use provided document or preview document from store
  const activeDocument = providedDocument || previewDocument;

  // Determine which chunks and statistics to display (WebSocket or REST)
  const displayChunks = useWebSocket && wsChunks.length > 0 ? wsChunks : previewChunks;
  const displayStatistics = useWebSocket && wsStatistics ? wsStatistics : previewStatistics;
  const displayLoading = useWebSocket ? (wsProgress !== null) : previewLoading;
  const displayError = useWebSocket ? wsError?.message : previewError;

  useEffect(() => {
    if (providedDocument && (!previewDocument || previewDocument.id !== providedDocument.id)) {
      setPreviewDocument(providedDocument);
      
      // Clear WebSocket data when document changes
      clearWebSocketData();
      
      // Try WebSocket first, fall back to REST API
      if (useWebSocket && isConnected && providedDocument.id) {
        startWebSocketPreview(
          providedDocument.id,
          selectedStrategy,
          strategyConfig.parameters
        );
      } else {
        loadPreview();
      }
    }
  }, [providedDocument, previewDocument, setPreviewDocument, loadPreview, 
      useWebSocket, isConnected, startWebSocketPreview, clearWebSocketData,
      selectedStrategy, strategyConfig]);

  // Calculate chunk boundaries in original text
  const chunkBoundaries = useMemo(() => {
    if (!activeDocument?.content || !displayChunks.length) return [];
    
    return displayChunks.map(chunk => ({
      id: chunk.id,
      start: chunk.startIndex,
      end: chunk.endIndex,
      overlapStart: chunk.overlapWithPrevious ? chunk.startIndex : null,
      overlapEnd: chunk.overlapWithNext ? chunk.endIndex - (chunk.overlapWithNext || 0) : null
    }));
  }, [activeDocument, displayChunks]);

  // Render original document with chunk boundaries highlighted
  const renderOriginalWithBoundaries = (): React.ReactNode => {
    if (!activeDocument?.content) return null;

    const content = activeDocument.content;
    const elements: React.ReactElement[] = [];
    let lastIndex = 0;

    chunkBoundaries.forEach((boundary, index) => {
      // Add text before this chunk
      if (boundary.start > lastIndex) {
        elements.push(
          <span key={`text-${lastIndex}`} className="text-gray-400">
            {content.substring(lastIndex, boundary.start)}
          </span>
        );
      }

      // Add the chunk text with highlighting
      const isHighlighted = highlightedChunkId === boundary.id;
      const isSelected = selectedChunkIndex === index;
      
      elements.push(
        <span
          key={`chunk-${boundary.id}`}
          className={`
            relative cursor-pointer transition-all duration-200
            ${isHighlighted ? 'bg-yellow-200' : ''}
            ${isSelected ? 'bg-blue-100 border-b-2 border-blue-500' : 'hover:bg-gray-100'}
          `}
          onClick={() => {
            setSelectedChunkIndex(index);
            scrollToChunk(index, 'right');
          }}
          onMouseEnter={() => setHighlightedChunkId(boundary.id)}
          onMouseLeave={() => setHighlightedChunkId(null)}
        >
          {/* Overlap indicator */}
          {boundary.overlapStart && (
            <span className="text-orange-600 bg-orange-100">
              {content.substring(boundary.overlapStart, boundary.start)}
            </span>
          )}
          
          {/* Main chunk content */}
          <span>{content.substring(boundary.start, boundary.end)}</span>
          
          {/* Chunk boundary marker */}
          <span className="absolute -right-2 top-0 text-xs text-blue-600 font-mono">
            [{index + 1}]
          </span>
        </span>
      );

      lastIndex = boundary.end;
    });

    // Add remaining text
    if (lastIndex < content.length) {
      elements.push(
        <span key={`text-${lastIndex}`} className="text-gray-400">
          {content.substring(lastIndex)}
        </span>
      );
    }

    return <>{elements}</>;
  };

  const scrollToChunk = (index: number, panel: 'left' | 'right') => {
    const chunkElement = document.getElementById(
      panel === 'left' ? `original-chunk-${index}` : `preview-chunk-${index}`
    );
    if (chunkElement) {
      chunkElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };

  const handleChunkSelect = (index: number) => {
    setSelectedChunkIndex(index);
    // Sync scroll in split view
    if (viewMode === 'split') {
      scrollToChunk(index, 'left');
    }
  };

  const copyChunkToClipboard = async (chunk: ChunkPreview) => {
    try {
      await navigator.clipboard.writeText(chunk.content);
      setCopiedChunkId(chunk.id);
      setTimeout(() => setCopiedChunkId(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const handleZoom = (delta: number) => {
    setFontSize(prev => Math.max(10, Math.min(24, prev + delta)));
  };

  if (!activeDocument) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
        <div className="text-center">
          <FileText className="h-12 w-12 text-gray-400 mx-auto mb-3" />
          <p className="text-sm text-gray-600">No document selected for preview</p>
          {onDocumentSelect && (
            <button
              onClick={onDocumentSelect}
              className="mt-2 text-sm text-blue-600 hover:text-blue-700 font-medium"
            >
              Select Document
            </button>
          )}
        </div>
      </div>
    );
  }

  if (displayLoading) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
        <div className="text-center">
          <svg className="animate-spin h-8 w-8 text-blue-600 mx-auto mb-3" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
          <p className="text-sm text-gray-600">
            {wsProgress ? `Processing chunks: ${wsProgress.currentChunk}/${wsProgress.totalChunks} (${wsProgress.percentage}%)` : 'Generating chunk preview...'}
          </p>
        </div>
      </div>
    );
  }

  if (displayError) {
    return (
      <div className="flex items-center justify-center h-64 bg-red-50 rounded-lg">
        <div className="text-center">
          <div className="text-red-600 mb-3">⚠️</div>
          <p className="text-sm text-red-700">{displayError}</p>
          <div className="flex items-center justify-center space-x-2 mt-2">
            <button
              onClick={() => {
                if (useWebSocket && activeDocument?.id) {
                  clearWebSocketData();
                  startWebSocketPreview(activeDocument.id, selectedStrategy, strategyConfig.parameters);
                } else {
                  loadPreview(true);
                }
              }}
              className="text-sm text-red-600 hover:text-red-700 font-medium"
            >
              Retry
            </button>
            {useWebSocket && !isConnected && (
              <button
                onClick={() => setUseWebSocket(false)}
                className="text-sm text-gray-600 hover:text-gray-700 font-medium"
              >
                Use REST API
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden" style={{ height }}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 bg-gray-50">
        <div className="flex items-center space-x-4">
          {/* Connection Status Indicator */}
          <div className="flex items-center space-x-2">
            {connectionStatus === 'connected' && (
              <div className="flex items-center text-green-600">
                <Wifi className="h-4 w-4 mr-1" />
                <span className="text-xs font-medium">Live</span>
              </div>
            )}
            {connectionStatus === 'connecting' && (
              <div className="flex items-center text-yellow-600">
                <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                <span className="text-xs font-medium">Connecting...</span>
              </div>
            )}
            {connectionStatus === 'reconnecting' && (
              <div className="flex items-center text-orange-600">
                <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                <span className="text-xs font-medium">Reconnecting...</span>
              </div>
            )}
            {(connectionStatus === 'disconnected' || connectionStatus === 'error') && (
              <div className="flex items-center text-gray-500">
                <WifiOff className="h-4 w-4 mr-1" />
                <span className="text-xs font-medium">
                  {useWebSocket ? 'Offline' : 'REST Mode'}
                </span>
                {useWebSocket && (
                  <button
                    onClick={connectWebSocket}
                    className="ml-1 text-xs text-blue-600 hover:text-blue-700"
                  >
                    Reconnect
                  </button>
                )}
              </div>
            )}
          </div>

          {/* View Mode Selector */}
          <div className="flex items-center bg-white rounded-md border border-gray-300">
            <button
              onClick={() => setViewMode('split')}
              className={`px-3 py-1 text-sm font-medium transition-colors ${
                viewMode === 'split' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              Split View
            </button>
            <button
              onClick={() => setViewMode('chunks')}
              className={`px-3 py-1 text-sm font-medium transition-colors ${
                viewMode === 'chunks' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              Chunks Only
            </button>
            <button
              onClick={() => setViewMode('original')}
              className={`px-3 py-1 text-sm font-medium transition-colors ${
                viewMode === 'original' 
                  ? 'bg-blue-600 text-white' 
                  : 'text-gray-700 hover:bg-gray-100'
              }`}
            >
              Original Only
            </button>
          </div>

          {/* Zoom Controls */}
          <div className="flex items-center space-x-1">
            <button
              onClick={() => handleZoom(-2)}
              className="p-1 text-gray-600 hover:text-gray-800"
              title="Decrease font size"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <span className="text-xs text-gray-600 font-mono w-8 text-center">
              {fontSize}
            </span>
            <button
              onClick={() => handleZoom(2)}
              className="p-1 text-gray-600 hover:text-gray-800"
              title="Increase font size"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Statistics */}
        {displayStatistics && (
          <div className="flex items-center space-x-4 text-sm text-gray-600">
            <div className="flex items-center">
              <Hash className="h-4 w-4 mr-1" />
              <span>{displayStatistics.totalChunks} chunks</span>
            </div>
            <div className="flex items-center">
              <BarChart3 className="h-4 w-4 mr-1" />
              <span>Avg: {displayStatistics.avgChunkSize} chars</span>
            </div>
            {displayStatistics.overlapPercentage !== undefined && (
              <div className="flex items-center">
                <div className="group relative">
                  <Info className="h-4 w-4 mr-1" />
                  <div className="absolute bottom-full left-0 mb-2 hidden group-hover:block z-10">
                    <div className="bg-gray-900 text-white text-xs rounded-lg py-2 px-3 whitespace-nowrap">
                      Overlap between chunks
                      <div className="absolute top-full left-4 -mt-1">
                        <div className="border-4 border-transparent border-t-gray-900"></div>
                      </div>
                    </div>
                  </div>
                </div>
                <span>{displayStatistics.overlapPercentage}% overlap</span>
              </div>
            )}
            {wsProgress && (
              <div className="flex items-center text-blue-600">
                <RefreshCw className="h-4 w-4 mr-1 animate-spin" />
                <span>{wsProgress.percentage}%</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Content Area */}
      <div className="flex h-[calc(100%-48px)]">
        {/* Original Document Panel */}
        {(viewMode === 'split' || viewMode === 'original') && (
          <div
            ref={leftPanelRef}
            className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} 
              border-r border-gray-200 overflow-y-auto p-4`}
            style={{ fontSize: `${fontSize}px` }}
          >
            <h3 className="text-sm font-medium text-gray-700 mb-3 sticky top-0 bg-white pb-2">
              Original Document
              {activeDocument.name && (
                <span className="text-gray-500 font-normal ml-2">
                  ({activeDocument.name})
                </span>
              )}
            </h3>
            <pre className="whitespace-pre-wrap font-mono text-gray-800 leading-relaxed">
              {viewMode === 'original' ? activeDocument.content : renderOriginalWithBoundaries()}
            </pre>
          </div>
        )}

        {/* Chunks Panel */}
        {(viewMode === 'split' || viewMode === 'chunks') && (
          <div
            ref={rightPanelRef}
            className={`${viewMode === 'split' ? 'w-1/2' : 'w-full'} 
              overflow-y-auto`}
          >
            <div className="sticky top-0 bg-white z-10 px-4 py-2 border-b border-gray-100">
              <h3 className="text-sm font-medium text-gray-700">
                Chunks ({displayChunks.length})
              </h3>
            </div>
            
            <div className="divide-y divide-gray-200">
              {displayChunks.map((chunk, index) => (
                <div
                  key={chunk.id}
                  id={`preview-chunk-${index}`}
                  className={`p-4 transition-colors cursor-pointer
                    ${selectedChunkIndex === index ? 'bg-blue-50' : 'hover:bg-gray-50'}
                    ${highlightedChunkId === chunk.id ? 'bg-yellow-50' : ''}
                  `}
                  onClick={() => handleChunkSelect(index)}
                  onMouseEnter={() => setHighlightedChunkId(chunk.id)}
                  onMouseLeave={() => setHighlightedChunkId(null)}
                >
                  {/* Chunk Header */}
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-sm font-medium text-gray-700">
                        Chunk {index + 1}
                      </span>
                      <span className="text-xs text-gray-500">
                        ({chunk.startIndex}-{chunk.endIndex})
                      </span>
                      {chunk.tokens && (
                        <span className="text-xs text-gray-500">
                          • {chunk.tokens} tokens
                        </span>
                      )}
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        copyChunkToClipboard(chunk);
                      }}
                      className="p-1 text-gray-400 hover:text-gray-600"
                      title="Copy chunk"
                    >
                      {copiedChunkId === chunk.id ? (
                        <Check className="h-4 w-4 text-green-600" />
                      ) : (
                        <Copy className="h-4 w-4" />
                      )}
                    </button>
                  </div>

                  {/* Overlap Indicators */}
                  {(chunk.overlapWithPrevious || chunk.overlapWithNext) && (
                    <div className="flex items-center space-x-2 mb-2">
                      {chunk.overlapWithPrevious && (
                        <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded">
                          ↑ {chunk.overlapWithPrevious} chars overlap
                        </span>
                      )}
                      {chunk.overlapWithNext && (
                        <span className="text-xs bg-orange-100 text-orange-700 px-2 py-1 rounded">
                          ↓ {chunk.overlapWithNext} chars overlap
                        </span>
                      )}
                    </div>
                  )}

                  {/* Chunk Content */}
                  <pre
                    className="whitespace-pre-wrap font-mono text-gray-800 leading-relaxed"
                    style={{ fontSize: `${fontSize}px` }}
                  >
                    {chunk.content}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Navigation Controls */}
      {viewMode !== 'original' && displayChunks.length > 1 && (
        <div className="absolute bottom-4 right-4 flex items-center space-x-2 bg-white rounded-lg shadow-lg px-3 py-2">
          <button
            onClick={() => {
              const newIndex = Math.max(0, selectedChunkIndex - 1);
              setSelectedChunkIndex(newIndex);
              scrollToChunk(newIndex, 'right');
            }}
            disabled={selectedChunkIndex === 0}
            className="p-1 text-gray-600 hover:text-gray-800 disabled:text-gray-400"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
          <span className="text-sm text-gray-600 font-medium">
            {selectedChunkIndex + 1} / {displayChunks.length}
          </span>
          <button
            onClick={() => {
              const newIndex = Math.min(displayChunks.length - 1, selectedChunkIndex + 1);
              setSelectedChunkIndex(newIndex);
              scrollToChunk(newIndex, 'right');
            }}
            disabled={selectedChunkIndex === displayChunks.length - 1}
            className="p-1 text-gray-600 hover:text-gray-800 disabled:text-gray-400"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}
    </div>
  );
}