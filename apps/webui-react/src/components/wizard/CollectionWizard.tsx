// apps/webui-react/src/components/wizard/CollectionWizard.tsx
import { useState, useCallback, useMemo } from 'react';
import { X } from 'lucide-react';
import { StepProgressIndicator } from './StepProgressIndicator';
import { BasicsStep } from './steps/BasicsStep';
import { ConfigureStep } from './steps/ConfigureStep';
import { AnalysisStep } from './steps/AnalysisStep';
import { ReviewStep } from './steps/ReviewStep';
import { useWizardKeyboard } from './hooks/useWizardKeyboard';
import ErrorBoundary from '../ErrorBoundary';
import { getInitialWizardState, MANUAL_STEPS, ASSISTED_STEPS } from '../../types/wizard';
import { useCreateCollection } from '../../hooks/useCollections';
import { useAddSource } from '../../hooks/useCollectionOperations';
import { useStartAssistedFlow } from '../../hooks/useAssistedFlow';
import { useUIStore } from '../../stores/uiStore';
import { waitForCollectionReady } from '../../services/api/v2/collections';
import type { WizardState, WizardFlow } from '../../types/wizard';
import type { PipelineDAG } from '../../types/pipeline';
import type { SyncMode } from '../../types/collection';
import { ensurePathNames } from '../../utils/pipelineUtils';

interface CollectionWizardProps {
  onClose: () => void;
  onSuccess: () => void;
  /** Optional conversation ID to resume an in-progress setup */
  resumeConversationId?: string;
}

// Default pipeline DAG
function getDefaultDAG(): PipelineDAG {
  return {
    id: 'default',
    version: '1',
    nodes: [
      { id: 'parser1', type: 'parser', plugin_id: 'text', config: {} },
      { id: 'chunker1', type: 'chunker', plugin_id: 'semantic', config: {} },
      // Note: plugin_id is the embedding provider plugin, model is selected in config
      { id: 'embedder1', type: 'embedder', plugin_id: 'dense_local', config: {} },
    ],
    edges: [
      { from_node: '_source', to_node: 'parser1', when: null },
      { from_node: 'parser1', to_node: 'chunker1', when: null },
      { from_node: 'chunker1', to_node: 'embedder1', when: null },
    ],
  };
}

export function CollectionWizard({ onClose, onSuccess, resumeConversationId }: CollectionWizardProps) {
  // Determine initial state based on whether we're resuming
  const isResuming = Boolean(resumeConversationId);
  const initialFlow = isResuming ? 'assisted' : 'manual';
  const initialStep = isResuming ? 2 : 0; // Skip to analysis step if resuming

  const [wizardState, setWizardState] = useState<WizardState>(() => {
    const state = getInitialWizardState(initialFlow);
    if (isResuming) {
      // Mark previous steps as complete when resuming
      return {
        ...state,
        currentStep: initialStep,
        steps: state.steps.map((s, i) => ({ ...s, isComplete: i < initialStep })),
      };
    }
    return state;
  });

  const createCollectionMutation = useCreateCollection();
  const addSourceMutation = useAddSource();
  const startAssistedFlowMutation = useStartAssistedFlow();
  const { addToast } = useUIStore();

  // Assisted flow state - initialize with resume ID if provided
  const [conversationId, setConversationId] = useState<string | null>(resumeConversationId || null);
  const [agentSummary, setAgentSummary] = useState('');

  // Step 1 (Basics) state
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [connectorType, setConnectorType] = useState<string>('none');
  const [configValues, setConfigValues] = useState<Record<string, unknown>>({});
  const [secrets, setSecrets] = useState<Record<string, string>>({});
  const [syncMode, setSyncMode] = useState<SyncMode>('one_time');
  const [syncIntervalMinutes, setSyncIntervalMinutes] = useState(60);
  const [errors, setErrors] = useState<Record<string, string>>({});

  // Step 3 (Configure) state
  const [dag, setDag] = useState<PipelineDAG>(getDefaultDAG());

  // Submission state
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Determine modal size based on current step
  const isExpanded = wizardState.currentStep >= 2;

  // Mobile: full screen when expanded, responsive otherwise
  const sizeClasses = isExpanded
    ? 'w-full lg:w-[90vw] max-w-7xl h-[100dvh] lg:h-[85vh]'
    : 'w-full max-w-2xl max-h-[90vh]';

  const validateBasics = useCallback(() => {
    const newErrors: Record<string, string> = {};

    if (!name.trim()) {
      newErrors.name = 'Collection name is required';
    } else if (name.length > 100) {
      newErrors.name = 'Name must be 100 characters or less';
    }

    if (description && description.length > 500) {
      newErrors.description = 'Description must be 500 characters or less';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  }, [name, description]);

  const handleClose = useCallback(() => {
    onClose();
  }, [onClose]);

  const handleNext = useCallback(async () => {
    // Validate current step before proceeding
    if (wizardState.currentStep === 0 && !validateBasics()) {
      return;
    }

    // Create conversation when entering assisted step 3
    if (
      wizardState.currentStep === 1 &&
      wizardState.flow === 'assisted' &&
      connectorType !== 'none' &&
      !conversationId
    ) {
      try {
        const response = await startAssistedFlowMutation.mutateAsync({
          inline_source: {
            source_type: connectorType,
            source_config: configValues,
          },
          secrets: Object.keys(secrets).length > 0 ? secrets : undefined,
        });
        setConversationId(response.session_id);
      } catch (error) {
        addToast({
          message: error instanceof Error ? error.message : 'Failed to start analysis',
          type: 'error',
        });
        return;
      }
    }

    setWizardState(prev => {
      const newSteps = [...prev.steps];
      newSteps[prev.currentStep] = { ...newSteps[prev.currentStep], isComplete: true };
      return {
        ...prev,
        currentStep: Math.min(prev.currentStep + 1, prev.steps.length - 1),
        steps: newSteps,
      };
    });
  }, [wizardState.currentStep, wizardState.flow, validateBasics, connectorType, configValues, secrets, conversationId, startAssistedFlowMutation, addToast]);

  const handleBack = useCallback(() => {
    setWizardState(prev => ({
      ...prev,
      currentStep: Math.max(prev.currentStep - 1, 0),
    }));
  }, []);

  const handleFlowChange = useCallback((flow: WizardFlow) => {
    setWizardState(prev => ({
      ...prev,
      flow,
      steps: flow === 'manual'
        ? MANUAL_STEPS.map((s, i) => ({ ...s, isComplete: i < prev.currentStep }))
        : ASSISTED_STEPS.map((s, i) => ({ ...s, isComplete: i < prev.currentStep })),
    }));
  }, []);

  const handleSwitchToAssisted = useCallback(() => {
    handleFlowChange('assisted');
    // Go back to step 2 (mode selection) when switching
    setWizardState(prev => ({
      ...prev,
      currentStep: 1,
      flow: 'assisted',
      steps: ASSISTED_STEPS.map((s, i) => ({ ...s, isComplete: i < 1 })),
    }));
  }, [handleFlowChange]);

  // Handle switching from assisted to manual
  const handleSwitchToManual = useCallback(() => {
    handleFlowChange('manual');
    // Go back to step 2 (mode selection) when switching
    setWizardState(prev => ({
      ...prev,
      currentStep: 1,
      flow: 'manual',
      steps: MANUAL_STEPS.map((s, i) => ({ ...s, isComplete: i < 1 })),
    }));
  }, [handleFlowChange]);

  // Handle agent analysis completion
  const handleAgentComplete = useCallback(() => {
    // Agent analysis is done - "Next" button will advance to review step
  }, []);

  const handleCreate = useCallback(async () => {
    setIsSubmitting(true);

    try {
      // Ensure parallel edges have path_names before submission
      const dagWithPathNames = ensurePathNames(dag);

      // Extract config from DAG
      const chunkerNode = dagWithPathNames.nodes.find(n => n.type === 'chunker');
      const embedderNode = dagWithPathNames.nodes.find(n => n.type === 'embedder');

      // Get embedding model from config (if using dense_local plugin) or fall back to plugin_id for legacy compatibility
      const embeddingModel = (embedderNode?.config?.model as string) || embedderNode?.plugin_id || 'sentence-transformers/all-MiniLM-L6-v2';
      const quantization = (embedderNode?.config?.quantization as string) || 'float16';

      const response = await createCollectionMutation.mutateAsync({
        name: name.trim(),
        description: description.trim() || undefined,
        embedding_model: embeddingModel,
        quantization: quantization,
        chunking_strategy: chunkerNode?.plugin_id || 'semantic',
        chunking_config: (chunkerNode?.config || {}) as Record<string, string | number | boolean>,
        sync_mode: syncMode,
        sync_interval_minutes: syncMode === 'continuous' ? syncIntervalMinutes : undefined,
        // Pass full pipeline DAG for custom routing configuration
        pipeline_config: dagWithPathNames as unknown as Record<string, unknown>,
      });

      // Add source if configured
      if (connectorType !== 'none' && Object.keys(configValues).length > 0) {
        const sourcePath = (configValues.path as string) || (configValues.repo_url as string) || '';
        if (sourcePath) {
          // Wait for collection to become ready before adding source
          // This prevents 409 Conflict errors when the initial CREATE operation is still in progress
          await waitForCollectionReady(response.id, {
            timeout: 30000,
            pollInterval: 500,
          });

          await addSourceMutation.mutateAsync({
            collectionId: response.id,
            sourceType: connectorType,
            sourceConfig: configValues,
            secrets: Object.keys(secrets).length > 0 ? secrets : undefined,
            sourcePath,
          });
        }
      }

      addToast({
        message: 'Collection created successfully!',
        type: 'success',
      });

      onSuccess();
    } catch (error) {
      addToast({
        message: error instanceof Error ? error.message : 'Failed to create collection',
        type: 'error',
      });
    } finally {
      setIsSubmitting(false);
    }
  }, [
    name, description, dag, connectorType, configValues, secrets,
    syncMode, syncIntervalMinutes, createCollectionMutation, addSourceMutation,
    addToast, onSuccess,
  ]);

  // Handle final step action
  const handleFinalAction = useCallback(() => {
    if (wizardState.currentStep === wizardState.steps.length - 1) {
      handleCreate();
    } else {
      handleNext();
    }
  }, [wizardState.currentStep, wizardState.steps.length, handleCreate, handleNext]);

  const isNextDisabled = useMemo(() => {
    if (wizardState.currentStep === 0) {
      return !name.trim();
    }
    return false;
  }, [wizardState.currentStep, name]);

  const isFinalStep = wizardState.currentStep === wizardState.steps.length - 1;

  // Keyboard navigation and focus management
  const { modalRef } = useWizardKeyboard({
    onClose: handleClose,
    onNext: handleFinalAction,
    onBack: handleBack,
    canAdvance: !isNextDisabled,
    isSubmitting,
  });

  return (
    <div className="fixed inset-0 bg-black/50 dark:bg-black/80 flex items-center justify-center p-4 z-50">
      <div
        ref={modalRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby="wizard-title"
        data-testid="create-collection-modal"
        className={`
          panel relative rounded-2xl shadow-2xl
          flex flex-col overflow-hidden
          transition-all duration-300 ease-out
          ${sizeClasses}
        `}
      >
        {/* Header */}
        <div className="px-6 py-4 border-b border-[var(--border)] bg-[var(--bg-secondary)] shrink-0">
          <div className="flex items-center justify-between">
            <h2 id="wizard-title" className="text-lg font-semibold text-[var(--text-primary)]">
              Create Collection
            </h2>
            <button
              onClick={handleClose}
              disabled={isSubmitting}
              aria-label="Close"
              className="p-2 rounded-lg hover:bg-[var(--bg-tertiary)] text-[var(--text-muted)] disabled:opacity-50"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          <StepProgressIndicator
            steps={wizardState.steps}
            currentStep={wizardState.currentStep}
          />
        </div>

        {/* Content area */}
        <div className="flex-1 overflow-y-auto">
          <ErrorBoundary level="section">
          {/* Step 1: Basics & Source */}
          {wizardState.currentStep === 0 && (
            <div className="px-6 py-4">
              <BasicsStep
                name={name}
                description={description}
                connectorType={connectorType}
                configValues={configValues}
                secrets={secrets}
                syncMode={syncMode}
                syncIntervalMinutes={syncIntervalMinutes}
                onNameChange={setName}
                onDescriptionChange={setDescription}
                onConnectorTypeChange={setConnectorType}
                onConfigChange={setConfigValues}
                onSecretsChange={setSecrets}
                onSyncModeChange={setSyncMode}
                onSyncIntervalChange={setSyncIntervalMinutes}
                errors={errors}
              />
            </div>
          )}

          {/* Step 2: Mode Selection */}
          {wizardState.currentStep === 1 && (
            <div className="px-6 py-4 space-y-4">
              <p className="text-sm text-[var(--text-secondary)]">
                Choose how you want to configure your pipeline:
              </p>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <button
                  onClick={() => handleFlowChange('assisted')}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all
                    ${wizardState.flow === 'assisted'
                      ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                      : 'border-[var(--border)] hover:border-gray-300 dark:hover:border-gray-600'
                    }
                  `}
                >
                  <span className="text-lg">Assisted</span>
                  <p className="text-sm text-[var(--text-muted)] mt-1">
                    AI analyzes your source and recommends optimal settings
                  </p>
                </button>
                <button
                  onClick={() => handleFlowChange('manual')}
                  className={`
                    p-4 rounded-lg border-2 text-left transition-all
                    ${wizardState.flow === 'manual'
                      ? 'border-gray-400 dark:border-white bg-gray-100 dark:bg-white/10'
                      : 'border-[var(--border)] hover:border-gray-300 dark:hover:border-gray-600'
                    }
                  `}
                >
                  <span className="text-lg">Manual</span>
                  <p className="text-sm text-[var(--text-muted)] mt-1">
                    Configure pipeline settings yourself
                  </p>
                </button>
              </div>
            </div>
          )}

          {/* Step 3: Configure (manual) or Analysis (assisted) */}
          {wizardState.currentStep === 2 && (
            <div className="h-full">
              {wizardState.flow === 'manual' ? (
                <ConfigureStep
                  dag={dag}
                  onDagChange={setDag}
                  sourceAnalysis={null}
                  onSwitchToAssisted={handleSwitchToAssisted}
                />
              ) : conversationId ? (
                <AnalysisStep
                  conversationId={conversationId}
                  dag={dag}
                  onDagChange={setDag}
                  onAgentComplete={handleAgentComplete}
                  onSwitchToManual={handleSwitchToManual}
                  onSummaryChange={setAgentSummary}
                />
              ) : (
                <div className="flex items-center justify-center h-full">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[var(--text-muted)]" />
                </div>
              )}
            </div>
          )}

          {/* Step 4: Review (assisted only) */}
          {wizardState.currentStep === 3 && wizardState.flow === 'assisted' && (
            <ReviewStep
              dag={dag}
              onDagChange={setDag}
              agentSummary={agentSummary || 'The agent has analyzed your source and recommended the pipeline configuration shown.'}
              conversationId={conversationId || undefined}
              onBackToAnalysis={() => setWizardState(prev => ({ ...prev, currentStep: 2 }))}
            />
          )}
          </ErrorBoundary>
        </div>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-[var(--border)] bg-[var(--bg-secondary)] flex justify-between shrink-0">
          <button
            onClick={wizardState.currentStep === 0 ? handleClose : handleBack}
            disabled={isSubmitting}
            className="px-4 py-2 rounded-lg border border-[var(--border)] hover:bg-[var(--bg-tertiary)] text-[var(--text-secondary)] disabled:opacity-50"
          >
            {wizardState.currentStep === 0 ? 'Cancel' : 'Back'}
          </button>

          <button
            onClick={handleFinalAction}
            disabled={isNextDisabled || isSubmitting}
            className="px-4 py-2 rounded-lg bg-gray-200 dark:bg-white text-gray-900 dark:text-gray-900 font-medium disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {isSubmitting && (
              <div className="animate-spin rounded-full h-4 w-4 border-2 border-gray-600 border-t-transparent" />
            )}
            {isFinalStep ? 'Create Collection' : 'Next'}
          </button>
        </div>
      </div>
    </div>
  );
}
