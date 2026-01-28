"""Spawn tools for launching sub-agents from the orchestrator.

These tools allow the main orchestrator to spawn specialized sub-agents
with their own context windows for complex, multi-step tasks.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from webui.services.agent.tools.base import BaseTool

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from shared.llm.factory import LLMServiceFactory
    from shared.pipeline.types import FileReference
    from webui.services.agent.models import AgentConversation
    from webui.services.agent.subagents.base import SubAgentResult

logger = logging.getLogger(__name__)


class SpawnSourceAnalyzerTool(BaseTool):
    """Spawn a SourceAnalyzer sub-agent to investigate a data source.

    The SourceAnalyzer systematically analyzes a source to understand:
    - File composition (counts by type, size distribution)
    - Content characteristics (languages, document types)
    - Quality issues (scanned PDFs, corrupted files)
    - Parser recommendations for each file type

    The tool can work with either:
    - An existing source (via source_id parameter)
    - An inline source configuration stored in the conversation

    If source_id is not provided, the tool will use the conversation's
    inline_source_config if available.

    Context Requirements:
        - session: Database session
        - user_id: Current user ID
        - conversation: Current AgentConversation
        - llm_factory: LLMServiceFactory for sub-agent

    Returns:
        Structured SourceAnalysis with recommendations
    """

    NAME: ClassVar[str] = "spawn_source_analyzer"
    DESCRIPTION: ClassVar[str] = (
        "Analyze a data source to understand its contents. Spawns a specialized "
        "sub-agent that enumerates files, samples by type, tries parsers, and "
        "returns a structured analysis with parser recommendations. "
        "If no source_id is provided, uses the conversation's inline source config."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "integer",
                "description": "ID of the source to analyze. Optional if conversation has inline_source_config.",
            },
            "user_intent": {
                "type": "string",
                "description": "Optional description of what the user wants to do",
            },
        },
        "required": [],
    }

    async def execute(
        self,
        source_id: int | None = None,
        user_intent: str = "",
    ) -> dict[str, Any]:
        """Spawn the SourceAnalyzer sub-agent.

        Args:
            source_id: ID of the source to analyze. Optional if conversation has inline_source_config.
            user_intent: Optional user goal description

        Returns:
            Dictionary with analysis results
        """
        try:
            session: AsyncSession | None = self.context.get("session")
            user_id = self.context.get("user_id")
            conversation: AgentConversation | None = self.context.get("conversation")
            llm_factory: LLMServiceFactory | None = self.context.get("llm_factory")

            if not session:
                return {
                    "success": False,
                    "error": "No database session available",
                }

            if not user_id:
                return {
                    "success": False,
                    "error": "No user ID available",
                }

            if not llm_factory:
                return {
                    "success": False,
                    "error": "No LLM factory available for sub-agent",
                }

            # Lazy imports to avoid circular imports
            from shared.llm.types import LLMQualityTier
            from webui.services.agent.subagents.source_analyzer import SourceAnalyzer
            from webui.services.connector_factory import ConnectorFactory

            source_type: str
            source_config: dict[str, Any]
            effective_source_id: int | None = source_id

            if source_id is not None:
                # Using existing source - fetch from database
                from shared.database.repositories.collection_source_repository import CollectionSourceRepository

                source_repo = CollectionSourceRepository(session)
                source = await source_repo.get_by_id(source_id)

                if not source:
                    return {
                        "success": False,
                        "error": f"Source not found: {source_id}",
                    }

                source_type = source.source_type
                source_config = source.source_config or {}
                effective_source_id = int(source.id)

            elif conversation and conversation.inline_source_config:
                # Using inline source config from conversation
                inline_config = conversation.inline_source_config
                source_type = inline_config.get("source_type", "")
                source_config = inline_config.get("source_config", {})

                # Merge pending secrets into config for connector creation
                if "_pending_secrets" in inline_config:
                    source_config = {**source_config, **inline_config["_pending_secrets"]}

                if not source_type:
                    return {
                        "success": False,
                        "error": "Invalid inline source config: missing source_type",
                    }

            else:
                return {
                    "success": False,
                    "error": "No source_id provided and no inline_source_config in conversation",
                }

            # Create connector
            try:
                connector = ConnectorFactory.get_connector(
                    source_type,
                    source_config,
                )
            except ValueError as e:
                return {
                    "success": False,
                    "error": f"Failed to create connector: {e}",
                }

            # Create LLM provider for sub-agent (separate context window)
            try:
                llm_provider = await llm_factory.create_provider_for_tier(
                    user_id,
                    LLMQualityTier.HIGH,
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create LLM provider: {e}",
                }

            # Import the connection manager for independent session
            from shared.database.postgres_database import pg_connection_manager

            # Build sub-agent context - NOTE: We'll create an independent session below
            # to avoid the request lifecycle cancellation issue
            subagent_context = {
                "source_id": effective_source_id,  # May be None for inline sources
                "source_type": source_type,
                "connector": connector,
                "user_intent": user_intent,
                "user_id": user_id,
                # session will be set below with independent session
            }

            # Run the sub-agent with an INDEPENDENT database session
            # This prevents cancellation when SSE connection closes
            async with pg_connection_manager.get_session() as subagent_session:
                subagent_context["session"] = subagent_session

                async with llm_provider:
                    agent = SourceAnalyzer(llm_provider, subagent_context)
                    result: SubAgentResult = await agent.run()

                analysis_data = dict(result.data or {})
                analysis_data.setdefault("source_id", effective_source_id)
                analysis_data.setdefault("source_type", source_type)

                # Add a small, durable set of sample FileReferences to enable PipelineValidator
                # without forcing re-enumeration of the source.
                analysis_data = self._augment_analysis_with_samples(
                    analysis=analysis_data,
                    enumerated=subagent_context.get("_enumerated_files", []),
                    samples_per_extension=3,
                    max_total_samples=50,
                )

                # Store analysis in conversation state using orchestrator hook when available,
                # otherwise fall back to repository persistence.
                if conversation and result.success:
                    conversation.source_analysis = analysis_data

                    orchestrator = self.context.get("orchestrator")
                    if orchestrator is not None and hasattr(orchestrator, "add_subagent_result"):
                        await orchestrator.add_subagent_result(
                            subagent_type="source_analyzer",
                            result=analysis_data,
                            summary=result.summary,
                        )
                    else:
                        from webui.services.agent.repository import AgentConversationRepository

                        repo = AgentConversationRepository(subagent_session)
                        await repo.update_source_analysis(
                            conversation_id=conversation.id,
                            user_id=user_id,
                            source_analysis=analysis_data,
                        )

                # Store uncertainties in conversation
                persisted_uncertainties: list[dict[str, Any]] = []
                if conversation and result.uncertainties:
                    from webui.services.agent.models import UncertaintySeverity
                    from webui.services.agent.repository import AgentConversationRepository

                    repo = AgentConversationRepository(subagent_session)
                    for uncertainty in result.uncertainties:
                        # Normalize severity to valid enum value (LLM may generate invalid values like "critical")
                        try:
                            severity = UncertaintySeverity(uncertainty.severity)
                        except ValueError:
                            # Map common invalid values or default to INFO
                            severity_map = {
                                "critical": UncertaintySeverity.BLOCKING,
                                "warning": UncertaintySeverity.NOTABLE,
                            }
                            severity = severity_map.get(uncertainty.severity, UncertaintySeverity.INFO)

                        created = await repo.add_uncertainty(
                            conversation_id=conversation.id,
                            user_id=user_id,
                            severity=severity,
                            message=uncertainty.message,
                            context=uncertainty.context,
                        )
                        persisted_uncertainties.append(
                            {
                                "id": created.id,
                                "severity": severity.value,
                                "message": created.message,
                                "resolved": bool(created.resolved),
                                "context": created.context,
                            }
                        )
                # Commit happens automatically via the context manager

                return {
                    "success": result.success,
                    "analysis": analysis_data,
                    "summary": result.summary,
                    "error": result.summary if not result.success else None,
                    "uncertainties": (
                        persisted_uncertainties
                        if persisted_uncertainties
                        else [{"severity": u.severity, "message": u.message} for u in result.uncertainties]
                    ),
                }

        except Exception as e:
            logger.error(f"Failed to spawn SourceAnalyzer: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _augment_analysis_with_samples(
        self,
        analysis: dict[str, Any],
        enumerated: list[FileReference],
        samples_per_extension: int,
        max_total_samples: int,
    ) -> dict[str, Any]:
        """Add a small set of sample FileReference dicts to the analysis.

        The pipeline validator requires full FileReference objects. Persisting a small
        sample set (instead of the full enumeration) enables validation without
        re-enumerating the source.
        """
        if not enumerated:
            return analysis

        by_extension: dict[str, list[dict[str, Any]]] = {}
        total_samples: list[dict[str, Any]] = []

        for ref in enumerated:
            if len(total_samples) >= max_total_samples:
                break
            ext = (ref.extension or "(no ext)").lower()
            samples = by_extension.setdefault(ext, [])
            if len(samples) >= samples_per_extension:
                continue
            ref_dict = ref.to_dict()
            samples.append(ref_dict)
            total_samples.append(ref_dict)

        analysis.setdefault("sample_files", total_samples)
        analysis.setdefault("by_extension", {})
        for ext, samples in by_extension.items():
            ext_entry = analysis["by_extension"].setdefault(ext, {})
            ext_entry.setdefault("sample_files", samples)
            ext_entry.setdefault("sample_uris", [s.get("uri") for s in samples if s.get("uri")])

        return analysis


class SpawnPipelineValidatorTool(BaseTool):
    """Spawn a PipelineValidator sub-agent to validate a pipeline configuration.

    The PipelineValidator tests a pipeline against sample files to:
    - Validate that files can be processed through all stages
    - Identify and categorize failures
    - Try alternative configurations for fixable issues
    - Assess chunk quality
    - Produce a structured validation report

    Context Requirements:
        - session: Database session
        - user_id: Current user ID
        - conversation: Current AgentConversation (with source_analysis)
        - llm_factory: LLMServiceFactory for sub-agent

    Returns:
        Structured ValidationReport with assessment and recommendations
    """

    NAME: ClassVar[str] = "spawn_pipeline_validator"
    DESCRIPTION: ClassVar[str] = (
        "Validate a pipeline configuration against sample files. Spawns a specialized "
        "sub-agent that runs dry-run validation, investigates failures, and returns "
        "a structured validation report with assessment and suggested fixes. "
        "IMPORTANT: You must first call build_pipeline or get_pipeline_state to obtain "
        "the pipeline object, then pass it as the 'pipeline' argument to this tool."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pipeline": {
                "type": "object",
                "description": (
                    "Pipeline DAG configuration to validate. Obtain this from the "
                    "'pipeline' field returned by build_pipeline or get_pipeline_state."
                ),
            },
            "sample_count": {
                "type": "integer",
                "description": "Number of sample files to test (default 50)",
                "default": 50,
            },
        },
        "required": ["pipeline"],
    }

    async def execute(
        self,
        pipeline: dict[str, Any] | None = None,
        sample_count: int = 50,
    ) -> dict[str, Any]:
        """Spawn the PipelineValidator sub-agent.

        Args:
            pipeline: Pipeline DAG configuration to validate. Required.
            sample_count: Number of samples to test

        Returns:
            Dictionary with validation results
        """
        # Validate required argument - this allows graceful error handling
        # instead of Python raising TypeError when LLM omits the argument
        if pipeline is None:
            return {
                "success": False,
                "error": (
                    "Missing required argument 'pipeline'. "
                    "Call build_pipeline or get_pipeline_state first to obtain "
                    "the pipeline configuration, then pass it to this tool."
                ),
            }

        try:
            session: AsyncSession | None = self.context.get("session")
            user_id = self.context.get("user_id")
            conversation: AgentConversation | None = self.context.get("conversation")
            llm_factory: LLMServiceFactory | None = self.context.get("llm_factory")

            if not session:
                return {
                    "success": False,
                    "error": "No database session available",
                }

            if not user_id:
                return {
                    "success": False,
                    "error": "No user ID available",
                }

            if not llm_factory:
                return {
                    "success": False,
                    "error": "No LLM factory available for sub-agent",
                }

            # Get source analysis from conversation to select samples
            source_analysis = None
            if conversation:
                source_analysis = getattr(conversation, "source_analysis", None)

            if not source_analysis:
                return {
                    "success": False,
                    "error": "No source analysis available. Run spawn_source_analyzer first.",
                }

            # Select representative samples from analysis (before connector creation).
            # This allows us to return a clear error if the analysis doesn't contain any
            # usable sample references.
            sample_files = self._select_samples(source_analysis, sample_count)
            sample_uris: list[str] = []
            if not sample_files:
                sample_uris = self._select_sample_uris(source_analysis, sample_count)

            if not sample_files and not sample_uris:
                return {
                    "success": False,
                    "error": "No sample files available from source analysis",
                }

            # Get connector from context or create from source
            connector = self.context.get("connector")
            if not connector:
                inline_config: dict[str, Any] | None = None

                # Prefer the conversation's persisted source_id if available
                source_id = getattr(conversation, "source_id", None) if conversation else None
                if not isinstance(source_id, int):
                    source_id = None

                # Fallback: try to get from source analysis context
                if not source_id:
                    candidate = source_analysis.get("source_id")
                    source_id = candidate if isinstance(candidate, int) else None

                if source_id:
                    from shared.database.repositories.collection_source_repository import CollectionSourceRepository
                    from webui.services.connector_factory import ConnectorFactory

                    source_repo = CollectionSourceRepository(session)
                    source = await source_repo.get_by_id(source_id)
                    if source:
                        source_type = source.source_type
                        source_config = source.source_config or {}
                        connector = ConnectorFactory.get_connector(source_type, source_config)
                else:
                    inline_config = getattr(conversation, "inline_source_config", None) if conversation else None
                    if not isinstance(inline_config, dict):
                        inline_config = None

                if not connector and inline_config:
                    # Inline source config (source not created yet)
                    from webui.services.connector_factory import ConnectorFactory

                    source_type = inline_config.get("source_type", "")
                    source_config = inline_config.get("source_config", {})

                    # Merge pending secrets into config for connector creation
                    if "_pending_secrets" in inline_config:
                        source_config = {**source_config, **inline_config["_pending_secrets"]}

                    if source_type:
                        connector = ConnectorFactory.get_connector(source_type, source_config)

            if not connector:
                return {
                    "success": False,
                    "error": "No connector available for validation",
                }

            # Fallback: reconstruct samples from sample_uris by enumerating until matched
            if not sample_files:
                source_id_for_enum = getattr(conversation, "source_id", None) if conversation else None
                if not isinstance(source_id_for_enum, int):
                    source_id_for_enum = None
                if not source_id_for_enum:
                    candidate = source_analysis.get("source_id")
                    source_id_for_enum = candidate if isinstance(candidate, int) else None
                sample_files = await self._resolve_samples_from_uris(
                    connector=connector,
                    source_id=source_id_for_enum,
                    sample_uris=sample_uris,
                    sample_count=sample_count,
                )

            if not sample_files:
                return {
                    "success": False,
                    "error": "No sample files available from source analysis",
                }

            # Create LLM provider for sub-agent (separate context window)
            from shared.llm.types import LLMQualityTier

            try:
                llm_provider = await llm_factory.create_provider_for_tier(
                    user_id,
                    LLMQualityTier.HIGH,
                )
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Failed to create LLM provider: {e}",
                }

            # Import the connection manager for independent session
            from shared.database.postgres_database import pg_connection_manager

            # Build sub-agent context - NOTE: We'll create an independent session below
            # to avoid the request lifecycle cancellation issue
            subagent_context = {
                "pipeline": pipeline,
                "sample_files": sample_files,
                "connector": connector,
                "user_id": user_id,
                # session will be set below with independent session
            }

            # Run the sub-agent with an INDEPENDENT database session
            # This prevents cancellation when SSE connection closes
            from webui.services.agent.subagents.pipeline_validator import PipelineValidator

            async with pg_connection_manager.get_session() as subagent_session:
                subagent_context["session"] = subagent_session

                async with llm_provider:
                    agent = PipelineValidator(llm_provider, subagent_context)
                    result: SubAgentResult = await agent.run()

                # Store validation report in conversation state (best-effort, not persisted)
                if conversation and result.success:
                    conversation.current_pipeline_validation = result.data

                # Store uncertainties in conversation
                persisted_uncertainties: list[dict[str, Any]] = []
                if conversation and result.uncertainties:
                    from webui.services.agent.models import UncertaintySeverity
                    from webui.services.agent.repository import AgentConversationRepository

                    repo = AgentConversationRepository(subagent_session)
                    for uncertainty in result.uncertainties:
                        # Normalize severity to valid enum value (LLM may generate invalid values like "critical")
                        try:
                            severity = UncertaintySeverity(uncertainty.severity)
                        except ValueError:
                            # Map common invalid values or default to INFO
                            severity_map = {
                                "critical": UncertaintySeverity.BLOCKING,
                                "warning": UncertaintySeverity.NOTABLE,
                            }
                            severity = severity_map.get(uncertainty.severity, UncertaintySeverity.INFO)

                        created = await repo.add_uncertainty(
                            conversation_id=conversation.id,
                            user_id=user_id,
                            severity=severity,
                            message=uncertainty.message,
                            context=uncertainty.context,
                        )
                        persisted_uncertainties.append(
                            {
                                "id": created.id,
                                "severity": severity.value,
                                "message": created.message,
                                "resolved": bool(created.resolved),
                                "context": created.context,
                            }
                        )
                # Commit happens automatically via the context manager

                return {
                    "success": result.success,
                    "report": result.data,
                    "summary": result.summary,
                    "uncertainties": (
                        persisted_uncertainties
                        if persisted_uncertainties
                        else [{"severity": u.severity, "message": u.message} for u in result.uncertainties]
                    ),
                }

        except Exception as e:
            logger.error(f"Failed to spawn PipelineValidator: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }

    def _select_samples(
        self,
        source_analysis: dict[str, Any],
        sample_count: int,
    ) -> list[FileReference]:
        """Select representative sample files from source analysis.

        Prefers persisted sample FileReference dicts (from SourceAnalyzer), falling back
        to minimal, lossy samples only if needed.

        Args:
            source_analysis: Source analysis data from SourceAnalyzer
            sample_count: Target number of samples

        Returns:
            List of FileReference objects
        """
        from shared.pipeline.types import FileReference

        sample_refs: list[FileReference] = []

        # Preferred: full FileReference dicts persisted by SourceAnalyzer spawn tool
        sample_dicts = source_analysis.get("sample_files") or []
        for d in sample_dicts:
            if len(sample_refs) >= sample_count:
                break
            if not isinstance(d, dict):
                continue
            try:
                sample_refs.append(FileReference.from_dict(d))
            except Exception:
                continue

        if sample_refs:
            return sample_refs[:sample_count]

        # Fallback: attempt to reconstruct from by_extension sample_files
        by_extension = source_analysis.get("by_extension", {})
        if isinstance(by_extension, dict):
            for ext_data in by_extension.values():
                if len(sample_refs) >= sample_count:
                    break
                for d in ext_data.get("sample_files") or []:
                    if len(sample_refs) >= sample_count:
                        break
                    if not isinstance(d, dict):
                        continue
                    try:
                        sample_refs.append(FileReference.from_dict(d))
                    except Exception:
                        continue

        return sample_refs[:sample_count]

    def _select_sample_uris(self, source_analysis: dict[str, Any], sample_count: int) -> list[str]:
        """Select sample URIs from analysis (lossy fallback)."""
        uris: list[str] = []
        by_extension = source_analysis.get("by_extension", {})
        if isinstance(by_extension, dict):
            for ext_data in by_extension.values():
                for uri in ext_data.get("sample_uris") or []:
                    if isinstance(uri, str) and uri and uri not in uris:
                        uris.append(uri)
                    if len(uris) >= sample_count:
                        return uris
        return uris

    async def _resolve_samples_from_uris(
        self,
        connector: Any,
        source_id: int | None,
        sample_uris: list[str],
        sample_count: int,
    ) -> list[FileReference]:
        """Resolve FileReference objects by enumerating until URIs are found."""
        if not sample_uris:
            return []

        target = set(sample_uris)
        found: list[FileReference] = []
        fallback: list[FileReference] = []

        # Limit scan effort to avoid walking huge sources in worst-case scenarios.
        max_scan = max(500, sample_count * 50)
        scanned = 0

        async for ref in connector.enumerate(source_id=source_id):
            scanned += 1
            if ref.uri in target:
                found.append(ref)
                target.discard(ref.uri)
                if not target and len(found) >= min(sample_count, len(sample_uris)):
                    break
            elif len(fallback) < sample_count:
                fallback.append(ref)

            if scanned >= max_scan:
                break

        # Prefer exact matches, then fill with fallback enumerated refs
        if len(found) < sample_count:
            seen = {f.uri for f in found}
            for ref in fallback:
                if len(found) >= sample_count:
                    break
                if ref.uri not in seen:
                    found.append(ref)
                    seen.add(ref.uri)

        return found[:sample_count]
