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
                from shared.database.repositories.collection_source_repository import (
                    CollectionSourceRepository,
                )

                source_repo = CollectionSourceRepository(session)
                source = await source_repo.get_by_id(source_id)

                if not source:
                    return {
                        "success": False,
                        "error": f"Source not found: {source_id}",
                    }

                source_type = source.source_type
                source_config = source.source_config or {}

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

                # Store analysis in conversation state using the subagent's session
                if conversation and result.success:
                    conversation.source_analysis = result.data
                    # Update the conversation in the database
                    from webui.services.agent.repository import AgentConversationRepository

                    repo = AgentConversationRepository(subagent_session)
                    await repo.update_source_analysis(
                        conversation_id=conversation.id,
                        user_id=user_id,
                        source_analysis=result.data,
                    )

                # Store uncertainties in conversation
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
                            severity_map = {"critical": UncertaintySeverity.BLOCKING, "warning": UncertaintySeverity.NOTABLE}
                            severity = severity_map.get(uncertainty.severity, UncertaintySeverity.INFO)

                        await repo.add_uncertainty(
                            conversation_id=conversation.id,
                            user_id=user_id,
                            severity=severity,
                            message=uncertainty.message,
                            context=uncertainty.context,
                        )
                # Commit happens automatically via the context manager

                return {
                    "success": result.success,
                    "analysis": result.data,
                    "summary": result.summary,
                    "error": result.summary if not result.success else None,
                    "uncertainties": [{"severity": u.severity, "message": u.message} for u in result.uncertainties],
                }

        except Exception as e:
            logger.error(f"Failed to spawn SourceAnalyzer: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }


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
        pipeline: dict[str, Any],
        sample_count: int = 50,
    ) -> dict[str, Any]:
        """Spawn the PipelineValidator sub-agent.

        Args:
            pipeline: Pipeline DAG configuration to validate
            sample_count: Number of samples to test

        Returns:
            Dictionary with validation results
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

            # Get source analysis from conversation to select samples
            source_analysis = None
            if conversation:
                source_analysis = getattr(conversation, "source_analysis", None)

            if not source_analysis:
                return {
                    "success": False,
                    "error": "No source analysis available. Run spawn_source_analyzer first.",
                }

            # Select representative sample files
            sample_files = self._select_samples(source_analysis, sample_count)

            if not sample_files:
                return {
                    "success": False,
                    "error": "No sample files available from source analysis",
                }

            # Get connector from context or create from source
            connector = self.context.get("connector")
            if not connector:
                # Try to get from source analysis context
                source_id = source_analysis.get("source_id")
                if source_id:
                    from shared.database.repositories.collection_source_repository import (
                        CollectionSourceRepository,
                    )
                    from webui.services.connector_factory import ConnectorFactory

                    source_repo = CollectionSourceRepository(session)
                    source = await source_repo.get_by_id(source_id)
                    if source:
                        source_type = source.source_type
                        source_config = source.source_config or {}
                        connector = ConnectorFactory.get_connector(source_type, source_config)

            if not connector:
                return {
                    "success": False,
                    "error": "No connector available for validation",
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

                # Store validation report in conversation state
                if conversation and result.success:
                    conversation.current_pipeline_validation = result.data

                # Store uncertainties in conversation
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
                            severity_map = {"critical": UncertaintySeverity.BLOCKING, "warning": UncertaintySeverity.NOTABLE}
                            severity = severity_map.get(uncertainty.severity, UncertaintySeverity.INFO)

                        await repo.add_uncertainty(
                            conversation_id=conversation.id,
                            user_id=user_id,
                            severity=severity,
                            message=uncertainty.message,
                            context=uncertainty.context,
                        )
                # Commit happens automatically via the context manager

                return {
                    "success": result.success,
                    "report": result.data,
                    "summary": result.summary,
                    "uncertainties": [{"severity": u.severity, "message": u.message} for u in result.uncertainties],
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
    ) -> list[Any]:
        """Select representative sample files from source analysis.

        Tries to get a balanced sample across file types.

        Args:
            source_analysis: Source analysis data from SourceAnalyzer
            sample_count: Target number of samples

        Returns:
            List of FileReference objects (or equivalent dicts)
        """
        # Check for enumerated files in the analysis
        by_extension = source_analysis.get("by_extension", {})
        if not by_extension:
            return []

        samples: list[dict[str, str]] = []

        # If we have sample URIs stored, use those
        for ext_data in by_extension.values():
            sample_uris = ext_data.get("sample_uris", [])
            for uri in sample_uris:
                if len(samples) >= sample_count:
                    break
                # Create a minimal file reference dict
                samples.append({"uri": uri})

        # If no samples found, return empty
        return samples[:sample_count]
