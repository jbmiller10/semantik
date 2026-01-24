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
        "returns a structured analysis with parser recommendations."
    )
    PARAMETERS: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "source_id": {
                "type": "integer",
                "description": "ID of the source to analyze",
            },
            "user_intent": {
                "type": "string",
                "description": "Optional description of what the user wants to do",
            },
        },
        "required": ["source_id"],
    }

    async def execute(
        self,
        source_id: int,
        user_intent: str = "",
    ) -> dict[str, Any]:
        """Spawn the SourceAnalyzer sub-agent.

        Args:
            source_id: ID of the source to analyze
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

            # Get the source and create connector
            # Lazy imports to avoid circular imports
            from shared.database.repositories.collection_source_repository import (
                CollectionSourceRepository,
            )
            from shared.llm.types import LLMQualityTier
            from webui.services.agent.subagents.source_analyzer import SourceAnalyzer
            from webui.services.connector_factory import ConnectorFactory

            source_repo = CollectionSourceRepository(session)
            source = await source_repo.get_by_id(source_id)

            if not source:
                return {
                    "success": False,
                    "error": f"Source not found: {source_id}",
                }

            # Create connector
            # Note: SQLAlchemy Column types resolve to actual values at runtime
            source_type: str = source.source_type  # type: ignore[assignment]
            source_config: dict[str, Any] = source.source_config or {}  # type: ignore[assignment]
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

            # Build sub-agent context
            subagent_context = {
                "source_id": source_id,
                "connector": connector,
                "user_intent": user_intent,
                "session": session,
                "user_id": user_id,
            }

            # Run the sub-agent
            async with llm_provider:
                agent = SourceAnalyzer(llm_provider, subagent_context)
                result: SubAgentResult = await agent.run()

            # Store analysis in conversation state
            if conversation and result.success:
                conversation.source_analysis = result.data

            # Store uncertainties in conversation
            if conversation and result.uncertainties:
                from webui.services.agent.repository import AgentConversationRepository

                repo = AgentConversationRepository(session)
                for uncertainty in result.uncertainties:
                    await repo.add_uncertainty(
                        conversation_id=conversation.id,
                        user_id=user_id,
                        severity=uncertainty.severity,
                        message=uncertainty.message,
                        context=uncertainty.context,
                    )

            return {
                "success": result.success,
                "analysis": result.data,
                "summary": result.summary,
                "uncertainties": [{"severity": u.severity, "message": u.message} for u in result.uncertainties],
            }

        except Exception as e:
            logger.error(f"Failed to spawn SourceAnalyzer: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
            }
