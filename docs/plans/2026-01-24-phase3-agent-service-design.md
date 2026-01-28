# Phase 3: Agent Service with Sub-Agents

**Date:** 2026-01-24
**Status:** Draft
**Parent Document:** `2026-01-22-agentic-pipeline-builder-design.md`

## Overview

Phase 3 implements the agent service that enables conversational pipeline building. The architecture uses a main orchestrator that delegates complex tasks to focused sub-agents, each with their own context window and specialized toolset.

**Outcome:** Working agent that builds pipelines through conversation, with sub-agents handling source analysis and pipeline validation.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              User (Chat UI)                              │
└─────────────────────────────────────┬───────────────────────────────────┘
                                      │
                    POST /api/v2/agent/conversations/{id}/messages
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         AgentOrchestrator                                │
│                                                                          │
│  • Manages conversation flow                                             │
│  • Answers simple questions directly                                     │
│  • Spawns sub-agents for complex tasks                                   │
│  • Synthesizes sub-agent results for user                                │
│                                                                          │
│  Tools: list_plugins, get_template, get_pipeline_state                   │
└───────────┬─────────────────────────────────────┬───────────────────────┘
            │                                     │
            │ spawn                               │ spawn
            ▼                                     ▼
┌─────────────────────────┐         ┌─────────────────────────┐
│   SourceAnalyzer        │         │   PipelineValidator     │
│                         │         │                         │
│ Own context window      │         │ Own context window      │
│ Specialized prompt      │         │ Specialized prompt      │
│                         │         │                         │
│ Tools:                  │         │ Tools:                  │
│ • enumerate_files       │         │ • run_dry_run           │
│ • sample_files          │         │ • get_failure_details   │
│ • try_parser            │         │ • try_alternative       │
│ • detect_patterns       │         │ • compare_configs       │
└─────────────────────────┘         └─────────────────────────┘
```

### Key Principles

- **Orchestrator is lightweight** - handles conversation, delegates heavy analysis
- **Sub-agents are focused** - each has a specialized system prompt and toolset
- **Clean handoff** - sub-agents return structured results, orchestrator synthesizes for user
- **Parallel-capable** - independent sub-agents can run concurrently (future optimization)

---

## Conversation State Model

The conversation uses hybrid storage: PostgreSQL for durable state, Redis for ephemeral message history.

### PostgreSQL (durable)

```python
# webui/services/agent/models.py

class AgentConversation(Base):
    """Persistent conversation state."""

    __tablename__ = "agent_conversations"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    user_id: Mapped[UUID] = mapped_column(ForeignKey("users.id"), index=True)
    source_id: Mapped[UUID | None] = mapped_column(ForeignKey("sources.id"))
    collection_id: Mapped[UUID | None] = mapped_column(ForeignKey("collections.id"))

    # Current state
    status: Mapped[str] = mapped_column(default="active")  # active, applied, abandoned
    current_pipeline: Mapped[dict | None] = mapped_column(JSON)  # Serialized PipelineDAG
    source_analysis: Mapped[dict | None] = mapped_column(JSON)  # SourceAnalyzer result

    # For conversation recovery when Redis expires
    summary: Mapped[str | None] = mapped_column(Text)

    # Tracking
    created_at: Mapped[datetime] = mapped_column(default=func.now())
    updated_at: Mapped[datetime] = mapped_column(default=func.now(), onupdate=func.now())


class ConversationUncertainty(Base):
    """Uncertainties flagged during conversation."""

    __tablename__ = "conversation_uncertainties"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    conversation_id: Mapped[UUID] = mapped_column(ForeignKey("agent_conversations.id"))

    severity: Mapped[str]  # "blocking", "notable", "info"
    message: Mapped[str]
    context: Mapped[dict | None] = mapped_column(JSON)  # Related data (file refs, etc.)
    resolved: Mapped[bool] = mapped_column(default=False)
    resolved_by: Mapped[str | None]  # "user_confirmed", "pipeline_adjusted", etc.
```

### Redis (ephemeral)

```python
# Key: agent:conversation:{id}:messages
# TTL: 24 hours (extended on activity)

@dataclass
class ConversationMessage:
    role: Literal["user", "assistant", "tool", "subagent"]
    content: str
    timestamp: datetime
    metadata: dict | None = None  # tool_name, subagent_type, etc.
```

### Recovery Flow

When Redis expires but user returns, orchestrator reconstructs context from `summary` + `source_analysis` + `current_pipeline` and continues naturally.

---

## Sub-Agent Architecture

Sub-agents are independent agent loops with their own context, system prompt, and tools. The orchestrator spawns them and receives structured results.

### Base Sub-Agent Definition

```python
# webui/services/agent/subagents/base.py

@dataclass
class Uncertainty:
    """An uncertainty flagged by the agent."""
    severity: Literal["blocking", "notable", "info"]
    message: str
    context: dict[str, Any] | None = None


@dataclass
class SubAgentResult:
    """Structured result from a sub-agent."""
    success: bool
    data: dict[str, Any]
    uncertainties: list[Uncertainty] = field(default_factory=list)
    summary: str = ""  # Human-readable summary for orchestrator to relay


class SubAgent(ABC):
    """Base class for sub-agents."""

    AGENT_ID: ClassVar[str]
    SYSTEM_PROMPT: ClassVar[str]
    TOOLS: ClassVar[list[type[BaseTool]]]
    MAX_TURNS: ClassVar[int] = 20  # Safety limit
    TIMEOUT_SECONDS: ClassVar[int] = 300  # 5 minutes max

    def __init__(
        self,
        llm_provider: LLMProvider,
        context: dict[str, Any],  # Passed from orchestrator
    ):
        self.llm = llm_provider
        self.context = context
        self.messages: list[Message] = []
        self.tools = {t.NAME: t(context) for t in self.TOOLS}

    async def run(self) -> SubAgentResult:
        """Execute the agent loop until complete or max_turns."""
        try:
            return await asyncio.wait_for(
                self._run_loop(),
                timeout=self.TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return SubAgentResult(
                success=False,
                data=self._get_partial_result(),
                summary=f"Timed out after {self.TIMEOUT_SECONDS}s",
            )

    async def _run_loop(self) -> SubAgentResult:
        self.messages.append(self._build_initial_message())

        for turn in range(self.MAX_TURNS):
            response = await self.llm.generate(
                system=self.SYSTEM_PROMPT,
                messages=self.messages,
                tools=self._tool_schemas(),
            )

            self.messages.append(response)

            if response.stop_reason == "end_turn":
                return self._extract_result(response)

            if response.tool_calls:
                tool_results = await self._execute_tools(response.tool_calls)
                self.messages.append(tool_results)

        return SubAgentResult(
            success=False,
            data={},
            summary="Sub-agent reached max turns without completing",
        )

    @abstractmethod
    def _build_initial_message(self) -> Message:
        """Build the initial user message with task context."""
        ...

    @abstractmethod
    def _extract_result(self, response: Message) -> SubAgentResult:
        """Extract structured result from final response."""
        ...

    def _get_partial_result(self) -> dict:
        """Get partial result for timeout/failure cases."""
        return {}
```

### Spawning from Orchestrator

```python
# In orchestrator
async def _spawn_subagent(
    self,
    agent_class: type[SubAgent],
    context: dict[str, Any],
) -> SubAgentResult:
    """Spawn a sub-agent with isolated context."""

    # Sub-agent gets fresh LLM provider (own context window)
    provider = await self.llm_factory.create_provider_for_tier(
        self.user_id,
        LLMQualityTier.HIGH,
    )

    async with provider:
        agent = agent_class(provider, context)
        result = await agent.run()

    # Store uncertainties in conversation state
    for uncertainty in result.uncertainties:
        await self._record_uncertainty(uncertainty)

    return result
```

---

## SourceAnalyzer Sub-Agent

The workhorse sub-agent. Investigates a source to understand what's there and how to handle it.

```python
# webui/services/agent/subagents/source_analyzer.py

class SourceAnalyzer(SubAgent):
    AGENT_ID = "source_analyzer"
    MAX_TURNS = 30  # May need many tool calls for large sources

    SYSTEM_PROMPT = """You are a source analysis agent for Semantik, a semantic search engine.

Your job: Investigate a data source and produce a comprehensive analysis that helps
the pipeline builder choose the right parsing, chunking, and embedding strategy.

You have tools to enumerate files, sample by type, and try parsers. Use them
systematically:

1. First, enumerate to understand the source composition (file types, sizes, counts)
2. Sample representative files from each major type
3. Try parsing samples to verify they work and understand content characteristics
4. Look for patterns: Are PDFs scanned? Is there mixed language content? Code vs prose?
5. Note any issues or uncertainties

Be thorough but efficient. Don't try every file - sample intelligently.

When done, produce a structured analysis with:
- Source composition (counts by type, size distribution)
- Content characteristics (languages, document types, quality issues)
- Recommended parsers for each file type with confidence
- Any uncertainties the user should know about

Always end with a structured JSON result in your final message."""

    TOOLS = [
        EnumerateFilesTool,
        SampleFilesTool,
        TryParserTool,
        DetectLanguageTool,
        GetFileContentPreviewTool,
    ]

    def _build_initial_message(self) -> Message:
        source_id = self.context["source_id"]
        user_intent = self.context.get("user_intent", "")

        return Message(
            role="user",
            content=f"""Analyze source {source_id} for pipeline configuration.

User's goal: {user_intent or "Not specified - infer from content"}

Produce a comprehensive analysis I can use to recommend a pipeline."""
        )

    def _extract_result(self, response: Message) -> SubAgentResult:
        # Parse the structured JSON from the response
        analysis = self._parse_analysis_json(response.content)

        return SubAgentResult(
            success=True,
            data=analysis,
            uncertainties=analysis.get("uncertainties", []),
            summary=analysis.get("summary", ""),
        )
```

### SourceAnalyzer Tools

| Tool | Purpose |
|------|---------|
| `EnumerateFilesTool` | List all files with metadata, return counts by type/size |
| `SampleFilesTool` | Get N files matching criteria (extension, size range, random) |
| `TryParserTool` | Attempt to parse a file with a specific parser, return success + stats |
| `DetectLanguageTool` | Detect language of text content |
| `GetFileContentPreviewTool` | Get first N bytes/chars to inspect content |

### Result Structure

```python
@dataclass
class FileTypeStats:
    count: int
    total_size_bytes: int
    sample_results: list[dict]  # Parser attempts on samples


@dataclass
class ContentCharacteristics:
    languages: list[str]
    document_types: list[str]  # "academic", "code", "documentation", etc.
    quality_issues: list[str]  # "scanned_pdfs", "corrupted_files", etc.


@dataclass
class ParserRecommendation:
    extension: str
    parser_id: str
    confidence: float  # 0.0 - 1.0
    notes: str | None


@dataclass
class SourceAnalysis:
    total_files: int
    total_size_bytes: int

    by_extension: dict[str, FileTypeStats]

    content_characteristics: ContentCharacteristics

    parser_recommendations: list[ParserRecommendation]

    uncertainties: list[Uncertainty]

    summary: str  # "247 files, mostly academic PDFs. Some scanned docs detected."
```

---

## PipelineValidator Sub-Agent

Validates a proposed pipeline by running it against samples and investigating any failures.

```python
# webui/services/agent/subagents/pipeline_validator.py

class PipelineValidator(SubAgent):
    AGENT_ID = "pipeline_validator"
    MAX_TURNS = 25

    SYSTEM_PROMPT = """You are a pipeline validation agent for Semantik, a semantic search engine.

Your job: Test a proposed pipeline against sample files and determine if it's ready
for production use.

You have tools to run dry-run validation, inspect failures, and try alternatives.
Work systematically:

1. Run dry-run validation on the provided samples
2. If failures occur, investigate each one to understand why
3. Categorize failures: parser issue, file corruption, unsupported format, config problem
4. For fixable issues, try alternative configurations
5. Assess overall quality: What percentage works? Are failures acceptable edge cases or systemic?

Quality thresholds:
- >95% success: Ready to apply
- 90-95% success: Notable issues, surface to user
- <90% success: Blocking issues, need pipeline adjustment

When done, produce a validation report with:
- Success/failure counts and rates
- Failure categories with examples
- Recommended fixes (if any)
- Overall assessment: ready, needs_adjustment, or blocking_issues

Always end with a structured JSON result in your final message."""

    TOOLS = [
        RunDryRunTool,
        GetFailureDetailsTool,
        TryAlternativeConfigTool,
        CompareParserOutputTool,
        InspectChunksTool,
    ]

    def _build_initial_message(self) -> Message:
        pipeline = self.context["pipeline"]
        sample_files = self.context["sample_files"]

        return Message(
            role="user",
            content=f"""Validate this pipeline configuration:

```json
{json.dumps(pipeline.to_dict(), indent=2)}
```

Test against {len(sample_files)} sample files and report on quality.

Sample files: {json.dumps([f.uri for f in sample_files[:10]])}{'...' if len(sample_files) > 10 else ''}"""
        )

    def _extract_result(self, response: Message) -> SubAgentResult:
        report = self._parse_validation_json(response.content)

        # Convert to uncertainties based on severity
        uncertainties = []
        if report["success_rate"] < 0.90:
            uncertainties.append(Uncertainty(
                severity="blocking",
                message=f"{report['failure_count']} files failed ({100 - report['success_rate']*100:.0f}%)",
                context=report["failure_categories"],
            ))
        elif report["success_rate"] < 0.95:
            uncertainties.append(Uncertainty(
                severity="notable",
                message=f"Some files failed: {report['failure_summary']}",
                context=report["failure_categories"],
            ))

        return SubAgentResult(
            success=True,
            data=report,
            uncertainties=uncertainties,
            summary=report.get("summary", ""),
        )
```

### PipelineValidator Tools

| Tool | Purpose |
|------|---------|
| `RunDryRunTool` | Execute pipeline in dry_run mode on files, return results |
| `GetFailureDetailsTool` | Get detailed error info for a specific failure |
| `TryAlternativeConfigTool` | Re-run a failed file with different parser/config |
| `CompareParserOutputTool` | Compare output of two parsers on same file |
| `InspectChunksTool` | Examine chunk output (sizes, count, content preview) |

### Result Structure

```python
@dataclass
class FailureCategory:
    count: int
    examples: list[str]  # File URIs
    suggested_fix: str | None


@dataclass
class PipelineFix:
    description: str
    pipeline_diff: dict  # What to change


@dataclass
class ValidationReport:
    files_tested: int
    files_succeeded: int
    files_failed: int
    success_rate: float

    failure_categories: dict[str, FailureCategory]

    chunk_stats: ChunkStats  # From dry_run
    stage_timings: dict[str, float]

    assessment: Literal["ready", "needs_adjustment", "blocking_issues"]
    recommended_fixes: list[PipelineFix]

    summary: str  # "98% success. 2 scanned PDFs need OCR - add fallback?"
```

---

## AgentOrchestrator

The main agent that manages conversation flow and delegates to sub-agents.

```python
# webui/services/agent/orchestrator.py

class AgentOrchestrator:
    """Main agent that handles user conversation and spawns sub-agents."""

    SYSTEM_PROMPT = """You are a pipeline configuration assistant for Semantik, a semantic search engine.

You help users set up document ingestion pipelines through conversation. You can:
- Answer questions about parsers, chunkers, embedders, and templates
- Analyze their data sources (by delegating to SourceAnalyzer)
- Recommend pipeline configurations based on their needs
- Validate configurations work (by delegating to PipelineValidator)
- Apply the final pipeline to create a collection

## Tools

You have simple tools for quick queries:
- list_plugins: Get available plugins by type
- get_plugin_details: Get full details on a specific plugin
- list_templates: Get available pipeline templates
- get_pipeline_state: Get current proposed pipeline

For complex tasks, you spawn sub-agents:
- spawn_source_analyzer: Investigate a source (returns structured analysis)
- spawn_pipeline_validator: Validate a pipeline against samples (returns report)

## Conversation Flow

Typical flow:
1. User describes their source and goals
2. You spawn SourceAnalyzer to understand the data
3. You recommend a pipeline based on analysis + user intent
4. You spawn PipelineValidator to verify it works
5. User refines if needed
6. You apply the pipeline

## Guidelines

- Be concise. Users want results, not lectures.
- Surface uncertainties clearly. If something might not work, say so.
- When sub-agents report issues, explain them simply and offer solutions.
- Don't over-explain technical details unless asked.
- If the user's intent is unclear, ask ONE clarifying question."""

    TOOLS = [
        ListPluginsTool,
        GetPluginDetailsTool,
        ListTemplatesTool,
        GetPipelineStateTool,
        SpawnSourceAnalyzerTool,
        SpawnPipelineValidatorTool,
        BuildPipelineTool,
        ApplyPipelineTool,
    ]

    def __init__(
        self,
        conversation: AgentConversation,
        llm_factory: LLMServiceFactory,
        session: AsyncSession,
    ):
        self.conversation = conversation
        self.llm_factory = llm_factory
        self.session = session
        self.redis = get_redis_client()

    async def handle_message(self, user_message: str) -> AsyncIterator[str]:
        """Process user message and yield response chunks (for streaming)."""

        # Load message history from Redis
        messages = await self._load_messages()
        messages.append(Message(role="user", content=user_message))

        # Get LLM provider
        provider = await self.llm_factory.create_provider_for_tier(
            self.conversation.user_id,
            LLMQualityTier.HIGH,
        )

        async with provider:
            while True:
                response = await provider.generate(
                    system=self.SYSTEM_PROMPT,
                    messages=messages,
                    tools=self._tool_schemas(),
                    stream=True,
                )

                # Stream text content to user
                full_response = ""
                async for chunk in response:
                    if chunk.type == "text":
                        yield chunk.content
                        full_response += chunk.content

                messages.append(Message(role="assistant", content=full_response))

                # Handle tool calls
                if response.tool_calls:
                    tool_results = await self._execute_tools(response.tool_calls)
                    messages.append(tool_results)
                    # Continue loop to let LLM process results
                else:
                    break

        # Persist messages to Redis
        await self._save_messages(messages)

        # Update summary if significant state change
        if self._state_changed():
            await self._update_summary(messages)
```

### Orchestrator Spawn Tools

```python
# webui/services/agent/tools/spawn.py

class SpawnSourceAnalyzerTool(BaseTool):
    NAME = "spawn_source_analyzer"
    DESCRIPTION = "Analyze a data source to understand its contents. Returns structured analysis."

    PARAMETERS = {
        "source_id": {"type": "string", "description": "Source to analyze"},
        "user_intent": {"type": "string", "description": "What user wants to do (optional)"},
    }

    async def execute(self, source_id: str, user_intent: str = "") -> dict:
        result = await self.context["orchestrator"]._spawn_subagent(
            SourceAnalyzer,
            context={"source_id": source_id, "user_intent": user_intent},
        )

        # Store analysis in conversation state
        self.context["conversation"].source_analysis = result.data

        return {
            "success": result.success,
            "analysis": result.data,
            "summary": result.summary,
            "uncertainties": [u.message for u in result.uncertainties],
        }


class SpawnPipelineValidatorTool(BaseTool):
    NAME = "spawn_pipeline_validator"
    DESCRIPTION = "Validate a pipeline against sample files. Returns validation report."

    PARAMETERS = {
        "pipeline": {"type": "object", "description": "Pipeline DAG to validate"},
        "sample_count": {"type": "integer", "description": "Number of samples to test (default 50)"},
    }

    async def execute(self, pipeline: dict, sample_count: int = 50) -> dict:
        # Get sample files from source analysis
        source_analysis = self.context["conversation"].source_analysis
        sample_files = self._select_samples(source_analysis, sample_count)

        result = await self.context["orchestrator"]._spawn_subagent(
            PipelineValidator,
            context={"pipeline": pipeline, "sample_files": sample_files},
        )

        return {
            "success": result.success,
            "report": result.data,
            "summary": result.summary,
            "uncertainties": [u.message for u in result.uncertainties],
        }
```

---

## API Endpoints

REST API for managing agent conversations.

```python
# webui/api/v2/agent.py

router = APIRouter(prefix="/api/v2/agent", tags=["agent"])


@router.post("/conversations", response_model=ConversationResponse)
async def create_conversation(
    request: CreateConversationRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Start a new pipeline builder conversation."""

    # Check LLM is configured
    llm_factory = LLMServiceFactory(session)
    if not await llm_factory.has_provider_for_tier(user.id, LLMQualityTier.HIGH):
        raise HTTPException(
            status_code=400,
            detail="LLM provider not configured. Set up an LLM in Settings first.",
        )

    # Verify source exists and user owns it
    source = await get_source(session, request.source_id, user.id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    # Create conversation
    conversation = AgentConversation(
        user_id=user.id,
        source_id=source.id,
        status="active",
    )
    session.add(conversation)
    await session.commit()

    return ConversationResponse(
        id=conversation.id,
        status=conversation.status,
        source_id=conversation.source_id,
        created_at=conversation.created_at,
    )


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: UUID,
    request: SendMessageRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Send a message and stream the response."""

    conversation = await get_conversation(session, conversation_id, user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if conversation.status != "active":
        raise HTTPException(status_code=400, detail="Conversation is not active")

    orchestrator = AgentOrchestrator(
        conversation=conversation,
        llm_factory=LLMServiceFactory(session),
        session=session,
    )

    async def generate():
        async for chunk in orchestrator.handle_message(request.message):
            yield f"data: {json.dumps({'type': 'text', 'content': chunk})}\n\n"
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation_detail(
    conversation_id: UUID,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Get conversation state including current pipeline."""

    conversation = await get_conversation(session, conversation_id, user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Load messages from Redis (may be empty if expired)
    messages = await load_messages_from_redis(conversation_id)

    # Load uncertainties
    uncertainties = await get_uncertainties(session, conversation_id)

    return ConversationDetailResponse(
        id=conversation.id,
        status=conversation.status,
        source_id=conversation.source_id,
        collection_id=conversation.collection_id,
        current_pipeline=conversation.current_pipeline,
        source_analysis=conversation.source_analysis,
        uncertainties=[UncertaintyResponse.from_model(u) for u in uncertainties],
        messages=[MessageResponse.from_model(m) for m in messages],
        summary=conversation.summary,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
    )


@router.post("/conversations/{conversation_id}/apply", response_model=ApplyResponse)
async def apply_pipeline(
    conversation_id: UUID,
    request: ApplyPipelineRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    """Apply the proposed pipeline, creating a collection."""

    conversation = await get_conversation(session, conversation_id, user.id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")

    if not conversation.current_pipeline:
        raise HTTPException(status_code=400, detail="No pipeline configured")

    # Check for blocking uncertainties
    blocking = await get_blocking_uncertainties(session, conversation_id)
    if blocking and not request.force:
        raise HTTPException(
            status_code=400,
            detail=f"Blocking issues unresolved: {blocking[0].message}",
        )

    # Create collection with pipeline
    collection = await create_collection_with_pipeline(
        session=session,
        user_id=user.id,
        source_id=conversation.source_id,
        name=request.collection_name,
        pipeline_config=conversation.current_pipeline,
    )

    # Update conversation
    conversation.collection_id = collection.id
    conversation.status = "applied"
    await session.commit()

    # Trigger indexing
    await dispatch_index_operation(collection.id)

    return ApplyResponse(
        collection_id=collection.id,
        collection_name=collection.name,
        status="indexing",
    )
```

### Request/Response Schemas

```python
# webui/api/v2/schemas/agent.py

class CreateConversationRequest(BaseModel):
    source_id: UUID


class SendMessageRequest(BaseModel):
    message: str


class ApplyPipelineRequest(BaseModel):
    collection_name: str
    force: bool = False  # Apply even with blocking uncertainties


class UncertaintyResponse(BaseModel):
    severity: Literal["blocking", "notable", "info"]
    message: str
    resolved: bool


class ConversationResponse(BaseModel):
    id: UUID
    status: str
    source_id: UUID
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    id: UUID
    status: str
    source_id: UUID
    collection_id: UUID | None
    current_pipeline: dict | None
    source_analysis: dict | None
    uncertainties: list[UncertaintyResponse]
    messages: list[MessageResponse]
    summary: str | None
    created_at: datetime
    updated_at: datetime


class ApplyResponse(BaseModel):
    collection_id: UUID
    collection_name: str
    status: str
```

---

## File Structure

```
packages/webui/
├── api/v2/
│   ├── agent.py                    # API endpoints
│   └── schemas/
│       └── agent.py                # Request/response models
│
├── services/agent/
│   ├── __init__.py
│   ├── orchestrator.py             # AgentOrchestrator
│   ├── models.py                   # AgentConversation, ConversationUncertainty
│   ├── repository.py               # Database operations
│   ├── message_store.py            # Redis message persistence
│   ├── exceptions.py               # AgentError hierarchy
│   │
│   ├── subagents/
│   │   ├── __init__.py
│   │   ├── base.py                 # SubAgent base class, SubAgentResult
│   │   ├── source_analyzer.py      # SourceAnalyzer
│   │   └── pipeline_validator.py   # PipelineValidator
│   │
│   └── tools/
│       ├── __init__.py
│       ├── base.py                 # BaseTool
│       ├── plugins.py              # ListPluginsTool, GetPluginDetailsTool
│       ├── templates.py            # ListTemplatesTool
│       ├── pipeline.py             # GetPipelineStateTool, BuildPipelineTool, ApplyPipelineTool
│       ├── spawn.py                # SpawnSourceAnalyzerTool, SpawnPipelineValidatorTool
│       │
│       └── subagent_tools/
│           ├── __init__.py
│           ├── source.py           # EnumerateFilesTool, SampleFilesTool, etc.
│           └── validation.py       # RunDryRunTool, GetFailureDetailsTool, etc.
```

---

## Database Migration

```python
# alembic/versions/202601XX_add_agent_conversations.py

"""Add agent conversation tables.

Revision ID: xxxx
Revises: 202601231000 (pipeline DAG support)
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON


def upgrade():
    # Agent conversations
    op.create_table(
        'agent_conversations',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('user_id', UUID, sa.ForeignKey('users.id', ondelete='CASCADE'),
                  nullable=False, index=True),
        sa.Column('source_id', UUID, sa.ForeignKey('sources.id', ondelete='SET NULL'),
                  nullable=True, index=True),
        sa.Column('collection_id', UUID, sa.ForeignKey('collections.id', ondelete='SET NULL'),
                  nullable=True),

        # State
        sa.Column('status', sa.String(20), nullable=False, default='active'),
        sa.Column('current_pipeline', JSON, nullable=True),
        sa.Column('source_analysis', JSON, nullable=True),
        sa.Column('summary', sa.Text, nullable=True),

        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now(), onupdate=sa.func.now()),
    )

    # Conversation uncertainties
    op.create_table(
        'conversation_uncertainties',
        sa.Column('id', UUID, primary_key=True),
        sa.Column('conversation_id', UUID,
                  sa.ForeignKey('agent_conversations.id', ondelete='CASCADE'),
                  nullable=False, index=True),

        sa.Column('severity', sa.String(20), nullable=False),  # blocking, notable, info
        sa.Column('message', sa.Text, nullable=False),
        sa.Column('context', JSON, nullable=True),
        sa.Column('resolved', sa.Boolean, nullable=False, default=False),
        sa.Column('resolved_by', sa.String(50), nullable=True),

        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    # Index for finding unresolved blocking issues
    op.create_index(
        'ix_uncertainties_blocking_unresolved',
        'conversation_uncertainties',
        ['conversation_id', 'severity', 'resolved'],
        postgresql_where=sa.text("severity = 'blocking' AND resolved = false"),
    )


def downgrade():
    op.drop_table('conversation_uncertainties')
    op.drop_table('agent_conversations')
```

### Redis Key Structure

```
agent:conversation:{id}:messages     # List of messages, TTL 24h
agent:conversation:{id}:lock         # Mutex for concurrent message handling
agent:subagent:{id}:state            # Sub-agent intermediate state (short TTL)
```

---

## Error Handling

Errors can occur at multiple levels. Each needs appropriate handling.

### Exception Hierarchy

```python
# webui/services/agent/exceptions.py

class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class LLMNotConfiguredError(AgentError):
    """User hasn't configured an LLM provider."""
    pass


class SubAgentFailedError(AgentError):
    """Sub-agent failed to complete its task."""
    def __init__(self, agent_id: str, reason: str, partial_result: dict | None = None):
        self.agent_id = agent_id
        self.reason = reason
        self.partial_result = partial_result


class ConversationNotActiveError(AgentError):
    """Conversation is already applied or abandoned."""
    pass


class BlockingUncertaintyError(AgentError):
    """Cannot proceed due to unresolved blocking issues."""
    def __init__(self, uncertainties: list[Uncertainty]):
        self.uncertainties = uncertainties
```

### Error Handling by Layer

```python
# Orchestrator level - handles sub-agent failures gracefully
class AgentOrchestrator:
    async def _execute_tools(self, tool_calls: list[ToolCall]) -> Message:
        results = []

        for call in tool_calls:
            try:
                result = await self.tools[call.name].execute(**call.arguments)
                results.append(ToolResult(name=call.name, success=True, data=result))

            except SubAgentFailedError as e:
                # Sub-agent failed - return error to LLM so it can handle
                results.append(ToolResult(
                    name=call.name,
                    success=False,
                    error=f"Sub-agent {e.agent_id} failed: {e.reason}",
                    partial_data=e.partial_result,
                ))

            except ToolExecutionError as e:
                # Tool failed - return error to LLM
                results.append(ToolResult(
                    name=call.name,
                    success=False,
                    error=str(e),
                ))

        return Message(role="tool", content=results)


# Sub-agent level - captures failures with context
class SubAgent:
    async def run(self) -> SubAgentResult:
        try:
            # ... agent loop ...

        except LLMProviderError as e:
            # LLM error - can't recover
            raise SubAgentFailedError(
                agent_id=self.AGENT_ID,
                reason=f"LLM error: {e}",
                partial_result=self._get_partial_result(),
            )


# API level - converts to HTTP responses
@router.post("/conversations/{id}/messages")
async def send_message(...):
    try:
        async for chunk in orchestrator.handle_message(request.message):
            yield chunk

    except LLMNotConfiguredError:
        raise HTTPException(400, "LLM not configured")

    except LLMRateLimitError:
        raise HTTPException(429, "Rate limited. Try again shortly.")

    except LLMProviderError as e:
        logger.error(f"LLM error in conversation {id}: {e}")
        raise HTTPException(502, "LLM provider error")
```

### Retry Strategy

| Error Type | Retry? | Strategy |
|------------|--------|----------|
| LLM rate limit | Yes | Exponential backoff, max 3 attempts |
| LLM timeout | Yes | Once, with longer timeout |
| Tool timeout | No | Return error to agent, let it decide |
| Parse error | No | Return error to agent |
| Source inaccessible | No | Return error to agent |

---

## Testing Strategy

### Unit Tests - Isolated Components

```python
# tests/unit/agent/test_orchestrator.py

class TestOrchestrator:
    @pytest.fixture
    def mock_llm(self):
        provider = AsyncMock(spec=LLMProvider)
        provider.generate = AsyncMock(return_value=Message(
            role="assistant",
            content="I'll analyze your source.",
            tool_calls=[ToolCall(name="spawn_source_analyzer", arguments={"source_id": "123"})],
        ))
        return provider

    @pytest.fixture
    def orchestrator(self, mock_llm, db_session):
        conversation = AgentConversation(user_id=uuid4(), source_id=uuid4())
        factory = AsyncMock()
        factory.create_provider_for_tier = AsyncMock(return_value=mock_llm)
        return AgentOrchestrator(conversation, factory, db_session)

    async def test_spawns_source_analyzer_on_analyze_request(self, orchestrator, mock_llm):
        with patch.object(orchestrator, '_spawn_subagent') as mock_spawn:
            mock_spawn.return_value = SubAgentResult(
                success=True,
                data={"total_files": 100},
                summary="Found 100 files",
            )

            chunks = [c async for c in orchestrator.handle_message("Analyze my docs")]

            mock_spawn.assert_called_once()
            assert mock_spawn.call_args[0][0] == SourceAnalyzer
```

### Integration Tests - Real LLM with Mocked Tools

```python
# tests/integration/agent/test_conversation_flow.py

@pytest.mark.integration
class TestConversationFlow:
    async def test_full_conversation_flow(self, api_client, test_source):
        # Create conversation
        resp = await api_client.post("/api/v2/agent/conversations", json={
            "source_id": str(test_source.id),
        })
        conv_id = resp.json()["id"]

        # Send initial message
        resp = await api_client.post(
            f"/api/v2/agent/conversations/{conv_id}/messages",
            json={"message": "Help me index these research papers"},
        )

        # Verify source analysis was triggered
        conv = await api_client.get(f"/api/v2/agent/conversations/{conv_id}")
        assert conv.json()["source_analysis"] is not None
```

### Sub-Agent Tool Tests

```python
# tests/unit/agent/tools/test_source_tools.py

class TestEnumerateFilesTool:
    async def test_returns_file_counts_by_type(self, mock_connector):
        mock_connector.enumerate.return_value = async_iter([
            FileReference(uri="file:///a.pdf", extension=".pdf", ...),
            FileReference(uri="file:///b.pdf", extension=".pdf", ...),
            FileReference(uri="file:///c.md", extension=".md", ...),
        ])

        tool = EnumerateFilesTool({"connector": mock_connector})
        result = await tool.execute()

        assert result["total_files"] == 3
        assert result["by_extension"][".pdf"]["count"] == 2
        assert result["by_extension"][".md"]["count"] == 1
```

---

## Implementation Sub-Phases

### Sub-phase 3a: Foundation (Database + Base Classes)

| Deliverable | Location |
|-------------|----------|
| `AgentConversation` model | `webui/services/agent/models.py` |
| `ConversationUncertainty` model | `webui/services/agent/models.py` |
| Alembic migration | `alembic/versions/` |
| `AgentConversationRepository` | `webui/services/agent/repository.py` |
| Redis message store | `webui/services/agent/message_store.py` |
| `BaseTool` class | `webui/services/agent/tools/base.py` |
| `SubAgent` base class | `webui/services/agent/subagents/base.py` |
| `SubAgentResult`, `Uncertainty` types | `webui/services/agent/subagents/base.py` |

**Acceptance criteria:**
- [ ] Migration runs cleanly
- [ ] Can create/read/update conversations in Postgres
- [ ] Can store/retrieve messages in Redis with TTL
- [ ] Base classes have full type hints and docstrings

**Dependencies:** None (can start immediately)

---

### Sub-phase 3b: Orchestrator Tools

| Deliverable | Location |
|-------------|----------|
| `ListPluginsTool` | `webui/services/agent/tools/plugins.py` |
| `GetPluginDetailsTool` | `webui/services/agent/tools/plugins.py` |
| `ListTemplatesTool` | `webui/services/agent/tools/templates.py` |
| `GetPipelineStateTool` | `webui/services/agent/tools/pipeline.py` |
| `BuildPipelineTool` | `webui/services/agent/tools/pipeline.py` |
| `ApplyPipelineTool` | `webui/services/agent/tools/pipeline.py` |
| Unit tests for each tool | `tests/unit/agent/tools/` |

**Acceptance criteria:**
- [ ] Each tool has schema, execute method, and tests
- [ ] `BuildPipelineTool` produces valid `PipelineDAG`
- [ ] `ApplyPipelineTool` creates collection with pipeline config

**Dependencies:** 3a

---

### Sub-phase 3c: SourceAnalyzer Sub-Agent

| Deliverable | Location |
|-------------|----------|
| `EnumerateFilesTool` | `webui/services/agent/tools/subagent_tools/source.py` |
| `SampleFilesTool` | `webui/services/agent/tools/subagent_tools/source.py` |
| `TryParserTool` | `webui/services/agent/tools/subagent_tools/source.py` |
| `DetectLanguageTool` | `webui/services/agent/tools/subagent_tools/source.py` |
| `GetFileContentPreviewTool` | `webui/services/agent/tools/subagent_tools/source.py` |
| `SourceAnalyzer` sub-agent | `webui/services/agent/subagents/source_analyzer.py` |
| `SpawnSourceAnalyzerTool` | `webui/services/agent/tools/spawn.py` |
| `SourceAnalysis` result type | `webui/services/agent/subagents/source_analyzer.py` |
| Unit + integration tests | `tests/unit/agent/subagents/`, `tests/integration/agent/` |

**Acceptance criteria:**
- [ ] Sub-agent can enumerate and sample files from a source
- [ ] Sub-agent tries parsers and reports success/failure
- [ ] Returns structured `SourceAnalysis` with recommendations
- [ ] Handles large sources (1000+ files) without timeout

**Dependencies:** 3a, 3b (needs BaseTool)

---

### Sub-phase 3d: PipelineValidator Sub-Agent

| Deliverable | Location |
|-------------|----------|
| `RunDryRunTool` | `webui/services/agent/tools/subagent_tools/validation.py` |
| `GetFailureDetailsTool` | `webui/services/agent/tools/subagent_tools/validation.py` |
| `TryAlternativeConfigTool` | `webui/services/agent/tools/subagent_tools/validation.py` |
| `CompareParserOutputTool` | `webui/services/agent/tools/subagent_tools/validation.py` |
| `InspectChunksTool` | `webui/services/agent/tools/subagent_tools/validation.py` |
| `PipelineValidator` sub-agent | `webui/services/agent/subagents/pipeline_validator.py` |
| `SpawnPipelineValidatorTool` | `webui/services/agent/tools/spawn.py` |
| `ValidationReport` result type | `webui/services/agent/subagents/pipeline_validator.py` |
| Unit + integration tests | `tests/unit/agent/subagents/`, `tests/integration/agent/` |

**Acceptance criteria:**
- [ ] Sub-agent runs dry_run on sample files
- [ ] Investigates failures and categorizes them
- [ ] Returns structured `ValidationReport` with assessment
- [ ] Suggests fixes for common failure patterns

**Dependencies:** 3a, 3b, Phase 1 executor (dry_run mode)

---

### Sub-phase 3e: Orchestrator + API

| Deliverable | Location |
|-------------|----------|
| `AgentOrchestrator` class | `webui/services/agent/orchestrator.py` |
| API endpoints | `webui/api/v2/agent.py` |
| Request/response schemas | `webui/api/v2/schemas/agent.py` |
| Streaming response handling | `webui/api/v2/agent.py` |
| Conversation recovery logic | `webui/services/agent/orchestrator.py` |
| Error handling | `webui/services/agent/exceptions.py` |
| Integration tests | `tests/integration/agent/` |

**Acceptance criteria:**
- [ ] Can create conversation and send messages
- [ ] Responses stream correctly via SSE
- [ ] Orchestrator spawns sub-agents appropriately
- [ ] Conversation state persists across requests
- [ ] Recovery works when Redis messages expire

**Dependencies:** 3a, 3b, 3c, 3d

---

### Sub-phase 3f: Chat UI

| Deliverable | Location |
|-------------|----------|
| `AgentChat` component | `apps/webui-react/src/components/agent/AgentChat.tsx` |
| Message list component | `apps/webui-react/src/components/agent/MessageList.tsx` |
| Pipeline preview panel | `apps/webui-react/src/components/agent/PipelinePreview.tsx` |
| Uncertainty display | `apps/webui-react/src/components/agent/UncertaintyBanner.tsx` |
| API hooks | `apps/webui-react/src/hooks/useAgentConversation.ts` |
| SSE streaming hook | `apps/webui-react/src/hooks/useAgentStream.ts` |
| Entry point in collection creation | Modified `CollectionCreate.tsx` |

**Acceptance criteria:**
- [ ] Can start conversation from "Guided setup" button
- [ ] Messages stream in real-time
- [ ] Current pipeline shown in side panel
- [ ] Uncertainties displayed with appropriate severity styling
- [ ] Can apply pipeline from UI

**Dependencies:** 3e (API must be complete)

---

### Dependency Graph

```
3a (Foundation)
 │
 ├── 3b (Orchestrator Tools)
 │    │
 │    ├── 3c (SourceAnalyzer)
 │    │    │
 │    │    └──┬── 3e (Orchestrator + API)
 │    │       │
 │    └── 3d (PipelineValidator)
 │              │
 │              └── 3f (Chat UI)
```

**Parallel work possible:**
- 3c and 3d can run in parallel after 3b
- 3f can start UI scaffolding while 3e is in progress

---

## Complete Deliverables Checklist

### Sub-phase 3a: Foundation
- [ ] `webui/services/agent/__init__.py`
- [ ] `webui/services/agent/models.py`
- [ ] `webui/services/agent/repository.py`
- [ ] `webui/services/agent/message_store.py`
- [ ] `webui/services/agent/exceptions.py`
- [ ] `webui/services/agent/tools/__init__.py`
- [ ] `webui/services/agent/tools/base.py`
- [ ] `webui/services/agent/subagents/__init__.py`
- [ ] `webui/services/agent/subagents/base.py`
- [ ] `alembic/versions/XXXXXX_add_agent_conversations.py`
- [ ] `tests/unit/agent/test_models.py`
- [ ] `tests/unit/agent/test_repository.py`
- [ ] `tests/unit/agent/test_message_store.py`

### Sub-phase 3b: Orchestrator Tools
- [ ] `webui/services/agent/tools/plugins.py`
- [ ] `webui/services/agent/tools/templates.py`
- [ ] `webui/services/agent/tools/pipeline.py`
- [ ] `tests/unit/agent/tools/test_plugins.py`
- [ ] `tests/unit/agent/tools/test_templates.py`
- [ ] `tests/unit/agent/tools/test_pipeline.py`

### Sub-phase 3c: SourceAnalyzer Sub-Agent
- [ ] `webui/services/agent/tools/subagent_tools/__init__.py`
- [ ] `webui/services/agent/tools/subagent_tools/source.py`
- [ ] `webui/services/agent/subagents/source_analyzer.py`
- [ ] `webui/services/agent/tools/spawn.py`
- [ ] `tests/unit/agent/tools/subagent_tools/test_source.py`
- [ ] `tests/unit/agent/subagents/test_source_analyzer.py`
- [ ] `tests/integration/agent/test_source_analyzer.py`

### Sub-phase 3d: PipelineValidator Sub-Agent
- [ ] `webui/services/agent/tools/subagent_tools/validation.py`
- [ ] `webui/services/agent/subagents/pipeline_validator.py`
- [ ] `webui/services/agent/tools/spawn.py` (add SpawnPipelineValidatorTool)
- [ ] `tests/unit/agent/tools/subagent_tools/test_validation.py`
- [ ] `tests/unit/agent/subagents/test_pipeline_validator.py`
- [ ] `tests/integration/agent/test_pipeline_validator.py`

### Sub-phase 3e: Orchestrator + API
- [ ] `webui/services/agent/orchestrator.py`
- [ ] `webui/api/v2/agent.py`
- [ ] `webui/api/v2/schemas/agent.py`
- [ ] Register router in `webui/api/v2/__init__.py`
- [ ] `tests/unit/agent/test_orchestrator.py`
- [ ] `tests/integration/agent/test_api.py`
- [ ] `tests/integration/agent/test_conversation_flow.py`

### Sub-phase 3f: Chat UI
- [ ] `apps/webui-react/src/components/agent/AgentChat.tsx`
- [ ] `apps/webui-react/src/components/agent/MessageList.tsx`
- [ ] `apps/webui-react/src/components/agent/MessageBubble.tsx`
- [ ] `apps/webui-react/src/components/agent/PipelinePreview.tsx`
- [ ] `apps/webui-react/src/components/agent/UncertaintyBanner.tsx`
- [ ] `apps/webui-react/src/components/agent/index.ts`
- [ ] `apps/webui-react/src/hooks/useAgentConversation.ts`
- [ ] `apps/webui-react/src/hooks/useAgentStream.ts`
- [ ] `apps/webui-react/src/api/agent.ts`
- [ ] Modify `apps/webui-react/src/pages/CollectionCreate.tsx`
- [ ] `apps/webui-react/src/pages/AgentChat.tsx`
- [ ] Add route in `apps/webui-react/src/router.tsx`
- [ ] `apps/webui-react/src/__tests__/agent/AgentChat.test.tsx`

---

## Summary

| Sub-phase | Files | Tests | Estimate |
|-----------|-------|-------|----------|
| 3a Foundation | 10 | 3 | 2-3 days |
| 3b Orchestrator Tools | 3 | 3 | 2 days |
| 3c SourceAnalyzer | 3 | 3 | 3-4 days |
| 3d PipelineValidator | 3 | 3 | 3-4 days |
| 3e Orchestrator + API | 4 | 3 | 3-4 days |
| 3f Chat UI | 12 | 1 | 3-4 days |
| **Total** | **35** | **16** | **16-21 days** |

**With parallelization (3c + 3d concurrent):** 13-17 days

---

## Key Design Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| Sub-agent pattern | Orchestrator + focused sub-agents | Context isolation for intensive tasks; clean separation |
| Sub-agent count | Start with 2 (SourceAnalyzer, PipelineValidator) | Cover main use cases; add more if needed |
| State storage | Postgres durable + Redis ephemeral | Messages can expire; core state persists |
| Streaming | SSE for responses | Simple, well-supported, works with proxies |
| LLM requirement | Gate at conversation creation | Fail fast; no degraded mode |
| Uncertainty model | Three severities (blocking/notable/info) | Clear escalation; blocking must be resolved |
| Sub-agent timeout | 5 minutes max | Prevent runaway; return partial results |
| Tool errors | Return to LLM, let it decide | Agent can retry or inform user |

---

## Open Questions

1. **Sub-agent parallelism:** Should we run SourceAnalyzer and PipelineValidator concurrently when possible? Adds complexity but could speed up flow.

2. **Conversation persistence:** How long to keep applied/abandoned conversations? Suggest 30 days, then archive.

3. **Cost tracking:** Should we track LLM token usage per conversation? Useful for billing/analytics.

4. **Rate limiting:** Per-user rate limits on agent conversations? Prevents abuse.
