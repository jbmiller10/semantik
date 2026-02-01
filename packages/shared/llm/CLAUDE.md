<!-- IMPORTANT: If you make any changes that affect the information in this CLAUDE.md file,
     you MUST update this documentation accordingly. This includes:
     - Adding/removing LLM providers
     - Changing the quality tier system
     - Modifying the factory pattern
     - Altering exception handling
     - Updating the model registry
     - Changing HyDE implementation
     Keep this documentation in sync with the actual implementation! -->

<component>
  <name>LLM Service Layer</name>
  <purpose>Multi-provider LLM abstraction with quality tiers for HyDE search, summarization, and AI features</purpose>
  <location>packages/shared/llm/</location>
</component>

<architecture>
  <pattern>Factory + Provider abstraction with quality-tier based model selection</pattern>
  <key-principle>Caller owns provider lifecycle via async context manager</key-principle>
  <data-flow>
    1. Feature requests provider for quality tier (HIGH or LOW)
    2. Factory loads user's LLMProviderConfig from database
    3. Factory selects provider + model for tier, decrypts API key
    4. Provider initialized and returned to caller
    5. Caller uses `async with provider:` for automatic cleanup
  </data-flow>
</architecture>

<providers>
  <provider name="anthropic">
    <class>AnthropicLLMProvider</class>
    <models>Claude Opus 4.5 (high), Sonnet 4.5 (low)</models>
    <requires>API key from user settings</requires>
    <timeout>60s default, capped at 120s</timeout>
  </provider>

  <provider name="openai">
    <class>OpenAILLMProvider</class>
    <models>GPT-4o (high), GPT-4o Mini (low)</models>
    <requires>API key from user settings</requires>
    <timeout>60s default, capped at 120s</timeout>
  </provider>

  <provider name="local">
    <class>LocalLLMProvider</class>
    <models>Qwen 2.5 family (0.5B-7B)</models>
    <requires>Nothing - calls VecPipe internally</requires>
    <timeout>120s default (model loading time)</timeout>
    <critical>
      LocalLLMProvider is an HTTP client to VecPipe, NOT direct GPU access.
      GPU memory is managed by VecPipe's GPUMemoryGovernor.
      Flow: WebUI → LocalLLMProvider → HTTP → VecPipe /llm/generate → GPU
    </critical>
  </provider>
</providers>

<quality-tiers>
  <tier name="HIGH">
    <purpose>Complex tasks requiring best quality</purpose>
    <use-cases>Document summarization, entity extraction, complex reasoning</use-cases>
    <cost>Higher token costs, slower responses</cost>
  </tier>
  <tier name="LOW">
    <purpose>Simple tasks where speed/cost matter</purpose>
    <use-cases>HyDE query expansion, keyword extraction</use-cases>
    <cost>Lower token costs, faster responses</cost>
  </tier>
  <selection>
    Users configure provider + model per tier in WebUI settings.
    Features explicitly request which tier they need.
  </selection>
</quality-tiers>

<exceptions>
  <hierarchy>
    LLMError (base)
    ├── LLMNotConfiguredError: User hasn't set up LLM settings
    │   → Show settings prompt, don't retry
    ├── LLMAuthenticationError: Invalid or missing API key
    │   → Show re-auth prompt, don't retry
    ├── LLMRateLimitError: Provider rate limit exceeded
    │   → Retryable, has retry_after field
    ├── LLMProviderError: General API/network error
    │   → May retry with backoff, check status_code
    ├── LLMTimeoutError: Request timed out
    │   → Retryable with longer timeout
    └── LLMContextLengthError: Input exceeds model context
        → Reduce input size, don't retry same request
  </hierarchy>

  <retry-guidance>
    <retryable>LLMRateLimitError, LLMTimeoutError, LLMProviderError (5xx)</retryable>
    <not-retryable>LLMNotConfiguredError, LLMAuthenticationError, LLMContextLengthError</not-retryable>
    <special-case>
      LLMProviderError with status_code=507 from local provider means GPU OOM.
      Consider smaller model or lower quantization, not simple retry.
    </special-case>
  </retry-guidance>
</exceptions>

<factory-usage>
  <pattern>
    from shared.llm.factory import LLMServiceFactory
    from shared.llm.types import LLMQualityTier

    factory = LLMServiceFactory(session)
    provider = await factory.create_provider_for_tier(user_id, LLMQualityTier.LOW)

    async with provider:
        response = await provider.generate(
            prompt="Your prompt here",
            system_prompt="Optional system prompt",
            max_tokens=256,
        )
        print(response.content)
  </pattern>

  <critical-rules>
    1. ALWAYS use async context manager (`async with provider:`) for cleanup
    2. Factory requires active database session for config lookup
    3. Provider is bound to ONE model - create new provider for different tier
    4. Check has_provider_configured() before features that require LLM
  </critical-rules>
</factory-usage>

<hyde-integration>
  <purpose>Generate hypothetical documents for improved semantic search</purpose>
  <location>hyde.py</location>
  <usage>
    from shared.llm.hyde import HyDEConfig, generate_hyde_expansion

    # Provider must already be initialized (caller manages lifecycle)
    async with provider:
        config = HyDEConfig(timeout_seconds=30, max_tokens=256)
        result, response = await generate_hyde_expansion(provider, query, config=config)

        if result.success:
            dense_query = result.expanded_query  # Use for embedding
        else:
            dense_query = result.original_query  # Fallback gracefully
  </usage>
  <graceful-degradation>
    HyDE failures return original query with success=False and warning message.
    Search continues without HyDE enhancement - never fails the search.
  </graceful-degradation>
</hyde-integration>

<model-registry>
  <location>model_registry.yaml + model_registry.py</location>
  <purpose>Curated list of recommended models with memory estimates</purpose>
  <functions>
    - get_default_model(provider, tier): Default model for provider/tier
    - get_all_models(): Flat list for API/UI
    - get_model_by_id(model_id): Lookup model metadata
  </functions>
  <local-model-memory>
    Local models include memory_mb estimates per quantization:
    - float16: Full precision (highest quality, most memory)
    - int8: 8-bit quantization (good balance)
    - int4: 4-bit quantization (lowest memory, some quality loss)
  </local-model-memory>
</model-registry>

<usage-tracking>
  <purpose>Record token usage for billing/analytics</purpose>
  <location>usage_tracking.py</location>
  <usage>
    from shared.llm.usage_tracking import record_llm_usage

    response = await provider.generate(prompt="...")
    await record_llm_usage(
        session,
        user_id=user_id,
        response=response,
        feature="hyde",        # Feature name for grouping
        quality_tier="low",    # Tier used
    )
  </usage>
</usage-tracking>

<common-pitfalls>
  <pitfall>
    <issue>Forgetting async context manager</issue>
    <wrong>
      provider = await factory.create_provider_for_tier(user_id, tier)
      response = await provider.generate(prompt)
      # HTTP client never closed!
    </wrong>
    <correct>
      provider = await factory.create_provider_for_tier(user_id, tier)
      async with provider:
          response = await provider.generate(prompt)
      # Cleanup happens automatically
    </correct>
  </pitfall>

  <pitfall>
    <issue>Assuming local provider has direct GPU access</issue>
    <consequence>Confusion about memory management and errors</consequence>
    <reality>LocalLLMProvider is HTTP client → VecPipe handles GPU</reality>
  </pitfall>

  <pitfall>
    <issue>Not handling LLMNotConfiguredError in user-facing features</issue>
    <consequence>Cryptic errors instead of "Please configure LLM settings"</consequence>
    <solution>Catch and show helpful configuration prompt</solution>
  </pitfall>

  <pitfall>
    <issue>Using wrong quality tier for feature</issue>
    <consequence>Overpaying for simple tasks OR poor quality on complex ones</consequence>
    <guidance>HyDE/keywords → LOW, summaries/extraction → HIGH</guidance>
  </pitfall>

  <pitfall>
    <issue>Retrying LLMContextLengthError without reducing input</issue>
    <consequence>Infinite retry loop</consequence>
    <solution>Truncate or chunk input before retry</solution>
  </pitfall>
</common-pitfalls>

<development>
  <commands>
    - Test: `uv run pytest tests/shared/llm/`
    - Type check: `uv run mypy packages/shared/llm/`
  </commands>
</development>
