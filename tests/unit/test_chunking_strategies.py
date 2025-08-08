#!/usr/bin/env python3
"""
Unit tests for ChunkingStrategyRegistry.

This module tests the chunking strategy registry and its methods.
"""


from packages.webui.api.v2.chunking_schemas import ChunkingStrategy
from packages.webui.services.chunking_strategies import ChunkingStrategyRegistry


class TestChunkingStrategyRegistry:
    """Test suite for ChunkingStrategyRegistry."""

    def test_get_strategy_definition_valid(self):
        """Test getting a valid strategy definition."""
        # Test each defined strategy
        for strategy in ChunkingStrategy:
            definition = ChunkingStrategyRegistry.get_strategy_definition(strategy)

            # All strategies should have definitions
            assert definition is not None
            assert isinstance(definition, dict)

            # Check required fields
            assert "name" in definition
            assert "description" in definition
            assert "best_for" in definition
            assert "pros" in definition
            assert "cons" in definition
            assert "performance_characteristics" in definition

            # Validate field types
            assert isinstance(definition["name"], str)
            assert isinstance(definition["description"], str)
            assert isinstance(definition["best_for"], list)
            assert isinstance(definition["pros"], list)
            assert isinstance(definition["cons"], list)
            assert isinstance(definition["performance_characteristics"], dict)

    def test_get_strategy_definition_fixed_size(self):
        """Test getting the FIXED_SIZE strategy definition."""
        definition = ChunkingStrategyRegistry.get_strategy_definition(ChunkingStrategy.FIXED_SIZE)

        assert definition["name"] == "Fixed Size Chunking"
        assert "Simple fixed-size chunking" in definition["description"]
        assert "txt" in definition["best_for"]
        assert "Predictable chunk sizes" in definition["pros"]
        assert "May split sentences" in definition["cons"][0]
        assert definition["performance_characteristics"]["speed"] == "very_fast"
        assert definition["performance_characteristics"]["memory_usage"] == "low"

    def test_get_strategy_definition_semantic(self):
        """Test getting the SEMANTIC strategy definition."""
        definition = ChunkingStrategyRegistry.get_strategy_definition(ChunkingStrategy.SEMANTIC)

        assert definition["name"] == "Semantic"
        assert "embeddings" in definition["description"]
        assert "pdf" in definition["best_for"]
        assert "Maintains semantic coherence" in definition["pros"]
        assert definition["performance_characteristics"]["quality"] == "excellent"

    def test_get_all_definitions(self):
        """Test getting all strategy definitions."""
        all_definitions = ChunkingStrategyRegistry.get_all_definitions()

        # Should return a dictionary
        assert isinstance(all_definitions, dict)

        # Should contain all strategies
        assert len(all_definitions) == len(ChunkingStrategy)

        # All strategies should be present
        for strategy in ChunkingStrategy:
            assert strategy in all_definitions
            assert isinstance(all_definitions[strategy], dict)

        # Verify it's a copy (not the original)
        all_definitions[ChunkingStrategy.FIXED_SIZE]["test_field"] = "test"
        original = ChunkingStrategyRegistry.STRATEGY_DEFINITIONS[ChunkingStrategy.FIXED_SIZE]
        assert "test_field" not in original

    def test_get_recommended_strategy_pdf_files(self):
        """Test getting recommended strategy for PDF files."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["pdf"])

        # PDF is best for SEMANTIC or DOCUMENT_STRUCTURE
        assert strategy in [ChunkingStrategy.SEMANTIC, ChunkingStrategy.DOCUMENT_STRUCTURE]

    def test_get_recommended_strategy_text_files(self):
        """Test getting recommended strategy for text files."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["txt"])

        # TXT can use multiple strategies
        assert strategy in [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.RECURSIVE, ChunkingStrategy.SLIDING_WINDOW]

    def test_get_recommended_strategy_mixed_files(self):
        """Test getting recommended strategy for mixed file types."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["pdf", "txt", "md"])

        # Should return a strategy that handles multiple types well
        assert strategy in ChunkingStrategy

    def test_get_recommended_strategy_unknown_files(self):
        """Test getting recommended strategy for unknown file types."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["xyz", "abc"])

        # Should default to RECURSIVE
        assert strategy == ChunkingStrategy.RECURSIVE

    def test_get_recommended_strategy_empty_list(self):
        """Test getting recommended strategy with empty file list."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy([])

        # Should default to RECURSIVE
        assert strategy == ChunkingStrategy.RECURSIVE

    def test_get_recommended_strategy_code_files(self):
        """Test getting recommended strategy for code files."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["py", "js", "code files"])

        # Code files are best with RECURSIVE
        assert strategy == ChunkingStrategy.RECURSIVE

    def test_get_recommended_strategy_structured_documents(self):
        """Test getting recommended strategy for structured documents."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["docx", "html", "epub"])

        # Structured documents work well with DOCUMENT_STRUCTURE
        assert strategy == ChunkingStrategy.DOCUMENT_STRUCTURE

    def test_get_recommended_strategy_logs(self):
        """Test getting recommended strategy for log files."""
        strategy = ChunkingStrategyRegistry.get_recommended_strategy(["log"])

        # Logs can use FIXED_SIZE or SLIDING_WINDOW
        assert strategy in [ChunkingStrategy.FIXED_SIZE, ChunkingStrategy.SLIDING_WINDOW]

    def test_strategy_performance_characteristics(self):
        """Test that all strategies have valid performance characteristics."""
        for strategy in ChunkingStrategy:
            definition = ChunkingStrategyRegistry.get_strategy_definition(strategy)
            perf = definition["performance_characteristics"]

            # Check that performance metrics are defined
            assert "speed" in perf
            assert "memory_usage" in perf
            assert "quality" in perf

            # Validate speed values
            valid_speeds = ["very_fast", "fast", "moderate", "slow"]
            assert perf["speed"] in valid_speeds

            # Validate memory usage values
            valid_memory = ["low", "moderate", "high", "very_high"]
            assert perf["memory_usage"] in valid_memory

            # Validate quality values
            valid_quality = ["moderate", "good", "very_good", "excellent"]
            assert perf["quality"] in valid_quality

    def test_strategy_best_for_completeness(self):
        """Test that common file types are covered by strategies."""
        common_file_types = ["txt", "pdf", "docx", "md", "html", "json", "csv", "log"]

        for file_type in common_file_types:
            # Check that at least one strategy handles this file type
            found = False
            for strategy in ChunkingStrategy:
                definition = ChunkingStrategyRegistry.get_strategy_definition(strategy)
                if file_type in definition.get("best_for", []):
                    found = True
                    break

            assert found, f"No strategy found for file type: {file_type}"

    def test_hybrid_strategy_characteristics(self):
        """Test specific characteristics of HYBRID strategy."""
        definition = ChunkingStrategyRegistry.get_strategy_definition(ChunkingStrategy.HYBRID)

        assert definition["name"] == "Hybrid"
        assert "mixed content" in definition["best_for"]
        assert "Adaptive to content" in definition["pros"]
        assert definition["performance_characteristics"]["memory_usage"] == "very_high"
        assert definition["performance_characteristics"]["quality"] == "excellent"

    def test_sliding_window_strategy_characteristics(self):
        """Test specific characteristics of SLIDING_WINDOW strategy."""
        definition = ChunkingStrategyRegistry.get_strategy_definition(ChunkingStrategy.SLIDING_WINDOW)

        assert definition["name"] == "Sliding Window"
        assert "transcript" in definition["best_for"]
        assert "No information loss at boundaries" in definition["pros"]
        assert "Redundant information" in definition["cons"]

    def test_document_structure_strategy_characteristics(self):
        """Test specific characteristics of DOCUMENT_STRUCTURE strategy."""
        definition = ChunkingStrategyRegistry.get_strategy_definition(ChunkingStrategy.DOCUMENT_STRUCTURE)

        assert definition["name"] == "Document Structure"
        assert "epub" in definition["best_for"]
        assert "tex" in definition["best_for"]
        assert "Preserves document hierarchy" in definition["pros"]
        assert definition["performance_characteristics"]["quality"] == "very_good"

    def test_recursive_strategy_characteristics(self):
        """Test specific characteristics of RECURSIVE strategy."""
        definition = ChunkingStrategyRegistry.get_strategy_definition(ChunkingStrategy.RECURSIVE)

        assert definition["name"] == "Recursive"
        assert "rst" in definition["best_for"]
        assert "Respects document structure" in definition["pros"]
        assert definition["performance_characteristics"]["speed"] == "fast"
