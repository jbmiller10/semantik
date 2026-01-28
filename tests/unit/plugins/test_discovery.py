"""Unit tests for the Plugin Discovery API."""

from unittest.mock import MagicMock, patch

from shared.plugins.discovery import (
    _calculate_overlap_score,
    _has_overlapping_inputs,
    _input_type_specificity,
    _matches_input_type,
    _patterns_overlap,
    find_plugins_for_input,
    get_alternative_plugins,
    list_plugins_for_agent,
)
from shared.plugins.manifest import AgentHints, PluginManifest


class TestMatchesInputType:
    """Tests for _matches_input_type helper function."""

    def test_exact_match(self):
        """Test exact MIME type matching."""
        assert _matches_input_type("application/pdf", ["application/pdf"])
        assert _matches_input_type("text/plain", ["text/plain"])

    def test_exact_match_case_insensitive(self):
        """Test case insensitivity."""
        assert _matches_input_type("Application/PDF", ["application/pdf"])
        assert _matches_input_type("text/plain", ["TEXT/PLAIN"])

    def test_wildcard_minor_type(self):
        """Test wildcard minor type matching (e.g., text/*)."""
        assert _matches_input_type("text/plain", ["text/*"])
        assert _matches_input_type("text/html", ["text/*"])
        assert not _matches_input_type("application/pdf", ["text/*"])

    def test_wildcard_major_type(self):
        """Test wildcard major type matching (e.g., */*)."""
        assert _matches_input_type("text/plain", ["*/*"])
        assert _matches_input_type("application/pdf", ["*/*"])
        assert _matches_input_type("image/png", ["*/*"])

    def test_multiple_accepted_types(self):
        """Test matching against multiple accepted types."""
        accepted = ["text/plain", "application/pdf", "text/html"]
        assert _matches_input_type("text/plain", accepted)
        assert _matches_input_type("application/pdf", accepted)
        assert _matches_input_type("text/html", accepted)
        assert not _matches_input_type("image/png", accepted)

    def test_no_match(self):
        """Test when there's no match."""
        assert not _matches_input_type("image/png", ["text/plain", "application/pdf"])


class TestPatternsOverlap:
    """Tests for _patterns_overlap helper function."""

    def test_exact_match(self):
        """Test exact pattern equality."""
        assert _patterns_overlap("text/plain", "text/plain")
        assert _patterns_overlap("application/pdf", "application/pdf")

    def test_wildcard_overlap(self):
        """Test wildcard pattern overlap."""
        assert _patterns_overlap("text/plain", "text/*")
        assert _patterns_overlap("text/*", "text/html")
        assert _patterns_overlap("text/*", "text/*")

    def test_full_wildcard_overlap(self):
        """Test full wildcard overlap."""
        assert _patterns_overlap("*/*", "text/plain")
        assert _patterns_overlap("application/pdf", "*/*")

    def test_no_overlap(self):
        """Test no overlap between different types."""
        assert not _patterns_overlap("text/plain", "application/pdf")
        assert not _patterns_overlap("image/png", "text/*")


class TestHasOverlappingInputs:
    """Tests for _has_overlapping_inputs helper function."""

    def test_direct_overlap(self):
        """Test direct MIME type overlap."""
        a = ["text/plain", "text/html"]
        b = ["text/plain", "application/pdf"]
        assert _has_overlapping_inputs(a, b)

    def test_wildcard_overlap(self):
        """Test overlap via wildcard."""
        a = ["text/*"]
        b = ["text/plain"]
        assert _has_overlapping_inputs(a, b)

    def test_no_overlap(self):
        """Test lists with no overlap."""
        a = ["text/plain", "text/html"]
        b = ["application/pdf", "image/png"]
        assert not _has_overlapping_inputs(a, b)

    def test_empty_lists(self):
        """Test empty input lists."""
        assert not _has_overlapping_inputs([], ["text/plain"])
        assert not _has_overlapping_inputs(["text/plain"], [])
        assert not _has_overlapping_inputs([], [])


class TestCalculateOverlapScore:
    """Tests for _calculate_overlap_score helper function."""

    def test_single_overlap(self):
        """Test single overlapping pattern."""
        a = ["text/plain"]
        b = ["text/plain"]
        assert _calculate_overlap_score(a, b) == 1

    def test_multiple_overlaps(self):
        """Test multiple overlapping patterns."""
        a = ["text/plain", "text/html"]
        b = ["text/plain", "text/html", "text/css"]
        # text/plain matches text/plain (1)
        # text/html matches text/html (1)
        assert _calculate_overlap_score(a, b) >= 2

    def test_wildcard_increases_score(self):
        """Test that wildcard patterns can match multiple entries."""
        a = ["text/*"]
        b = ["text/plain", "text/html"]
        # text/* matches both text/plain and text/html
        assert _calculate_overlap_score(a, b) == 2

    def test_no_overlap(self):
        """Test zero score when no overlap."""
        a = ["text/plain"]
        b = ["application/pdf"]
        assert _calculate_overlap_score(a, b) == 0


class TestInputTypeSpecificity:
    """Tests for _input_type_specificity helper function."""

    def test_exact_match_highest_specificity(self):
        """Test exact match gives highest score (100)."""
        manifest = PluginManifest(
            id="test",
            type="parser",
            version="1.0.0",
            display_name="Test",
            description="Test",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
                input_types=["application/pdf"],
            ),
        )
        assert _input_type_specificity("application/pdf", manifest) == 100

    def test_wildcard_minor_medium_specificity(self):
        """Test wildcard minor type gives medium score (50)."""
        manifest = PluginManifest(
            id="test",
            type="parser",
            version="1.0.0",
            display_name="Test",
            description="Test",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
                input_types=["text/*"],
            ),
        )
        assert _input_type_specificity("text/plain", manifest) == 50

    def test_full_wildcard_lowest_specificity(self):
        """Test full wildcard gives lowest score (10)."""
        manifest = PluginManifest(
            id="test",
            type="parser",
            version="1.0.0",
            display_name="Test",
            description="Test",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
                input_types=["*/*"],
            ),
        )
        assert _input_type_specificity("application/pdf", manifest) == 10

    def test_no_agent_hints_zero_specificity(self):
        """Test missing agent_hints gives zero score."""
        manifest = PluginManifest(
            id="test",
            type="parser",
            version="1.0.0",
            display_name="Test",
            description="Test",
        )
        assert _input_type_specificity("application/pdf", manifest) == 0

    def test_no_input_types_zero_specificity(self):
        """Test missing input_types gives zero score."""
        manifest = PluginManifest(
            id="test",
            type="parser",
            version="1.0.0",
            display_name="Test",
            description="Test",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
                # No input_types
            ),
        )
        assert _input_type_specificity("application/pdf", manifest) == 0


class TestListPluginsForAgent:
    """Tests for list_plugins_for_agent function."""

    def test_returns_plugins_with_agent_hints(self):
        """Test that only plugins with agent_hints are returned."""
        manifest_with_hints = PluginManifest(
            id="with-hints",
            type="parser",
            version="1.0.0",
            display_name="With Hints",
            description="Has hints",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
            ),
        )
        manifest_without_hints = PluginManifest(
            id="without-hints",
            type="parser",
            version="1.0.0",
            display_name="Without Hints",
            description="No hints",
        )

        mock_record_with = MagicMock()
        mock_record_with.manifest = manifest_with_hints

        mock_record_without = MagicMock()
        mock_record_without.manifest = manifest_without_hints

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [mock_record_with, mock_record_without]

            result = list_plugins_for_agent()

            assert len(result) == 1
            assert result[0].id == "with-hints"

    def test_filters_by_plugin_type(self):
        """Test filtering by plugin type."""
        manifest_parser = PluginManifest(
            id="parser-plugin",
            type="parser",
            version="1.0.0",
            display_name="Parser",
            description="Parser",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
            ),
        )
        manifest_chunking = PluginManifest(
            id="chunking-plugin",
            type="chunking",
            version="1.0.0",
            display_name="Chunking",
            description="Chunking",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["test"],
                not_recommended_for=["test"],
            ),
        )

        mock_record_parser = MagicMock()
        mock_record_parser.manifest = manifest_parser

        mock_record_chunking = MagicMock()
        mock_record_chunking.manifest = manifest_chunking

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [mock_record_parser]

            result = list_plugins_for_agent(plugin_type="parser")

            mock_registry.list_records.assert_called_with(plugin_type="parser")
            assert len(result) == 1
            assert result[0].id == "parser-plugin"

    def test_returns_sorted_by_type_and_id(self):
        """Test that results are sorted by type then by ID."""
        manifests = [
            PluginManifest(
                id="zebra",
                type="parser",
                version="1.0.0",
                display_name="Zebra",
                description="Z",
                agent_hints=AgentHints(purpose="test", best_for=["x"], not_recommended_for=["y"]),
            ),
            PluginManifest(
                id="alpha",
                type="parser",
                version="1.0.0",
                display_name="Alpha",
                description="A",
                agent_hints=AgentHints(purpose="test", best_for=["x"], not_recommended_for=["y"]),
            ),
            PluginManifest(
                id="beta",
                type="chunking",
                version="1.0.0",
                display_name="Beta",
                description="B",
                agent_hints=AgentHints(purpose="test", best_for=["x"], not_recommended_for=["y"]),
            ),
        ]

        mock_records = [MagicMock(manifest=m) for m in manifests]

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = mock_records

            result = list_plugins_for_agent()

            # Should be sorted by type first (chunking before parser), then by ID
            assert result[0].id == "beta"  # chunking
            assert result[1].id == "alpha"  # parser
            assert result[2].id == "zebra"  # parser


class TestFindPluginsForInput:
    """Tests for find_plugins_for_input function."""

    def test_finds_plugins_matching_mime_type(self):
        """Test finding plugins that match a MIME type."""
        manifest = PluginManifest(
            id="pdf-parser",
            type="parser",
            version="1.0.0",
            display_name="PDF Parser",
            description="Parses PDFs",
            agent_hints=AgentHints(
                purpose="parse pdfs",
                best_for=["pdfs"],
                not_recommended_for=["text"],
                input_types=["application/pdf"],
            ),
        )

        mock_record = MagicMock()
        mock_record.manifest = manifest

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [mock_record]

            result = find_plugins_for_input("application/pdf")

            assert len(result) == 1
            assert result[0].id == "pdf-parser"

    def test_excludes_plugins_without_matching_input(self):
        """Test that plugins without matching input are excluded."""
        manifest = PluginManifest(
            id="text-parser",
            type="parser",
            version="1.0.0",
            display_name="Text Parser",
            description="Parses text",
            agent_hints=AgentHints(
                purpose="parse text",
                best_for=["text"],
                not_recommended_for=["binary"],
                input_types=["text/plain"],
            ),
        )

        mock_record = MagicMock()
        mock_record.manifest = manifest

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [mock_record]

            result = find_plugins_for_input("application/pdf")

            assert len(result) == 0

    def test_excludes_plugins_without_input_types(self):
        """Test that plugins without input_types are excluded."""
        manifest = PluginManifest(
            id="no-input-parser",
            type="parser",
            version="1.0.0",
            display_name="No Input Parser",
            description="No input types",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                # No input_types
            ),
        )

        mock_record = MagicMock()
        mock_record.manifest = manifest

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = [mock_record]

            result = find_plugins_for_input("application/pdf")

            assert len(result) == 0

    def test_sorts_by_specificity(self):
        """Test that results are sorted by specificity."""
        manifest_exact = PluginManifest(
            id="exact-match",
            type="parser",
            version="1.0.0",
            display_name="Exact",
            description="Exact",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["text/plain"],
            ),
        )
        manifest_wildcard = PluginManifest(
            id="wildcard-match",
            type="parser",
            version="1.0.0",
            display_name="Wildcard",
            description="Wildcard",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["text/*"],
            ),
        )

        mock_records = [
            MagicMock(manifest=manifest_wildcard),
            MagicMock(manifest=manifest_exact),
        ]

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.list_records.return_value = mock_records

            result = find_plugins_for_input("text/plain")

            # Exact match (100) should come before wildcard (50)
            assert result[0].id == "exact-match"
            assert result[1].id == "wildcard-match"


class TestGetAlternativePlugins:
    """Tests for get_alternative_plugins function."""

    def test_finds_alternatives_with_overlapping_inputs(self):
        """Test finding alternatives with overlapping input types."""
        original_manifest = PluginManifest(
            id="original",
            type="parser",
            version="1.0.0",
            display_name="Original",
            description="Original",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["application/pdf", "text/plain"],
            ),
        )
        alternative_manifest = PluginManifest(
            id="alternative",
            type="parser",
            version="1.0.0",
            display_name="Alternative",
            description="Alternative",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["application/pdf"],  # Overlaps with original
            ),
        )

        mock_original_record = MagicMock()
        mock_original_record.manifest = original_manifest
        mock_original_record.plugin_type = "parser"
        mock_original_record.plugin_id = "original"

        mock_alternative_record = MagicMock()
        mock_alternative_record.manifest = alternative_manifest
        mock_alternative_record.plugin_type = "parser"
        mock_alternative_record.plugin_id = "alternative"

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = mock_original_record
            mock_registry.list_records.return_value = [mock_original_record, mock_alternative_record]

            result = get_alternative_plugins("original")

            assert len(result) == 1
            assert result[0].id == "alternative"

    def test_excludes_original_plugin(self):
        """Test that the original plugin is excluded from results."""
        manifest = PluginManifest(
            id="original",
            type="parser",
            version="1.0.0",
            display_name="Original",
            description="Original",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["application/pdf"],
            ),
        )

        mock_record = MagicMock()
        mock_record.manifest = manifest
        mock_record.plugin_type = "parser"
        mock_record.plugin_id = "original"

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = mock_record
            mock_registry.list_records.return_value = [mock_record]

            result = get_alternative_plugins("original")

            assert len(result) == 0

    def test_returns_empty_for_unknown_plugin(self):
        """Test that empty list is returned for unknown plugin ID."""
        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = None

            result = get_alternative_plugins("nonexistent")

            assert result == []

    def test_returns_empty_when_no_agent_hints(self):
        """Test that empty list is returned when original has no agent_hints."""
        manifest = PluginManifest(
            id="no-hints",
            type="parser",
            version="1.0.0",
            display_name="No Hints",
            description="No hints",
        )

        mock_record = MagicMock()
        mock_record.manifest = manifest
        mock_record.plugin_type = "parser"
        mock_record.plugin_id = "no-hints"

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = mock_record

            result = get_alternative_plugins("no-hints")

            assert result == []

    def test_returns_empty_when_no_input_types(self):
        """Test that empty list is returned when original has no input_types."""
        manifest = PluginManifest(
            id="no-inputs",
            type="parser",
            version="1.0.0",
            display_name="No Inputs",
            description="No inputs",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                # No input_types
            ),
        )

        mock_record = MagicMock()
        mock_record.manifest = manifest
        mock_record.plugin_type = "parser"
        mock_record.plugin_id = "no-inputs"

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = mock_record

            result = get_alternative_plugins("no-inputs")

            assert result == []

    def test_sorts_alternatives_by_overlap_score(self):
        """Test that alternatives are sorted by overlap score."""
        original_manifest = PluginManifest(
            id="original",
            type="parser",
            version="1.0.0",
            display_name="Original",
            description="Original",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["application/pdf", "text/plain", "text/html"],
            ),
        )
        # High overlap - matches 2 types
        high_overlap_manifest = PluginManifest(
            id="high-overlap",
            type="parser",
            version="1.0.0",
            display_name="High",
            description="High",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["application/pdf", "text/plain"],
            ),
        )
        # Low overlap - matches 1 type
        low_overlap_manifest = PluginManifest(
            id="low-overlap",
            type="parser",
            version="1.0.0",
            display_name="Low",
            description="Low",
            agent_hints=AgentHints(
                purpose="test",
                best_for=["x"],
                not_recommended_for=["y"],
                input_types=["application/pdf"],
            ),
        )

        mock_original = MagicMock()
        mock_original.manifest = original_manifest
        mock_original.plugin_type = "parser"
        mock_original.plugin_id = "original"

        mock_high = MagicMock()
        mock_high.manifest = high_overlap_manifest
        mock_high.plugin_type = "parser"
        mock_high.plugin_id = "high-overlap"

        mock_low = MagicMock()
        mock_low.manifest = low_overlap_manifest
        mock_low.plugin_type = "parser"
        mock_low.plugin_id = "low-overlap"

        with patch("shared.plugins.discovery.plugin_registry") as mock_registry:
            mock_registry.find_by_id.return_value = mock_original
            mock_registry.list_records.return_value = [mock_original, mock_low, mock_high]

            result = get_alternative_plugins("original")

            # High overlap should come first
            assert len(result) == 2
            assert result[0].id == "high-overlap"
            assert result[1].id == "low-overlap"
