"""Tests for AddSourceRequest schema validation."""

from webui.api.schemas import AddSourceRequest


class TestAddSourceRequestValidation:
    """Test AddSourceRequest schema validation and normalization."""

    def test_new_format_with_source_type_and_config(self) -> None:
        """Test new format with source_type and source_config."""
        request = AddSourceRequest(
            source_type="directory",
            source_config={"path": "/data/docs", "recursive": True},
        )
        assert request.source_type == "directory"
        assert request.source_config == {"path": "/data/docs", "recursive": True}
        assert request.source_path is None

    def test_legacy_format_converts_source_path(self) -> None:
        """Test legacy source_path is converted to source_config."""
        request = AddSourceRequest(source_path="/data/docs")
        assert request.source_type == "directory"  # Default
        assert request.source_config == {"path": "/data/docs"}
        assert request.source_path == "/data/docs"

    def test_legacy_format_does_not_override_explicit_config(self) -> None:
        """Test explicit source_config is not overridden by source_path."""
        request = AddSourceRequest(
            source_path="/legacy/path",
            source_config={"path": "/explicit/path"},
        )
        assert request.source_config == {"path": "/explicit/path"}

    def test_web_source_type(self) -> None:
        """Test non-directory source type."""
        request = AddSourceRequest(
            source_type="web",
            source_config={"url": "https://example.com", "depth": 2},
        )
        assert request.source_type == "web"
        assert request.source_config["url"] == "https://example.com"

    def test_empty_request_defaults(self) -> None:
        """Test empty request uses defaults."""
        request = AddSourceRequest()
        assert request.source_type == "directory"
        assert request.source_config is None
        assert request.source_path is None

    def test_source_type_defaults_to_directory(self) -> None:
        """Test source_type defaults to directory when only source_config provided."""
        request = AddSourceRequest(source_config={"path": "/data/docs"})
        assert request.source_type == "directory"
        assert request.source_config == {"path": "/data/docs"}

    def test_additional_config_field(self) -> None:
        """Test config field (chunk settings, metadata) is preserved."""
        request = AddSourceRequest(
            source_type="directory",
            source_config={"path": "/data/docs"},
            config={"chunk_size": 1000, "metadata": {"department": "eng"}},
        )
        assert request.config == {"chunk_size": 1000, "metadata": {"department": "eng"}}

    def test_legacy_with_config_field(self) -> None:
        """Test legacy source_path with config field."""
        request = AddSourceRequest(
            source_path="/data/docs",
            config={"chunk_size": 500},
        )
        assert request.source_config == {"path": "/data/docs"}
        assert request.config == {"chunk_size": 500}

    def test_slack_source_type(self) -> None:
        """Test Slack source type with channel config."""
        request = AddSourceRequest(
            source_type="slack",
            source_config={"channel": "#engineering", "since_days": 30},
        )
        assert request.source_type == "slack"
        assert request.source_config["channel"] == "#engineering"
        assert request.source_config["since_days"] == 30

    def test_source_config_with_nested_objects(self) -> None:
        """Test source_config with nested objects."""
        request = AddSourceRequest(
            source_type="web",
            source_config={
                "url": "https://example.com",
                "options": {"max_depth": 3, "follow_redirects": True},
            },
        )
        assert request.source_config["options"]["max_depth"] == 3
