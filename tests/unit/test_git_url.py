"""Tests for git URL parsing utilities."""

import pytest

from shared.plugins.git_url import append_version_to_git_url, is_git_url, parse_git_url


class TestParseGitUrl:
    """Tests for parse_git_url function."""

    def test_https_url_with_git_extension(self) -> None:
        """Test parsing HTTPS URL with .git extension."""
        result = parse_git_url("https://github.com/org/repo.git")
        assert result is not None
        assert result.scheme == "https"
        assert result.host == "github.com"
        assert result.repo_name == "repo"
        assert result.ref is None

    def test_https_url_without_git_extension(self) -> None:
        """Test parsing HTTPS URL without .git extension."""
        result = parse_git_url("https://github.com/org/repo")
        assert result is not None
        assert result.scheme == "https"
        assert result.host == "github.com"
        assert result.repo_name == "repo"

    def test_git_plus_https(self) -> None:
        """Test parsing git+https:// URL."""
        result = parse_git_url("git+https://github.com/org/repo.git")
        assert result is not None
        assert result.scheme == "git+https"
        assert result.host == "github.com"
        assert result.repo_name == "repo"

    def test_git_plus_http(self) -> None:
        """Test parsing git+http:// URL."""
        result = parse_git_url("git+http://internal.example.com/org/repo.git")
        assert result is not None
        assert result.scheme == "git+http"
        assert result.host == "internal.example.com"
        assert result.repo_name == "repo"

    def test_url_with_existing_ref(self) -> None:
        """Test parsing URL with existing ref/version."""
        result = parse_git_url("git+https://github.com/org/repo.git@v1.0.0")
        assert result is not None
        assert result.ref == "v1.0.0"
        assert result.repo_name == "repo"

    def test_url_with_branch_ref(self) -> None:
        """Test parsing URL with branch name as ref."""
        result = parse_git_url("git+https://github.com/org/repo.git@main")
        assert result is not None
        assert result.ref == "main"

    def test_url_with_commit_hash_ref(self) -> None:
        """Test parsing URL with commit hash as ref."""
        result = parse_git_url("git+https://github.com/org/repo.git@abc123def")
        assert result is not None
        assert result.ref == "abc123def"

    def test_ssh_shorthand(self) -> None:
        """Test parsing SSH shorthand URL."""
        result = parse_git_url("git@github.com:org/repo.git")
        assert result is not None
        assert result.scheme == "ssh"
        assert result.host == "github.com"
        assert result.repo_name == "repo"
        assert result.ref is None

    def test_ssh_shorthand_with_ref(self) -> None:
        """Test parsing SSH shorthand URL with ref."""
        result = parse_git_url("git@github.com:org/repo.git@v1.0.0")
        assert result is not None
        assert result.scheme == "ssh"
        assert result.ref == "v1.0.0"

    def test_git_plus_ssh(self) -> None:
        """Test parsing git+ssh:// URL."""
        result = parse_git_url("git+ssh://git@github.com/org/repo.git")
        assert result is not None
        assert result.scheme == "git+ssh"
        assert result.host == "git@github.com"
        assert result.repo_name == "repo"

    def test_git_plus_ssh_with_ref(self) -> None:
        """Test parsing git+ssh:// URL with ref."""
        result = parse_git_url("git+ssh://git@github.com/org/repo.git@v2.0.0")
        assert result is not None
        assert result.scheme == "git+ssh"
        assert result.ref == "v2.0.0"

    def test_gitlab_url(self) -> None:
        """Test parsing GitLab URL."""
        result = parse_git_url("git+https://gitlab.com/group/subgroup/repo.git")
        assert result is not None
        assert result.host == "gitlab.com"
        assert result.repo_name == "repo"

    def test_bitbucket_url(self) -> None:
        """Test parsing Bitbucket URL."""
        result = parse_git_url("git+https://bitbucket.org/team/repo.git")
        assert result is not None
        assert result.host == "bitbucket.org"
        assert result.repo_name == "repo"

    def test_invalid_url_returns_none(self) -> None:
        """Test that invalid URLs return None."""
        assert parse_git_url("not-a-url") is None
        assert parse_git_url("https://example.com") is None
        assert parse_git_url("requests==2.28.0") is None
        assert parse_git_url("pip install something") is None
        assert parse_git_url("") is None

    def test_pypi_package_not_git_url(self) -> None:
        """Test that PyPI package names are not git URLs."""
        assert parse_git_url("semantik-plugin-openai") is None
        assert parse_git_url("requests") is None

    def test_whitespace_stripped(self) -> None:
        """Test that whitespace is stripped from URL."""
        result = parse_git_url("  git+https://github.com/org/repo.git  ")
        assert result is not None
        assert result.repo_name == "repo"


class TestIsGitUrl:
    """Tests for is_git_url function."""

    def test_valid_git_urls(self) -> None:
        """Test that valid git URLs are detected."""
        assert is_git_url("git+https://github.com/org/repo.git") is True
        assert is_git_url("https://github.com/org/repo.git") is True
        assert is_git_url("git@github.com:org/repo.git") is True
        assert is_git_url("git+ssh://git@github.com/org/repo.git") is True

    def test_invalid_git_urls(self) -> None:
        """Test that non-git URLs return False."""
        assert is_git_url("requests==2.28.0") is False
        assert is_git_url("semantik-plugin-openai") is False
        assert is_git_url("https://example.com") is False
        assert is_git_url("") is False


class TestAppendVersionToGitUrl:
    """Tests for append_version_to_git_url function."""

    def test_append_to_https(self) -> None:
        """Test appending version to HTTPS URL."""
        result = append_version_to_git_url(
            "git+https://github.com/org/repo.git",
            "v1.0.0",
        )
        assert result == "git+https://github.com/org/repo.git@v1.0.0"

    def test_append_to_https_without_git_extension(self) -> None:
        """Test appending version to HTTPS URL without .git."""
        result = append_version_to_git_url(
            "git+https://github.com/org/repo",
            "v1.0.0",
        )
        assert result == "git+https://github.com/org/repo@v1.0.0"

    def test_replace_existing_ref(self) -> None:
        """Test that existing ref is replaced."""
        result = append_version_to_git_url(
            "git+https://github.com/org/repo.git@main",
            "v2.0.0",
        )
        assert result == "git+https://github.com/org/repo.git@v2.0.0"
        assert "@main" not in result

    def test_replace_existing_version(self) -> None:
        """Test replacing existing version with new version."""
        result = append_version_to_git_url(
            "git+https://github.com/org/repo.git@v1.0.0",
            "v2.0.0",
        )
        assert result == "git+https://github.com/org/repo.git@v2.0.0"
        assert "v1.0.0" not in result

    def test_ssh_shorthand(self) -> None:
        """Test appending version to SSH shorthand URL."""
        result = append_version_to_git_url(
            "git@github.com:org/repo.git",
            "v1.0.0",
        )
        # SSH shorthand gets normalized to git+ssh://
        assert "@v1.0.0" in result
        assert "git+ssh://" in result

    def test_ssh_shorthand_replace_ref(self) -> None:
        """Test replacing ref in SSH shorthand URL."""
        result = append_version_to_git_url(
            "git@github.com:org/repo.git@main",
            "v2.0.0",
        )
        assert "@v2.0.0" in result
        assert "@main" not in result

    def test_invalid_url_raises(self) -> None:
        """Test that invalid URL raises ValueError."""
        with pytest.raises(ValueError, match="Invalid git URL"):
            append_version_to_git_url("not-a-url", "v1.0.0")

        with pytest.raises(ValueError, match="Invalid git URL"):
            append_version_to_git_url("requests==2.28.0", "v1.0.0")

    def test_special_version_formats(self) -> None:
        """Test various version format strings."""
        url = "git+https://github.com/org/repo.git"

        # Semantic version
        result = append_version_to_git_url(url, "v1.2.3")
        assert result.endswith("@v1.2.3")

        # Branch name
        result = append_version_to_git_url(url, "develop")
        assert result.endswith("@develop")

        # Commit hash
        result = append_version_to_git_url(url, "abc123def456")
        assert result.endswith("@abc123def456")


class TestGitUrlWithRef:
    """Tests for GitUrl.with_ref method."""

    def test_with_ref_simple(self) -> None:
        """Test with_ref on simple URL."""
        url = parse_git_url("git+https://github.com/org/repo.git")
        assert url is not None
        result = url.with_ref("v1.0.0")
        assert result == "git+https://github.com/org/repo.git@v1.0.0"

    def test_with_ref_replaces_existing(self) -> None:
        """Test with_ref replaces existing ref."""
        url = parse_git_url("git+https://github.com/org/repo.git@old-version")
        assert url is not None
        result = url.with_ref("new-version")
        assert result == "git+https://github.com/org/repo.git@new-version"
        assert "old-version" not in result
