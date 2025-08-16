"""Safe regex operations with ReDoS protection using the regex module.

This module provides timeout-protected regex operations using the regex module
which has native timeout support, avoiding the need for ThreadPoolExecutor.
"""

import logging
from re import Pattern

try:
    import regex

    HAS_REGEX = True
except ImportError:
    import re as regex

    HAS_REGEX = False
    logging.warning("regex module not available. Using standard re module without native timeout support.")

logger = logging.getLogger(__name__)


class RegexTimeoutError(Exception):
    """Raised when regex operation times out."""


# Backwards compatibility alias
RegexTimeout = RegexTimeoutError


def safe_regex_search(pattern: str, text: str, timeout: float = 1.0, flags: int = 0) -> regex.Match | None:
    """Execute regex search with timeout protection.

    Args:
        pattern: Regex pattern to search for
        text: Text to search in
        timeout: Maximum execution time in seconds (default: 1.0)
        flags: Regex compilation flags

    Returns:
        Match object if found, None otherwise

    Raises:
        RegexTimeout: If pattern execution exceeds timeout
        ValueError: If pattern is deemed dangerous
    """
    if analyze_pattern_complexity(pattern):
        # Try to simplify the pattern first
        pattern = simplify_pattern(pattern)
        if analyze_pattern_complexity(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous: {pattern[:50]}...")

    try:
        if HAS_REGEX:
            compiled = regex.compile(pattern, flags)
            return compiled.search(text, timeout=timeout)
        else:
            # Fallback to standard re without timeout
            compiled = regex.compile(pattern, flags)
            return compiled.search(text)
    except regex.error as e:
        if HAS_REGEX and "timeout" in str(e).lower():
            raise RegexTimeoutError(f"Pattern timed out after {timeout}s: {pattern[:50]}...") from e
        raise
    except Exception as e:
        logger.error(f"Regex search failed: {e}")
        return None


def safe_regex_match(pattern: str, text: str, timeout: float = 1.0, flags: int = 0) -> regex.Match | None:
    """Execute regex match with timeout protection.

    Args:
        pattern: Regex pattern to match
        text: Text to match against
        timeout: Maximum execution time in seconds (default: 1.0)
        flags: Regex compilation flags

    Returns:
        Match object if pattern matches at start of text, None otherwise

    Raises:
        RegexTimeout: If pattern execution exceeds timeout
        ValueError: If pattern is deemed dangerous
    """
    if analyze_pattern_complexity(pattern):
        pattern = simplify_pattern(pattern)
        if analyze_pattern_complexity(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous: {pattern[:50]}...")

    try:
        if HAS_REGEX:
            compiled = regex.compile(pattern, flags)
            return compiled.match(text, timeout=timeout)
        else:
            compiled = regex.compile(pattern, flags)
            return compiled.match(text)
    except regex.error as e:
        if HAS_REGEX and "timeout" in str(e).lower():
            raise RegexTimeoutError(f"Pattern timed out after {timeout}s: {pattern[:50]}...") from e
        raise
    except Exception as e:
        logger.error(f"Regex match failed: {e}")
        return None


def safe_regex_findall(
    pattern: str, text: str, timeout: float = 1.0, max_matches: int = 1000, flags: int = 0
) -> list[str]:
    """Find all matches with timeout and limit protection.

    Args:
        pattern: Regex pattern to find
        text: Text to search in
        timeout: Maximum execution time in seconds (default: 1.0)
        max_matches: Maximum number of matches to return (default: 1000)
        flags: Regex compilation flags

    Returns:
        List of matching strings (limited by max_matches)

    Raises:
        RegexTimeout: If pattern execution exceeds timeout
        ValueError: If pattern is deemed dangerous
    """
    if analyze_pattern_complexity(pattern):
        pattern = simplify_pattern(pattern)
        if analyze_pattern_complexity(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous: {pattern[:50]}...")

    try:
        if HAS_REGEX:
            compiled = regex.compile(pattern, flags)
            matches = []
            for match in compiled.finditer(text, timeout=timeout):
                matches.append(match.group())
                if len(matches) >= max_matches:
                    break
            return matches
        else:
            # Fallback without timeout
            compiled = regex.compile(pattern, flags)
            matches = compiled.findall(text)
            return matches[:max_matches]
    except regex.error as e:
        if HAS_REGEX and "timeout" in str(e).lower():
            raise RegexTimeoutError(f"Pattern timed out after {timeout}s: {pattern[:50]}...") from e
        raise
    except Exception as e:
        logger.error(f"Regex findall failed: {e}")
        return []


def analyze_pattern_complexity(pattern: str) -> bool:
    """Check if pattern might cause ReDoS.

    Dangerous patterns include:
    - Nested quantifiers: (a+)+, (a*)*
    - Alternation with overlap: (a|a)*
    - Multiple unbounded wildcards: .*.*
    - Excessive backtracking potential

    Args:
        pattern: Regex pattern to analyze

    Returns:
        True if pattern is potentially dangerous, False if safe
    """
    # Check pattern length
    if len(pattern) > 500:
        logger.warning(f"Pattern too long ({len(pattern)} chars), may cause issues")
        return True

    # Check for nested quantifiers using string operations
    dangerous_patterns = [
        # Nested quantifiers
        r"(\w+)*",
        r"(\w+)+",
        r"(\w*)*",
        r"(\w*)+",
        r'([^"]*)*',
        r'([^"]*)+',
        r"([^\']*)*",
        r"([^\']*)+",
        r"(\S+)*",
        r"(\S+)+",
        r"(\S*)*",
        r"(\S*)+",
        r"(.*)*",
        r"(.*)+",
        r"(.+)*",
        r"(.+)+",
        # Multiple wildcards
        ".*.*",
        ".+.+",
        ".*\\s*.*",
        ".+\\s+.+",
        # Catastrophic patterns
        "(a+)+",
        "(a*)*",
        "(a+)*",
        "(a*)+",
        "(\\w+)+",
        "(\\w*)*",
        "(\\d+)+",
        "(\\d*)*",
    ]

    pattern_lower = pattern.lower()
    for dangerous in dangerous_patterns:
        if dangerous in pattern_lower:
            logger.warning(f"Pattern contains dangerous construct: {dangerous}")
            return True

    # Check for alternation with quantifier
    if "|" in pattern:
        # Look for patterns like (a|b)* or (a|b)+
        import re

        if re.search(r"\([^)]*\|[^)]*\)[*+]", pattern):
            logger.warning("Pattern contains alternation with quantifier")
            return True

    # Check for unescaped dots with quantifiers
    if ".*" in pattern and pattern.count(".*") > 2:
        logger.warning("Pattern contains multiple .* constructs")
        return True

    if ".+" in pattern and pattern.count(".+") > 2:
        logger.warning("Pattern contains multiple .+ constructs")
        return True

    # Check for complex lookarounds (can be expensive)
    if "(?=" in pattern or "(?!" in pattern or "(?<=" in pattern or "(?<!" in pattern:
        lookaround_count = pattern.count("(?=") + pattern.count("(?!") + pattern.count("(?<=") + pattern.count("(?<!")
        if lookaround_count > 3:
            logger.warning(f"Pattern contains {lookaround_count} lookarounds")
            return True

    return False


def simplify_pattern(pattern: str) -> str:
    """Attempt to simplify a regex pattern to avoid ReDoS.

    This function tries to make patterns safer by:
    - Replacing nested quantifiers with simpler forms
    - Limiting repetition counts
    - Simplifying alternations

    Args:
        pattern: Original regex pattern

    Returns:
        Simplified pattern (may not be exactly equivalent but safer)
    """
    simplified = pattern

    # Replace nested quantifiers with bounded versions
    replacements = [
        (r"(\w+)+", r"\w+"),
        (r"(\w+)*", r"\w*"),
        (r"(\w*)+", r"\w*"),
        (r"(\w*)*", r"\w*"),
        (r"(\S+)+", r"\S+"),
        (r"(\S+)*", r"\S*"),
        (r"(\S*)+", r"\S*"),
        (r"(\S*)*", r"\S*"),
        (r'([^"]*)+', r'[^"]*'),
        (r'([^"]*)*', r'[^"]*'),
        (r"([^\']*)+", r"[^\']*"),
        (r"([^\']*)*", r"[^\']*"),
        (r"(.*)+", r".*"),
        (r"(.*)*", r".*"),
        (r"(.+)+", r".+"),
        (r"(.+)*", r".*"),
    ]

    for old, new in replacements:
        if old in simplified:
            simplified = simplified.replace(old, new)
            logger.debug(f"Simplified pattern: replaced '{old}' with '{new}'")

    # Replace multiple wildcards with single ones
    if ".*.*" in simplified:
        simplified = simplified.replace(".*.*", ".*")
        logger.debug("Simplified pattern: replaced '.*.*' with '.*'")

    if ".+.+" in simplified:
        simplified = simplified.replace(".+.+", ".+")
        logger.debug("Simplified pattern: replaced '.+.+' with '.+'")

    # Limit unbounded repetitions to reasonable bounds
    import re

    # Replace {n,} with {n,1000} to add upper bound
    simplified = re.sub(r"\{(\d+),\}", r"{\1,1000}", simplified)

    # If pattern is still complex, try more aggressive simplification
    if len(simplified) > 200:
        # Truncate very long character classes
        simplified = re.sub(r"\[[^\]]{50,}\]", r"[\w\W]", simplified)

    return simplified


def compile_safe(pattern: str, flags: int = 0, timeout: float = 1.0) -> Pattern:
    """Compile a regex pattern with safety checks.

    Args:
        pattern: Regex pattern to compile
        flags: Regex compilation flags
        timeout: Timeout value to use when executing the pattern

    Returns:
        Compiled pattern object

    Raises:
        ValueError: If pattern is deemed unsafe
    """
    if analyze_pattern_complexity(pattern):
        pattern = simplify_pattern(pattern)
        if analyze_pattern_complexity(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous: {pattern[:50]}...")

    try:
        if HAS_REGEX:
            # Store timeout as an attribute for later use
            compiled = regex.compile(pattern, flags)
            compiled._timeout = timeout
            return compiled
        else:
            return regex.compile(pattern, flags)
    except Exception as e:
        logger.error(f"Failed to compile pattern: {e}")
        raise


# Convenience function for replacing direct re.search calls
def search_with_fallback(pattern: str, text: str, timeout: float = 1.0, flags: int = 0) -> regex.Match | None:
    """Search with automatic fallback on timeout.

    If the pattern times out, tries a simplified version.
    If that also fails, returns None.

    Args:
        pattern: Regex pattern to search for
        text: Text to search in
        timeout: Maximum execution time in seconds
        flags: Regex compilation flags

    Returns:
        Match object if found, None otherwise
    """
    try:
        return safe_regex_search(pattern, text, timeout, flags)
    except RegexTimeoutError:
        logger.warning(f"Pattern timed out, trying simplified version: {pattern[:50]}...")
        simplified = simplify_pattern(pattern)
        if simplified != pattern:
            try:
                return safe_regex_search(simplified, text, timeout, flags)
            except RegexTimeoutError:
                logger.error(f"Simplified pattern also timed out: {simplified[:50]}...")
        return None
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return None
