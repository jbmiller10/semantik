#!/usr/bin/env python3
"""Safe regex operations with ReDoS protection."""

import logging
from re import Match, Pattern
from typing import Any

logger = logging.getLogger(__name__)

# Try to import regex module for native timeout support
try:
    import regex

    HAS_REGEX = True
except ImportError:
    import re as regex

    HAS_REGEX = False
    logger.warning("regex module not available. Using standard re without native timeout support.")

# Try to import RE2 as secondary option
try:
    import re2

    HAS_RE2 = True
except ImportError:
    HAS_RE2 = False


class RegexTimeoutError(Exception):
    """Raised when regex execution times out."""


# Backwards-compatibility alias expected by tests
RegexTimeout = RegexTimeoutError


class SafeRegex:
    """Safe regex operations with ReDoS protection."""

    # Patterns known to be safe (simple, no backtracking)
    SAFE_PATTERNS = {
        "word": r"\w+",
        "whitespace": r"\s+",
        "number": r"\d+",
        "newline": r"\n",
    }

    # Maximum regex execution time (seconds)
    DEFAULT_TIMEOUT = 1.0

    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        """Initialize SafeRegex with timeout configuration.

        Args:
            timeout: Maximum time allowed for regex operations (seconds)
        """
        self.timeout = timeout
        self._pattern_cache: dict[tuple[str, bool, int], Pattern[str] | Any] = {}

    def compile_safe(self, pattern: str, use_re2: bool = True, flags: int = 0) -> Pattern[str] | Any:
        """Compile pattern with safety checks.

        Args:
            pattern: Regex pattern string
            use_re2: Use RE2 engine (no backreferences but guaranteed linear time)
            flags: Regex compilation flags (e.g., re.MULTILINE)

        Returns:
            Compiled pattern object

        Raises:
            ValueError: If pattern is deemed unsafe
        """
        # Check cache first
        cache_key = (pattern, use_re2, flags)
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]

        # Check pattern complexity
        if self._is_pattern_dangerous(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous: {pattern}")

        if use_re2 and HAS_RE2:
            try:
                # RE2 doesn't support all Python regex features but is safe
                compiled = re2.compile(pattern, flags=flags)
            except Exception:
                # Fall back to regex module with timeout protection
                logger.debug(f"RE2 compilation failed for pattern: {pattern}. Using regex module.")
                compiled = regex.compile(pattern, flags=flags)
        else:
            compiled = regex.compile(pattern, flags=flags)

        # Cache the compiled pattern (limit cache size)
        if len(self._pattern_cache) < 100:
            self._pattern_cache[cache_key] = compiled

        return compiled

    def match_with_timeout(self, pattern: str, text: str, timeout: float | None = None) -> Match | None:
        """Match pattern with timeout protection.

        Args:
            pattern: Regex pattern
            text: Text to match against
            timeout: Maximum execution time

        Returns:
            Match object or None

        Raises:
            RegexTimeoutError: If pattern takes too long
        """
        timeout = timeout or self.timeout

        try:
            compiled = self.compile_safe(pattern)
            if HAS_REGEX:
                # Use native timeout support
                return compiled.match(text, timeout=timeout)
            # No timeout available with standard re
            return compiled.match(text)
        except regex.error as e:
            if HAS_REGEX and "timeout" in str(e).lower():
                raise RegexTimeoutError(f"Regex timeout after {timeout}s. Pattern: {pattern[:50]}...") from None
            raise
        except Exception as e:
            logger.error(f"Match failed: {e}")
            return None

    def findall_safe(self, pattern: str, text: str, max_matches: int = 1000, flags: int = 0) -> list[str]:
        """Find all matches with safety limits.

        Args:
            pattern: Regex pattern
            text: Text to search
            max_matches: Maximum matches to return
            flags: Regex compilation flags (e.g., regex.MULTILINE)

        Returns:
            List of matches (limited)
        """
        try:
            compiled = self.compile_safe(pattern, flags=flags)
            matches = []

            if HAS_REGEX:
                # Use finditer with timeout
                for match in compiled.finditer(text, timeout=self.timeout):
                    matches.append(match.group())
                    if len(matches) >= max_matches:
                        break
            else:
                # No timeout available
                for match in compiled.finditer(text):
                    matches.append(match.group())
                    if len(matches) >= max_matches:
                        break

            return matches
        except regex.error as e:
            if HAS_REGEX and "timeout" in str(e).lower():
                raise RegexTimeoutError(f"Regex timeout after {self.timeout}s. Pattern: {pattern[:50]}...") from None
            raise
        except Exception as e:
            logger.error(f"Findall failed: {e}")
            return []

    def search_with_timeout(self, pattern: str, text: str, timeout: float | None = None) -> Match | None:
        """Search for pattern with timeout protection.

        Args:
            pattern: Regex pattern
            text: Text to search in
            timeout: Maximum execution time

        Returns:
            Match object or None

        Raises:
            RegexTimeoutError: If pattern takes too long
        """
        timeout = timeout or self.timeout

        try:
            compiled = self.compile_safe(pattern)
            if HAS_REGEX:
                # Use native timeout support
                return compiled.search(text, timeout=timeout)
            # No timeout available with standard re
            return compiled.search(text)
        except regex.error as e:
            if HAS_REGEX and "timeout" in str(e).lower():
                raise RegexTimeoutError(f"Regex timeout after {timeout}s. Pattern: {pattern[:50]}...") from None
            raise
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return None

    def _is_pattern_dangerous(self, pattern: str) -> bool:
        """Check if pattern might cause ReDoS.

        Dangerous patterns include:
        - Nested quantifiers: (a+)+
        - Alternation with overlap: (a|a)*
        - Unbound repetition with complex groups

        Uses string operations instead of regex to avoid bootstrapping issues.
        """
        # Check pattern length (very long patterns are suspicious)
        if len(pattern) > 500:
            return True

        # Check for nested quantifiers using string operations
        # Look for patterns like (...)+ or (...)* or (...)?
        paren_depth = 0
        i = 0
        while i < len(pattern):
            if pattern[i] == "(":
                paren_depth += 1
                # Look ahead for quantifier after group
                j = i + 1
                while j < len(pattern) and paren_depth > 0:
                    if pattern[j] == "(":
                        paren_depth += 1
                    elif pattern[j] == ")":
                        paren_depth -= 1
                        if paren_depth == 0 and j + 1 < len(pattern):
                            # Check if there's a quantifier after the group
                            next_char = pattern[j + 1]
                            if next_char in "+*?":
                                # Check if the group contains quantifiers
                                group_content = pattern[i + 1 : j]
                                if any(q in group_content for q in ["+", "*", "?"]):
                                    # Found nested quantifier
                                    return True
                    j += 1
            elif pattern[i] == ")":
                paren_depth = max(0, paren_depth - 1)
            i += 1

        # Check for alternation with star: (...|...)*
        if "|" in pattern:
            # Find groups containing alternation
            i = 0
            while i < len(pattern):
                if pattern[i] == "(" and "|" in pattern[i:]:
                    # Find the closing parenthesis
                    paren_count = 1
                    j = i + 1
                    while j < len(pattern) and paren_count > 0:
                        if pattern[j] == "(":
                            paren_count += 1
                        elif pattern[j] == ")":
                            paren_count -= 1
                            # Check for quantifier after alternation group
                            if paren_count == 0 and j + 1 < len(pattern) and pattern[j + 1] in "*+":
                                return True
                        j += 1
                i += 1

        # Check for multiple unbounded wildcards: .*.*
        if pattern.count(".*") > 1 or pattern.count(".+") > 1:
            return True

        # Check for excessive backtracking potential
        # Look for patterns like: a*a* or a+a+ (overlapping quantifiers)
        for i in range(len(pattern) - 3):
            if pattern[i : i + 2] in [".*", ".+", r"\w*", r"\w+", r"\d*", r"\d+"]:
                for j in range(i + 2, min(i + 10, len(pattern) - 1)):
                    if pattern[j : j + 2] in [".*", ".+", r"\w*", r"\w+", r"\d*", r"\d+"]:
                        return True

        return False
