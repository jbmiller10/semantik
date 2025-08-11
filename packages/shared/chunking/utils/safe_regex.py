#!/usr/bin/env python3
"""Safe regex operations with ReDoS protection."""

import logging
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from re import Match, Pattern
from typing import Any

logger = logging.getLogger(__name__)

# Try to import RE2, fall back to standard re if not available
try:
    import re2

    HAS_RE2 = True
except ImportError:
    HAS_RE2 = False
    logger.warning("RE2 not available. Using standard regex with timeout protection only.")


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
        self.executor = ThreadPoolExecutor(max_workers=1)
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
                # Fall back to Python re with timeout protection
                logger.debug(f"RE2 compilation failed for pattern: {pattern}. Using standard re.")
                compiled = re.compile(pattern, flags=flags)
        else:
            compiled = re.compile(pattern, flags=flags)

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

        def _match() -> Match[str] | None:
            compiled = self.compile_safe(pattern)
            return compiled.match(text)

        try:
            future = self.executor.submit(_match)
            return future.result(timeout=timeout)
        except TimeoutError:
            raise RegexTimeoutError(f"Regex timeout after {timeout}s. Pattern: {pattern[:50]}...") from None

    def findall_safe(self, pattern: str, text: str, max_matches: int = 1000, flags: int = 0) -> list[str]:
        """Find all matches with safety limits.

        Args:
            pattern: Regex pattern
            text: Text to search
            max_matches: Maximum matches to return
            flags: Regex compilation flags (e.g., re.MULTILINE)

        Returns:
            List of matches (limited)
        """
        compiled = self.compile_safe(pattern, flags=flags)
        matches = []

        for match in compiled.finditer(text):
            matches.append(match.group())
            if len(matches) >= max_matches:
                break

        return matches

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

        def _search() -> Match[str] | None:
            compiled = self.compile_safe(pattern)
            return compiled.search(text)

        try:
            future = self.executor.submit(_search)
            return future.result(timeout=timeout)
        except TimeoutError:
            raise RegexTimeoutError(f"Regex timeout after {timeout}s. Pattern: {pattern[:50]}...") from None

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

    def __del__(self) -> None:
        """Clean up executor on deletion."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
