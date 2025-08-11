# SEC-001: Fix ReDoS Vulnerabilities in Regex Patterns

## Ticket Information
- **Priority**: HIGH
- **Estimated Time**: 3 hours  
- **Dependencies**: None
- **Risk Level**: HIGH - Can cause DoS attacks
- **Affected Files**:
  - `packages/shared/chunking/domain/services/chunking_strategies/markdown.py`
  - `packages/shared/chunking/domain/services/chunking_strategies/hierarchical.py`
  - `packages/shared/chunking/domain/services/chunking_strategies/hybrid.py`
  - `packages/shared/text_processing/strategies/*.py`

## Context

Multiple regex patterns in the chunking strategies are vulnerable to ReDoS (Regular Expression Denial of Service) attacks. Malicious input can cause these patterns to take exponential time, leading to CPU exhaustion and service unavailability.

### Vulnerable Patterns Found

```python
# VULNERABLE: Nested quantifiers with backtracking
HEADING_PATTERN = r'^(#{1,6})\s+(.+)$'  # (.+) is dangerous
CODE_BLOCK = r'```[\s\S]*?```'  # [\s\S]*? can backtrack
URL_PATTERN = r'https?://[^\s]+'  # [^\s]+ unbounded

# VULNERABLE: Alternation with overlap
SENTENCE_END = r'[.!?]+\s*'  # Multiple quantifiers
WORD_BOUNDARY = r'\b\w+\b|\S+'  # Overlapping alternatives
```

## Requirements

1. Replace all vulnerable regex patterns with safe alternatives
2. Implement regex timeout protection
3. Use linear-time regex engine where possible (RE2)
4. Add input validation before regex processing
5. Implement pattern complexity limits
6. Add regex performance monitoring

## Technical Details

### 1. Install and Configure RE2

```python
# requirements.txt
google-re2==1.1.0  # Linear time regex engine
regex-timeout==1.0.0  # Timeout protection

# packages/shared/chunking/utils/safe_regex.py

import re2
import re
import time
from typing import Optional, Pattern, Match, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from functools import lru_cache

class SafeRegex:
    """Safe regex operations with ReDoS protection"""
    
    # Patterns known to be safe (simple, no backtracking)
    SAFE_PATTERNS = {
        'word': r'\w+',
        'whitespace': r'\s+',
        'number': r'\d+',
        'newline': r'\n',
    }
    
    # Maximum regex execution time (seconds)
    DEFAULT_TIMEOUT = 1.0
    
    def __init__(self, timeout: float = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._pattern_cache = {}
    
    @lru_cache(maxsize=100)
    def compile_safe(
        self, 
        pattern: str,
        use_re2: bool = True
    ) -> Pattern:
        """
        Compile pattern with safety checks
        
        Args:
            pattern: Regex pattern string
            use_re2: Use RE2 engine (no backreferences but guaranteed linear time)
            
        Returns:
            Compiled pattern object
            
        Raises:
            ValueError: If pattern is deemed unsafe
        """
        # Check pattern complexity
        if self._is_pattern_dangerous(pattern):
            raise ValueError(f"Pattern rejected as potentially dangerous: {pattern}")
        
        if use_re2:
            try:
                # RE2 doesn't support all Python regex features but is safe
                return re2.compile(pattern)
            except re2.error:
                # Fall back to Python re with timeout protection
                return re.compile(pattern)
        else:
            return re.compile(pattern)
    
    def match_with_timeout(
        self,
        pattern: str,
        text: str,
        timeout: Optional[float] = None
    ) -> Optional[Match]:
        """
        Match pattern with timeout protection
        
        Args:
            pattern: Regex pattern
            text: Text to match against
            timeout: Maximum execution time
            
        Returns:
            Match object or None
            
        Raises:
            RegexTimeout: If pattern takes too long
        """
        timeout = timeout or self.timeout
        
        def _match():
            compiled = self.compile_safe(pattern)
            return compiled.match(text)
        
        try:
            future = self.executor.submit(_match)
            return future.result(timeout=timeout)
        except TimeoutError:
            raise RegexTimeout(
                f"Regex timeout after {timeout}s. "
                f"Pattern: {pattern[:50]}..."
            )
    
    def findall_safe(
        self,
        pattern: str,
        text: str,
        max_matches: int = 1000
    ) -> List[str]:
        """
        Find all matches with safety limits
        
        Args:
            pattern: Regex pattern
            text: Text to search
            max_matches: Maximum matches to return
            
        Returns:
            List of matches (limited)
        """
        compiled = self.compile_safe(pattern)
        matches = []
        
        for match in compiled.finditer(text):
            matches.append(match.group())
            if len(matches) >= max_matches:
                break
        
        return matches
    
    def _is_pattern_dangerous(self, pattern: str) -> bool:
        """
        Check if pattern might cause ReDoS
        
        Dangerous patterns include:
        - Nested quantifiers: (a+)+
        - Alternation with overlap: (a|a)*
        - Unbound repetition with complex groups
        """
        dangerous_patterns = [
            r'(\(.*\+\))\+',  # Nested quantifiers
            r'(\(.*\*\))\*',  # Nested stars
            r'(\(.*\?\))\?',  # Nested optional
            r'\([^)]*\|[^)]*\)\*',  # Alternation with star
            r'\.[\+\*].*\.[\+\*]',  # Multiple wildcards
        ]
        
        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern):
                return True
        
        # Check pattern length (very long patterns are suspicious)
        if len(pattern) > 500:
            return True
        
        return False

class RegexTimeout(Exception):
    """Raised when regex execution times out"""
    pass
```

### 2. Fix Markdown Strategy Patterns

```python
# packages/shared/chunking/domain/services/chunking_strategies/markdown.py

from packages.shared.chunking.utils.safe_regex import SafeRegex, RegexTimeout

class SafeMarkdownChunker:
    """Markdown chunker with ReDoS protection"""
    
    def __init__(self):
        self.safe_regex = SafeRegex(timeout=1.0)
        
        # Safe patterns using RE2
        self.patterns = {
            # Use atomic groups and possessive quantifiers
            'heading': r'^#{1,6}\s+\S.*$',  # Non-greedy, bounded
            'code_block': r'^```[^`\n]*\n(?:[^`]|`(?!``))*\n```$',  # No backtracking
            'list_item': r'^[\*\-\+]\s+\S.*$',  # Bounded
            'blockquote': r'^>\s*\S.*$',  # Bounded
            'horizontal_rule': r'^(?:---|\*\*\*|___)$',  # Fixed alternatives
            'link': r'\[([^\]]+)\]\(([^)]+)\)',  # Bounded groups
            'image': r'!\[([^\]]*)\]\(([^)]+)\)',  # Bounded groups
            'bold': r'\*\*([^*]+)\*\*',  # No nested stars
            'italic': r'\*([^*]+)\*',  # No nested stars
            'code': r'`([^`]+)`',  # No nested backticks
        }
        
        # Compile all patterns with safety checks
        self.compiled_patterns = {}
        for name, pattern in self.patterns.items():
            self.compiled_patterns[name] = self.safe_regex.compile_safe(pattern)
    
    def find_headings(self, text: str) -> List[tuple]:
        """Find markdown headings safely"""
        headings = []
        
        # Limit input size
        if len(text) > 1_000_000:
            raise ValueError("Text too large for regex processing")
        
        # Process line by line to limit regex scope
        for line_num, line in enumerate(text.split('\n')):
            if len(line) > 1000:  # Skip very long lines
                continue
            
            try:
                match = self.safe_regex.match_with_timeout(
                    self.patterns['heading'],
                    line,
                    timeout=0.1  # 100ms per line
                )
                
                if match:
                    level = len(match.group(1))
                    title = match.group(2)
                    headings.append((line_num, level, title))
                    
            except RegexTimeout:
                # Log and skip this line
                logger.warning(f"Regex timeout on line {line_num}")
                continue
        
        return headings
    
    def find_code_blocks(self, text: str) -> List[tuple]:
        """Find code blocks without ReDoS risk"""
        code_blocks = []
        in_code_block = False
        current_block = []
        block_start = 0
        
        # State machine approach instead of regex
        for line_num, line in enumerate(text.split('\n')):
            if line.startswith('```'):
                if not in_code_block:
                    in_code_block = True
                    block_start = line_num
                    current_block = [line]
                else:
                    current_block.append(line)
                    code_blocks.append((
                        block_start,
                        line_num,
                        '\n'.join(current_block)
                    ))
                    in_code_block = False
                    current_block = []
            elif in_code_block:
                current_block.append(line)
        
        return code_blocks
```

### 3. Fix Hierarchical Strategy

```python
# packages/shared/chunking/domain/services/chunking_strategies/hierarchical.py

class SafeHierarchicalChunker:
    """Hierarchical chunker with safe patterns"""
    
    def __init__(self):
        self.safe_regex = SafeRegex()
        
        # Define hierarchy levels with safe patterns
        self.level_patterns = [
            # Level 1: Major sections (chapters, parts)
            {
                'name': 'chapter',
                'patterns': [
                    r'^Chapter\s+\d+',  # Simple, bounded
                    r'^Part\s+[IVXLCDM]+',  # Roman numerals, bounded
                    r'^#{1}\s+\S.*$',  # H1 headers
                ]
            },
            # Level 2: Sections
            {
                'name': 'section',
                'patterns': [
                    r'^Section\s+\d+',
                    r'^#{2}\s+\S.*$',  # H2 headers
                    r'^\d+\.\s+\S.*$',  # Numbered sections
                ]
            },
            # Level 3: Subsections
            {
                'name': 'subsection',  
                'patterns': [
                    r'^#{3}\s+\S.*$',  # H3 headers
                    r'^\d+\.\d+\.\s+\S.*$',  # Numbered subsections
                ]
            },
            # Level 4: Paragraphs
            {
                'name': 'paragraph',
                'patterns': [
                    r'^[A-Z].*\.$',  # Sentence starting with capital
                ]
            }
        ]
    
    def identify_structure(self, text: str) -> List[dict]:
        """Identify document structure safely"""
        structure = []
        
        # Pre-process text to remove potential ReDoS triggers
        text = self._sanitize_text(text)
        
        lines = text.split('\n')
        for line_num, line in enumerate(lines):
            # Skip empty or very long lines
            if not line.strip() or len(line) > 1000:
                continue
            
            for level_idx, level in enumerate(self.level_patterns):
                for pattern in level['patterns']:
                    try:
                        if self.safe_regex.match_with_timeout(
                            pattern, 
                            line,
                            timeout=0.05  # 50ms timeout per pattern
                        ):
                            structure.append({
                                'line': line_num,
                                'level': level_idx,
                                'type': level['name'],
                                'text': line[:200]  # Limit stored text
                            })
                            break  # Found match, skip other patterns
                    except RegexTimeout:
                        continue  # Try next pattern
        
        return structure
    
    def _sanitize_text(self, text: str) -> str:
        """Remove potential ReDoS triggers from text"""
        # Remove excessive whitespace
        text = re.sub(r'\s{100,}', ' ' * 99, text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.!?]{10,}', '...', text)
        
        # Remove excessive special characters
        text = re.sub(r'[*#-]{20,}', '*' * 19, text)
        
        return text
```

### 4. Add Input Validation

```python
# packages/shared/chunking/utils/input_validator.py

class ChunkingInputValidator:
    """Validate input before regex processing"""
    
    MAX_DOCUMENT_SIZE = 10_000_000  # 10MB
    MAX_LINE_LENGTH = 10_000
    MAX_WORD_LENGTH = 100
    
    @classmethod
    def validate_document(cls, text: str) -> None:
        """
        Validate document is safe for regex processing
        
        Raises:
            ValueError: If document is invalid
        """
        # Check size
        if len(text) > cls.MAX_DOCUMENT_SIZE:
            raise ValueError(
                f"Document too large: {len(text)} > {cls.MAX_DOCUMENT_SIZE}"
            )
        
        # Check for ReDoS triggers
        if cls._contains_redos_triggers(text):
            raise ValueError("Document contains potential ReDoS triggers")
        
        # Check line lengths
        lines = text.split('\n')
        for i, line in enumerate(lines):
            if len(line) > cls.MAX_LINE_LENGTH:
                raise ValueError(
                    f"Line {i} too long: {len(line)} > {cls.MAX_LINE_LENGTH}"
                )
        
        # Check for binary content
        if '\x00' in text or '\xff' in text:
            raise ValueError("Document appears to contain binary data")
    
    @classmethod
    def _contains_redos_triggers(cls, text: str) -> bool:
        """Check for common ReDoS trigger patterns"""
        triggers = [
            r'a{1000,}',  # Excessive repetition
            r'(.)\1{100,}',  # Repeated characters
            r'[\s]{1000,}',  # Excessive whitespace
            r'[^\w]{1000,}',  # Excessive special chars
        ]
        
        for trigger in triggers:
            if re.search(trigger, text):
                return True
        
        return False
```

### 5. Add Performance Monitoring

```python
# packages/shared/chunking/utils/regex_monitor.py

import time
from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

@dataclass
class RegexMetrics:
    pattern: str
    execution_time: float
    input_size: int
    matched: bool
    timed_out: bool

class RegexPerformanceMonitor:
    """Monitor regex performance to detect issues"""
    
    def __init__(self):
        self.metrics: List[RegexMetrics] = []
        self.slow_patterns: Dict[str, int] = {}
    
    def record_execution(
        self,
        pattern: str,
        execution_time: float,
        input_size: int,
        matched: bool = False,
        timed_out: bool = False
    ):
        """Record regex execution metrics"""
        metric = RegexMetrics(
            pattern=pattern[:100],  # Truncate long patterns
            execution_time=execution_time,
            input_size=input_size,
            matched=matched,
            timed_out=timed_out
        )
        
        self.metrics.append(metric)
        
        # Track slow patterns
        if execution_time > 0.1:  # 100ms threshold
            self.slow_patterns[pattern] = \
                self.slow_patterns.get(pattern, 0) + 1
        
        # Log warnings
        if timed_out:
            logger.error(
                f"Regex timeout: pattern='{pattern[:50]}...', "
                f"input_size={input_size}"
            )
        elif execution_time > 1.0:
            logger.warning(
                f"Slow regex: pattern='{pattern[:50]}...', "
                f"time={execution_time:.2f}s, input_size={input_size}"
            )
    
    def get_problematic_patterns(self) -> List[str]:
        """Get patterns that frequently cause issues"""
        return [
            pattern for pattern, count in self.slow_patterns.items()
            if count > 5
        ]
```

## Acceptance Criteria

1. **Pattern Safety**
   - [ ] All regex patterns reviewed and fixed
   - [ ] RE2 engine used where possible
   - [ ] No nested quantifiers or dangerous alternations
   - [ ] All patterns have bounded repetition

2. **Timeout Protection**
   - [ ] All regex operations have timeout
   - [ ] Timeout errors handled gracefully
   - [ ] No regex can run > 1 second

3. **Input Validation**
   - [ ] Document size limits enforced
   - [ ] Line length limits enforced
   - [ ] ReDoS triggers detected and rejected

4. **Performance**
   - [ ] No regex takes > 100ms on normal input
   - [ ] Pattern compilation cached
   - [ ] Performance metrics collected

## Testing Requirements

1. **Security Tests**
   ```python
   def test_redos_protection():
       # Known ReDoS payload
       evil_input = "a" * 1000 + "!" * 1000
       
       chunker = SafeMarkdownChunker()
       
       # Should timeout or reject, not hang
       with pytest.raises((RegexTimeout, ValueError)):
           chunker.find_headings(evil_input)
   
   def test_pattern_timeout():
       text = "x" * 100000
       pattern = r'(x+)+y'  # Catastrophic backtracking
       
       safe_regex = SafeRegex(timeout=0.1)
       
       with pytest.raises(RegexTimeout):
           safe_regex.match_with_timeout(pattern, text)
   ```

2. **Performance Tests**
   - Test with large documents (1MB+)
   - Test with malicious input
   - Verify all operations < 1 second
   - Check memory usage stays bounded

## Rollback Plan

1. Keep original patterns commented
2. Feature flag for safe regex mode
3. Monitor performance metrics
4. Quick revert if issues found

## Success Metrics

- Zero ReDoS vulnerabilities in security scan
- All regex operations complete < 1 second
- No CPU spikes from regex processing
- Pattern performance p99 < 100ms

## Notes for LLM Agent

- Prioritize safety over features - disable complex patterns if needed
- Use state machines instead of regex where possible
- Always bound repetition in patterns
- Test with fuzzing tools and evil inputs
- Monitor production for slow patterns
- Consider pre-filtering input before regex