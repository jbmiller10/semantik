"""
Comprehensive XSS prevention tests for chunking metadata sanitization.

This module tests various XSS attack vectors to ensure proper sanitization.
"""

import pytest

from shared.chunks.metadata_sanitizer import MetadataSanitizer


class TestXSSPrevention:
    """Test suite for XSS prevention in metadata sanitization."""

    # Common XSS attack vectors
    XSS_VECTORS = [
        # Script tags
        '<script>alert("XSS")</script>',
        '"><script>alert("XSS")</script>',
        '<ScRiPt>alert("XSS")</ScRiPt>',
        '<script src="http://evil.com/xss.js"></script>',
        # Event handlers
        '<img src=x onerror=alert("XSS")>',
        '<img src=x onload=alert("XSS")>',
        '<body onload=alert("XSS")>',
        "<div onclick=\"alert('XSS')\">Click me</div>",
        '<input onfocus=alert("XSS") autofocus>',
        # JavaScript protocols
        "<a href=\"javascript:alert('XSS')\">Click</a>",
        "<iframe src=\"javascript:alert('XSS')\">",
        "<img src=\"javascript:alert('XSS')\">",
        # Data URLs
        "<a href=\"data:text/html,<script>alert('XSS')</script>\">Click</a>",
        "<object data=\"data:text/html,<script>alert('XSS')</script>\">",
        # VBScript (IE)
        "<a href=\"vbscript:msgbox('XSS')\">Click</a>",
        # SVG attacks
        '<svg onload=alert("XSS")>',
        '<svg><script>alert("XSS")</script></svg>',
        '<svg><animate onbegin=alert("XSS") attributeName=x></svg>',
        # IFrame injection
        '<iframe src="http://evil.com"></iframe>',
        "<iframe srcdoc=\"<script>alert('XSS')</script>\"></iframe>",
        # Form injection
        '<form action="http://evil.com"><input type="submit"></form>',
        "<form><button formaction=\"javascript:alert('XSS')\">Submit</button></form>",
        # Meta refresh
        '<meta http-equiv="refresh" content="0;url=http://evil.com">',
        '<meta http-equiv="refresh" content="0;url=javascript:alert(\'XSS\')">',
        # Link injection
        '<link rel="stylesheet" href="http://evil.com/evil.css">',
        # Style injection with expressions
        '<style>body{background:expression(alert("XSS"))}</style>',
        "<div style=\"background:expression(alert('XSS'))\">",
        # Encoded attacks
        '&lt;script&gt;alert("XSS")&lt;/script&gt;',
        '&#60;script&#62;alert("XSS")&#60;/script&#62;',
        '%3Cscript%3Ealert("XSS")%3C/script%3E',
        # Marquee (annoying)
        "<marquee>XSS</marquee>",
        '<marquee onstart=alert("XSS")>',
        # Object and embed
        '<object data="http://evil.com/evil.swf">',
        '<embed src="http://evil.com/evil.swf">',
        '<applet code="Evil.class">',
        # Import/require (for Node.js contexts)
        'import { evil } from "evil-module"',
        'require("child_process").exec("rm -rf /")',
    ]

    def test_sanitize_string_basic(self):
        """Test basic string sanitization."""
        # Normal text should pass through with HTML escaping
        assert MetadataSanitizer.sanitize_string("Hello World") == "Hello World"

        # HTML special characters should be escaped
        # Plain div tags should be escaped but not removed (they're not dangerous without event handlers)
        assert MetadataSanitizer.sanitize_string("<div>Test</div>") == "&lt;div&gt;Test&lt;/div&gt;"
        # Dangerous tags should be removed completely
        assert MetadataSanitizer.sanitize_string("<script>alert('XSS')</script>") == "[Content removed for security]"
        assert MetadataSanitizer.sanitize_string("Test & Done") == "Test &amp; Done"
        assert MetadataSanitizer.sanitize_string('Test "quoted"') == "Test &quot;quoted&quot;"
        assert MetadataSanitizer.sanitize_string("Test 'quoted'") == "Test &#x27;quoted&#x27;"

    def test_sanitize_string_xss_vectors(self):
        """Test sanitization against all XSS vectors."""
        for vector in self.XSS_VECTORS:
            result = MetadataSanitizer.sanitize_string(vector)
            # Should either escape or remove dangerous content
            assert result == "[Content removed for security]" or (
                "<script" not in result.lower()
                and "javascript:" not in result.lower()
                and "onerror" not in result.lower()
                and "onload" not in result.lower()
            ), f"XSS vector not properly sanitized: {vector}"

    def test_sanitize_string_null_bytes(self):
        """Test null byte removal."""
        input_str = "Hello\x00World"
        result = MetadataSanitizer.sanitize_string(input_str)
        assert "\x00" not in result
        assert "HelloWorld" in result or result == "[Content removed for security]"

    def test_sanitize_string_length_limit(self):
        """Test string length limiting."""
        long_string = "A" * 2000
        result = MetadataSanitizer.sanitize_string(long_string, max_length=1000)
        assert len(result) <= 1000

    def test_sanitize_metadata_dict(self):
        """Test metadata dictionary sanitization."""
        metadata = {
            "safe_key": "safe_value",
            "<script>evil</script>": "value",
            "key": "<script>alert('XSS')</script>",
            "nested": {"inner": "<img src=x onerror=alert('XSS')>"},
            "number": 123,
            "boolean": True,
            "list": ["item1", "<script>evil</script>", 456],
        }

        result = MetadataSanitizer.sanitize_metadata(metadata)

        # Safe values should be preserved
        assert result["safe_key"] == "safe_value"
        assert result["number"] == 123
        assert result["boolean"] is True

        # Dangerous content should be sanitized
        assert "<script>" not in str(result)
        assert "onerror" not in str(result)

        # Lists should be sanitized
        assert isinstance(result["list"], list)
        assert result["list"][0] == "item1"
        assert result["list"][2] == 456

    def test_sanitize_metadata_nested(self):
        """Test deeply nested metadata sanitization."""
        metadata = {"level1": {"level2": {"level3": {"xss": "<script>alert('XSS')</script>"}}}}

        result = MetadataSanitizer.sanitize_metadata(metadata)

        # Navigate to nested value
        nested_value = result["level1"]["level2"]["level3"]["xss"]
        assert nested_value == "[Content removed for security]"

    def test_validate_no_xss(self):
        """Test XSS validation function."""
        # Safe content should pass
        assert MetadataSanitizer.validate_no_xss("Hello World") is True
        assert MetadataSanitizer.validate_no_xss("Test 123") is True
        assert MetadataSanitizer.validate_no_xss("") is True

        # XSS content should fail
        # Note: Some vectors in XSS_VECTORS are already HTML-encoded, which won't match dangerous patterns
        dangerous_vectors = [
            vector
            for vector in self.XSS_VECTORS
            if not vector.startswith("&lt;") and not vector.startswith("&#") and not vector.startswith("%3C")
        ]
        for vector in dangerous_vectors:
            assert MetadataSanitizer.validate_no_xss(vector) is False, f"XSS vector passed validation: {vector}"

    def test_escape_for_json(self):
        """Test JSON-specific escaping."""
        # Forward slashes should be escaped to prevent </script> injection
        input_str = "</script>"
        result = MetadataSanitizer.escape_for_json(input_str)
        assert "<\\/script>" not in result  # Should be removed or escaped

        # Other escaping should still apply
        input_str = '"test" & <div>'
        result = MetadataSanitizer.escape_for_json(input_str)
        assert '"' not in result or "&quot;" in result
        assert "<" not in result or "&lt;" in result

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Empty input
        assert MetadataSanitizer.sanitize_string("") == ""
        assert MetadataSanitizer.sanitize_string(None) == ""
        assert MetadataSanitizer.sanitize_metadata({}) == {}
        assert MetadataSanitizer.sanitize_metadata(None) == {}

        # Very long keys
        long_key = "A" * 200
        metadata = {long_key: "value"}
        result = MetadataSanitizer.sanitize_metadata(metadata)
        # Long keys should be skipped
        assert long_key not in result

        # Large lists
        large_list = ["item"] * 200
        metadata = {"list": large_list}
        result = MetadataSanitizer.sanitize_metadata(metadata)
        # List should be truncated
        assert len(result["list"]) <= 100

    def test_mixed_content(self):
        """Test mixed safe and unsafe content."""
        input_str = "Hello <script>alert('XSS')</script> World"
        result = MetadataSanitizer.sanitize_string(input_str)
        # Entire string should be rejected if it contains dangerous patterns
        assert result == "[Content removed for security]"

    def test_unicode_handling(self):
        """Test proper Unicode handling."""
        # Unicode should be preserved
        unicode_str = "Hello 世界 🌍"
        result = MetadataSanitizer.sanitize_string(unicode_str)
        assert result == unicode_str

        # Unicode with XSS should be sanitized
        unicode_xss = "Hello 世界 <script>alert('XSS')</script>"
        result = MetadataSanitizer.sanitize_string(unicode_xss)
        assert result == "[Content removed for security]"


class TestCSPHeaders:
    """Test Content Security Policy headers are properly set."""

    @pytest.mark.asyncio()
    async def test_csp_headers_on_chunking_endpoints(self, test_client):
        """Test that CSP headers are present on chunking endpoints."""
        # This would require a test client setup
        # Placeholder for integration test

    @pytest.mark.asyncio()
    async def test_strict_csp_for_chunking(self, test_client):
        """Test that chunking endpoints have strict CSP."""
        # This would require a test client setup
        # Placeholder for integration test
