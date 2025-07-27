#!/usr/bin/env python3
"""Script to fix the failing tests in test_tasks_helpers_original.py"""

import re

# Read the original file
with open('/home/dockertest/semantik/tests/webui/test_tasks_helpers_original.py', 'r') as f:
    content = f.read()

# Fix 1: Replace the audit_log_operation_success test
old_test = '''    async def test_audit_log_operation_success(self):
        """Test successful audit log creation."""
        with patch("packages.shared.database.database.AsyncSessionLocal") as mock_session_local:
            # Setup session mock with proper async context manager
            mock_session = AsyncMock()
            mock_session.add = Mock()
            mock_session.commit = AsyncMock()

            # Create async context manager
            mock_session_cm = AsyncMock()
            mock_session_cm.__aenter__.return_value = mock_session
            mock_session_cm.__aexit__.return_value = None
            mock_session_local.return_value = mock_session_cm

            # Create audit log
            await _audit_log_operation(
                collection_id="col-123",
                operation_id=456,
                user_id=1,
                action="test_action",
                details={"key": "value", "password": "secret"},
            )

            # Verify session operations
            assert mock_session.add.called
            assert mock_session.commit.called

            # Check the audit log object that was added
            audit_log = mock_session.add.call_args[0][0]
            assert hasattr(audit_log, "collection_id")
            assert audit_log.collection_id == "col-123"
            assert audit_log.operation_id == 456
            assert audit_log.user_id == 1
            assert audit_log.action == "test_action"

            # Details should be sanitized
            assert "password" not in audit_log.details
            assert audit_log.details["key"] == "value"'''

new_test = '''    @patch("packages.shared.database.models.CollectionAuditLog")
    @patch("packages.shared.database.database.AsyncSessionLocal")
    async def test_audit_log_operation_success(self, mock_session_local, mock_audit_log_class):
        """Test successful audit log creation."""
        # Setup session mock with proper async context manager
        mock_session = MagicMock()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()
        
        # Create async context manager
        mock_session_cm = AsyncMock()
        mock_session_cm.__aenter__.return_value = mock_session
        mock_session_cm.__aexit__.return_value = None
        mock_session_local.return_value = mock_session_cm
        
        # Mock audit log instance
        mock_audit_log = MagicMock()
        mock_audit_log_class.return_value = mock_audit_log
        
        # Create audit log
        await _audit_log_operation(
            collection_id="col-123",
            operation_id=456,
            user_id=1,
            action="test_action",
            details={"key": "value", "password": "secret"}
        )
        
        # Verify audit log was created with correct parameters
        mock_audit_log_class.assert_called_once()
        call_kwargs = mock_audit_log_class.call_args[1]
        assert call_kwargs["collection_id"] == "col-123"
        assert call_kwargs["operation_id"] == 456
        assert call_kwargs["user_id"] == 1
        assert call_kwargs["action"] == "test_action"
        # Details should be sanitized
        assert "password" not in call_kwargs["details"]
        assert call_kwargs["details"]["key"] == "value"
        
        # Verify session operations
        mock_session.add.assert_called_once_with(mock_audit_log)
        mock_session.commit.assert_called_once()'''

content = content.replace(old_test, new_test)

# Write the fixed file
with open('/home/dockertest/semantik/tests/webui/test_tasks_helpers_original.py', 'w') as f:
    f.write(content)

print("Fixed test_audit_log_operation_success")