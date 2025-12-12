"""Unit tests for ImapConnector."""

import email
from email.header import make_header, Header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.connectors.imap import (
    DEFAULT_MAX_MESSAGES,
    DEFAULT_SINCE_DAYS,
    MAX_EMAIL_BODY_SIZE,
    ImapConnector,
    _decode_mime_header,
    _email_to_markdown,
    _format_email_date,
    _get_email_body,
)


class TestImapConnectorInit:
    """Test ImapConnector initialization and config validation."""

    def test_valid_config(self):
        """Test initialization with valid config."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        assert connector.host == "imap.example.com"
        assert connector.username == "user@example.com"

    def test_valid_config_with_all_options(self):
        """Test initialization with all options."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "port": 143,
            "use_ssl": False,
            "username": "user@example.com",
            "mailboxes": ["INBOX", "Sent"],
            "since_days": 60,
            "max_messages": 500,
        })
        assert connector.host == "imap.example.com"
        assert connector.port == 143
        assert connector.use_ssl is False
        assert connector.username == "user@example.com"
        assert connector.mailboxes == ["INBOX", "Sent"]
        assert connector.since_days == 60
        assert connector.max_messages == 500

    def test_missing_host(self):
        """Test initialization fails without host."""
        with pytest.raises(ValueError) as exc_info:
            ImapConnector({"username": "user@example.com"})
        assert "host" in str(exc_info.value)

    def test_missing_username(self):
        """Test initialization fails without username."""
        with pytest.raises(ValueError) as exc_info:
            ImapConnector({"host": "imap.example.com"})
        assert "username" in str(exc_info.value)


class TestImapConnectorProperties:
    """Test ImapConnector property methods."""

    @pytest.fixture()
    def connector(self):
        return ImapConnector({
            "host": "imap.gmail.com",
            "port": 993,
            "use_ssl": True,
            "username": "user@gmail.com",
            "mailboxes": ["INBOX", "Sent", "Drafts"],
            "since_days": 14,
            "max_messages": 200,
        })

    def test_host(self, connector):
        assert connector.host == "imap.gmail.com"

    def test_port(self, connector):
        assert connector.port == 993

    def test_port_default(self):
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        assert connector.port == 993

    def test_use_ssl(self, connector):
        assert connector.use_ssl is True

    def test_use_ssl_default(self):
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        assert connector.use_ssl is True

    def test_username(self, connector):
        assert connector.username == "user@gmail.com"

    def test_mailboxes(self, connector):
        assert connector.mailboxes == ["INBOX", "Sent", "Drafts"]

    def test_mailboxes_default(self):
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        assert connector.mailboxes == ["INBOX"]

    def test_mailboxes_empty_becomes_inbox(self):
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
            "mailboxes": [],
        })
        assert connector.mailboxes == ["INBOX"]

    def test_since_days(self, connector):
        assert connector.since_days == 14

    def test_since_days_default(self):
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        assert connector.since_days == DEFAULT_SINCE_DAYS

    def test_max_messages(self, connector):
        assert connector.max_messages == 200

    def test_max_messages_default(self):
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        assert connector.max_messages == DEFAULT_MAX_MESSAGES


class TestImapConnectorCredentials:
    """Test credential and cursor handling."""

    def test_set_credentials(self):
        """Test setting password credentials."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        connector.set_credentials(password="secret123")
        assert connector._password == "secret123"

    def test_set_credentials_none(self):
        """Test setting None credentials."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        connector.set_credentials(password=None)
        assert connector._password is None

    def test_set_cursor(self):
        """Test setting sync cursor."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        cursor = {
            "INBOX": {"uidvalidity": 12345, "last_uid": 500},
            "Sent": {"uidvalidity": 12346, "last_uid": 200},
        }
        connector.set_cursor(cursor)
        assert connector._cursor == cursor

    def test_set_cursor_none(self):
        """Test setting None cursor initializes empty dict."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        connector.set_cursor(None)
        assert connector._cursor == {}

    def test_get_cursor_returns_shallow_copy(self):
        """Test get_cursor returns a shallow copy (top-level keys only)."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        original_cursor = {"INBOX": {"uidvalidity": 100, "last_uid": 50}}
        connector.set_cursor(original_cursor)

        returned_cursor = connector.get_cursor()
        assert returned_cursor == original_cursor

        # Adding a new top-level key to returned copy shouldn't affect original
        returned_cursor["NewMailbox"] = {"uidvalidity": 200, "last_uid": 10}
        assert "NewMailbox" not in connector._cursor


class TestDecodeMimeHeader:
    """Test _decode_mime_header helper function."""

    def test_decode_plain_header(self):
        """Test decoding plain ASCII header."""
        result = _decode_mime_header("Hello World")
        assert result == "Hello World"

    def test_decode_none_header(self):
        """Test decoding None header."""
        result = _decode_mime_header(None)
        assert result == ""

    def test_decode_empty_header(self):
        """Test decoding empty header."""
        result = _decode_mime_header("")
        assert result == ""

    def test_decode_utf8_header(self):
        """Test decoding UTF-8 encoded header."""
        # Create a MIME-encoded header
        h = Header("Привет мир", "utf-8")
        encoded = str(h)
        result = _decode_mime_header(encoded)
        assert "Привет" in result

    def test_decode_iso8859_header(self):
        """Test decoding ISO-8859-1 encoded header."""
        h = Header("Café", "iso-8859-1")
        encoded = str(h)
        result = _decode_mime_header(encoded)
        assert "Caf" in result


class TestFormatEmailDate:
    """Test _format_email_date helper function."""

    def test_format_valid_date(self):
        """Test formatting valid email date."""
        result = _format_email_date("Mon, 12 Dec 2025 10:30:00 -0600")
        assert result is not None
        assert "Dec" in result
        assert "2025" in result

    def test_format_none_date(self):
        """Test formatting None date."""
        result = _format_email_date(None)
        assert result is None

    def test_format_invalid_date(self):
        """Test formatting invalid date returns original."""
        result = _format_email_date("not a date")
        assert result == "not a date"


class TestGetEmailBody:
    """Test _get_email_body helper function."""

    def test_get_plain_text_body(self):
        """Test extracting plain text body from simple message."""
        msg = email.message_from_string(
            "Subject: Test\n"
            "Content-Type: text/plain; charset=utf-8\n"
            "\n"
            "Hello, this is the body."
        )
        result = _get_email_body(msg)
        assert "Hello, this is the body." in result

    def test_get_multipart_body(self):
        """Test extracting body from multipart message."""
        msg = MIMEMultipart("alternative")
        msg.attach(MIMEText("Plain text body", "plain"))
        msg.attach(MIMEText("<html><body>HTML body</body></html>", "html"))

        result = _get_email_body(msg)
        assert "Plain text body" in result

    def test_get_html_fallback_body(self):
        """Test extracting HTML body when no plain text."""
        msg = MIMEMultipart()
        msg.attach(MIMEText("<html><body><p>HTML only body</p></body></html>", "html"))

        result = _get_email_body(msg)
        assert "HTML only body" in result
        # HTML tags should be stripped
        assert "<html>" not in result

    def test_skip_attachment(self):
        """Test that attachments are skipped."""
        msg = MIMEMultipart()
        msg.attach(MIMEText("Main body text", "plain"))

        attachment = MIMEText("Attachment content", "plain")
        attachment.add_header("Content-Disposition", "attachment", filename="test.txt")
        msg.attach(attachment)

        result = _get_email_body(msg)
        assert "Main body text" in result
        assert "Attachment content" not in result

    def test_truncate_large_body(self):
        """Test that large bodies are truncated."""
        large_content = "x" * (MAX_EMAIL_BODY_SIZE + 1000)
        msg = email.message_from_string(
            f"Subject: Test\n"
            f"Content-Type: text/plain; charset=utf-8\n"
            f"\n"
            f"{large_content}"
        )
        result = _get_email_body(msg)
        assert len(result) <= MAX_EMAIL_BODY_SIZE + 100  # Some buffer for truncation message
        assert "[Truncated" in result


class TestEmailToMarkdown:
    """Test _email_to_markdown helper function."""

    def test_full_email_to_markdown(self):
        """Test converting full email to markdown."""
        result = _email_to_markdown(
            subject="Test Subject",
            from_addr="sender@example.com",
            to_addr="recipient@example.com",
            date="Mon, 12 Dec 2025 10:30:00 -0600",
            body="This is the email body.",
        )

        assert "## Test Subject" in result
        assert "**From:** sender@example.com" in result
        assert "**To:** recipient@example.com" in result
        assert "**Date:** Mon, 12 Dec 2025 10:30:00 -0600" in result
        assert "---" in result
        assert "This is the email body." in result

    def test_email_to_markdown_no_date(self):
        """Test converting email without date."""
        result = _email_to_markdown(
            subject="Test Subject",
            from_addr="sender@example.com",
            to_addr="recipient@example.com",
            date=None,
            body="Body text.",
        )

        assert "## Test Subject" in result
        assert "**Date:**" not in result


class TestImapConnectorConnect:
    """Test _connect method."""

    def test_connect_ssl(self):
        """Test SSL connection creation."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "port": 993,
            "use_ssl": True,
            "username": "user@example.com",
        })

        with patch("shared.connectors.imap.imaplib.IMAP4_SSL") as mock_ssl:
            mock_ssl.return_value = MagicMock()
            conn = connector._connect()

            mock_ssl.assert_called_once()
            assert "imap.example.com" in str(mock_ssl.call_args)

    def test_connect_no_ssl(self):
        """Test non-SSL connection creation."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "port": 143,
            "use_ssl": False,
            "username": "user@example.com",
        })

        with patch("shared.connectors.imap.imaplib.IMAP4") as mock_imap:
            mock_imap.return_value = MagicMock()
            conn = connector._connect()

            mock_imap.assert_called_once_with("imap.example.com", 143)


class TestImapConnectorAuthenticate:
    """Test authenticate method."""

    @pytest.mark.asyncio()
    async def test_authenticate_success(self):
        """Test successful authentication."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        connector.set_credentials(password="secret")

        with patch.object(connector, "_connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.login.return_value = ("OK", [])
            mock_connect.return_value = mock_conn

            result = await connector.authenticate()

            assert result is True
            mock_conn.login.assert_called_once_with("user@example.com", "secret")
            mock_conn.logout.assert_called_once()

    @pytest.mark.asyncio()
    async def test_authenticate_no_password(self):
        """Test authentication fails without password."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })

        with pytest.raises(ValueError) as exc_info:
            await connector.authenticate()
        assert "Password not set" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_authenticate_failure(self):
        """Test authentication failure."""
        import imaplib

        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })
        connector.set_credentials(password="wrong")

        with patch.object(connector, "_connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.login.side_effect = imaplib.IMAP4.error("LOGIN failed")
            mock_connect.return_value = mock_conn

            with pytest.raises(ValueError) as exc_info:
                await connector.authenticate()
            assert "IMAP authentication failed" in str(exc_info.value)


class TestImapConnectorLoadDocuments:
    """Test load_documents method."""

    @pytest.mark.asyncio()
    async def test_load_documents_no_password(self):
        """Test load_documents fails without password."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })

        with pytest.raises(ValueError) as exc_info:
            async for _ in connector.load_documents():
                pass
        assert "Password not set" in str(exc_info.value)

    @pytest.mark.asyncio()
    async def test_load_documents_empty_mailbox(self):
        """Test load_documents with empty mailbox."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
            "mailboxes": ["INBOX"],
        })
        connector.set_credentials(password="secret")

        mock_conn = MagicMock()
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"0"])
        mock_conn.response.return_value = ("OK", [b"12345"])
        mock_conn.uid.return_value = ("OK", [b""])  # No messages

        with patch.object(connector, "_connect", return_value=mock_conn):
            documents = []
            async for doc in connector.load_documents():
                documents.append(doc)

            assert len(documents) == 0
            mock_conn.logout.assert_called()

    @pytest.mark.asyncio()
    async def test_load_documents_with_messages(self):
        """Test load_documents with messages."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
            "mailboxes": ["INBOX"],
        })
        connector.set_credentials(password="secret")

        # Create a test email
        test_email = (
            "From: sender@example.com\r\n"
            "To: user@example.com\r\n"
            "Subject: Test Email\r\n"
            "Date: Mon, 12 Dec 2025 10:30:00 -0600\r\n"
            "Message-ID: <test123@example.com>\r\n"
            "\r\n"
            "This is a test email body."
        )

        mock_conn = MagicMock()
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"1"])
        mock_conn.response.return_value = ("OK", [b"12345"])

        # Search returns one UID
        def mock_uid(cmd, *args):
            if cmd == "SEARCH":
                return ("OK", [b"100"])
            if cmd == "FETCH":
                return ("OK", [(b"100 (RFC822 {123}", test_email.encode())])
            return ("OK", [])

        mock_conn.uid = mock_uid

        with patch.object(connector, "_connect", return_value=mock_conn):
            documents = []
            async for doc in connector.load_documents():
                documents.append(doc)

            assert len(documents) == 1
            assert "Test Email" in documents[0].content
            assert documents[0].source_type == "imap"
            assert "imap://" in documents[0].unique_id

    @pytest.mark.asyncio()
    async def test_load_documents_max_messages_limit(self):
        """Test load_documents respects max_messages limit."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
            "mailboxes": ["INBOX"],
            "max_messages": 2,
        })
        connector.set_credentials(password="secret")

        # Create test emails
        def make_email(uid):
            return (
                f"From: sender@example.com\r\n"
                f"To: user@example.com\r\n"
                f"Subject: Test Email {uid}\r\n"
                f"Message-ID: <test{uid}@example.com>\r\n"
                f"\r\n"
                f"Body {uid}"
            ).encode()

        mock_conn = MagicMock()
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"5"])
        mock_conn.response.return_value = ("OK", [b"12345"])

        fetch_calls = []

        def mock_uid(cmd, *args):
            if cmd == "SEARCH":
                return ("OK", [b"100 101 102 103 104"])  # 5 messages available
            if cmd == "FETCH":
                uid = int(args[0])
                fetch_calls.append(uid)
                return ("OK", [(b"fetch response", make_email(uid))])
            return ("OK", [])

        mock_conn.uid = mock_uid

        with patch.object(connector, "_connect", return_value=mock_conn):
            documents = []
            async for doc in connector.load_documents():
                documents.append(doc)

            # Should only get 2 documents due to max_messages=2
            assert len(documents) == 2


class TestImapConnectorCursor:
    """Test cursor handling during sync."""

    @pytest.mark.asyncio()
    async def test_cursor_updated_after_sync(self):
        """Test cursor is updated after successful sync."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
            "mailboxes": ["INBOX"],
        })
        connector.set_credentials(password="secret")
        connector.set_cursor({})

        test_email = (
            "From: sender@example.com\r\n"
            "Subject: Test\r\n"
            "Message-ID: <test@example.com>\r\n"
            "\r\n"
            "Body"
        ).encode()

        mock_conn = MagicMock()
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"1"])
        mock_conn.response.return_value = ("OK", [b"99999"])

        def mock_uid(cmd, *args):
            if cmd == "SEARCH":
                return ("OK", [b"500"])
            if cmd == "FETCH":
                return ("OK", [(b"fetch", test_email)])
            return ("OK", [])

        mock_conn.uid = mock_uid

        with patch.object(connector, "_connect", return_value=mock_conn):
            async for _ in connector.load_documents():
                pass

            cursor = connector.get_cursor()
            assert "INBOX" in cursor
            assert cursor["INBOX"]["uidvalidity"] == 99999
            assert cursor["INBOX"]["last_uid"] == 500

    @pytest.mark.asyncio()
    async def test_uidvalidity_change_resets_cursor(self):
        """Test UIDVALIDITY change resets cursor."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
            "mailboxes": ["INBOX"],
        })
        connector.set_credentials(password="secret")
        # Set old cursor with different uidvalidity
        connector.set_cursor({
            "INBOX": {"uidvalidity": 11111, "last_uid": 1000},
        })

        mock_conn = MagicMock()
        mock_conn.login.return_value = ("OK", [])
        mock_conn.select.return_value = ("OK", [b"1"])
        mock_conn.response.return_value = ("OK", [b"22222"])  # Different uidvalidity

        def mock_uid(cmd, *args):
            if cmd == "SEARCH":
                # Search should use SINCE (initial) not UID range (incremental)
                # because uidvalidity changed
                if "SINCE" in str(args):
                    return ("OK", [b""])  # No results
                return ("OK", [b""])
            return ("OK", [])

        mock_conn.uid = mock_uid

        with patch.object(connector, "_connect", return_value=mock_conn):
            async for _ in connector.load_documents():
                pass

            cursor = connector.get_cursor()
            # Cursor should be updated with new uidvalidity
            assert cursor["INBOX"]["uidvalidity"] == 22222


class TestImapConnectorCleanup:
    """Test cleanup method."""

    def test_cleanup_with_connection(self):
        """Test cleanup closes connection."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })

        mock_conn = MagicMock()
        connector._connection = mock_conn

        connector.cleanup()

        mock_conn.logout.assert_called_once()
        assert connector._connection is None

    def test_cleanup_without_connection(self):
        """Test cleanup handles no connection gracefully."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })

        # Should not raise
        connector.cleanup()
        assert connector._connection is None

    def test_cleanup_handles_exception(self):
        """Test cleanup handles logout exception."""
        connector = ImapConnector({
            "host": "imap.example.com",
            "username": "user@example.com",
        })

        mock_conn = MagicMock()
        mock_conn.logout.side_effect = Exception("Connection lost")
        connector._connection = mock_conn

        # Should not raise
        connector.cleanup()
        assert connector._connection is None
