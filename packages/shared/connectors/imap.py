"""IMAP email connector for document sources."""

from __future__ import annotations

import asyncio
import contextlib
import email
import email.utils
import hashlib
import imaplib
import logging
import re
import ssl
from datetime import UTC, datetime
from email.header import decode_header
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from email.message import Message

from shared.connectors.base import BaseConnector
from shared.dtos.ingestion import IngestedDocument
from shared.utils.hashing import compute_content_hash

logger = logging.getLogger(__name__)

# Default max messages per sync
DEFAULT_MAX_MESSAGES = 1000

# Default days to look back for initial sync
DEFAULT_SINCE_DAYS = 30

# Max email body size (5 MB)
MAX_EMAIL_BODY_SIZE = 5 * 1024 * 1024


def _decode_mime_header(header: str | None) -> str:
    """Decode a MIME-encoded email header to string."""
    if not header:
        return ""

    parts = []
    for decoded, charset in decode_header(header):
        if isinstance(decoded, bytes):
            try:
                charset = charset or "utf-8"
                parts.append(decoded.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                parts.append(decoded.decode("utf-8", errors="replace"))
        else:
            parts.append(str(decoded))

    return " ".join(parts)


def _format_email_date(date_str: str | None) -> str | None:
    """Parse and format an email date header."""
    if not date_str:
        return None

    try:
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed.strftime("%a, %d %b %Y %H:%M:%S %z")
    except (ValueError, TypeError):
        return date_str


def _get_email_body(msg: Message) -> str:
    """Extract plain text body from email message."""
    body_parts: list[str] = []

    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition", ""))

            # Skip attachments
            if "attachment" in content_disposition:
                continue

            if content_type == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        body_parts.append(payload.decode(charset, errors="replace"))
                    except (LookupError, UnicodeDecodeError):
                        body_parts.append(payload.decode("utf-8", errors="replace"))

            elif content_type == "text/html" and not body_parts:
                # Fall back to HTML if no plain text
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    charset = part.get_content_charset() or "utf-8"
                    try:
                        html = payload.decode(charset, errors="replace")
                        # Basic HTML strip - just remove tags
                        text = re.sub(r"<[^>]+>", " ", html)
                        text = re.sub(r"\s+", " ", text)
                        body_parts.append(text.strip())
                    except (LookupError, UnicodeDecodeError):
                        pass
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            charset = msg.get_content_charset() or "utf-8"
            try:
                body_parts.append(payload.decode(charset, errors="replace"))
            except (LookupError, UnicodeDecodeError):
                body_parts.append(payload.decode("utf-8", errors="replace"))

    body = "\n\n".join(body_parts)

    # Truncate if too large
    if len(body) > MAX_EMAIL_BODY_SIZE:
        body = body[:MAX_EMAIL_BODY_SIZE] + "\n\n[Truncated - email body too large]"

    return body


def _email_to_markdown(
    subject: str,
    from_addr: str,
    to_addr: str,
    date: str | None,
    body: str,
) -> str:
    """Convert email components to markdown format."""
    lines = [
        f"## {subject}",
        "",
        f"**From:** {from_addr}",
        f"**To:** {to_addr}",
    ]

    if date:
        lines.append(f"**Date:** {date}")

    lines.extend([
        "",
        "---",
        "",
        body,
    ])

    return "\n".join(lines)


class ImapConnector(BaseConnector):
    """Connector for IMAP email sources.

    Connects to IMAP mailboxes and indexes emails as markdown documents
    with cursor-based incremental sync using UID VALIDITY and UID.

    Config keys:
        host (required): IMAP server hostname
        port (optional): IMAP port (default: 993 for SSL)
        use_ssl (optional): Use SSL connection (default: True)
        username (required): IMAP username/email
        mailboxes (optional): List of mailbox names to sync (default: ["INBOX"])
        since_days (optional): Days to look back for initial sync (default: 30)
        max_messages (optional): Max messages per sync (default: 1000)

    Secrets (set via set_credentials):
        password: IMAP password

    Cursor format (stored in source.meta.imap_cursor):
        {
            "INBOX": {"uidvalidity": 12345, "last_uid": 500},
            "Sent": {"uidvalidity": 12346, "last_uid": 200}
        }

    Example:
        ```python
        connector = ImapConnector({
            "host": "imap.gmail.com",
            "port": 993,
            "username": "user@example.com",
            "mailboxes": ["INBOX", "Sent"],
        })
        connector.set_credentials(password="app_password")
        connector.set_cursor(existing_cursor)

        if await connector.authenticate():
            async for doc in connector.load_documents():
                print(doc.unique_id)

        # Save updated cursor
        new_cursor = connector.get_cursor()
        ```
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the IMAP connector."""
        self._password: str | None = None
        self._cursor: dict[str, dict[str, int]] = {}
        self._connection: imaplib.IMAP4_SSL | imaplib.IMAP4 | None = None
        super().__init__(config)

    def validate_config(self) -> None:
        """Validate required config keys."""
        if "host" not in self._config:
            raise ValueError("ImapConnector requires 'host' in config")
        if "username" not in self._config:
            raise ValueError("ImapConnector requires 'username' in config")

    def set_credentials(self, password: str | None = None) -> None:
        """Set authentication credentials.

        Args:
            password: IMAP password
        """
        self._password = password

    def set_cursor(self, cursor: dict[str, dict[str, int]] | None) -> None:
        """Set the sync cursor from previous run.

        Args:
            cursor: Cursor dict from source.meta.imap_cursor
        """
        self._cursor = cursor or {}

    def get_cursor(self) -> dict[str, dict[str, int]]:
        """Get the updated cursor after sync.

        Returns:
            Cursor dict to store in source.meta.imap_cursor
        """
        return self._cursor.copy()

    @property
    def host(self) -> str:
        """Get the IMAP host."""
        return str(self._config["host"])

    @property
    def port(self) -> int:
        """Get the IMAP port."""
        return int(self._config.get("port", 993))

    @property
    def use_ssl(self) -> bool:
        """Check if SSL should be used."""
        return bool(self._config.get("use_ssl", True))

    @property
    def username(self) -> str:
        """Get the IMAP username."""
        return str(self._config["username"])

    @property
    def mailboxes(self) -> list[str]:
        """Get list of mailboxes to sync."""
        boxes = self._config.get("mailboxes", ["INBOX"])
        return list(boxes) if boxes else ["INBOX"]

    @property
    def since_days(self) -> int:
        """Get initial sync lookback days."""
        return int(self._config.get("since_days", DEFAULT_SINCE_DAYS))

    @property
    def max_messages(self) -> int:
        """Get max messages per sync."""
        return int(self._config.get("max_messages", DEFAULT_MAX_MESSAGES))

    def _connect(self) -> imaplib.IMAP4_SSL | imaplib.IMAP4:
        """Create IMAP connection."""
        if self.use_ssl:
            context = ssl.create_default_context()
            return imaplib.IMAP4_SSL(self.host, self.port, ssl_context=context)
        return imaplib.IMAP4(self.host, self.port)

    async def authenticate(self) -> bool:
        """Verify IMAP credentials by connecting and logging in.

        Returns:
            True if authentication succeeds.

        Raises:
            ValueError: If authentication fails.
        """
        if not self._password:
            raise ValueError("Password not set - call set_credentials() first")

        def _auth() -> bool:
            try:
                conn = self._connect()
                conn.login(self.username, self._password or "")
                conn.logout()
                return True
            except imaplib.IMAP4.error as e:
                raise ValueError(f"IMAP authentication failed: {e}") from e

        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, _auth)
        logger.info(f"Successfully authenticated to {self.host} as {self.username}")
        return result

    async def load_documents(
        self,
        source_id: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[IngestedDocument]:
        """Yield documents from IMAP mailboxes.

        Args:
            source_id: Optional source ID (unused for IMAP)

        Yields:
            IngestedDocument for each email.
        """
        if not self._password:
            raise ValueError("Password not set - call set_credentials() first")

        # Connect to IMAP
        def _connect_and_login() -> imaplib.IMAP4_SSL | imaplib.IMAP4:
            conn = self._connect()
            conn.login(self.username, self._password or "")
            return conn

        loop = asyncio.get_event_loop()
        self._connection = await loop.run_in_executor(None, _connect_and_login)

        try:
            total_fetched = 0

            for mailbox in self.mailboxes:
                if total_fetched >= self.max_messages:
                    break

                remaining = self.max_messages - total_fetched
                async for doc in self._process_mailbox(mailbox, remaining):
                    yield doc
                    total_fetched += 1

        finally:
            # Cleanup connection
            if self._connection:
                with contextlib.suppress(Exception):
                    self._connection.logout()
                self._connection = None

    async def _process_mailbox(
        self,
        mailbox: str,
        max_count: int,
    ) -> AsyncIterator[IngestedDocument]:
        """Process a single mailbox.

        Args:
            mailbox: Mailbox name
            max_count: Maximum messages to fetch

        Yields:
            IngestedDocument for each email
        """
        if not self._connection:
            return

        loop = asyncio.get_event_loop()

        # Select mailbox
        def _select() -> tuple[str, list[bytes | None]]:
            if not self._connection:
                raise ValueError("No connection")
            return self._connection.select(f'"{mailbox}"')

        try:
            status, data = await loop.run_in_executor(None, _select)
            if status != "OK":
                logger.warning(f"Cannot select mailbox {mailbox}: {data}")
                return
        except Exception as e:
            logger.warning(f"Failed to select mailbox {mailbox}: {e}")
            return

        # Get UIDVALIDITY
        def _get_uidvalidity() -> int | None:
            if not self._connection:
                return None
            status, data = self._connection.response("UIDVALIDITY")
            if status == "OK" and data and data[0]:
                try:
                    return int(data[0])
                except (ValueError, TypeError):
                    pass
            return None

        uidvalidity = await loop.run_in_executor(None, _get_uidvalidity)

        # Get cursor for this mailbox
        mailbox_cursor = self._cursor.get(mailbox, {})
        prev_uidvalidity = mailbox_cursor.get("uidvalidity")
        last_uid = mailbox_cursor.get("last_uid", 0)

        # Check if UIDVALIDITY changed (mailbox reset)
        if (
            uidvalidity is not None
            and prev_uidvalidity is not None
            and uidvalidity != prev_uidvalidity
        ):
            logger.warning(
                f"UIDVALIDITY changed for {mailbox}: {prev_uidvalidity} -> {uidvalidity}. "
                "Resetting cursor."
            )
            last_uid = 0

        # Build search criteria
        search_criteria: str
        if last_uid > 0:
            # Incremental: get messages with UID > last_uid
            search_criteria = f"UID {last_uid + 1}:*"
        else:
            # Initial: get messages from since_days ago
            since_date = datetime.now(UTC).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            from datetime import timedelta

            since_date = since_date - timedelta(days=self.since_days)
            date_str = since_date.strftime("%d-%b-%Y")
            search_criteria = f"SINCE {date_str}"

        # Search for messages
        def _search() -> list[bytes]:
            if not self._connection:
                return []
            status, data = self._connection.uid("SEARCH", "", search_criteria)
            if status == "OK" and data and data[0] and isinstance(data[0], bytes):
                return list(data[0].split())
            return []

        try:
            msg_uids = await loop.run_in_executor(None, _search)
        except Exception as e:
            logger.error(f"Failed to search mailbox {mailbox}: {e}")
            return

        if not msg_uids:
            logger.debug(f"No new messages in {mailbox}")
            # Still update cursor with uidvalidity
            if uidvalidity is not None:
                self._cursor[mailbox] = {
                    "uidvalidity": uidvalidity,
                    "last_uid": last_uid,
                }
            return

        # Limit messages
        msg_uids = msg_uids[:max_count]
        logger.info(f"Processing {len(msg_uids)} messages from {mailbox}")

        highest_uid = last_uid

        for uid_bytes in msg_uids:
            uid = int(uid_bytes)

            # Fetch message
            def _fetch(u: int = uid) -> bytes | None:
                if not self._connection:
                    return None
                status, data = self._connection.uid("FETCH", str(u), "(RFC822)")
                if status == "OK" and data and data[0] and isinstance(data[0], tuple):
                    return data[0][1] if isinstance(data[0][1], bytes) else None
                return None

            try:
                raw_email = await loop.run_in_executor(None, _fetch)
            except Exception as e:
                logger.warning(f"Failed to fetch UID {uid} from {mailbox}: {e}")
                continue

            if not raw_email:
                continue

            # Parse email
            try:
                msg = email.message_from_bytes(raw_email)
            except Exception as e:
                logger.warning(f"Failed to parse email UID {uid}: {e}")
                continue

            # Extract components
            subject = _decode_mime_header(msg.get("Subject")) or "(No Subject)"
            from_addr = _decode_mime_header(msg.get("From")) or ""
            to_addr = _decode_mime_header(msg.get("To")) or ""
            email_date = _format_email_date(msg.get("Date"))
            message_id = msg.get("Message-ID") or f"<uid-{uid}@{mailbox}>"

            # Get body
            body = _get_email_body(msg)

            # Convert to markdown
            content = _email_to_markdown(subject, from_addr, to_addr, email_date, body)

            # Build unique ID
            # Use message-id hash for uniqueness across mailboxes
            msg_hash = hashlib.sha256(message_id.encode()).hexdigest()[:16]
            unique_id = f"imap://{self.host}/{mailbox};uid={uid};hash={msg_hash}"

            # Compute content hash
            content_hash = compute_content_hash(content)

            # Build metadata
            metadata: dict[str, Any] = {
                "mailbox": mailbox,
                "uid": uid,
                "uidvalidity": uidvalidity,
                "message_id": message_id,
                "subject": subject,
                "from": from_addr,
                "to": to_addr,
                "date": email_date,
            }

            yield IngestedDocument(
                content=content,
                unique_id=unique_id,
                source_type="imap",
                metadata=metadata,
                content_hash=content_hash,
                file_path=None,
            )

            # Track highest UID
            if uid > highest_uid:
                highest_uid = uid

        # Update cursor
        if uidvalidity is not None:
            self._cursor[mailbox] = {
                "uidvalidity": uidvalidity,
                "last_uid": highest_uid,
            }

    def cleanup(self) -> None:
        """Close IMAP connection if open."""
        if self._connection:
            with contextlib.suppress(Exception):
                self._connection.logout()
            self._connection = None
