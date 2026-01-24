"""IMAP email connector for inbox document sources."""

from __future__ import annotations

import asyncio
import contextlib
import email
import email.utils
import imaplib
import logging
import re
import ssl
from datetime import UTC, datetime, timedelta
from email.header import decode_header
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from shared.connectors.base import BaseConnector
from shared.pipeline.types import FileReference

logger = logging.getLogger(__name__)

# Default max messages per sync
DEFAULT_MAX_MESSAGES = 1000

# Default days to look back for initial sync
DEFAULT_SINCE_DAYS = 30


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


class ImapConnector(BaseConnector):
    """Connector for IMAP email sources.

    Connects to IMAP mailboxes and enumerates emails as FileReference objects
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
            async for file_ref in connector.enumerate():
                print(file_ref.uri, file_ref.change_hint)

        # Save updated cursor
        new_cursor = connector.get_cursor()
        ```
    """

    PLUGIN_ID: ClassVar[str] = "imap"
    PLUGIN_TYPE: ClassVar[str] = "connector"
    METADATA: ClassVar[dict[str, Any]] = {
        "name": "Email (IMAP)",
        "description": "Connect to an IMAP mailbox and index emails",
        "icon": "mail",
        "supports_sync": True,
        "preview_endpoint": "/api/v2/connectors/preview/imap",
    }

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

    @classmethod
    def get_config_fields(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "host",
                "type": "text",
                "label": "IMAP Server",
                "description": "IMAP server hostname",
                "required": True,
                "placeholder": "imap.gmail.com",
            },
            {
                "name": "port",
                "type": "number",
                "label": "Port",
                "description": "IMAP server port",
                "default": 993,
                "min": 1,
                "max": 65535,
            },
            {
                "name": "use_ssl",
                "type": "boolean",
                "label": "Use SSL",
                "description": "Connect using SSL/TLS",
                "default": True,
            },
            {
                "name": "username",
                "type": "text",
                "label": "Username",
                "description": "IMAP username or email address",
                "required": True,
                "placeholder": "user@example.com",
            },
            {
                "name": "mailboxes",
                "type": "multiselect",
                "label": "Mailboxes",
                "description": "Mailboxes to sync (leave empty for INBOX only)",
                "placeholder": "INBOX, Sent",
            },
            {
                "name": "since_days",
                "type": "number",
                "label": "Initial Days",
                "description": "Days to look back for initial sync",
                "default": 30,
                "min": 1,
                "max": 365,
            },
            {
                "name": "max_messages",
                "type": "number",
                "label": "Max Messages",
                "description": "Maximum messages per sync",
                "default": 1000,
                "min": 10,
                "max": 10000,
            },
        ]

    @classmethod
    def get_secret_fields(cls) -> list[dict[str, Any]]:
        return [
            {
                "name": "password",
                "label": "Password",
                "description": "IMAP password or app password",
                "required": True,
            },
        ]

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

        loop = asyncio.get_running_loop()
        try:
            conn = await loop.run_in_executor(None, self._connect)
            await loop.run_in_executor(None, conn.login, self.username, self._password)
            await loop.run_in_executor(None, conn.logout)
            logger.info(f"Successfully authenticated to {self.host} as {self.username}")
            return True
        except imaplib.IMAP4.error as e:
            raise ValueError(f"IMAP authentication failed: {e}") from e

    async def enumerate(
        self,
        source_id: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[FileReference]:
        """Yield file references for emails in the mailboxes.

        Connects to the IMAP server and yields FileReference objects for
        each email with basic metadata. Full email body loading is deferred
        to the pipeline executor.

        Args:
            source_id: Optional source ID (unused for IMAP)

        Yields:
            FileReference for each email message.
        """
        if not self._password:
            raise ValueError("Password not set - call set_credentials() first")

        loop = asyncio.get_running_loop()
        conn = await loop.run_in_executor(None, self._connect)
        self._connection = conn

        try:
            await loop.run_in_executor(None, conn.login, self.username, self._password)
            total_fetched = 0

            for mailbox in self.mailboxes:
                if total_fetched >= self.max_messages:
                    break

                remaining = self.max_messages - total_fetched

                async for ref in self._enumerate_mailbox(mailbox, remaining):
                    yield ref
                    total_fetched += 1

            await loop.run_in_executor(None, conn.logout)

        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP error: {e}")
            raise ValueError(f"IMAP error: {e}") from e
        finally:
            if self._connection:
                with contextlib.suppress(Exception):
                    await loop.run_in_executor(None, self._connection.logout)
            self._connection = None

    async def _enumerate_mailbox(
        self,
        mailbox: str,
        max_count: int,
    ) -> AsyncIterator[FileReference]:
        """Enumerate emails from a single mailbox.

        Args:
            mailbox: Mailbox name
            max_count: Maximum messages to fetch

        Yields:
            FileReference for each email
        """
        if not self._connection:
            return

        loop = asyncio.get_running_loop()
        conn = self._connection

        # Select mailbox
        try:
            status, data = await loop.run_in_executor(None, conn.select, f'"{mailbox}"')
            if status != "OK":
                logger.warning(f"Cannot select mailbox {mailbox}: {data}")
                return
        except Exception as e:
            logger.warning(f"Failed to select mailbox {mailbox}: {e}")
            return

        # Get UIDVALIDITY
        status, uidvalidity_data = await loop.run_in_executor(None, conn.response, "UIDVALIDITY")
        uidvalidity: int | None = None
        if status == "OK" and uidvalidity_data and uidvalidity_data[0]:
            with contextlib.suppress(ValueError, TypeError):
                uidvalidity = int(uidvalidity_data[0])

        # Get cursor for this mailbox
        mailbox_cursor = self._cursor.get(mailbox, {})
        prev_uidvalidity = mailbox_cursor.get("uidvalidity")
        last_uid = mailbox_cursor.get("last_uid", 0)

        # Check if UIDVALIDITY changed (mailbox reset)
        if uidvalidity is not None and prev_uidvalidity is not None and uidvalidity != prev_uidvalidity:
            logger.warning(f"UIDVALIDITY changed for {mailbox}: {prev_uidvalidity} -> {uidvalidity}. Resetting cursor.")
            last_uid = 0

        # Build search criteria
        search_criteria: str
        if last_uid > 0:
            # Incremental: get messages with UID > last_uid
            search_criteria = f"UID {last_uid + 1}:*"
        else:
            # Initial: get messages from since_days ago
            since_date = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            since_date = since_date - timedelta(days=self.since_days)
            date_str = since_date.strftime("%d-%b-%Y")
            search_criteria = f"SINCE {date_str}"

        # Search for messages
        try:
            status, search_data = await loop.run_in_executor(
                None, conn.uid, "SEARCH", "", search_criteria
            )
            if status != "OK" or not search_data or not search_data[0]:
                logger.debug(f"No new messages in {mailbox}")
                # Still update cursor with uidvalidity
                if uidvalidity is not None:
                    self._cursor[mailbox] = {
                        "uidvalidity": uidvalidity,
                        "last_uid": last_uid,
                    }
                return
        except Exception as e:
            logger.error(f"Failed to search mailbox {mailbox}: {e}")
            return

        msg_uids = search_data[0].split() if isinstance(search_data[0], bytes) else []
        if not msg_uids:
            logger.debug(f"No new messages in {mailbox}")
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

            try:
                # Fetch headers only (no body - defer to pipeline executor)
                status, fetch_data = await loop.run_in_executor(
                    None, conn.uid, "FETCH", str(uid), "(BODY.PEEK[HEADER] RFC822.SIZE)"
                )
                if status != "OK" or not fetch_data or not fetch_data[0]:
                    continue

                # Parse response
                raw_header = None
                message_size = 0

                for item in fetch_data:
                    if isinstance(item, tuple) and len(item) >= 2:
                        raw_header = item[1]
                    elif isinstance(item, bytes):
                        # Parse RFC822.SIZE from response
                        size_str = item.decode("utf-8", errors="replace")
                        if "RFC822.SIZE" in size_str:
                            match = re.search(r"RFC822\.SIZE\s+(\d+)", size_str)
                            if match:
                                message_size = int(match.group(1))

                if not raw_header:
                    continue

                # Parse headers
                msg = email.message_from_bytes(raw_header)

                subject = _decode_mime_header(msg.get("Subject")) or "(No Subject)"
                from_addr = _decode_mime_header(msg.get("From")) or ""
                to_addr = _decode_mime_header(msg.get("To")) or ""
                email_date = _format_email_date(msg.get("Date"))
                message_id = msg.get("Message-ID") or f"<uid-{uid}@{mailbox}>"

                # Build unique URI
                uri = f"imap://{self.host}/{mailbox};uid={uid}"

                # Build filename from subject (sanitized)
                safe_subject = re.sub(r'[<>:"/\\|?*]', "_", subject[:50]) if subject else f"email_{uid}"

                yield FileReference(
                    uri=uri,
                    source_type="imap",
                    content_type="message",
                    filename=f"{safe_subject}.eml",
                    extension=".eml",
                    mime_type="message/rfc822",
                    size_bytes=message_size,
                    change_hint=str(uid),  # UID is stable within UIDVALIDITY
                    source_metadata={
                        "uid": uid,
                        "mailbox": mailbox,
                        "message_id": message_id,
                        "subject": subject,
                        "from": from_addr,
                        "to": to_addr,
                        "date": email_date,
                        "host": self.host,
                        "username": self.username,
                        "uidvalidity": uidvalidity,
                    },
                )

                # Track highest UID
                if uid > highest_uid:
                    highest_uid = uid

            except Exception as e:
                logger.warning(f"Failed to fetch UID {uid} from {mailbox}: {e}")
                continue

        # Update cursor
        if uidvalidity is not None:
            self._cursor[mailbox] = {
                "uidvalidity": uidvalidity,
                "last_uid": highest_uid,
            }

    async def load_content(self, file_ref: FileReference) -> bytes:
        """Load raw content bytes for an email message.

        Fetches the full RFC822 message content from the IMAP server.

        Args:
            file_ref: File reference with uid and mailbox in source_metadata

        Returns:
            Raw RFC822 email content bytes

        Raises:
            ValueError: If required metadata is missing or fetch fails
        """
        uid = file_ref.source_metadata.get("uid")
        mailbox = file_ref.source_metadata.get("mailbox")

        if uid is None or not mailbox:
            raise ValueError(f"Missing uid or mailbox in source_metadata for {file_ref.uri}")

        if not self._password:
            raise ValueError("Password not set - call set_credentials() first")

        loop = asyncio.get_running_loop()

        # Create a new connection for content loading
        conn = await loop.run_in_executor(None, self._connect)
        try:
            await loop.run_in_executor(None, conn.login, self.username, self._password)

            # Select the mailbox
            status, data = await loop.run_in_executor(None, conn.select, f'"{mailbox}"')
            if status != "OK":
                raise ValueError(f"Cannot select mailbox {mailbox}: {data}")

            # Fetch full message content
            status, fetch_data = await loop.run_in_executor(
                None, conn.uid, "FETCH", str(uid), "(RFC822)"
            )
            if status != "OK" or not fetch_data or not fetch_data[0]:
                raise ValueError(f"Cannot fetch message UID {uid} from {mailbox}")

            # Extract raw message bytes
            for item in fetch_data:
                if isinstance(item, tuple) and len(item) >= 2:
                    return item[1] if isinstance(item[1], bytes) else item[1].encode()

            raise ValueError(f"No content returned for UID {uid} from {mailbox}")

        finally:
            with contextlib.suppress(Exception):
                await loop.run_in_executor(None, conn.logout)

    def cleanup(self) -> None:
        """Close IMAP connection if open."""
        if self._connection:
            with contextlib.suppress(Exception):
                self._connection.logout()
            self._connection = None
