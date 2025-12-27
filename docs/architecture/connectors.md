# Connectors Architecture

> **Location:** `packages/shared/connectors/`

## Overview

The connector system provides a pluggable architecture for ingesting data from various sources. Each connector defines its configuration schema, secrets handling, and document iteration logic.

## Connector Definitions

### Structure
```python
CONNECTOR_DEFINITIONS = {
    "directory": ConnectorDefinition(
        name="Local Directory",
        description="Scan local filesystem directory",
        icon="folder",
        fields=[
            FieldDefinition(
                name="path",
                label="Directory Path",
                type="text",
                required=True,
                placeholder="/path/to/documents"
            ),
            FieldDefinition(
                name="file_extensions",
                label="File Extensions",
                type="multi-select",
                required=False,
                options=[".pdf", ".txt", ".md", ".docx", ".html"],
                default=[".pdf", ".txt", ".md"]
            ),
            FieldDefinition(
                name="recursive",
                label="Include Subdirectories",
                type="checkbox",
                default=True
            )
        ],
        secrets=[],
        supports_preview=False,
        supports_continuous_sync=True
    ),

    "git": ConnectorDefinition(
        name="Git Repository",
        description="Clone and index Git repository",
        icon="git-branch",
        fields=[
            FieldDefinition(
                name="url",
                label="Repository URL",
                type="text",
                required=True,
                placeholder="https://github.com/user/repo.git"
            ),
            FieldDefinition(
                name="branch",
                label="Branch",
                type="text",
                required=False,
                default="main"
            ),
            FieldDefinition(
                name="auth_method",
                label="Authentication",
                type="select",
                options=["none", "token", "ssh_key"],
                default="none"
            ),
            FieldDefinition(
                name="file_extensions",
                label="File Extensions",
                type="multi-select",
                options=[".py", ".js", ".ts", ".md", ".txt"],
                default=[".md", ".txt"]
            )
        ],
        secrets=[
            SecretDefinition(
                name="token",
                label="Access Token",
                type="password",
                required=False,
                show_when=[{"field": "auth_method", "equals": "token"}]
            ),
            SecretDefinition(
                name="ssh_key",
                label="SSH Private Key",
                type="textarea",
                required=False,
                show_when=[{"field": "auth_method", "equals": "ssh_key"}]
            )
        ],
        supports_preview=True,
        supports_continuous_sync=True
    ),

    "imap": ConnectorDefinition(
        name="Email (IMAP)",
        description="Index emails from IMAP server",
        icon="mail",
        fields=[
            FieldDefinition(
                name="host",
                label="IMAP Server",
                type="text",
                required=True,
                placeholder="imap.gmail.com"
            ),
            FieldDefinition(
                name="port",
                label="Port",
                type="number",
                required=True,
                default=993
            ),
            FieldDefinition(
                name="use_ssl",
                label="Use SSL",
                type="checkbox",
                default=True
            ),
            FieldDefinition(
                name="mailbox",
                label="Mailbox",
                type="text",
                required=True,
                default="INBOX"
            ),
            FieldDefinition(
                name="since_days",
                label="Index emails from last N days",
                type="number",
                required=False,
                default=30
            )
        ],
        secrets=[
            SecretDefinition(
                name="username",
                label="Username/Email",
                type="text",
                required=True
            ),
            SecretDefinition(
                name="password",
                label="Password/App Password",
                type="password",
                required=True
            )
        ],
        supports_preview=True,
        supports_continuous_sync=True
    )
}
```

## Base Connector

```python
class BaseConnector(ABC):
    """Abstract base class for all connectors."""

    def __init__(self, config: SourceConfig, secrets: dict | None = None):
        self.config = config
        self.secrets = secrets or {}

    @abstractmethod
    async def iterate_documents(self) -> AsyncIterator[SourceDocument]:
        """Iterate over all documents in the source."""
        pass

    @abstractmethod
    async def preview(self) -> PreviewResponse:
        """Test connection and return preview information."""
        pass

    async def get_document_hash(self, document: SourceDocument) -> str:
        """Generate unique hash for change detection."""
        content = f"{document.path}:{document.modified_at}"
        return hashlib.sha256(content.encode()).hexdigest()
```

## Connector Implementations

### DirectoryConnector
```python
class DirectoryConnector(BaseConnector):
    async def iterate_documents(self) -> AsyncIterator[SourceDocument]:
        base_path = Path(self.config.path)
        extensions = set(self.config.file_extensions or [])

        pattern = "**/*" if self.config.recursive else "*"

        for file_path in base_path.glob(pattern):
            if file_path.is_file():
                if not extensions or file_path.suffix in extensions:
                    yield SourceDocument(
                        path=str(file_path),
                        name=file_path.name,
                        size=file_path.stat().st_size,
                        modified_at=datetime.fromtimestamp(
                            file_path.stat().st_mtime
                        ),
                        content_type=mimetypes.guess_type(file_path)[0]
                    )

    async def preview(self) -> PreviewResponse:
        path = Path(self.config.path)
        if not path.exists():
            raise ConnectorError(f"Path does not exist: {path}")
        if not path.is_dir():
            raise ConnectorError(f"Path is not a directory: {path}")

        # Count matching files
        count = sum(1 for _ in self.iterate_documents())

        return PreviewResponse(
            success=True,
            message=f"Found {count} files",
            metadata={"file_count": count, "path": str(path)}
        )
```

### GitConnector
```python
class GitConnector(BaseConnector):
    async def iterate_documents(self) -> AsyncIterator[SourceDocument]:
        # Clone or update repository
        repo_path = await self._ensure_repo()

        extensions = set(self.config.file_extensions or [])

        for file_path in repo_path.rglob("*"):
            if file_path.is_file():
                # Skip .git directory
                if ".git" in file_path.parts:
                    continue

                if not extensions or file_path.suffix in extensions:
                    # Get last commit info for file
                    commit_info = await self._get_file_commit_info(file_path)

                    yield SourceDocument(
                        path=str(file_path.relative_to(repo_path)),
                        name=file_path.name,
                        size=file_path.stat().st_size,
                        modified_at=commit_info["date"],
                        content_type=mimetypes.guess_type(file_path)[0],
                        metadata={
                            "commit": commit_info["sha"],
                            "author": commit_info["author"]
                        }
                    )

    async def _ensure_repo(self) -> Path:
        """Clone or update repository."""
        cache_dir = Path(os.environ.get("GIT_CACHE_DIR", "/tmp/git_cache"))
        repo_hash = hashlib.md5(self.config.url.encode()).hexdigest()
        repo_path = cache_dir / repo_hash

        env = self._get_git_env()

        if repo_path.exists():
            # Pull latest changes
            await run_command(
                ["git", "pull"],
                cwd=repo_path,
                env=env
            )
        else:
            # Clone repository
            await run_command(
                ["git", "clone", "--depth", "1",
                 "-b", self.config.branch or "main",
                 self.config.url, str(repo_path)],
                env=env
            )

        return repo_path

    def _get_git_env(self) -> dict:
        """Get environment with authentication."""
        env = os.environ.copy()

        if self.secrets.get("token"):
            # Use token for HTTPS auth
            url = self.config.url
            if url.startswith("https://"):
                parts = url.split("://", 1)
                url = f"https://oauth2:{self.secrets['token']}@{parts[1]}"

        if self.secrets.get("ssh_key"):
            # Write SSH key to temp file
            key_file = self._write_ssh_key(self.secrets["ssh_key"])
            env["GIT_SSH_COMMAND"] = f"ssh -i {key_file} -o StrictHostKeyChecking=no"

        return env

    async def preview(self) -> GitPreviewResponse:
        """Test connection and list branches."""
        try:
            refs = await self._list_remote_refs()
            branches = [ref for ref in refs if ref.startswith("refs/heads/")]

            return GitPreviewResponse(
                success=True,
                branches=[b.replace("refs/heads/", "") for b in branches],
                default_branch=self._detect_default_branch(refs)
            )
        except Exception as e:
            return GitPreviewResponse(
                success=False,
                error=str(e)
            )
```

### ImapConnector
```python
class ImapConnector(BaseConnector):
    async def iterate_documents(self) -> AsyncIterator[SourceDocument]:
        async with self._connect() as imap:
            await imap.select(self.config.mailbox)

            # Search for emails
            since_date = datetime.now() - timedelta(
                days=self.config.since_days or 30
            )
            search_criteria = f'SINCE {since_date.strftime("%d-%b-%Y")}'

            _, message_ids = await imap.search(None, search_criteria)

            for msg_id in message_ids[0].split():
                _, msg_data = await imap.fetch(msg_id, "(RFC822)")
                email_message = email.message_from_bytes(msg_data[0][1])

                # Extract content
                body = self._extract_body(email_message)
                attachments = self._extract_attachments(email_message)

                yield SourceDocument(
                    path=f"email:{msg_id.decode()}",
                    name=email_message["Subject"] or "No Subject",
                    content=body,
                    modified_at=parsedate_to_datetime(email_message["Date"]),
                    metadata={
                        "from": email_message["From"],
                        "to": email_message["To"],
                        "subject": email_message["Subject"],
                        "attachments": len(attachments)
                    }
                )

    @asynccontextmanager
    async def _connect(self) -> aioimaplib.IMAP4_SSL:
        """Create authenticated IMAP connection."""
        if self.config.use_ssl:
            imap = aioimaplib.IMAP4_SSL(
                host=self.config.host,
                port=self.config.port
            )
        else:
            imap = aioimaplib.IMAP4(
                host=self.config.host,
                port=self.config.port
            )

        await imap.wait_hello_from_server()
        await imap.login(
            self.secrets["username"],
            self.secrets["password"]
        )

        try:
            yield imap
        finally:
            await imap.logout()

    async def preview(self) -> ImapPreviewResponse:
        """Test connection and list mailboxes."""
        try:
            async with self._connect() as imap:
                _, mailboxes = await imap.list()
                parsed = [self._parse_mailbox(m) for m in mailboxes]

                return ImapPreviewResponse(
                    success=True,
                    mailboxes=parsed
                )
        except Exception as e:
            return ImapPreviewResponse(
                success=False,
                error=str(e)
            )
```

## Secrets Management

### Encryption
```python
# Using Fernet symmetric encryption
from cryptography.fernet import Fernet

class SecretsManager:
    def __init__(self):
        key = os.environ["CONNECTOR_SECRETS_KEY"]
        self.fernet = Fernet(key.encode())

    def encrypt(self, secrets: dict) -> str:
        """Encrypt secrets for database storage."""
        json_bytes = json.dumps(secrets).encode()
        return self.fernet.encrypt(json_bytes).decode()

    def decrypt(self, encrypted: str) -> dict:
        """Decrypt secrets from database."""
        decrypted = self.fernet.decrypt(encrypted.encode())
        return json.loads(decrypted)
```

### Storage
```python
class ConnectorSecret(Base):
    __tablename__ = "connector_secrets"

    id: Mapped[int] = mapped_column(primary_key=True)
    collection_id: Mapped[str] = mapped_column(ForeignKey("collections.id"))
    connector_type: Mapped[str]
    encrypted_secrets: Mapped[str]  # Fernet-encrypted JSON
    created_at: Mapped[datetime]
    updated_at: Mapped[datetime]
```

## Frontend Integration

### ConnectorTypeSelector
```typescript
const connectorIcons = {
  directory: Folder,
  git: GitBranch,
  imap: Mail,
};

const displayOrder = ['directory', 'git', 'imap'];

function ConnectorTypeSelector({ catalog, selectedType, onSelect }) {
  const sortedTypes = Object.keys(catalog).sort(
    (a, b) => displayOrder.indexOf(a) - displayOrder.indexOf(b)
  );

  return (
    <div className="grid grid-cols-3 gap-4">
      {sortedTypes.map(type => (
        <ConnectorCard
          key={type}
          type={type}
          definition={catalog[type]}
          selected={selectedType === type}
          onClick={() => onSelect(type)}
        />
      ))}
    </div>
  );
}
```

### ConnectorForm
```typescript
function ConnectorForm({ catalog, connectorType, values, secrets, onChange }) {
  const definition = catalog[connectorType];

  return (
    <form className="space-y-4">
      {/* Regular fields */}
      {definition.fields
        .filter(field => shouldShowField(field, values))
        .map(field => (
          <DynamicField
            key={field.name}
            field={field}
            value={values[field.name]}
            onChange={val => onChange({ ...values, [field.name]: val })}
          />
        ))}

      {/* Secret fields */}
      {definition.secrets.length > 0 && (
        <div className="border-t pt-4">
          <h4>Authentication</h4>
          {definition.secrets
            .filter(secret => shouldShowField(secret, values))
            .map(secret => (
              <SecretField
                key={secret.name}
                field={secret}
                value={secrets[secret.name]}
                onChange={val => onSecretsChange({
                  ...secrets, [secret.name]: val
                })}
              />
            ))}
        </div>
      )}
    </form>
  );
}
```

## Extension Points

### Adding a New Connector
1. Add definition to `CONNECTOR_DEFINITIONS`
2. Implement connector class extending `BaseConnector`
3. Add icon mapping in frontend
4. Add to display order
5. Implement preview if supported
6. Add integration tests

### Adding Field to Existing Connector
1. Add to `fields` in connector definition
2. Frontend automatically renders new field
3. Update connector implementation to use new field
