# Connectors

Semantik ingests documents via *connectors*: per-source implementations that know how to enumerate and load content (directories, Git repos, IMAP mailboxes, etc.) and emit `shared.dtos.ingestion.IngestedDocument`.

## Built-in Connector Types

- `directory`: Index files from a local directory available to the WebUI/worker.
- `git`: Clone and index files from a remote Git repository.
- `imap`: Connect to an IMAP mailbox and index emails as markdown documents.

Connector metadata is derived directly from connector classes (`get_config_fields()` / `get_secret_fields()` and `METADATA`).
Connector plugins register via the unified `semantik.plugins` entry point group; use `GET /api/v2/connectors` to discover what is enabled.

## Document Parsing

Connectors use the parser system (`shared.text_processing.parsers`) to extract text from documents. The `parse_content()` function handles format detection and fallback automatically. For details, see [PARSERS.md](./PARSERS.md).

## Connector Secrets Encryption (`CONNECTOR_SECRETS_KEY`)

Connectors that require credentials (e.g., Git tokens, IMAP passwords, SSH keys) store those credentials in the database, encrypted at rest using Fernet.

- Set `CONNECTOR_SECRETS_KEY` to a valid Fernet key (44-char base64) to enable secrets encryption.
- Set it to empty/unset to disable secrets encryption (credentialed connectors cannot be configured).
- Generate a key:
  - `uv run python scripts/generate_secrets_key.py`
  - `uv run python scripts/generate_secrets_key.py --write --env-file .env`

## Connector Catalog & Preview Endpoints

- `GET /api/v2/connectors`: List all connectors and their schemas.
- `GET /api/v2/connectors/{connector_type}`: Get schema/metadata for one connector.
- `POST /api/v2/connectors/preview/git`: Validate Git access and list refs (branches/tags).
- `POST /api/v2/connectors/preview/imap`: Validate IMAP access and list mailboxes.

Preview endpoints validate connectivity only; they are not a substitute for securely storing secrets.

## Source Configuration Examples

### Directory
```json
{
  "source_type": "directory",
  "source_config": {
    "path": "/docs",
    "recursive": true,
    "include_patterns": ["*.md", "*.txt"],
    "exclude_patterns": ["**/.git/**"]
  }
}
```

### Git
```json
{
  "source_type": "git",
  "source_config": {
    "repo_url": "https://github.com/org/repo.git",
    "ref": "main",
    "auth_method": "https_token",
    "include_globs": ["docs/**", "*.md"],
    "exclude_globs": ["node_modules/**"],
    "max_file_size_mb": 10,
    "shallow_depth": 1
  }
}
```

**Runtime dependency:** the worker must have `git` available on `PATH` (the Docker image includes it).

## Managing Sources (recommended workflow)

1. Start an ingestion run and create/reuse the source record:
   - `POST /api/v2/collections/{collection_id}/sources`
2. List sources to get `source_id`:
   - `GET /api/v2/collections/{collection_id}/sources`
3. Store encrypted secrets and configure sync (optional):
   - `PATCH /api/v2/collections/{collection_id}/sources/{source_id}`
4. Trigger runs and scheduling (collection-level):
   - `POST /api/v2/collections/{collection_id}/sync/run`
   - `POST /api/v2/collections/{collection_id}/sync/pause`
   - `POST /api/v2/collections/{collection_id}/sync/resume`

### IMAP
```json
{
  "source_type": "imap",
  "source_config": {
    "host": "imap.gmail.com",
    "port": 993,
    "use_ssl": true,
    "username": "user@example.com",
    "mailboxes": ["INBOX", "Sent"],
    "since_days": 30,
    "max_messages": 1000
  }
}
```

## Serving Content for Non-File Sources

For Git/IMAP (and other non-filesystem sources), Semantik stores a canonical representation of content in the `document_artifacts` table. The document content endpoint prefers artifacts and falls back to filesystem serving for local directory sources:

- `GET /api/v2/collections/{collection_id}/documents/{document_id}/content`
