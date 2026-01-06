# MCP Integration Guide

This guide explains how to use Semantik's Model Context Protocol (MCP) integration to connect AI assistants like Claude Desktop and Cursor to search your document collections.

## What is MCP?

The [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard that enables AI assistants to securely access external tools and data sources. Semantik's MCP integration exposes your document collections as searchable tools that AI assistants can use during conversations.

## Quick Start

1. **Create a collection and add documents** in the Semantik web interface
2. **Create an MCP profile** in Settings > MCP Profiles
3. **Configure your AI client** (Claude Desktop, Cursor, etc.)
4. **Start searching** - your AI assistant can now search your documents

## Creating MCP Profiles

MCP profiles define which collections an AI assistant can access and how searches are configured.

### Profile Settings

| Setting | Description |
|---------|-------------|
| **Name** | Used as the tool name. Must be lowercase with letters, numbers, hyphens, or underscores (e.g., `coding-docs`, `work_notes`). |
| **Description** | Shown to the AI to help it understand what this profile searches. Be descriptive (e.g., "Search Python documentation and API references"). |
| **Enabled** | Toggle to enable/disable the profile. Disabled profiles won't appear as tools. |
| **Collections** | Select which collections this profile can search. Order matters - first collections are prioritized. |
| **Search Type** | Choose the search algorithm (see below). |
| **Result Count** | Default number of results to return (1-100). |
| **Use Reranker** | Enable cross-encoder reranking for improved result quality. |
| **Score Threshold** | Minimum relevance score (0-1). Results below this threshold are filtered out. |
| **Hybrid Alpha** | Balance between semantic (1.0) and keyword (0.0) search when using hybrid mode. |

### Search Types

| Type | Description | Best For |
|------|-------------|----------|
| **semantic** | Vector similarity search using embeddings | General purpose, conceptual queries |
| **hybrid** | Combines semantic and keyword search | Queries mixing concepts and specific terms |
| **keyword** | Traditional keyword matching | Exact phrase or term lookups |
| **question** | Optimized for question-answering | Direct questions about content |
| **code** | Optimized for code search | Function names, code patterns |

## Configuring Claude Desktop

1. Open Claude Desktop settings (gear icon)
2. Navigate to "Developer" settings
3. Click "Edit Config" to open `claude_desktop_config.json`
4. Add the Semantik server configuration:

```json
{
  "mcpServers": {
    "semantik-coding": {
      "command": "semantik-mcp",
      "args": ["serve", "--profile", "coding"],
      "env": {
        "SEMANTIK_WEBUI_URL": "http://localhost:8080",
        "SEMANTIK_AUTH_TOKEN": "<your-api-key>"
      }
    }
  }
}
```

5. Save and restart Claude Desktop

### Multiple Profiles

You can configure multiple profiles by adding multiple entries:

```json
{
  "mcpServers": {
    "semantik-coding": {
      "command": "semantik-mcp",
      "args": ["serve", "--profile", "coding"],
      "env": {
        "SEMANTIK_WEBUI_URL": "http://localhost:8080",
        "SEMANTIK_AUTH_TOKEN": "<your-api-key>"
      }
    },
    "semantik-work": {
      "command": "semantik-mcp",
      "args": ["serve", "--profile", "work"],
      "env": {
        "SEMANTIK_WEBUI_URL": "http://localhost:8080",
        "SEMANTIK_AUTH_TOKEN": "<your-api-key>"
      }
    }
  }
}
```

Or expose all profiles at once:

```json
{
  "mcpServers": {
    "semantik": {
      "command": "semantik-mcp",
      "args": ["serve"],
      "env": {
        "SEMANTIK_WEBUI_URL": "http://localhost:8080",
        "SEMANTIK_AUTH_TOKEN": "<your-api-key>"
      }
    }
  }
}
```

## Configuring Cursor

1. Open Cursor settings (`Cmd/Ctrl + ,`)
2. Search for "MCP" in the settings
3. Click "Add MCP Server"
4. Configure with the same settings as Claude Desktop

## Available Tools

When connected, the MCP server exposes these tools:

| Tool | Description |
|------|-------------|
| `search_{profile}` | Search the profile's collections. Returns relevant chunks with scores. |
| `get_document` | Get document metadata (filename, size, status, chunk count). |
| `get_document_content` | Get the raw content of a document. |
| `get_chunk` | Get the full text of a specific chunk. |
| `list_documents` | List documents in a collection (paginated). |
| `diagnostics` | Show server status, profiles, and connection info. |

### Search Tool Parameters

The search tool accepts these parameters:

```
query: string (required) - The search query
k: integer (1-100) - Number of results (default: profile setting)
search_type: string - Override search type
use_reranker: boolean - Override reranker setting
score_threshold: number (0-1) - Override score threshold
hybrid_alpha: number (0-1) - Override hybrid alpha
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SEMANTIK_WEBUI_URL` | Base URL for Semantik WebUI | `http://localhost:8080` |
| `SEMANTIK_AUTH_TOKEN` | API key or JWT access token | (required) |
| `SEMANTIK_MCP_LOG_LEVEL` | Logging level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

## Command Line Options

```bash
semantik-mcp serve [OPTIONS]

Options:
  --profile, -p TEXT    Profile(s) to expose (can be repeated)
  --webui-url TEXT      Semantik WebUI base URL
  --auth-token TEXT     Auth token (JWT or API key)
  --log-level TEXT      Logging level
  --verbose, -v         Enable DEBUG logging
```

Examples:

```bash
# Serve a single profile
semantik-mcp serve --profile coding

# Serve multiple profiles
semantik-mcp serve --profile coding --profile work

# Serve all enabled profiles (no filter)
semantik-mcp serve

# Enable verbose logging for debugging
semantik-mcp serve --profile coding --verbose
```

## Troubleshooting

### "Connection refused"

**Cause:** Cannot connect to the Semantik WebUI server.

**Solutions:**
- Verify `SEMANTIK_WEBUI_URL` is correct
- Check if Semantik is running (`docker compose ps`)
- Ensure the port is not blocked by a firewall

### "Authentication failed" / 401 Unauthorized

**Cause:** Invalid or expired auth token.

**Solutions:**
- Verify `SEMANTIK_AUTH_TOKEN` is set correctly
- Generate a new API key in Semantik settings
- If using JWT, ensure it hasn't expired

### "Profile not found"

**Cause:** The specified profile doesn't exist or is disabled.

**Solutions:**
- Check the profile name matches exactly (case-sensitive)
- Verify the profile is enabled in the web interface
- Run `diagnostics` tool to see available profiles

### "No results" from search

**Cause:** Query didn't match any documents above the score threshold.

**Solutions:**
- Verify the collection has indexed documents
- Try lowering the `score_threshold`
- Try a different search type
- Check if documents are in "completed" status

### Viewing debug logs

Enable verbose logging to see detailed diagnostics:

```bash
semantik-mcp serve --profile coding --verbose
```

Or use the `diagnostics` tool in Claude:

> "Use the diagnostics tool to show the MCP server status"

This will show:
- Available profiles and their collection counts
- Connection status to Semantik
- Authentication status
- Cache status

## Security Considerations

- **Token Security:** Never commit auth tokens to version control. Use environment variables.
- **Collection Scope:** Each profile only has access to its configured collections.
- **Read-Only:** The MCP server only provides read access - it cannot modify documents.
- **User Isolation:** Profiles are scoped to the authenticated user.

## Performance Tips

- **Use score thresholds** to filter low-quality results and reduce noise
- **Enable reranking** for better result quality (may be slower)
- **Limit result count** if you only need top few results
- **Use specific profiles** rather than exposing all collections
- The server caches profile information for 10 seconds to reduce API calls
