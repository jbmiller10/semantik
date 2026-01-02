# Plugin Registry

The Semantik Plugin Registry is a curated directory of available plugins that users can browse and install. It provides a central place to discover plugins for embedding providers, rerankers, chunking strategies, extractors, and connectors.

## Overview

The registry is a YAML file that lists available plugins with their metadata, compatibility information, and installation instructions. Semantik fetches this registry and displays it in the UI's "Available" tab under Plugin Settings.

## Registry URL

By default, Semantik fetches the registry from:

```
https://raw.githubusercontent.com/semantik/plugin-registry/main/registry.yaml
```

You can override this by setting the `SEMANTIK_PLUGIN_REGISTRY_URL` environment variable:

```bash
export SEMANTIK_PLUGIN_REGISTRY_URL="https://your-registry.example.com/registry.yaml"
```

## Bundled Registry

Semantik includes a bundled registry that works offline. If the remote registry cannot be fetched (network issues, timeout, etc.), Semantik falls back to the bundled version. This ensures users can always browse available plugins.

The bundled registry is located at:
```
packages/shared/plugins/data/registry.yaml
```

## Registry Schema (v1.1)

```yaml
registry_version: "1.1"
last_updated: "2026-01-02T00:00:00Z"
plugins:
  - id: plugin-unique-id
    type: embedding | chunking | connector | reranker | extractor
    name: "Human-Readable Plugin Name"
    description: "Brief description of what the plugin does"
    author: "author-name"
    repository: "https://github.com/owner/repo"
    install_command: "pip install git+https://github.com/owner/repo.git"  # Optional
    pypi: "package-name-on-pypi"
    verified: true | false
    min_semantik_version: "0.7.5"  # Optional
    tags:
      - api
      - local
      - gpu
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Unique identifier for the plugin (lowercase, hyphenated) |
| `type` | Yes | Plugin type: `embedding`, `chunking`, `connector`, `reranker`, or `extractor` |
| `name` | Yes | Human-readable display name |
| `description` | Yes | Brief description (1-2 sentences) |
| `author` | Yes | Author or organization name |
| `repository` | Yes | GitHub repository URL |
| `install_command` | No | Full `pip install ...` command (e.g., `pip install git+https://...`) |
| `pypi` | Yes | PyPI package name for installation |
| `verified` | No | Whether the plugin is verified (default: `false`) |
| `min_semantik_version` | No | Minimum Semantik version required (semver) |
| `tags` | No | List of tags for categorization and search |

## Example Registry Entry

```yaml
plugins:
  - id: openai-embeddings
    type: embedding
    name: "OpenAI Embeddings"
    description: "text-embedding-3-small and text-embedding-3-large models via OpenAI API"
    author: "semantik"
    repository: "https://github.com/semantik-plugins/openai-embeddings"
    install_command: "pip install semantik-plugin-openai"
    pypi: "semantik-plugin-openai"
    verified: true
    min_semantik_version: "0.7.5"
    tags:
      - api
      - openai
      - cloud
```

## Version Compatibility

Plugins can specify a `min_semantik_version` to indicate the minimum Semantik version they support. The registry client checks compatibility and shows:

- **Compatible**: Plugin works with the current Semantik version
- **Incompatible**: Plugin requires a newer Semantik version (shows error message)

If no `min_semantik_version` is specified, the plugin is considered compatible with all versions.

## Plugin Verification

Plugins can be marked as `verified: true` to indicate they have been reviewed and tested. Unverified plugins are still listed but display an "Unverified" badge.

### Verification Criteria

To be verified, a plugin should:

1. Follow Semantik's plugin API and best practices
2. Have a public repository with source code
3. Be published on PyPI
4. Include documentation and usage examples
5. Not contain malicious code

## Using the Registry in the UI

### Browsing Plugins

1. Go to **Settings > Plugins**
2. Click the **Available** tab
3. Browse plugins grouped by type

### Filtering and Search

- **Search**: Filter by name, description, author, or tags
- **Type**: Filter by plugin type (Embedding, Reranker, etc.)
- **Verified Only**: Show only verified plugins

### Installing Plugins

Each plugin card shows an install command:

```bash
pip install semantik-plugin-openai
```

Click the copy button to copy the command to your clipboard.

## API Endpoints

### List Available Plugins

```
GET /api/v2/plugins/available
```

Query parameters:
- `plugin_type`: Filter by type (e.g., `embedding`)
- `verified_only`: Show only verified plugins (boolean)
- `force_refresh`: Bypass cache and fetch fresh data (boolean)

Response:
```json
{
  "plugins": [...],
  "registry_version": "1.1",
  "last_updated": "2026-01-02T00:00:00Z",
  "registry_source": "remote" | "bundled",
  "semantik_version": "0.7.5"
}
```

### Refresh Registry

```
POST /api/v2/plugins/available/refresh
```

Forces a cache refresh and fetches the latest registry data.

## Caching

The registry is cached for 1 hour to minimize network requests. The cache is automatically refreshed when:

- The cache expires (after 1 hour)
- A user clicks the "Refresh" button in the UI
- The `force_refresh` parameter is set to `true`

## Contributing Plugins

To add your plugin to the registry:

1. Create a plugin following Semantik's plugin API
2. Publish the plugin to PyPI
3. Submit a pull request to the registry repository adding your plugin entry
4. Wait for review and verification

### Registry Repository

The official registry is maintained at:
```
https://github.com/semantik/plugin-registry
```

## Troubleshooting

### Registry Not Loading

If the registry fails to load:

1. Check your network connection
2. Verify the registry URL is accessible
3. Check the Semantik logs for error messages
4. The bundled registry will be used as fallback

### Plugin Shows as Incompatible

If a plugin shows as incompatible:

1. Check the `min_semantik_version` requirement
2. Upgrade Semantik to a compatible version
3. Contact the plugin author if you believe this is an error

### Custom Registry

To use a custom registry:

1. Host your registry YAML file at a publicly accessible URL
2. Set `SEMANTIK_PLUGIN_REGISTRY_URL` environment variable
3. Restart Semantik
