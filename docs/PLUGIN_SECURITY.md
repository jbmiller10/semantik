# Plugin Security Guide

This document describes the security model for Semantik plugins, including limitations and best practices for both users and plugin authors.

## Security Model Overview

### Trust Model

Semantik plugins use a **trusted plugin model**:

- **Plugins run in-process** with the main Semantik application
- **No sandboxing or isolation** - plugins have full access to the Python runtime
- **Only install plugins you trust** - from verified sources or that you've audited

### What Plugins Can Access

When you install a plugin, it has access to:

| Resource | Access Level |
|----------|--------------|
| Environment variables | Full (including secrets) |
| Filesystem | Full read/write within container |
| Network | Unrestricted |
| Database | Via available connections |
| GPU/Memory | Shared with application |

### Why This Model?

Semantik is designed for **experimentation** by trusted developers (primarily yourself). Full sandboxing would require:

1. Out-of-process execution (subprocess, WASM, containers)
2. Complex IPC for plugin communication
3. Significant performance overhead
4. Limited plugin capabilities

For the target use case (self-hosted semantic search with self-authored plugins), the complexity isn't justified.

## Audit Logging

All plugin operations are logged for security auditing.

### Logged Events

| Event | Description |
|-------|-------------|
| `plugin.registered.builtin` | Built-in plugin loaded |
| `plugin.registered.external` | External plugin loaded |
| `plugin.load.failed` | Plugin failed to load |
| `plugin.config.updated` | Plugin configuration changed |
| `plugin.enabled` | Plugin enabled |
| `plugin.disabled` | Plugin disabled |
| `plugin.health_check` | Health check performed |

### Log Format

Plugin audit logs use structured logging with the `PLUGIN_AUDIT` prefix:

```
PLUGIN_AUDIT: my-embedding-plugin - plugin.registered.external
```

With JSON structured data in the `extra` field:
```json
{
  "plugin_id": "my-embedding-plugin",
  "action": "plugin.registered.external",
  "plugin_type": "embedding",
  "version": "1.0.0",
  "entry_point": "my_pkg.embedding:MyPlugin",
  "timestamp": "2026-01-01T12:00:00Z"
}
```

### Viewing Audit Logs

Filter logs by the PLUGIN_AUDIT prefix:

```bash
# Docker logs
docker logs semantik-webui 2>&1 | grep PLUGIN_AUDIT

# If using a log aggregator, search for:
# - message contains "PLUGIN_AUDIT"
# - extra.action starts with "plugin."
```

## Environment Variable Protection

### Cooperative Filtering

Plugins that want to be "good citizens" can use the `get_sanitized_environment()` utility:

```python
from shared.plugins.security import get_sanitized_environment

# Returns env vars without sensitive values
safe_env = get_sanitized_environment()
# Excludes: *PASSWORD*, *SECRET*, *KEY*, *TOKEN*, *CREDENTIAL*, etc.
```

### Limitation

This is **cooperative only**. A malicious plugin can still access `os.environ` directly. This utility exists for plugins that want to avoid accidentally logging or exposing secrets.

### Filtered Patterns

The following patterns in environment variable names trigger filtering:

- `PASSWORD`
- `SECRET`
- `KEY`
- `TOKEN`
- `CREDENTIAL`
- `API_KEY`
- `PRIVATE`
- `AUTH`

## Best Practices

### For Users

1. **Only install trusted plugins**
   - From official Semantik registry (verified badge)
   - From sources you've audited
   - From your own development

2. **Review plugin source code** before installation
   - Check for suspicious network calls
   - Look for environment variable access
   - Verify no data exfiltration

3. **Use minimal permissions**
   - Run Semantik in containers with limited privileges
   - Don't mount sensitive host directories
   - Use secrets management, not env vars when possible

4. **Monitor plugin activity**
   - Review audit logs regularly
   - Watch for unexpected network traffic
   - Monitor resource usage

### For Plugin Authors

1. **Use the sanitized environment helper**
   ```python
   from shared.plugins.security import get_sanitized_environment
   env = get_sanitized_environment()
   ```

2. **Never log sensitive configuration**
   ```python
   # BAD: logging API key
   logger.info("Config: %s", config)

   # GOOD: log only non-sensitive keys
   logger.info("Configured with keys: %s", list(config.keys()))
   ```

3. **Request only necessary configuration**
   - Use environment variable references (`_env` suffix) instead of raw secrets
   - Document what access your plugin needs

4. **Handle errors gracefully**
   - Don't expose internal details in error messages
   - Fail safely without logging credentials

## API Reference

### `audit_log(plugin_id, action, details=None, *, level=INFO)`

Log a plugin action for security auditing.

```python
from shared.plugins.security import audit_log

audit_log(
    "my-plugin",
    "plugin.custom.action",
    {"custom_key": "value"},
)
```

This function never raises exceptions - logging failures are caught and logged at WARNING level.

### `get_sanitized_environment()`

Return environment variables with sensitive values filtered out.

```python
from shared.plugins.security import get_sanitized_environment

env = get_sanitized_environment()
# Safe to log or pass to external services
```

### `SENSITIVE_ENV_PATTERNS`

The set of patterns used to identify sensitive environment variables:

```python
from shared.plugins.security import SENSITIVE_ENV_PATTERNS
# frozenset({'PASSWORD', 'SECRET', 'KEY', 'TOKEN', ...})
```

## Future Roadmap

The following security features are deferred until untrusted third-party plugins become a concern:

- **Plugin sandboxing** - Out-of-process execution with resource limits
- **Permission system** - Declared permissions with user consent
- **Plugin verification** - Code signing and security scanning
- **Tiered trust model** - Different isolation levels based on trust

## Questions?

If you have security concerns or need guidance on plugin security:

1. Check the [plugin development guide](./EMBEDDING_PLUGINS.md)
2. Review [existing plugins](./PLUGIN_REGISTRY.md) for patterns
3. Open a GitHub issue for security discussions
