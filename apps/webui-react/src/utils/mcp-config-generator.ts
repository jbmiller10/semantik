/**
 * MCP config generator for different client tools.
 * Generates properly formatted config strings (JSON or TOML) based on tool format type.
 * Supports both stdio (local) and http (remote/Docker) transports.
 *
 * HTTP transport notes:
 * - The MCP server requires client authentication via Authorization header
 * - Users must create an API key in Settings > API Keys
 * - The API key is included in the generated config as `headers.Authorization`
 */

import type { MCPClientConfig } from '../types/mcp-profile';
import type { FormatType } from '../types/mcp-client-tools';

/**
 * Generate config string for the given MCP client tool format.
 * @param config - The MCP client configuration
 * @param formatType - The output format type
 * @param apiKey - Optional API key to include in config
 *   - stdio transport: substituted for placeholder in env vars
 *   - http transport: included as Authorization header
 */
export function generateMCPConfig(
  config: MCPClientConfig,
  formatType: FormatType,
  apiKey?: string
): string {
  // If HTTP transport, use HTTP-specific generators with auth header
  if (config.transport === 'http') {
    return generateHTTPConfig(config, formatType, apiKey);
  }

  // stdio transport: substitute API key if provided
  const effectiveConfig = apiKey ? substituteApiKey(config, apiKey) : config;

  switch (formatType) {
    case 'standard':
      return generateStandardConfig(effectiveConfig);
    case 'cline':
      return generateClineConfig(effectiveConfig);
    case 'copilot':
      return generateCopilotConfig(effectiveConfig);
    case 'amp':
      return generateAmpConfig(effectiveConfig);
    case 'opencode':
      return generateOpencodeConfig(effectiveConfig);
    case 'codex':
      return generateCodexConfig(effectiveConfig);
    case 'claude-code':
      return generateClaudeCodeConfig(effectiveConfig);
    default:
      return generateStandardConfig(effectiveConfig);
  }
}

/**
 * Generate HTTP transport config for the given format type.
 * @param config - The MCP client configuration
 * @param formatType - The output format type
 * @param apiKey - Optional API key for Authorization header
 */
function generateHTTPConfig(
  config: MCPClientConfig,
  formatType: FormatType,
  apiKey?: string
): string {
  const url = config.url || 'http://localhost:9090/mcp';
  const headers = apiKey ? { Authorization: `Bearer ${apiKey}` } : undefined;
  const authPlaceholder = '<your-api-key>';

  switch (formatType) {
    case 'standard':
      // Standard format with SSE type for HTTP transport
      return JSON.stringify(
        {
          mcpServers: {
            [config.server_name]: {
              type: 'sse',
              url: url,
              ...(headers
                ? { headers }
                : { headers: { Authorization: `Bearer ${authPlaceholder}` } }),
            },
          },
        },
        null,
        2
      );
    case 'cline':
      // Cline format with SSE type
      return JSON.stringify(
        {
          mcpServers: {
            [config.server_name]: {
              type: 'sse',
              url: url,
              ...(headers
                ? { headers }
                : { headers: { Authorization: `Bearer ${authPlaceholder}` } }),
              timeout: 60,
              disabled: false,
            },
          },
        },
        null,
        2
      );
    case 'copilot':
      // Copilot format with remote type
      return JSON.stringify(
        {
          mcpServers: {
            [config.server_name]: {
              type: 'remote',
              url: url,
              ...(headers
                ? { headers }
                : { headers: { Authorization: `Bearer ${authPlaceholder}` } }),
              tools: ['*'],
            },
          },
        },
        null,
        2
      );
    case 'amp':
      // Amp format
      return JSON.stringify(
        {
          'amp.mcpServers': {
            [config.server_name]: {
              type: 'sse',
              url: url,
              ...(headers
                ? { headers }
                : { headers: { Authorization: `Bearer ${authPlaceholder}` } }),
            },
          },
        },
        null,
        2
      );
    case 'opencode':
      // opencode format with remote type
      return JSON.stringify(
        {
          $schema: 'https://opencode.ai/config.json',
          mcp: {
            [config.server_name]: {
              type: 'remote',
              url: url,
              ...(headers
                ? { headers }
                : { headers: { Authorization: `Bearer ${authPlaceholder}` } }),
              enabled: true,
            },
          },
        },
        null,
        2
      );
    case 'codex': {
      // Codex TOML format
      const authHeader = apiKey ? `Bearer ${apiKey}` : `Bearer ${authPlaceholder}`;
      return [
        `[mcp_servers.${config.server_name}]`,
        `type = "sse"`,
        `url = ${tomlString(url)}`,
        '',
        `[mcp_servers.${config.server_name}.headers]`,
        `Authorization = ${tomlString(authHeader)}`,
      ].join('\n');
    }
    case 'claude-code': {
      // Claude Code CLI command for HTTP transport
      const authValue = apiKey || authPlaceholder;
      return `claude mcp add --transport http \\\n  --header "Authorization: Bearer ${authValue}" \\\n  -- \\\n  ${config.server_name} \\\n  ${url}`;
    }
    default:
      return generateHTTPConfig(config, 'standard', apiKey);
  }
}

/**
 * Substitute API key placeholder with actual value in config env.
 */
function substituteApiKey(config: MCPClientConfig, apiKey: string): MCPClientConfig {
  if (!config.env) return config;

  const newEnv = { ...config.env };
  for (const key of Object.keys(newEnv)) {
    if (newEnv[key] === '<your-access-token-or-api-key>') {
      newEnv[key] = apiKey;
    }
  }
  return {
    ...config,
    env: newEnv,
  };
}

/**
 * Standard format used by Claude Desktop, Cursor, Kiro.
 */
function generateStandardConfig(config: MCPClientConfig): string {
  return JSON.stringify(
    {
      mcpServers: {
        [config.server_name]: {
          command: config.command || '',
          args: config.args || [],
          env: config.env || {},
        },
      },
    },
    null,
    2
  );
}

/**
 * Cline format - adds type, timeout, disabled fields.
 */
function generateClineConfig(config: MCPClientConfig): string {
  return JSON.stringify(
    {
      mcpServers: {
        [config.server_name]: {
          type: 'stdio',
          command: config.command || '',
          args: config.args || [],
          env: config.env || {},
          timeout: 60,
          disabled: false,
        },
      },
    },
    null,
    2
  );
}

/**
 * GitHub Copilot format - adds type: "local", tools: ["*"].
 */
function generateCopilotConfig(config: MCPClientConfig): string {
  return JSON.stringify(
    {
      mcpServers: {
        [config.server_name]: {
          type: 'local',
          command: config.command || '',
          args: config.args || [],
          env: config.env || {},
          tools: ['*'],
        },
      },
    },
    null,
    2
  );
}

/**
 * Amp format - wraps in amp.mcpServers key.
 */
function generateAmpConfig(config: MCPClientConfig): string {
  return JSON.stringify(
    {
      'amp.mcpServers': {
        [config.server_name]: {
          command: config.command || '',
          args: config.args || [],
          env: config.env || {},
        },
      },
    },
    null,
    2
  );
}

/**
 * opencode format - uses $schema, "mcp" key, command as array.
 */
function generateOpencodeConfig(config: MCPClientConfig): string {
  // opencode expects command and args combined into a single command array
  const commandArray = [config.command || '', ...(config.args || [])];

  return JSON.stringify(
    {
      $schema: 'https://opencode.ai/config.json',
      mcp: {
        [config.server_name]: {
          type: 'local',
          command: commandArray,
          enabled: true,
        },
      },
    },
    null,
    2
  );
}

/**
 * Codex format - TOML with [mcp_servers.name] sections.
 */
function generateCodexConfig(config: MCPClientConfig): string {
  const lines: string[] = [];

  // Server section header
  lines.push(`[mcp_servers.${config.server_name}]`);
  lines.push(`command = ${tomlString(config.command || '')}`);
  lines.push(`args = ${tomlStringArray(config.args || [])}`);

  // Environment variables section (if any)
  const env = config.env || {};
  const envKeys = Object.keys(env);
  if (envKeys.length > 0) {
    lines.push('');
    lines.push(`[mcp_servers.${config.server_name}.env]`);
    for (const key of envKeys) {
      lines.push(`${key} = ${tomlString(env[key])}`);
    }
  }

  return lines.join('\n');
}

/**
 * Claude Code format - CLI command for easy setup.
 */
function generateClaudeCodeConfig(config: MCPClientConfig): string {
  const parts: string[] = ['claude mcp add'];

  // Add environment variables (must come before --)
  const env = config.env || {};
  const envKeys = Object.keys(env);
  for (const key of envKeys) {
    parts.push(`-e ${key}=${shellEscape(env[key])}`);
  }

  // Separator between options and positional args
  parts.push('--');

  // Server name (first positional arg)
  parts.push(config.server_name);

  // Command and args
  parts.push(config.command || '');
  parts.push(...(config.args || []));

  return parts.join(' \\\n  ');
}

/**
 * Escape a string for shell usage.
 */
function shellEscape(value: string): string {
  // If the value contains special characters, wrap in quotes
  if (/[^a-zA-Z0-9_\-.:/@]/.test(value)) {
    // Escape any existing quotes and wrap in double quotes
    return `"${value.replace(/"/g, '\\"')}"`;
  }
  return value;
}

/**
 * Escape and quote a string for TOML.
 */
function tomlString(value: string): string {
  // Escape backslashes and quotes
  const escaped = value.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `"${escaped}"`;
}

/**
 * Format a string array for TOML.
 */
function tomlStringArray(values: string[]): string {
  const items = values.map(tomlString).join(', ');
  return `[${items}]`;
}

/**
 * Get the label for the config code block based on format type.
 */
export function getConfigBlockLabel(formatType: FormatType): string {
  if (formatType === 'claude-code') {
    return 'Run in terminal';
  }
  if (formatType === 'codex') {
    return 'Add to config.toml';
  }
  if (formatType === 'amp') {
    return 'Add to settings.json';
  }
  if (formatType === 'opencode') {
    return 'Add to opencode.json';
  }
  return 'Add to mcpServers';
}

/**
 * Get the language for syntax highlighting (if needed in future).
 */
export function getConfigLanguage(formatType: FormatType): 'json' | 'toml' | 'bash' {
  if (formatType === 'claude-code') {
    return 'bash';
  }
  return formatType === 'codex' ? 'toml' : 'json';
}
