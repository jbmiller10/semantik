/**
 * MCP config generator for different client tools.
 * Generates properly formatted config strings (JSON or TOML) based on tool format type.
 */

import type { MCPClientConfig } from '../types/mcp-profile';
import type { FormatType } from '../types/mcp-client-tools';

/**
 * Generate config string for the given MCP client tool format.
 */
export function generateMCPConfig(
  config: MCPClientConfig,
  formatType: FormatType
): string {
  switch (formatType) {
    case 'standard':
      return generateStandardConfig(config);
    case 'cline':
      return generateClineConfig(config);
    case 'copilot':
      return generateCopilotConfig(config);
    case 'amp':
      return generateAmpConfig(config);
    case 'opencode':
      return generateOpencodeConfig(config);
    case 'codex':
      return generateCodexConfig(config);
    case 'claude-code':
      return generateClaudeCodeConfig(config);
    default:
      return generateStandardConfig(config);
  }
}

/**
 * Standard format used by Claude Desktop, Cursor, Kiro.
 */
function generateStandardConfig(config: MCPClientConfig): string {
  return JSON.stringify(
    {
      mcpServers: {
        [config.server_name]: {
          command: config.command,
          args: config.args,
          env: config.env,
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
          command: config.command,
          args: config.args,
          env: config.env,
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
          command: config.command,
          args: config.args,
          env: config.env,
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
          command: config.command,
          args: config.args,
          env: config.env,
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
  const commandArray = [config.command, ...config.args];

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
  lines.push(`command = ${tomlString(config.command)}`);
  lines.push(`args = ${tomlStringArray(config.args)}`);

  // Environment variables section (if any)
  const envKeys = Object.keys(config.env);
  if (envKeys.length > 0) {
    lines.push('');
    lines.push(`[mcp_servers.${config.server_name}.env]`);
    for (const key of envKeys) {
      lines.push(`${key} = ${tomlString(config.env[key])}`);
    }
  }

  return lines.join('\n');
}

/**
 * Claude Code format - CLI command for easy setup.
 */
function generateClaudeCodeConfig(config: MCPClientConfig): string {
  const parts: string[] = ['claude mcp add'];

  // Add environment variables
  const envKeys = Object.keys(config.env);
  for (const key of envKeys) {
    parts.push(`--env ${key}=${shellEscape(config.env[key])}`);
  }

  // Add server name
  parts.push(config.server_name);

  // Add command separator and command with args
  parts.push('--');
  parts.push(config.command);
  parts.push(...config.args);

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
