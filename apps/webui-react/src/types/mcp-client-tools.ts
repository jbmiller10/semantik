/**
 * MCP Client Tool definitions for config generation.
 * Each tool has its own config file location(s) and format requirements.
 */

export type FormatType = 'standard' | 'cline' | 'copilot' | 'amp' | 'opencode' | 'codex';

export interface MCPClientTool {
  id: string;
  name: string;
  configPaths: {
    macos: string | null;
    linux: string | null;
    windows: string | null;
  };
  formatType: FormatType;
  notes?: string;
}

export const MCP_CLIENT_TOOLS: MCPClientTool[] = [
  {
    id: 'claude-desktop',
    name: 'Claude Desktop',
    configPaths: {
      macos: '~/Library/Application Support/Claude/claude_desktop_config.json',
      linux: '~/.config/Claude/claude_desktop_config.json',
      windows: '%APPDATA%\\Claude\\claude_desktop_config.json',
    },
    formatType: 'standard',
    notes: 'Restart Claude Desktop after updating config',
  },
  {
    id: 'cursor',
    name: 'Cursor',
    configPaths: {
      macos: '~/.cursor/mcp.json',
      linux: '~/.cursor/mcp.json',
      windows: '%USERPROFILE%\\.cursor\\mcp.json',
    },
    formatType: 'standard',
    notes: 'Use ~/.cursor/mcp.json for global, .cursor/mcp.json for project-specific',
  },
  {
    id: 'cline',
    name: 'Cline',
    configPaths: {
      macos: 'cline_mcp_settings.json (via extension)',
      linux: 'cline_mcp_settings.json (via extension)',
      windows: 'cline_mcp_settings.json (via extension)',
    },
    formatType: 'cline',
    notes: 'Access via MCP Servers icon → Configure tab in Cline extension',
  },
  {
    id: 'copilot',
    name: 'GitHub Copilot',
    configPaths: {
      macos: '.vscode/mcp.json',
      linux: '.vscode/mcp.json',
      windows: '.vscode\\mcp.json',
    },
    formatType: 'copilot',
    notes: 'Requires "MCP servers in Copilot" policy enabled for org users',
  },
  {
    id: 'amp',
    name: 'Amp',
    configPaths: {
      macos: '~/.config/amp/settings.json',
      linux: '~/.config/amp/settings.json',
      windows: '%APPDATA%\\amp\\settings.json',
    },
    formatType: 'amp',
    notes: 'Use .amp/settings.json for project-specific config',
  },
  {
    id: 'opencode',
    name: 'opencode',
    configPaths: {
      macos: '~/.config/opencode/opencode.json',
      linux: '~/.config/opencode/opencode.json',
      windows: '%APPDATA%\\opencode\\opencode.json',
    },
    formatType: 'opencode',
    notes: 'Use opencode.json in project root for project-specific config',
  },
  {
    id: 'codex',
    name: 'Codex',
    configPaths: {
      macos: '~/.codex/config.toml',
      linux: '~/.codex/config.toml',
      windows: '%USERPROFILE%\\.codex\\config.toml',
    },
    formatType: 'codex',
    notes: 'Add under [mcp_servers] section (underscore, not hyphen)',
  },
  {
    id: 'kiro',
    name: 'Kiro',
    configPaths: {
      macos: '~/.kiro/settings/mcp.json',
      linux: '~/.kiro/settings/mcp.json',
      windows: '%USERPROFILE%\\.kiro\\settings\\mcp.json',
    },
    formatType: 'standard',
    notes: 'Use .kiro/settings/mcp.json for project-specific config',
  },
];

export const DEFAULT_MCP_CLIENT_TOOL_ID = 'claude-desktop';

export function getMCPClientTool(id: string): MCPClientTool | undefined {
  return MCP_CLIENT_TOOLS.find((tool) => tool.id === id);
}
