// apps/webui-react/src/utils/__tests__/mcp-config-generator.test.ts
import { describe, it, expect } from 'vitest';
import { generateMCPConfig, getConfigBlockLabel, getConfigLanguage } from '../mcp-config-generator';
import type { MCPClientConfig } from '../../types/mcp-profile';

const mockStdioConfig: MCPClientConfig = {
  transport: 'stdio',
  server_name: 'semantik-test',
  command: 'semantik-mcp',
  args: ['serve', '--profile', 'test'],
  env: {
    SEMANTIK_WEBUI_URL: 'http://localhost:8080',
    SEMANTIK_AUTH_TOKEN: '<your-access-token-or-api-key>',
  },
};

const mockHttpConfig: MCPClientConfig = {
  transport: 'http',
  server_name: 'semantik-test',
  url: 'http://localhost:9090/mcp',
};

describe('generateMCPConfig', () => {
  describe('stdio transport with apiKey parameter', () => {
    it('should insert API key into standard format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'standard', 'smtk_test123_secret');
      expect(result).toContain('"SEMANTIK_AUTH_TOKEN": "smtk_test123_secret"');
      expect(result).not.toContain('<your-access-token-or-api-key>');
    });

    it('should insert API key into cline format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'cline', 'smtk_test123_secret');
      expect(result).toContain('"SEMANTIK_AUTH_TOKEN": "smtk_test123_secret"');
    });

    it('should insert API key into codex TOML format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'codex', 'smtk_test123_secret');
      expect(result).toContain('SEMANTIK_AUTH_TOKEN = "smtk_test123_secret"');
    });

    it('should insert API key into claude-code CLI format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'claude-code', 'smtk_test123_secret');
      expect(result).toContain('SEMANTIK_AUTH_TOKEN=smtk_test123_secret');
    });

    it('should keep placeholder when apiKey is undefined', () => {
      const result = generateMCPConfig(mockStdioConfig, 'standard');
      expect(result).toContain('<your-access-token-or-api-key>');
    });

    it('should insert API key into copilot format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'copilot', 'smtk_test123_secret');
      expect(result).toContain('"SEMANTIK_AUTH_TOKEN": "smtk_test123_secret"');
      expect(result).toContain('"type": "local"');
      // Verify tools array via JSON parsing (format uses multi-line arrays)
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers['semantik-test'].tools).toEqual(['*']);
    });

    it('should insert API key into amp format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'amp', 'smtk_test123_secret');
      expect(result).toContain('"SEMANTIK_AUTH_TOKEN": "smtk_test123_secret"');
      expect(result).toContain('"amp.mcpServers"');
    });

    it('should insert API key into opencode format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'opencode', 'smtk_test123_secret');
      // opencode format doesn't include env vars in stdio mode, just command array
      expect(result).toContain('"$schema": "https://opencode.ai/config.json"');
      expect(result).toContain('"type": "local"');
      expect(result).toContain('"enabled": true');
    });
  });

  describe('HTTP transport - all format types', () => {
    it('should generate standard HTTP config with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'standard');
      expect(result).toContain('"type": "sse"');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
      expect(result).toContain('"Authorization": "Bearer <your-api-key>"');
    });

    it('should generate standard HTTP config with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'standard', 'smtk_apikey123');
      expect(result).toContain('"type": "sse"');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
      expect(result).toContain('"Authorization": "Bearer smtk_apikey123"');
      expect(result).not.toContain('<your-api-key>');
    });

    it('should generate cline HTTP config with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'cline');
      expect(result).toContain('"type": "sse"');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
      expect(result).toContain('"timeout": 60');
      expect(result).toContain('"disabled": false');
      expect(result).toContain('"Authorization": "Bearer <your-api-key>"');
    });

    it('should generate cline HTTP config with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'cline', 'smtk_apikey123');
      expect(result).toContain('"Authorization": "Bearer smtk_apikey123"');
      expect(result).toContain('"timeout": 60');
    });

    it('should generate copilot HTTP config with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'copilot');
      expect(result).toContain('"type": "remote"');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
      expect(result).toContain('"Authorization": "Bearer <your-api-key>"');
      // Verify tools array via JSON parsing (format uses multi-line arrays)
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers['semantik-test'].tools).toEqual(['*']);
    });

    it('should generate copilot HTTP config with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'copilot', 'smtk_apikey123');
      expect(result).toContain('"type": "remote"');
      expect(result).toContain('"Authorization": "Bearer smtk_apikey123"');
    });

    it('should generate amp HTTP config with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'amp');
      expect(result).toContain('"amp.mcpServers"');
      expect(result).toContain('"type": "sse"');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
      expect(result).toContain('"Authorization": "Bearer <your-api-key>"');
    });

    it('should generate amp HTTP config with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'amp', 'smtk_apikey123');
      expect(result).toContain('"Authorization": "Bearer smtk_apikey123"');
    });

    it('should generate opencode HTTP config with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'opencode');
      expect(result).toContain('"$schema": "https://opencode.ai/config.json"');
      expect(result).toContain('"type": "remote"');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
      expect(result).toContain('"enabled": true');
      expect(result).toContain('"Authorization": "Bearer <your-api-key>"');
    });

    it('should generate opencode HTTP config with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'opencode', 'smtk_apikey123');
      expect(result).toContain('"Authorization": "Bearer smtk_apikey123"');
    });

    it('should generate codex HTTP config (TOML) with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'codex');
      expect(result).toContain('[mcp_servers.semantik-test]');
      expect(result).toContain('type = "sse"');
      expect(result).toContain('url = "http://localhost:9090/mcp"');
      expect(result).toContain('[mcp_servers.semantik-test.headers]');
      expect(result).toContain('Authorization = "Bearer <your-api-key>"');
    });

    it('should generate codex HTTP config (TOML) with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'codex', 'smtk_apikey123');
      expect(result).toContain('Authorization = "Bearer smtk_apikey123"');
      expect(result).not.toContain('<your-api-key>');
    });

    it('should generate claude-code HTTP config (CLI) with placeholder', () => {
      const result = generateMCPConfig(mockHttpConfig, 'claude-code');
      expect(result).toContain('claude mcp add --transport http');
      expect(result).toContain('--header "Authorization: Bearer <your-api-key>"');
      expect(result).toContain('semantik-test');
      expect(result).toContain('http://localhost:9090/mcp');
    });

    it('should generate claude-code HTTP config (CLI) with API key', () => {
      const result = generateMCPConfig(mockHttpConfig, 'claude-code', 'smtk_apikey123');
      expect(result).toContain('--header "Authorization: Bearer smtk_apikey123"');
      expect(result).not.toContain('<your-api-key>');
    });

    it('should handle HTTP config with default URL when url is undefined', () => {
      const configWithoutUrl: MCPClientConfig = {
        transport: 'http',
        server_name: 'semantik-test',
      };
      const result = generateMCPConfig(configWithoutUrl, 'standard');
      expect(result).toContain('"url": "http://localhost:9090/mcp"');
    });
  });

  describe('stdio transport - format structures', () => {
    it('should generate standard format structure', () => {
      const result = generateMCPConfig(mockStdioConfig, 'standard');
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers).toBeDefined();
      expect(parsed.mcpServers['semantik-test']).toBeDefined();
      expect(parsed.mcpServers['semantik-test'].command).toBe('semantik-mcp');
      expect(parsed.mcpServers['semantik-test'].args).toEqual(['serve', '--profile', 'test']);
    });

    it('should generate cline format with extra fields', () => {
      const result = generateMCPConfig(mockStdioConfig, 'cline');
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers['semantik-test'].type).toBe('stdio');
      expect(parsed.mcpServers['semantik-test'].timeout).toBe(60);
      expect(parsed.mcpServers['semantik-test'].disabled).toBe(false);
    });

    it('should generate copilot format with tools array', () => {
      const result = generateMCPConfig(mockStdioConfig, 'copilot');
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers['semantik-test'].type).toBe('local');
      expect(parsed.mcpServers['semantik-test'].tools).toEqual(['*']);
    });

    it('should generate amp format with amp.mcpServers key', () => {
      const result = generateMCPConfig(mockStdioConfig, 'amp');
      const parsed = JSON.parse(result);
      expect(parsed['amp.mcpServers']).toBeDefined();
      expect(parsed['amp.mcpServers']['semantik-test']).toBeDefined();
    });

    it('should generate opencode format with combined command array', () => {
      const result = generateMCPConfig(mockStdioConfig, 'opencode');
      const parsed = JSON.parse(result);
      expect(parsed.$schema).toBe('https://opencode.ai/config.json');
      expect(parsed.mcp['semantik-test'].type).toBe('local');
      expect(parsed.mcp['semantik-test'].command).toEqual(['semantik-mcp', 'serve', '--profile', 'test']);
      expect(parsed.mcp['semantik-test'].enabled).toBe(true);
    });

    it('should generate codex TOML format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'codex');
      expect(result).toContain('[mcp_servers.semantik-test]');
      expect(result).toContain('command = "semantik-mcp"');
      expect(result).toContain('args = ["serve", "--profile", "test"]');
      expect(result).toContain('[mcp_servers.semantik-test.env]');
      expect(result).toContain('SEMANTIK_WEBUI_URL = "http://localhost:8080"');
    });

    it('should generate claude-code CLI format', () => {
      const result = generateMCPConfig(mockStdioConfig, 'claude-code');
      expect(result).toContain('claude mcp add');
      expect(result).toContain('-e SEMANTIK_WEBUI_URL=http://localhost:8080');
      expect(result).toContain('-e SEMANTIK_AUTH_TOKEN=');
      expect(result).toContain('-- \\');
      expect(result).toContain('semantik-test');
      expect(result).toContain('semantik-mcp');
      expect(result).toContain('serve');
    });
  });

  describe('edge cases', () => {
    it('should handle config with empty env', () => {
      const configNoEnv: MCPClientConfig = {
        transport: 'stdio',
        server_name: 'test',
        command: 'test-cmd',
        args: [],
        env: {},
      };
      const result = generateMCPConfig(configNoEnv, 'codex');
      expect(result).toContain('[mcp_servers.test]');
      expect(result).toContain('command = "test-cmd"');
      expect(result).not.toContain('[mcp_servers.test.env]');
    });

    it('should handle config without env property', () => {
      const configUndefinedEnv: MCPClientConfig = {
        transport: 'stdio',
        server_name: 'test',
        command: 'test-cmd',
        args: [],
      };
      const result = generateMCPConfig(configUndefinedEnv, 'standard');
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers.test.env).toEqual({});
    });

    it('should handle config without args', () => {
      const configNoArgs: MCPClientConfig = {
        transport: 'stdio',
        server_name: 'test',
        command: 'test-cmd',
      };
      const result = generateMCPConfig(configNoArgs, 'standard');
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers.test.args).toEqual([]);
    });

    it('should escape special characters in claude-code format', () => {
      const configWithSpecialChars: MCPClientConfig = {
        transport: 'stdio',
        server_name: 'test',
        command: 'test-cmd',
        args: [],
        env: {
          SPECIAL: 'value with spaces and "quotes"',
        },
      };
      const result = generateMCPConfig(configWithSpecialChars, 'claude-code');
      expect(result).toContain('-e SPECIAL="value with spaces and \\"quotes\\""');
    });

    it('should escape backslashes and quotes in codex TOML format', () => {
      const configWithEscapes: MCPClientConfig = {
        transport: 'stdio',
        server_name: 'test',
        command: 'test-cmd',
        args: [],
        env: {
          PATH: 'C:\\Users\\Test',
          QUOTED: 'say "hello"',
        },
      };
      const result = generateMCPConfig(configWithEscapes, 'codex');
      expect(result).toContain('PATH = "C:\\\\Users\\\\Test"');
      expect(result).toContain('QUOTED = "say \\"hello\\""');
    });

    it('should fallback to standard format for unknown format type', () => {
      const result = generateMCPConfig(mockStdioConfig, 'unknown-format' as any);
      const parsed = JSON.parse(result);
      expect(parsed.mcpServers).toBeDefined();
      expect(parsed.mcpServers['semantik-test'].command).toBe('semantik-mcp');
    });
  });
});

describe('getConfigBlockLabel', () => {
  it('should return "Run in terminal" for claude-code', () => {
    expect(getConfigBlockLabel('claude-code')).toBe('Run in terminal');
  });

  it('should return "Add to config.toml" for codex', () => {
    expect(getConfigBlockLabel('codex')).toBe('Add to config.toml');
  });

  it('should return "Add to settings.json" for amp', () => {
    expect(getConfigBlockLabel('amp')).toBe('Add to settings.json');
  });

  it('should return "Add to opencode.json" for opencode', () => {
    expect(getConfigBlockLabel('opencode')).toBe('Add to opencode.json');
  });

  it('should return "Add to mcpServers" for standard', () => {
    expect(getConfigBlockLabel('standard')).toBe('Add to mcpServers');
  });

  it('should return "Add to mcpServers" for cline', () => {
    expect(getConfigBlockLabel('cline')).toBe('Add to mcpServers');
  });

  it('should return "Add to mcpServers" for copilot', () => {
    expect(getConfigBlockLabel('copilot')).toBe('Add to mcpServers');
  });
});

describe('getConfigLanguage', () => {
  it('should return "bash" for claude-code', () => {
    expect(getConfigLanguage('claude-code')).toBe('bash');
  });

  it('should return "toml" for codex', () => {
    expect(getConfigLanguage('codex')).toBe('toml');
  });

  it('should return "json" for standard', () => {
    expect(getConfigLanguage('standard')).toBe('json');
  });

  it('should return "json" for cline', () => {
    expect(getConfigLanguage('cline')).toBe('json');
  });

  it('should return "json" for copilot', () => {
    expect(getConfigLanguage('copilot')).toBe('json');
  });

  it('should return "json" for amp', () => {
    expect(getConfigLanguage('amp')).toBe('json');
  });

  it('should return "json" for opencode', () => {
    expect(getConfigLanguage('opencode')).toBe('json');
  });
});
