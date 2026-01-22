// apps/webui-react/src/utils/__tests__/mcp-config-generator.test.ts
import { describe, it, expect } from 'vitest';
import { generateMCPConfig } from '../mcp-config-generator';
import type { MCPClientConfig } from '../../types/mcp-profile';

const mockConfig: MCPClientConfig = {
  server_name: 'semantik-test',
  command: 'semantik-mcp',
  args: ['serve', '--profile', 'test'],
  env: {
    SEMANTIK_WEBUI_URL: 'http://localhost:8080',
    SEMANTIK_AUTH_TOKEN: '<your-access-token-or-api-key>',
  },
};

describe('generateMCPConfig', () => {
  describe('with apiKey parameter', () => {
    it('should insert API key into standard format', () => {
      const result = generateMCPConfig(mockConfig, 'standard', 'smtk_test123_secret');
      expect(result).toContain('"SEMANTIK_AUTH_TOKEN": "smtk_test123_secret"');
      expect(result).not.toContain('<your-access-token-or-api-key>');
    });

    it('should insert API key into cline format', () => {
      const result = generateMCPConfig(mockConfig, 'cline', 'smtk_test123_secret');
      expect(result).toContain('"SEMANTIK_AUTH_TOKEN": "smtk_test123_secret"');
    });

    it('should insert API key into codex TOML format', () => {
      const result = generateMCPConfig(mockConfig, 'codex', 'smtk_test123_secret');
      expect(result).toContain('SEMANTIK_AUTH_TOKEN = "smtk_test123_secret"');
    });

    it('should insert API key into claude-code CLI format', () => {
      const result = generateMCPConfig(mockConfig, 'claude-code', 'smtk_test123_secret');
      expect(result).toContain('SEMANTIK_AUTH_TOKEN=smtk_test123_secret');
    });

    it('should keep placeholder when apiKey is undefined', () => {
      const result = generateMCPConfig(mockConfig, 'standard');
      expect(result).toContain('<your-access-token-or-api-key>');
    });
  });
});
