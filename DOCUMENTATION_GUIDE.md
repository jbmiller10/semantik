# Semantik Documentation Guide

## Documentation Structure

This codebase uses context-aware CLAUDE.md files placed strategically throughout the project. When an LLM reads any file in a directory, it automatically loads the corresponding CLAUDE.md to provide context.

## Documentation Locations

### Core Services
- `/packages/webui/CLAUDE.md` - WebUI FastAPI service
- `/packages/shared/CLAUDE.md` - Shared models and utilities  
- `/packages/vecpipe/CLAUDE.md` - Vector embedding service

### Frontend
- `/apps/webui-react/src/CLAUDE.md` - React application
- `/apps/webui-react/src/components/chunking/CLAUDE.md` - Chunking UI components

### Infrastructure
- `/alembic/CLAUDE.md` - Database migrations
- `/tests/CLAUDE.md` - Testing infrastructure
- `/CLAUDE_DOCKER.md` - Docker configuration (root level)

### Project Root
- `/CLAUDE.md` - Main project instructions and architecture overview

## Documentation Format

All documentation uses XML-style tags for better structure:
- `<component>` - Component identification
- `<architecture>` - Architectural patterns
- `<critical-rule>` - Important rules to follow
- `<anti-pattern>` - What NOT to do
- `<common-pitfalls>` - Frequent mistakes

## Benefits

1. **Context-aware**: Documentation loads automatically when working in a directory
2. **Concise**: XML format is more compact than markdown
3. **Structured**: Clear hierarchy and relationships
4. **Maintainable**: Documentation lives next to the code it describes

## Usage

When working on a specific component:
1. The LLM automatically reads the local CLAUDE.md
2. No need to manually load documentation
3. Context is always relevant to the current task
4. Reduces context window usage

## Maintenance

When making significant changes:
1. Update the relevant CLAUDE.md file
2. Keep documentation close to the code
3. Focus on critical information only
4. Use XML tags for structure