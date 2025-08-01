# Code Style and Conventions

## Python Code Style
- **Formatter**: Black with line-length=120
- **Import Sorting**: isort
- **Linter**: Ruff with extensive rule set (pycodestyle, pyflakes, pep8-naming, etc.)
- **Type Checking**: mypy with --ignore-missing-imports
- **Target Python Version**: 3.11+

## Ruff Configuration
- Line length: 120
- Extensive lint rules enabled including:
  - E/W (pycodestyle errors/warnings)
  - F (pyflakes)
  - N (pep8-naming)
  - UP (pyupgrade)
  - B (flake8-bugbear)
  - And many more quality checks
- Ignores: E501 (line too long), B008 (function calls in defaults for FastAPI)

## TypeScript/React Code Style
- **Linter**: ESLint
- **Type Checking**: TypeScript strict mode
- **Build Tool**: Vite
- **Testing**: Vitest for unit tests, Playwright for E2E

## Architectural Patterns
- **Separation of Concerns**: 
  - API routers (packages/webui/api/) contain ONLY routing logic
  - Business logic goes in services (packages/webui/services/)
  - Database calls through repository pattern
  - Frontend state management centralized in Zustand stores
- **Async/Await**: Consistently use async patterns, no sync I/O in async functions
- **Error Handling**: Service layer exceptions mapped to appropriate HTTP status codes

## Naming Conventions
- **Python**: snake_case for functions/variables, PascalCase for classes
- **TypeScript**: camelCase for functions/variables, PascalCase for components/types
- **API Routes**: RESTful conventions with v2 versioning
- **Database**: snake_case for tables and columns

## Security Best Practices
- Never hardcode secrets or API keys
- Always validate and sanitize user inputs
- Use parameterized queries (SQLAlchemy)
- JWT tokens for authentication
- CORS properly configured