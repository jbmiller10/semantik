# Task Completion Checklist

When you complete a coding task in Semantik, you MUST:

## 1. Code Quality Checks
- Run `make format` to format Python code with Black and isort
- Run `make lint` to check for linting errors with Ruff
- Run `make type-check` to verify type annotations with mypy
- For frontend changes: run `npm run lint` in apps/webui-react

## 2. Testing
- Write tests for new functionality
- Run `make test` to ensure all tests pass
- For API changes, consider adding integration tests
- For frontend changes, run `npm run test` in apps/webui-react

## 3. Architecture Compliance
- Verify API routers contain ONLY routing logic (no business logic)
- Ensure business logic is in service layer
- Check that database calls go through repositories
- Frontend state changes should be in Zustand stores

## 4. Documentation
- Update docstrings for new functions/classes
- Add type hints to all function signatures
- Update API documentation if endpoints changed

## 5. Security Review
- No hardcoded secrets or keys
- Input validation and sanitization implemented
- Error messages don't expose sensitive information
- Authentication/authorization properly handled

## 6. Refactoring Progress
- Use "operation" terminology instead of "job"
- Follow "collection-centric" architecture patterns
- Remove any legacy "job-centric" code when encountered

## 7. Final Verification
Run the comprehensive check command:
```bash
make check  # Runs format, lint, and test
```

Only consider the task complete when all checks pass!