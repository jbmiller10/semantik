"""Domain-specific exceptions for repository operations.

These exceptions provide clear error handling and better context for database operations.
"""


class RepositoryError(Exception):
    """Base exception for all repository-related errors."""


class EntityNotFoundError(RepositoryError):
    """Raised when an entity is not found in the repository."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        """Initialize with entity type and ID."""
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with ID '{entity_id}' not found")


class EntityAlreadyExistsError(RepositoryError):
    """Raised when trying to create an entity that already exists."""

    def __init__(self, entity_type: str, identifier: str) -> None:
        """Initialize with entity type and identifier."""
        self.entity_type = entity_type
        self.identifier = identifier
        super().__init__(f"{entity_type} with identifier '{identifier}' already exists")


class InvalidUserIdError(RepositoryError):
    """Raised when a user ID is invalid or in wrong format."""

    def __init__(self, user_id: str) -> None:
        """Initialize with the invalid user ID."""
        self.user_id = user_id
        super().__init__(f"Invalid user ID format: '{user_id}' must be numeric")


class AccessDeniedError(RepositoryError):
    """Raised when a user doesn't have access to a resource."""

    def __init__(self, user_id: str, resource_type: str, resource_id: str) -> None:
        """Initialize with user and resource information."""
        self.user_id = user_id
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(f"User '{user_id}' does not have access to {resource_type} '{resource_id}'")


class ValidationError(RepositoryError):
    """Raised when validation fails for an operation."""

    def __init__(self, message: str, field: str | None = None) -> None:
        """Initialize with validation message and optional field."""
        self.message = message
        self.field = field
        if field:
            super().__init__(f"Validation error on field '{field}': {message}")
        else:
            super().__init__(f"Validation error: {message}")


class DatabaseOperationError(RepositoryError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, entity_type: str, details: str | None = None) -> None:
        """Initialize with operation details."""
        self.operation = operation
        self.entity_type = entity_type
        self.details = details
        message = f"Failed to {operation} {entity_type}"
        if details:
            message += f": {details}"
        super().__init__(message)


class TransactionError(RepositoryError):
    """Raised when a transaction fails."""

    def __init__(self, message: str, rollback_successful: bool = False) -> None:
        """Initialize with transaction error details."""
        self.rollback_successful = rollback_successful
        if rollback_successful:
            super().__init__(f"Transaction failed and was rolled back: {message}")
        else:
            super().__init__(f"Transaction failed: {message}")


class ConcurrencyError(RepositoryError):
    """Raised when a concurrent modification is detected."""

    def __init__(self, entity_type: str, entity_id: str) -> None:
        """Initialize with entity information."""
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} '{entity_id}' was modified by another process")
