#!/usr/bin/env python3
"""
Operation status value object with state transition rules.

This module defines the valid states for a chunking operation and enforces
the business rules for state transitions.
"""

from enum import Enum


class OperationStatus(Enum):
    """Status of a chunking operation with state machine rules."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    def can_transition_to(self, new_status: "OperationStatus") -> bool:
        """
        Check if transition to new status is allowed by business rules.

        Args:
            new_status: The target status

        Returns:
            True if transition is allowed, False otherwise
        """
        transitions = {
            OperationStatus.PENDING: [
                OperationStatus.PROCESSING,
                OperationStatus.CANCELLED,
                OperationStatus.FAILED,
            ],
            OperationStatus.PROCESSING: [
                OperationStatus.COMPLETED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
            ],
            OperationStatus.COMPLETED: [],  # Terminal state
            OperationStatus.FAILED: [],  # Terminal state
            OperationStatus.CANCELLED: [],  # Terminal state
        }
        return new_status in transitions[self]

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self in [
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
        ]

    def is_active(self) -> bool:
        """Check if the operation is actively running."""
        return self == OperationStatus.PROCESSING

    def is_pending(self) -> bool:
        """Check if the operation is waiting to start."""
        return self == OperationStatus.PENDING
