"""End-to-end tests for collection deletion functionality.

These tests verify the complete flow from UI interaction to database cleanup.
"""

import pytest
from playwright.sync_api import Page, expect
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from shared.database.models import Collection, Document, Operation


@pytest.mark.e2e()
class TestCollectionDeletionE2E:
    """End-to-end tests for collection deletion."""

    def test_delete_collection_from_ui(self, page: Page, test_collection_with_data) -> None:
        """Test deleting a collection through the UI."""
        collection_name = test_collection_with_data["name"]

        # Navigate to collections page
        page.goto("/collections")

        # Find and click on the collection card
        collection_card = page.locator(f"text={collection_name}").first
        collection_card.click()

        # Wait for collection details modal
        page.wait_for_selector("text=Delete Collection")

        # Click delete button
        delete_button = page.locator("button:has-text('Delete')")
        delete_button.click()

        # Wait for delete confirmation modal
        page.wait_for_selector("text=Type DELETE to confirm")

        # Type DELETE in confirmation input
        confirmation_input = page.locator("input[placeholder='Type DELETE here']")
        confirmation_input.fill("DELETE")

        # Click the delete button
        confirm_delete_button = page.locator("button:has-text('Delete Collection')").last
        confirm_delete_button.click()

        # Wait for success toast
        expect(page.locator("text=Collection deleted successfully")).to_be_visible(timeout=5000)

        # Verify collection is gone from the list
        expect(page.locator(f"text={collection_name}")).not_to_be_visible()

    def test_delete_collection_keyboard_shortcut(self, page: Page, test_collection_with_data) -> None:
        """Test deleting a collection using keyboard navigation."""
        collection_name = test_collection_with_data["name"]

        # Navigate to collections page
        page.goto("/collections")

        # Find and click on the collection card
        collection_card = page.locator(f"text={collection_name}").first
        collection_card.click()

        # Wait for modal and click delete
        page.wait_for_selector("text=Delete Collection")
        page.keyboard.press("Tab")  # Navigate to delete button
        page.keyboard.press("Tab")
        page.keyboard.press("Tab")
        page.keyboard.press("Enter")

        # Type DELETE
        page.wait_for_selector("input[placeholder='Type DELETE here']")
        page.keyboard.type("DELETE")

        # Submit with Enter
        page.keyboard.press("Enter")

        # Verify deletion
        expect(page.locator("text=Collection deleted successfully")).to_be_visible(timeout=5000)

    @pytest.mark.asyncio()
    async def test_delete_collection_database_cleanup(
        self, page: Page, db_session: AsyncSession, test_collection_with_data
    ) -> None:
        """Test that deletion properly cleans up all database records."""
        collection_id = test_collection_with_data["id"]
        collection_name = test_collection_with_data["name"]

        # Verify initial state - collection exists with documents
        result = await db_session.execute(select(Collection).where(Collection.id == collection_id))
        assert result.scalar_one_or_none() is not None

        # Check documents exist
        docs_result = await db_session.execute(select(Document).where(Document.collection_id == collection_id))
        documents = docs_result.scalars().all()
        assert len(documents) > 0

        # Delete through UI
        page.goto("/collections")
        collection_card = page.locator(f"text={collection_name}").first
        collection_card.click()

        page.wait_for_selector("text=Delete Collection")
        delete_button = page.locator("button:has-text('Delete')")
        delete_button.click()

        confirmation_input = page.locator("input[placeholder='Type DELETE here']")
        confirmation_input.fill("DELETE")

        confirm_delete_button = page.locator("button:has-text('Delete Collection')").last
        confirm_delete_button.click()

        # Wait for deletion to complete
        expect(page.locator("text=Collection deleted successfully")).to_be_visible(timeout=5000)

        # Verify database cleanup
        await db_session.commit()  # Ensure we see the latest state

        # Collection should be gone
        result = await db_session.execute(select(Collection).where(Collection.id == collection_id))
        assert result.scalar_one_or_none() is None

        # Documents should be gone
        docs_result = await db_session.execute(select(Document).where(Document.collection_id == collection_id))
        assert len(docs_result.scalars().all()) == 0

        # Operations should be gone
        ops_result = await db_session.execute(select(Operation).where(Operation.collection_id == collection_id))
        assert len(ops_result.scalars().all()) == 0

    def test_delete_collection_permission_denied(self, page: Page, other_user_collection) -> None:
        """Test that users cannot delete collections they don't own."""
        collection_name = other_user_collection["name"]

        # Navigate to collections page
        page.goto("/collections")

        # The collection should not be visible if it's not public
        if not other_user_collection.get("is_public", False):
            expect(page.locator(f"text={collection_name}")).not_to_be_visible()
        else:
            # If it's public, try to delete it
            collection_card = page.locator(f"text={collection_name}").first
            collection_card.click()

            # Delete button should be disabled or hidden
            delete_button = page.locator("button:has-text('Delete')")
            expect(delete_button).to_be_disabled()

    def test_delete_collection_with_active_operation(self, page: Page, collection_with_active_operation) -> None:
        """Test that collections with active operations cannot be deleted."""
        collection_name = collection_with_active_operation["name"]

        # Navigate and try to delete
        page.goto("/collections")
        collection_card = page.locator(f"text={collection_name}").first
        collection_card.click()

        delete_button = page.locator("button:has-text('Delete')")
        delete_button.click()

        confirmation_input = page.locator("input[placeholder='Type DELETE here']")
        confirmation_input.fill("DELETE")

        confirm_delete_button = page.locator("button:has-text('Delete Collection')").last
        confirm_delete_button.click()

        # Should see error message
        expect(page.locator("text=operations are in progress")).to_be_visible(timeout=5000)

    def test_delete_collection_cancel(self, page: Page, test_collection_with_data) -> None:
        """Test canceling collection deletion."""
        collection_name = test_collection_with_data["name"]

        # Navigate and open delete modal
        page.goto("/collections")
        collection_card = page.locator(f"text={collection_name}").first
        collection_card.click()

        delete_button = page.locator("button:has-text('Delete')")
        delete_button.click()

        # Click cancel
        cancel_button = page.locator("button:has-text('Cancel')")
        cancel_button.click()

        # Modal should close, collection should still exist
        expect(page.locator("text=Type DELETE to confirm")).not_to_be_visible()

        # Go back to collections list
        page.keyboard.press("Escape")  # Close details modal

        # Collection should still be in the list
        expect(page.locator(f"text={collection_name}")).to_be_visible()

    def test_delete_multiple_collections_sequentially(self, page: Page, multiple_test_collections) -> None:
        """Test deleting multiple collections one after another."""
        for collection in multiple_test_collections:
            collection_name = collection["name"]

            # Navigate to collections if not already there
            if page.url.endswith("/collections"):
                page.reload()  # Refresh to see latest state
            else:
                page.goto("/collections")

            # Delete the collection
            collection_card = page.locator(f"text={collection_name}").first
            collection_card.click()

            delete_button = page.locator("button:has-text('Delete')")
            delete_button.click()

            confirmation_input = page.locator("input[placeholder='Type DELETE here']")
            confirmation_input.fill("DELETE")

            confirm_delete_button = page.locator("button:has-text('Delete Collection')").last
            confirm_delete_button.click()

            # Wait for success
            expect(page.locator("text=Collection deleted successfully")).to_be_visible(timeout=5000)

            # Verify it's gone
            expect(page.locator(f"text={collection_name}")).not_to_be_visible()


# Playwright fixtures for e2e tests
@pytest.fixture()
def test_collection_with_data(create_test_collection_with_documents) -> None:
    """Create a test collection with some documents."""
    return create_test_collection_with_documents(name="E2E Test Collection", document_count=5)


@pytest.fixture()
def other_user_collection(create_collection_for_other_user) -> None:
    """Create a collection owned by another user."""
    return create_collection_for_other_user(name="Other User Collection", is_public=False)


@pytest.fixture()
def collection_with_active_operation(create_collection_with_active_operation) -> None:
    """Create a collection with an active operation."""
    return create_collection_with_active_operation(name="Collection with Active Op", operation_type="index")


@pytest.fixture()
def multiple_test_collections(create_test_collection_with_documents) -> None:
    """Create multiple test collections."""
    collections = []
    for i in range(3):
        collection = create_test_collection_with_documents(name=f"Test Collection {i + 1}", document_count=2)
        collections.append(collection)
    return collections
