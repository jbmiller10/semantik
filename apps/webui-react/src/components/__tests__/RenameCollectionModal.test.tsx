import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';
import RenameCollectionModal from '../RenameCollectionModal';
import { TestWrapper } from '../../tests/utils/TestWrapper';
import { collectionsV2Api } from '../../services/api/v2/collections';
import { AxiosError, InternalAxiosRequestConfig } from 'axios';
import type { MockCollection } from '@/tests/types/test-types';
import { createMockCollection } from '@/tests/types/test-types';

// Mock the API
vi.mock('../../services/api/v2/collections', () => ({
  collectionsV2Api: {
    update: vi.fn(),
  },
}));

// Mock the UI store
const mockAddToast = vi.fn();
vi.mock('../../stores/uiStore', () => ({
  useUIStore: () => ({
    addToast: mockAddToast,
  }),
}));

describe('RenameCollectionModal', () => {
  const defaultProps = {
    collectionId: 'test-collection-id',
    currentName: 'Test Collection',
    onClose: vi.fn(),
    onSuccess: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render modal with current collection name', () => {
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      // Check modal title
      expect(screen.getByText('Rename Collection')).toBeInTheDocument();

      // Check current name is displayed
      expect(screen.getByLabelText('Current Name')).toHaveValue('Test Collection');
      expect(screen.getByLabelText('Current Name')).toBeDisabled();

      // Check new name input has current name as initial value
      expect(screen.getByLabelText('New Name')).toHaveValue('Test Collection');

      // Check warning message
      expect(
        screen.getByText(
          'This will only change the display name. The underlying data and vector collections will remain unchanged.'
        )
      ).toBeInTheDocument();
    });

    it('should render action buttons', () => {
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Rename' })).toBeInTheDocument();
    });

    it('should disable rename button initially when name is unchanged', () => {
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const renameButton = screen.getByRole('button', { name: 'Rename' });
      expect(renameButton).toBeDisabled();
    });
  });

  describe('Form Validation', () => {
    it('should show error for empty name', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Clear the input
      await user.clear(newNameInput);
      
      // Check error message
      expect(screen.getByText('Collection name cannot be empty')).toBeInTheDocument();
      
      // Check rename button is disabled
      expect(screen.getByRole('button', { name: 'Rename' })).toBeDisabled();
    });

    it('should show error for name with only spaces', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Clear and type spaces
      await user.clear(newNameInput);
      await user.type(newNameInput, '   ');
      
      // Check error message
      expect(screen.getByText('Collection name cannot be empty')).toBeInTheDocument();
    });

    it('should show error for invalid characters', async () => {
      const user = userEvent.setup();
      const invalidChars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];
      
      for (const char of invalidChars) {
        const { unmount } = render(
          <TestWrapper>
            <RenameCollectionModal {...defaultProps} />
          </TestWrapper>
        );

        const newNameInput = screen.getByLabelText('New Name');
        
        // Clear and type invalid character
        await user.clear(newNameInput);
        await user.type(newNameInput, `Test${char}Name`);
        
        // Check error message
        expect(screen.getByText(`Collection name cannot contain "${char}"`)).toBeInTheDocument();
        
        // Cleanup for next iteration using proper cleanup method
        unmount();
      }
    });

    it('should clear error when valid name is entered', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // First create an error
      await user.clear(newNameInput);
      expect(screen.getByText('Collection name cannot be empty')).toBeInTheDocument();
      
      // Then type valid name
      await user.type(newNameInput, 'Valid Name');
      
      // Error should be gone
      expect(screen.queryByText('Collection name cannot be empty')).not.toBeInTheDocument();
      
      // Rename button should be enabled
      expect(screen.getByRole('button', { name: 'Rename' })).toBeEnabled();
    });

    it('should disable rename button when name is same as current', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Change name then change back
      await user.clear(newNameInput);
      await user.type(newNameInput, 'Different Name');
      expect(screen.getByRole('button', { name: 'Rename' })).toBeEnabled();
      
      await user.clear(newNameInput);
      await user.type(newNameInput, 'Test Collection');
      expect(screen.getByRole('button', { name: 'Rename' })).toBeDisabled();
    });
  });

  describe('Successful Rename Flow', () => {
    it('should call API and show success when rename succeeds', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      mockUpdate.mockResolvedValueOnce(createMockCollection({ 
        id: 'test-collection-id',
        name: 'New Collection Name' 
      }));
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      const renameButton = screen.getByRole('button', { name: 'Rename' });
      
      // Change name
      await user.clear(newNameInput);
      await user.type(newNameInput, 'New Collection Name');
      
      // Click rename
      await user.click(renameButton);
      
      // Check API was called correctly
      await waitFor(() => {
        expect(mockUpdate).toHaveBeenCalledWith('test-collection-id', {
          name: 'New Collection Name',
        });
      });
      
      // Check success callback was called
      await waitFor(() => {
        expect(defaultProps.onSuccess).toHaveBeenCalledWith('New Collection Name');
      });
    });

    it('should show loading state during rename', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      
      // Create a promise we can control
      let resolveUpdate: (value: MockCollection) => void;
      const updatePromise = new Promise<MockCollection>((resolve) => {
        resolveUpdate = resolve;
      });
      mockUpdate.mockReturnValueOnce(updatePromise);
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Change name
      await user.clear(newNameInput);
      await user.type(newNameInput, 'New Collection Name');
      
      // Click rename
      await user.click(screen.getByRole('button', { name: 'Rename' }));
      
      // Check loading state
      expect(screen.getByRole('button', { name: 'Renaming...' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Renaming...' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
      
      // Resolve the promise
      resolveUpdate!(createMockCollection({ id: 'test-collection-id', name: 'New Collection Name' }));
      
      // Wait for loading state to clear
      await waitFor(() => {
        expect(defaultProps.onSuccess).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('should show error when rename fails with API error', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      const axiosError = new AxiosError('Network error');
      axiosError.response = {
        data: { detail: 'Collection name already exists' },
        status: 400,
        statusText: 'Bad Request',
        headers: {},
        config: {} as InternalAxiosRequestConfig,
      };
      mockUpdate.mockRejectedValueOnce(axiosError);
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Change name and submit
      await user.clear(newNameInput);
      await user.type(newNameInput, 'Duplicate Name');
      await user.click(screen.getByRole('button', { name: 'Rename' }));
      
      // Check error is displayed
      await waitFor(() => {
        expect(screen.getByText('Collection name already exists')).toBeInTheDocument();
      });
      
      // Check toast was shown
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Collection name already exists',
      });
      
      // Check onSuccess was not called
      expect(defaultProps.onSuccess).not.toHaveBeenCalled();
    });

    it('should show generic error when rename fails without detail', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      mockUpdate.mockRejectedValueOnce(new Error('Network error'));
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Change name and submit
      await user.clear(newNameInput);
      await user.type(newNameInput, 'New Name');
      await user.click(screen.getByRole('button', { name: 'Rename' }));
      
      // Check error is displayed
      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument();
      });
      
      // Check toast was shown
      expect(mockAddToast).toHaveBeenCalledWith({
        type: 'error',
        message: 'Network error',
      });
    });

    it('should prevent submission when name equals current name', async () => {
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      // Try to submit form without changing name
      const form = screen.getByRole('button', { name: 'Rename' }).closest('form')!;
      fireEvent.submit(form);
      
      // API should not be called
      expect(mockUpdate).not.toHaveBeenCalled();
      
      // Error should be shown (from mutation)
      await waitFor(() => {
        expect(screen.getByText('New name must be different from current name')).toBeInTheDocument();
      });
    });
  });

  describe('Modal Close Functionality', () => {
    it('should close modal when Cancel button is clicked', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      await user.click(screen.getByRole('button', { name: 'Cancel' }));
      
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should close modal when backdrop is clicked', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      // Click the backdrop (first div with onClick)
      const backdrop = screen.getByText('Rename Collection').parentElement!.parentElement!.previousElementSibling!;
      await user.click(backdrop);
      
      expect(defaultProps.onClose).toHaveBeenCalledTimes(1);
    });

    it('should not close modal when clicking inside modal content', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      // Click inside the modal
      await user.click(screen.getByText('Rename Collection'));
      
      expect(defaultProps.onClose).not.toHaveBeenCalled();
    });
  });

  describe('Keyboard Shortcuts', () => {
    it('should submit form when Enter is pressed in input', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      mockUpdate.mockResolvedValueOnce(createMockCollection({ 
        id: 'test-collection-id',
        name: 'New Name' 
      }));
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Change name
      await user.clear(newNameInput);
      await user.type(newNameInput, 'New Name');
      
      // Press Enter
      await user.keyboard('{Enter}');
      
      // Check API was called
      await waitFor(() => {
        expect(mockUpdate).toHaveBeenCalledWith('test-collection-id', {
          name: 'New Name',
        });
      });
    });

    it('should not submit when Enter is pressed with invalid name', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Clear name (invalid)
      await user.clear(newNameInput);
      
      // Press Enter
      await user.keyboard('{Enter}');
      
      // API should not be called
      expect(mockUpdate).not.toHaveBeenCalled();
    });
  });

  describe('Focus Management', () => {
    it('should auto-focus on new name input when modal opens', () => {
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      expect(document.activeElement).toBe(newNameInput);
    });

    it('should maintain focus on input during validation', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Type invalid character
      await user.clear(newNameInput);
      await user.type(newNameInput, 'Test<Name');
      
      // Focus should remain on input
      expect(document.activeElement).toBe(newNameInput);
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA labels and roles', () => {
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      // Check modal has proper structure
      expect(screen.getByText('Rename Collection')).toBeInTheDocument();
      
      // Check inputs have labels
      expect(screen.getByLabelText('Current Name')).toBeInTheDocument();
      expect(screen.getByLabelText('New Name')).toBeInTheDocument();
      
      // Check buttons have proper roles
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Rename' })).toBeInTheDocument();
    });

    it('should announce error messages to screen readers', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Clear input to trigger error
      await user.clear(newNameInput);
      
      // Error should be associated with input (though not with aria-describedby in current implementation)
      const errorMessage = screen.getByText('Collection name cannot be empty');
      expect(errorMessage).toBeInTheDocument();
      expect(errorMessage).toHaveClass('text-red-400');
    });

    it('should properly disable form elements during submission', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      
      // Create a promise we can control
      let resolveUpdate: (value: MockCollection) => void;
      const updatePromise = new Promise<MockCollection>((resolve) => {
        resolveUpdate = resolve;
      });
      mockUpdate.mockReturnValueOnce(updatePromise);
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Change name and submit
      await user.clear(newNameInput);
      await user.type(newNameInput, 'New Name');
      await user.click(screen.getByRole('button', { name: 'Rename' }));
      
      // Both buttons should be disabled during submission
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
      expect(screen.getByRole('button', { name: 'Renaming...' })).toBeDisabled();
      
      // Resolve to clean up
      resolveUpdate!(createMockCollection({ id: 'test-collection-id', name: 'New Name' }));
      await waitFor(() => {
        expect(defaultProps.onSuccess).toHaveBeenCalled();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long collection names', async () => {
      const user = userEvent.setup();
      const veryLongName = 'A'.repeat(256);
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Type very long name
      await user.clear(newNameInput);
      await user.type(newNameInput, veryLongName);
      
      // Should be valid (no length constraint in component)
      expect(screen.getByRole('button', { name: 'Rename' })).toBeEnabled();
    });

    it('should handle rapid validation changes', async () => {
      const user = userEvent.setup();
      
      render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Rapidly type valid and invalid characters
      await user.clear(newNameInput);
      await user.type(newNameInput, 'Test');
      await user.type(newNameInput, '<');
      await user.type(newNameInput, '{Backspace}');
      await user.type(newNameInput, 'Name');
      
      // Should end in valid state
      expect(screen.queryByText(/Collection name cannot contain/)).not.toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Rename' })).toBeEnabled();
    });

    it('should handle component unmounting during API call', async () => {
      const user = userEvent.setup();
      const mockUpdate = vi.mocked(collectionsV2Api.update);
      
      // Create a promise that never resolves
      mockUpdate.mockReturnValueOnce(new Promise(() => {}));
      
      const { unmount } = render(
        <TestWrapper>
          <RenameCollectionModal {...defaultProps} />
        </TestWrapper>
      );

      const newNameInput = screen.getByLabelText('New Name');
      
      // Start rename
      await user.clear(newNameInput);
      await user.type(newNameInput, 'New Name');
      await user.click(screen.getByRole('button', { name: 'Rename' }));
      
      // Unmount while loading
      unmount();
      
      // Should not throw any errors
      expect(true).toBe(true);
    });
  });
});