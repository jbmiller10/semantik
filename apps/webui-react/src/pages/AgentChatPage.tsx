/**
 * Page component for the agent chat interface.
 * Wraps the AgentChat component with routing integration.
 */

import { useParams, useNavigate } from 'react-router-dom';
import { AgentChat } from '../components/agent';

export function AgentChatPage() {
  const { conversationId } = useParams<{ conversationId: string }>();
  const navigate = useNavigate();

  const handleClose = () => {
    navigate('/');
  };

  const handleApplySuccess = (collectionId: string) => {
    // Navigate to the newly created collection
    navigate(`/collections/${collectionId}`);
  };

  if (!conversationId) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-lg font-medium text-[var(--text-primary)] mb-2">
            Invalid Conversation
          </h2>
          <p className="text-sm text-[var(--text-secondary)] mb-4">
            No conversation ID provided.
          </p>
          <button onClick={handleClose} className="btn-secondary">
            Go Back
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-64px)]">
      <AgentChat
        conversationId={conversationId}
        onClose={handleClose}
        onApplySuccess={handleApplySuccess}
      />
    </div>
  );
}

export default AgentChatPage;
