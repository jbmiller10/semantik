import { useUIStore } from '../stores/uiStore';

function Toast() {
  const { toasts, removeToast } = useUIStore();

  if (!toasts || toasts.length === 0) return null;

  const getBorderColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'border-l-error';
      case 'success':
        return 'border-l-success';
      case 'warning':
        return 'border-l-warning';
      default:
        return 'border-l-info';
    }
  };

  const getTitle = (type: string) => {
    switch (type) {
      case 'error':
        return 'Error';
      case 'success':
        return 'Success';
      case 'warning':
        return 'Warning';
      default:
        return 'Info';
    }
  };

  return (
    <div className="fixed bottom-0 right-0 p-6 space-y-4 z-50">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          data-testid="toast"
          className={`max-w-sm w-full bg-[var(--bg-elevated)] shadow-lg rounded-lg pointer-events-auto border border-[var(--border)] overflow-hidden animate-slide-up border-l-4 ${getBorderColor(toast.type)}`}
        >
          <div className="p-4">
            <div className="flex items-start">
              <div className="flex-1">
                <p className="text-sm font-medium text-[var(--text-primary)]">
                  {getTitle(toast.type)}
                </p>
                <p className="mt-1 text-sm text-[var(--text-secondary)]">{toast.message}</p>
              </div>
              <button
                onClick={() => removeToast(toast.id)}
                className="ml-4 text-[var(--text-muted)] hover:text-[var(--text-primary)] focus:outline-none transition-colors"
              >
                <svg
                  className="h-5 w-5"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                    clipRule="evenodd"
                  />
                </svg>
              </button>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default Toast;
