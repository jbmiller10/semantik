import { useUIStore } from '../stores/uiStore';

function Toast() {
  const { toasts, removeToast } = useUIStore();

  if (toasts.length === 0) return null;

  return (
    <div className="fixed bottom-0 right-0 p-6 space-y-4 z-50">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          data-testid="toast"
          className={`max-w-sm w-full bg-white shadow-lg rounded-lg pointer-events-auto ring-1 ring-black ring-opacity-5 overflow-hidden toast-${toast.type} ${
            toast.type === 'error'
              ? 'border-l-4 border-red-500'
              : toast.type === 'success'
              ? 'border-l-4 border-green-500'
              : toast.type === 'warning'
              ? 'border-l-4 border-yellow-500'
              : 'border-l-4 border-blue-500'
          }`}
        >
          <div className="p-4">
            <div className="flex items-start">
              <div className="flex-1">
                <p className="text-sm font-medium text-gray-900">
                  {toast.type === 'error' && 'Error'}
                  {toast.type === 'success' && 'Success'}
                  {toast.type === 'warning' && 'Warning'}
                  {toast.type === 'info' && 'Info'}
                </p>
                <p className="mt-1 text-sm text-gray-500">{toast.message}</p>
              </div>
              <button
                onClick={() => removeToast(toast.id)}
                className="ml-4 text-gray-400 hover:text-gray-500 focus:outline-none"
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