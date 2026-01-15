import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';
import { authApi } from '../services/api/v2';
import { getErrorMessage } from '../utils/errorUtils';

function LoginPage() {
  const navigate = useNavigate();
  const setAuth = useAuthStore((state) => state.setAuth);
  const addToast = useUIStore((state) => state.addToast);

  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);

    try {
      if (isLogin) {
        // Login flow
        const response = await authApi.login({ username, password });

        if (response.data.access_token) {
          // First set the auth token so subsequent API calls work
          // We'll use a temporary user object and update it after
          setAuth(response.data.access_token, {
            id: 0,
            username: username,
            email: '',
            is_active: true,
            is_superuser: false,
            created_at: new Date().toISOString()
          }, response.data.refresh_token);

          // Now get the actual user info
          try {
            const userResponse = await authApi.me();
            // Update with real user data
            setAuth(response.data.access_token, userResponse.data, response.data.refresh_token);
          } catch (error) {
            console.error('Failed to fetch user details:', error);
            // Continue with login even if user details fetch fails
          }

          addToast({
            type: 'success',
            message: 'Logged in successfully',
          });
          navigate('/');
        }
      } else {
        // Registration flow
        const response = await authApi.register({
          username,
          email,
          password,
          full_name: fullName || undefined
        });

        if (response.data.id) {
          addToast({
            type: 'success',
            message: 'Registration successful! Please log in with your credentials.',
          });
          // Switch to login mode and clear password
          setIsLogin(true);
          setPassword('');
          // Keep username filled for convenience
        }
      }
    } catch (error) {
      addToast({
        type: 'error',
        message: getErrorMessage(error),
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8">
      {/* Ambient background glow for login */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-signal-600/10 blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] rounded-full bg-data-teal/5 blur-[120px]" />
      </div>

      <div className="max-w-md w-full space-y-8 glass-panel p-10 rounded-none border-l-0 border-r-0 sm:border sm:rounded-2xl animate-fade-in relative z-10">
        <div>
          <div className="flex flex-col items-center">
            {/* Text Logo for Swiss Style */}
            <h1 className="text-4xl font-bold tracking-tight text-white mb-2">SEMANTIK</h1>
            <div className="h-1 w-12 bg-signal-600 rounded-full"></div>
          </div>
          <h2 className="mt-8 text-center text-xl font-medium text-gray-200">
            {isLogin ? 'Welcome back' : 'Initialize account'}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-500">
            {isLogin ? 'Enter your credentials to access the void' : 'Create your secure identity'}
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-xs uppercase tracking-wider font-semibold text-gray-500 mb-1">
                Username
              </label>
              <input
                id="username"
                name="username"
                type="text"
                autoComplete="username"
                required
                minLength={3}
                pattern="^[A-Za-z0-9_]+$"
                title="Use at least 3 characters. Letters, numbers, and underscores only."
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="block w-full px-4 py-3 input-glass rounded-lg text-sm"
                placeholder="username"
                aria-describedby={!isLogin ? 'username-help' : undefined}
              />
            </div>
            {!isLogin && (
              <>
                <div>
                  <label htmlFor="email" className="block text-xs uppercase tracking-wider font-semibold text-gray-500 mb-1">
                    Email address
                  </label>
                  <input
                    id="email"
                    name="email"
                    type="email"
                    autoComplete="email"
                    required
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    className="block w-full px-4 py-3 input-glass rounded-lg text-sm"
                    placeholder="name@example.com"
                  />
                </div>
                <div>
                  <label htmlFor="fullName" className="block text-xs uppercase tracking-wider font-semibold text-gray-500 mb-1">
                    Full Name
                  </label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    autoComplete="name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="block w-full px-4 py-3 input-glass rounded-lg text-sm"
                    placeholder="John Doe (Optional)"
                  />
                </div>
              </>
            )}
            <div>
              <label htmlFor="password" className="block text-xs uppercase tracking-wider font-semibold text-gray-500 mb-1">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                minLength={isLogin ? undefined : 8}
                title={isLogin ? undefined : 'Use at least 8 characters.'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="block w-full px-4 py-3 input-glass rounded-lg text-sm"
                placeholder="••••••••"
                aria-describedby={!isLogin ? 'password-help' : undefined}
              />
            </div>
          </div>

          {!isLogin && (
            <div className="text-xs text-gray-500 bg-void-900/50 p-3 rounded-lg border border-white/5">
              <p id="username-help" className="mb-1">• Usernames: letters, numbers, underscores (3+ chars)</p>
              <p id="password-help">• Passwords: 8+ characters required</p>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-3 px-4 border border-transparent text-sm font-bold uppercase tracking-wide rounded-lg text-white bg-signal-600 hover:bg-signal-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-void-900 focus:ring-signal-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-signal-600/20"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Authenticating...
                </span>
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>
          </div>

          <div className="text-center pt-2">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-sm font-medium text-gray-400 hover:text-white transition-colors"
            >
              {isLogin
                ? "First time? Create access credentials"
                : 'Already verified? Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;
