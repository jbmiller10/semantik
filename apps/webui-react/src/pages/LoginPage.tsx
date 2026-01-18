import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';
import { authApi } from '../services/api/v2';
import { getErrorMessage } from '../utils/errorUtils';
import ThemeToggle from '../components/ThemeToggle';

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
            message: 'Signed in successfully',
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
            message: 'Account created. Please sign in.',
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
      {/* Theme toggle in corner */}
      <div className="fixed top-4 right-4">
        <ThemeToggle />
      </div>

      <div className="max-w-md w-full space-y-8 panel p-8 sm:p-10 animate-fade-in">
        <div>
          <div className="flex flex-col items-center">
            {/* Logo */}
            <div className="h-12 w-12 bg-ink-900 dark:bg-paper-100 rounded-xl flex items-center justify-center mb-4">
              <span className="text-paper-100 dark:text-ink-900 font-serif font-bold text-2xl">S</span>
            </div>
            <h1 className="text-3xl font-serif font-semibold text-[var(--text-primary)] tracking-tight">Semantik</h1>
            <div className="h-0.5 w-10 bg-ink-900 dark:bg-paper-100 rounded-full mt-3"></div>
          </div>
          <h2 className="mt-8 text-center text-lg font-medium text-[var(--text-primary)]">
            {isLogin ? 'Welcome back' : 'Create an account'}
          </h2>
          <p className="mt-2 text-center text-sm text-[var(--text-muted)]">
            {isLogin ? 'Sign in to continue' : 'Get started with Semantik'}
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5">
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
                className="input-field w-full"
                placeholder="username"
                aria-describedby={!isLogin ? 'username-help' : undefined}
              />
            </div>
            {!isLogin && (
              <>
                <div>
                  <label htmlFor="email" className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5">
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
                    className="input-field w-full"
                    placeholder="name@example.com"
                  />
                </div>
                <div>
                  <label htmlFor="fullName" className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5">
                    Full Name <span className="text-[var(--text-muted)]">(optional)</span>
                  </label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    autoComplete="name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="input-field w-full"
                    placeholder="John Doe"
                  />
                </div>
              </>
            )}
            <div>
              <label htmlFor="password" className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5">
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
                className="input-field w-full"
                placeholder="Enter your password"
                aria-describedby={!isLogin ? 'password-help' : undefined}
              />
            </div>
          </div>

          {!isLogin && (
            <div className="text-xs text-[var(--text-muted)] bg-[var(--bg-tertiary)] p-3 rounded-lg border border-[var(--border-subtle)]">
              <p id="username-help" className="mb-1">Username: letters, numbers, underscores (3+ characters)</p>
              <p id="password-help">Password: 8+ characters required</p>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3 text-sm font-semibold"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {isLogin ? 'Signing in...' : 'Creating account...'}
                </span>
              ) : (
                isLogin ? 'Sign in' : 'Create account'
              )}
            </button>
          </div>

          <div className="text-center pt-2">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors"
            >
              {isLogin
                ? "Don't have an account? Sign up"
                : 'Already have an account? Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;
