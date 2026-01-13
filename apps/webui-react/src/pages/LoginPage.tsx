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
    <div className="min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 bg-cover bg-center">
      <div className="max-w-md w-full space-y-8 glass-panel p-10 rounded-2xl animate-fade-in shadow-2xl shadow-brand-500/10">
        <div>
          <div className="mx-auto h-16 w-16 bg-gradient-to-br from-brand-500 to-accent-500 rounded-2xl shadow-lg flex items-center justify-center transform -rotate-6 transition-transform hover:rotate-0 duration-300">
            <span className="text-white font-bold text-3xl">S</span>
          </div>
          <h2 className="mt-8 text-center text-3xl font-bold heading-gradient">
            {isLogin ? 'Welcome Back' : 'Join Semantik'}
          </h2>
          <p className="mt-2 text-center text-sm text-gray-500">
            {isLogin ? 'Sign in to access your document pipeline' : 'Create your account to get started'}
          </p>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="space-y-4">
            <div>
              <label htmlFor="username" className="block text-sm font-semibold text-gray-700 mb-1">
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
                className="appearance-none block w-full px-4 py-3 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all sm:text-sm bg-white/50 focus:bg-white"
                placeholder="Enter your username"
                aria-describedby={!isLogin ? 'username-help' : undefined}
              />
            </div>
            {!isLogin && (
              <>
                <div>
                  <label htmlFor="email" className="block text-sm font-semibold text-gray-700 mb-1">
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
                    className="appearance-none block w-full px-4 py-3 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all sm:text-sm bg-white/50 focus:bg-white"
                    placeholder="Enter your email"
                  />
                </div>
                <div>
                  <label htmlFor="fullName" className="block text-sm font-semibold text-gray-700 mb-1">
                    Full Name
                  </label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    autoComplete="name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="appearance-none block w-full px-4 py-3 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all sm:text-sm bg-white/50 focus:bg-white"
                    placeholder="Enter your full name (optional)"
                  />
                </div>
              </>
            )}
            <div>
              <label htmlFor="password" className="block text-sm font-semibold text-gray-700 mb-1">
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
                className="appearance-none block w-full px-4 py-3 border border-gray-200 rounded-xl placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-brand-500/20 focus:border-brand-500 transition-all sm:text-sm bg-white/50 focus:bg-white"
                placeholder="Enter your password"
                aria-describedby={!isLogin ? 'password-help' : undefined}
              />
            </div>
          </div>

          {!isLogin && (
            <div className="text-xs text-gray-500 bg-gray-50 p-3 rounded-lg border border-gray-100">
              <p id="username-help" className="mb-1">• Usernames: letters, numbers, underscores (3+ chars)</p>
              <p id="password-help">• Passwords: 8+ characters required</p>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-3.5 px-4 border border-transparent text-sm font-bold rounded-xl text-white bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 hover:shadow-lg hover:shadow-brand-500/30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-brand-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 transform hover:-translate-y-0.5"
            >
              {loading ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
              ) : (
                isLogin ? 'Sign in' : 'Create Account'
              )}
            </button>
          </div>

          <div className="text-center pt-2">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="text-sm font-semibold text-brand-600 hover:text-brand-500 transition-colors"
            >
              {isLogin
                ? "Don't have an account? Create one"
                : 'Already have an account? Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;
