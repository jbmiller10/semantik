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
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            {isLogin ? 'Sign in to Semantik' : 'Create a Semantik account'}
          </h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <div className="rounded-md shadow-sm -space-y-px">
            <div>
              <label htmlFor="username" className="sr-only">
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
                className={`appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 ${
                  isLogin ? 'rounded-t-md' : ''
                } focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm`}
                placeholder="Username"
                aria-describedby={!isLogin ? 'username-help' : undefined}
              />
            </div>
            {!isLogin && (
              <>
                <div>
                  <label htmlFor="email" className="sr-only">
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
                    className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                    placeholder="Email address"
                  />
                </div>
                <div>
                  <label htmlFor="fullName" className="sr-only">
                    Full Name
                  </label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    autoComplete="name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    className="appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm"
                    placeholder="Full Name (optional)"
                  />
                </div>
              </>
            )}
            <div>
              <label htmlFor="password" className="sr-only">
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
                className={`appearance-none rounded-none relative block w-full px-3 py-2 border border-gray-300 placeholder-gray-500 text-gray-900 ${
                  isLogin ? 'rounded-b-md' : 'rounded-b-md'
                } focus:outline-none focus:ring-blue-500 focus:border-blue-500 focus:z-10 sm:text-sm`}
                placeholder="Password"
                aria-describedby={!isLogin ? 'password-help' : undefined}
              />
            </div>
          </div>

          {!isLogin && (
            <div className="text-xs text-gray-500 space-y-1">
              <p id="username-help">Usernames can include letters, numbers, or underscores and must be at least 3 characters.</p>
              <p id="password-help">Passwords must be at least 8 characters long.</p>
            </div>
          )}

          <div>
            <button
              type="submit"
              disabled={loading}
              className="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Processing...' : isLogin ? 'Sign in' : 'Register'}
            </button>
          </div>

          <div className="text-center">
            <button
              type="button"
              onClick={() => setIsLogin(!isLogin)}
              className="font-medium text-blue-600 hover:text-blue-500"
            >
              {isLogin
                ? "Don't have an account? Register"
                : 'Already have an account? Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;
