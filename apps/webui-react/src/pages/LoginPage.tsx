import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { useUIStore } from '../stores/uiStore';
import { authApi } from '../services/api/v2';
import { getErrorMessage } from '../utils/errorUtils';
import ThemeToggle from '../components/ThemeToggle';
import { Eye, EyeOff, Check, X } from 'lucide-react';

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

  // New state for UX improvements
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [touched, setTouched] = useState({
    username: false,
    email: false,
    password: false,
    confirmPassword: false,
  });
  const [validation, setValidation] = useState({
    username: { valid: null as boolean | null, message: '' },
    email: { valid: null as boolean | null, message: '' },
    password: { valid: null as boolean | null, message: '' },
    confirmPassword: { valid: null as boolean | null, message: '' },
  });

  // Validation functions
  const validateUsername = (value: string) => {
    if (!value) {
      setValidation((prev) => ({ ...prev, username: { valid: null, message: '' } }));
      return;
    }
    const isValid = value.length >= 3 && /^[A-Za-z0-9_]+$/.test(value);
    setValidation((prev) => ({
      ...prev,
      username: {
        valid: isValid,
        message: isValid
          ? ''
          : value.length < 3
            ? 'At least 3 characters'
            : 'Letters, numbers, underscores only',
      },
    }));
  };

  const validateEmail = (value: string) => {
    if (!value) {
      setValidation((prev) => ({ ...prev, email: { valid: null, message: '' } }));
      return;
    }
    const isValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
    setValidation((prev) => ({
      ...prev,
      email: { valid: isValid, message: isValid ? '' : 'Enter a valid email' },
    }));
  };

  const validatePassword = (value: string) => {
    if (!value) {
      setValidation((prev) => ({ ...prev, password: { valid: null, message: '' } }));
      return;
    }
    const isValid = value.length >= 8;
    setValidation((prev) => ({
      ...prev,
      password: { valid: isValid, message: isValid ? '' : 'At least 8 characters' },
    }));
    if (confirmPassword) validateConfirmPassword(confirmPassword, value);
  };

  const validateConfirmPassword = (value: string, pwd = password) => {
    if (!value) {
      setValidation((prev) => ({ ...prev, confirmPassword: { valid: null, message: '' } }));
      return;
    }
    const isValid = value === pwd;
    setValidation((prev) => ({
      ...prev,
      confirmPassword: { valid: isValid, message: isValid ? '' : 'Passwords do not match' },
    }));
  };

  const resetValidationState = () => {
    setConfirmPassword('');
    setValidation({
      username: { valid: null, message: '' },
      email: { valid: null, message: '' },
      password: { valid: null, message: '' },
      confirmPassword: { valid: null, message: '' },
    });
    setTouched({ username: false, email: false, password: false, confirmPassword: false });
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // Pre-submit validation for registration
    if (!isLogin && password !== confirmPassword) {
      addToast({ type: 'error', message: 'Passwords do not match' });
      return;
    }

    setLoading(true);

    try {
      if (isLogin) {
        // Login flow
        const response = await authApi.login({ username, password });

        if (response.data.access_token) {
          // First set the auth token so subsequent API calls work
          // We'll use a temporary user object and update it after
          setAuth(
            response.data.access_token,
            {
              id: 0,
              username: username,
              email: '',
              is_active: true,
              is_superuser: false,
              created_at: new Date().toISOString(),
            },
            response.data.refresh_token
          );

          // Now get the actual user info
          try {
            const userResponse = await authApi.me();
            // Update with real user data
            setAuth(response.data.access_token, userResponse.data, response.data.refresh_token);
          } catch (error) {
            console.error('Failed to fetch user details:', error);
            // Continue with login even if user details fetch fails
          }

          // Handle remember me
          if (!rememberMe) {
            localStorage.removeItem('auth-storage');
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
          full_name: fullName || undefined,
        });

        if (response.data.id) {
          // Auto-login after registration
          try {
            const loginResponse = await authApi.login({ username, password });

            if (loginResponse.data.access_token) {
              setAuth(
                loginResponse.data.access_token,
                response.data,
                loginResponse.data.refresh_token
              );

              // Handle remember me
              if (!rememberMe) {
                localStorage.removeItem('auth-storage');
              }

              addToast({ type: 'success', message: 'Account created and signed in!' });
              navigate('/');
            }
          } catch (loginError) {
            // Fall back to manual login if auto-login fails
            console.error('Auto-login failed:', loginError);
            addToast({ type: 'success', message: 'Account created. Please sign in.' });
            setIsLogin(true);
            setPassword('');
            setConfirmPassword('');
          }
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

  // Helper to get input validation class
  const getValidationClass = (field: keyof typeof validation) => {
    if (isLogin || !touched[field] || validation[field].valid === null) return '';
    return validation[field].valid
      ? 'border-green-500 focus:ring-green-500/50'
      : 'border-red-500 focus:ring-red-500/50';
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
              <span className="text-paper-100 dark:text-ink-900 font-serif font-bold text-2xl">
                S
              </span>
            </div>
            <h1 className="text-3xl font-serif font-semibold text-[var(--text-primary)] tracking-tight">
              Semantik
            </h1>
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
            {/* Username field */}
            <div>
              <label
                htmlFor="username"
                className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5"
              >
                Username
              </label>
              <div className="relative">
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
                  onChange={(e) => {
                    setUsername(e.target.value);
                    if (!isLogin) validateUsername(e.target.value);
                  }}
                  onBlur={() => !isLogin && setTouched((prev) => ({ ...prev, username: true }))}
                  disabled={loading}
                  className={`input-field w-full pr-10 ${loading ? 'opacity-60 cursor-not-allowed' : ''} ${getValidationClass('username')}`}
                  placeholder="username"
                  aria-describedby={!isLogin ? 'username-help' : undefined}
                />
                {!isLogin && touched.username && validation.username.valid !== null && (
                  <span
                    className={`absolute inset-y-0 right-3 flex items-center ${validation.username.valid ? 'text-green-500' : 'text-red-500'}`}
                  >
                    {validation.username.valid ? (
                      <Check className="h-4 w-4" />
                    ) : (
                      <X className="h-4 w-4" />
                    )}
                  </span>
                )}
              </div>
              {!isLogin && (
                <>
                  <p id="username-help" className="mt-1 text-xs text-[var(--text-muted)]">
                    Letters, numbers, and underscores (3+ characters)
                  </p>
                  {touched.username && validation.username.valid === false && (
                    <p className="mt-1 text-xs text-red-500">{validation.username.message}</p>
                  )}
                </>
              )}
            </div>

            {/* Email and Full Name - registration only */}
            {!isLogin && (
              <>
                <div>
                  <label
                    htmlFor="email"
                    className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5"
                  >
                    Email address
                  </label>
                  <div className="relative">
                    <input
                      id="email"
                      name="email"
                      type="email"
                      autoComplete="email"
                      required
                      value={email}
                      onChange={(e) => {
                        setEmail(e.target.value);
                        validateEmail(e.target.value);
                      }}
                      onBlur={() => setTouched((prev) => ({ ...prev, email: true }))}
                      disabled={loading}
                      className={`input-field w-full pr-10 ${loading ? 'opacity-60 cursor-not-allowed' : ''} ${getValidationClass('email')}`}
                      placeholder="name@example.com"
                    />
                    {touched.email && validation.email.valid !== null && (
                      <span
                        className={`absolute inset-y-0 right-3 flex items-center ${validation.email.valid ? 'text-green-500' : 'text-red-500'}`}
                      >
                        {validation.email.valid ? (
                          <Check className="h-4 w-4" />
                        ) : (
                          <X className="h-4 w-4" />
                        )}
                      </span>
                    )}
                  </div>
                  {touched.email && validation.email.valid === false && (
                    <p className="mt-1 text-xs text-red-500">{validation.email.message}</p>
                  )}
                </div>
                <div>
                  <label
                    htmlFor="fullName"
                    className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5"
                  >
                    Full Name <span className="text-[var(--text-muted)]">(optional)</span>
                  </label>
                  <input
                    id="fullName"
                    name="fullName"
                    type="text"
                    autoComplete="name"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    disabled={loading}
                    className={`input-field w-full ${loading ? 'opacity-60 cursor-not-allowed' : ''}`}
                    placeholder="John Doe"
                  />
                </div>
              </>
            )}

            {/* Password field */}
            <div>
              <label
                htmlFor="password"
                className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5"
              >
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  autoComplete={isLogin ? 'current-password' : 'new-password'}
                  required
                  minLength={isLogin ? undefined : 8}
                  title={isLogin ? undefined : 'Use at least 8 characters.'}
                  value={password}
                  onChange={(e) => {
                    setPassword(e.target.value);
                    if (!isLogin) validatePassword(e.target.value);
                  }}
                  onBlur={() => !isLogin && setTouched((prev) => ({ ...prev, password: true }))}
                  disabled={loading}
                  className={`input-field w-full pr-20 ${loading ? 'opacity-60 cursor-not-allowed' : ''} ${getValidationClass('password')}`}
                  placeholder="Enter your password"
                  aria-describedby={!isLogin ? 'password-help' : undefined}
                />
                <div className="absolute inset-y-0 right-0 flex items-center pr-3 gap-1">
                  {!isLogin && touched.password && validation.password.valid !== null && (
                    <span
                      className={validation.password.valid ? 'text-green-500' : 'text-red-500'}
                    >
                      {validation.password.valid ? (
                        <Check className="h-4 w-4" />
                      ) : (
                        <X className="h-4 w-4" />
                      )}
                    </span>
                  )}
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    disabled={loading}
                    className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] disabled:opacity-50"
                    aria-label={showPassword ? 'Hide password' : 'Show password'}
                  >
                    {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </button>
                </div>
              </div>
              {!isLogin && (
                <>
                  <p id="password-help" className="mt-1 text-xs text-[var(--text-muted)]">
                    Must be at least 8 characters
                  </p>
                  {touched.password && validation.password.valid === false && (
                    <p className="mt-1 text-xs text-red-500">{validation.password.message}</p>
                  )}
                </>
              )}
            </div>

            {/* Confirm Password - registration only */}
            {!isLogin && (
              <div>
                <label
                  htmlFor="confirmPassword"
                  className="block text-sm font-medium text-[var(--text-secondary)] mb-1.5"
                >
                  Confirm Password
                </label>
                <div className="relative">
                  <input
                    id="confirmPassword"
                    name="confirmPassword"
                    type={showConfirmPassword ? 'text' : 'password'}
                    autoComplete="new-password"
                    required
                    value={confirmPassword}
                    onChange={(e) => {
                      setConfirmPassword(e.target.value);
                      validateConfirmPassword(e.target.value);
                    }}
                    onBlur={() => setTouched((prev) => ({ ...prev, confirmPassword: true }))}
                    disabled={loading}
                    className={`input-field w-full pr-20 ${loading ? 'opacity-60 cursor-not-allowed' : ''} ${getValidationClass('confirmPassword')}`}
                    placeholder="Confirm your password"
                  />
                  <div className="absolute inset-y-0 right-0 flex items-center pr-3 gap-1">
                    {touched.confirmPassword && validation.confirmPassword.valid !== null && (
                      <span
                        className={
                          validation.confirmPassword.valid ? 'text-green-500' : 'text-red-500'
                        }
                      >
                        {validation.confirmPassword.valid ? (
                          <Check className="h-4 w-4" />
                        ) : (
                          <X className="h-4 w-4" />
                        )}
                      </span>
                    )}
                    <button
                      type="button"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                      disabled={loading}
                      className="text-[var(--text-muted)] hover:text-[var(--text-secondary)] disabled:opacity-50"
                      aria-label={showConfirmPassword ? 'Hide password' : 'Show password'}
                    >
                      {showConfirmPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </button>
                  </div>
                </div>
                {touched.confirmPassword && validation.confirmPassword.valid === false && (
                  <p className="mt-1 text-xs text-red-500">{validation.confirmPassword.message}</p>
                )}
              </div>
            )}
          </div>

          {/* Remember me checkbox */}
          <div className="flex items-center">
            <input
              id="remember-me"
              name="remember-me"
              type="checkbox"
              checked={rememberMe}
              onChange={(e) => setRememberMe(e.target.checked)}
              disabled={loading}
              className="h-4 w-4 rounded border-[var(--border)] bg-[var(--bg-secondary)] text-gray-600 focus:ring-gray-500 disabled:opacity-50"
            />
            <label htmlFor="remember-me" className="ml-2 text-sm text-[var(--text-secondary)]">
              Remember me
            </label>
          </div>

          <div>
            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full py-3 text-sm font-semibold"
            >
              {loading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle
                      className="opacity-25"
                      cx="12"
                      cy="12"
                      r="10"
                      stroke="currentColor"
                      strokeWidth="4"
                    ></circle>
                    <path
                      className="opacity-75"
                      fill="currentColor"
                      d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                    ></path>
                  </svg>
                  {isLogin ? 'Signing in...' : 'Creating account...'}
                </span>
              ) : isLogin ? (
                'Sign in'
              ) : (
                'Create account'
              )}
            </button>
          </div>

          <div className="text-center pt-2">
            <button
              type="button"
              onClick={() => {
                setIsLogin(!isLogin);
                if (!isLogin) {
                  // Switching to login mode - clear registration fields
                  resetValidationState();
                }
              }}
              disabled={loading}
              className={`text-sm font-medium text-[var(--text-secondary)] hover:text-[var(--text-primary)] transition-colors ${loading ? 'opacity-60 cursor-not-allowed' : ''}`}
            >
              {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

export default LoginPage;
