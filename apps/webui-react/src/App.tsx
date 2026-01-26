import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './stores/authStore';
import ErrorBoundary from './components/ErrorBoundary';
import Layout from './components/Layout';
import LoginPage from './pages/LoginPage';
import HomePage from './pages/HomePage';
import SettingsPage from './pages/SettingsPage';
import VerificationPage from './pages/VerificationPage';
import AgentChatPage from './pages/AgentChatPage';
import PipelineBuilderPage from './pages/PipelineBuilderPage';
import { queryClient } from './services/queryClient';
import { AnimationProvider } from './contexts/AnimationContext';

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const token = useAuthStore((state) => state.token);
  
  if (!token) {
    return <Navigate to="/login" replace />;
  }
  
  return <>{children}</>;
}

function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <AnimationProvider>
          <Router>
            <Routes>
              <Route path="/login" element={<LoginPage />} />
              <Route path="/verification" element={<VerificationPage />} />
              <Route
                path="/"
                element={
                  <ProtectedRoute>
                    <Layout />
                  </ProtectedRoute>
                }
              >
                <Route index element={<HomePage />} />
                <Route path="collections/:collectionId" element={<HomePage />} />
                <Route path="agent/:conversationId" element={<AgentChatPage />} />
                <Route path="pipeline/:conversationId" element={<PipelineBuilderPage />} />
                <Route path="settings" element={<SettingsPage />} />
              </Route>
            </Routes>
          </Router>
        </AnimationProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}

export default App;
