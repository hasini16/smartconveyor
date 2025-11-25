// client/src/App.tsx
import { useState } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import NavBar from './components/NavBar';
import About from './pages/About';
import UserPage from './pages/UserPage';
import AdminPage from './pages/AdminPage';
import EmployeePage from './pages/EmployeePage';
import './App.css';

// --- Global State Types ---
type AuthRole = 'none' | 'admin' | 'employee';

type AuthState = {
  role: AuthRole;
  username: string;
  userId: number | null; // Used for profile updates and API calls
}

function App() {
  // Initialize state with null user ID
  const [auth, setAuth] = useState<AuthState>({ role: 'none', username: '', userId: null });

  const handleLogin = (role: AuthRole, username: string, userId: number | null) => {
    setAuth({ role, username, userId });
  };

  const handleLogout = () => {
    setAuth({ role: 'none', username: '', userId: null });
  };
  
  // A simple guard component for protected routes
  const ProtectedRoute = ({ children, allowedRoles }: { children: JSX.Element, allowedRoles: AuthRole[] }) => {
    if (!allowedRoles.includes(auth.role)) {
      // Redirect to home/public page if unauthorized
      return <Navigate to="/" replace />;
    }
    return children;
  };

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <NavBar auth={auth} onLogout={handleLogout} />
        <main className="pt-20 p-4"> {/* Padding for fixed navbar */}
          <Routes>
            <Route path="/" element={<UserPage />} /> {/* Default/Home */}
            <Route path="/about" element={<About />} />
            
            {/* Login Routes (Show login form if not authenticated, otherwise redirect to dashboard) */}
            <Route 
              path="/admin-login" 
              element={auth.role === 'admin' ? <Navigate to="/admin" replace /> : <AdminPage auth={auth} onLogin={handleLogin} onLogout={handleLogout} />} 
            />
            <Route 
              path="/employee-login" 
              element={auth.role === 'employee' ? <Navigate to="/employee" replace /> : <EmployeePage auth={auth} onLogin={handleLogin} onLogout={handleLogout} />} 
            />

            {/* Protected Routes (Dashboards) */}
            <Route 
              path="/admin" 
              element={
                <ProtectedRoute allowedRoles={['admin']}>
                  <AdminPage auth={auth} onLogin={handleLogin} onLogout={handleLogout} />
                </ProtectedRoute>
              } 
            />
            
            <Route 
              path="/employee" 
              element={
                <ProtectedRoute allowedRoles={['employee']}>
                  <EmployeePage auth={auth} onLogin={handleLogin} onLogout={handleLogout} />
                </ProtectedRoute>
              } 
            />
            
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;