// client/src/components/NavBar.tsx
import React from 'react';
import { Link, useNavigate } from 'react-router-dom';

type AuthRole = 'none' | 'admin' | 'employee';
type AuthState = { role: AuthRole, username: string };

interface NavBarProps {
  auth: AuthState;
  onLogout: () => void;
}

const NavBar: React.FC<NavBarProps> = ({ auth, onLogout }) => {
  const navigate = useNavigate();

  const handleLogout = () => {
    onLogout();
    navigate('/');
  };

  const navItems = [
    { name: 'Home (User)', path: '/' },
    { name: 'About', path: '/about' },
  ];

  if (auth.role === 'admin') {
    navItems.push({ name: 'Admin Dashboard', path: '/admin' });
  } else if (auth.role === 'employee') {
    navItems.push({ name: 'Employee Panel', path: '/employee' });
  } else {
    // Show login options when logged out
    navItems.push({ name: 'Admin Login', path: '/admin-login' });
    navItems.push({ name: 'Employee Login', path: '/employee-login' });
  }

  return (
    <header className="fixed top-0 left-0 w-full bg-white shadow-lg z-10">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <Link to="/" className="text-xl font-bold text-indigo-600">
            SmartConveyor
          </Link>
          <div className="flex space-x-4 items-center">
            {navItems.map((item) => (
              <Link
                key={item.name}
                to={item.path}
                className="text-gray-600 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium transition duration-150"
              >
                {item.name}
              </Link>
            ))}
            
            {auth.role !== 'none' && (
              <div className="text-sm text-gray-700">
                Hi, **{auth.username}**!
              </div>
            )}

            {auth.role !== 'none' && (
              <button
                onClick={handleLogout}
                className="ml-4 bg-red-500 hover:bg-red-600 text-white px-3 py-1 rounded-md text-sm font-medium transition duration-150"
              >
                Logout
              </button>
            )}
          </div>
        </div>
      </nav>
    </header>
  );
};

export default NavBar;