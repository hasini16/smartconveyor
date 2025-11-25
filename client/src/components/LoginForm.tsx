// client/src/components/LoginForm.tsx
import React, { useState } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

const API_URL = 'http://127.0.0.1:8000'; 

interface LoginFormProps {
  role: 'admin' | 'employee';
  onLogin: (role: 'admin' | 'employee', username: string, userId: number) => void;
}

const LoginForm: React.FC<LoginFormProps> = ({ role, onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const endpoint = role === 'admin' ? '/auth/admin/login' : '/auth/employee/login';
      const response = await axios.post(`${API_URL}${endpoint}`, { username, password });
      
      const { username: loggedInUsername, user_id: userId } = response.data;
      onLogin(role, loggedInUsername, userId);
      
      navigate(role === 'admin' ? '/admin' : '/employee'); 

    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        setError(`Login failed: ${err.response.data.detail}`);
      } else {
        setError('An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-white p-8 rounded-lg shadow-xl w-full max-w-md">
      <h2 className={`text-2xl font-bold mb-6 text-center ${role === 'admin' ? 'text-red-600' : 'text-green-600'}`}>
        {role === 'admin' ? 'Admin Login' : 'Employee Login'}
      </h2>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-700">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
          />
        </div>
        {error && <p className="text-red-500 text-sm">{error}</p>}
        <button
          type="submit"
          disabled={loading}
          className={`w-full py-2 px-4 border border-transparent rounded-md text-sm font-medium text-white transition duration-150 ${
            loading 
              ? 'bg-gray-400 cursor-not-allowed' 
              : role === 'admin' 
                ? 'bg-red-600 hover:bg-red-700' 
                : 'bg-green-600 hover:bg-green-700'
          }`}
        >
          {loading ? 'Logging in...' : 'Login'}
        </button>
      </form>
    </div>
  );
};

export default LoginForm;