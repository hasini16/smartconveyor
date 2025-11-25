// client/src/pages/AdminPage.tsx
import React, { useState } from 'react';
import axios from 'axios';
import LoginForm from '../components/LoginForm';

const API_URL = 'http://127.0.0.1:8000';

type AuthRole = 'none' | 'admin' | 'employee';
type AuthState = { role: AuthRole, username: string, userId: number | null };

interface AdminPageProps {
  auth: AuthState;
  onLogin: (role: AuthRole, username: string, userId: number | null) => void;
  onLogout: () => void;
}

// --- Helper Component for Adding Admin/Employee ---
const AddUserForm: React.FC<{ userRole: 'admin' | 'employee' }> = ({ userRole }) => {
  const [email, setEmail] = useState('');
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
  const [newCreds, setNewCreds] = useState<{ username: string, password: string } | null>(null);
  const [loading, setLoading] = useState(false);

  const handleAddUser = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setMessage(null);
    setNewCreds(null);

    try {
      const endpoint = userRole === 'admin' ? '/admin/new_admin' : '/admin/new_employee';
      const response = await axios.post(`${API_URL}${endpoint}`, { email });
      
      setMessage({ type: 'success', text: response.data.message });
      setNewCreds({ username: response.data.username, password: response.data.password });
      setEmail('');
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        setMessage({ type: 'error', text: `Failed: ${err.response.data.detail}` });
      } else {
        setMessage({ type: 'error', text: 'An unexpected error occurred.' });
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 bg-gray-50 rounded-lg shadow-inner w-full max-w-xl mx-auto">
      <h3 className="text-xl font-semibold mb-4 text-gray-800">
        Add New {userRole.charAt(0).toUpperCase() + userRole.slice(1)}
      </h3>
      <form onSubmit={handleAddUser} className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700">Email (for credential generation)</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
          />
        </div>
        <button
          type="submit"
          disabled={loading || !email}
          className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 text-white rounded-md transition duration-150"
        >
          {loading ? 'Processing...' : `Generate & Add ${userRole.charAt(0).toUpperCase() + userRole.slice(1)}`}
        </button>
      </form>
      {message && (
        <div className={`mt-4 p-3 rounded-md ${message.type === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
          <p>{message.text}</p>
          {newCreds && (
            <div className="mt-2 text-sm">
              <p>Generated Username: **{newCreds.username}**</p>
              <p>One-Time Password: **{newCreds.password}**</p>
              <p className="text-xs italic text-red-500 mt-1">
                *The system generates a random password. The new user should copy this and change it upon first login.*
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// --- Helper Component for Editing Profile ---
const AdminEditProfile: React.FC<AdminPageProps> = ({ auth, onLogin }) => {
    const [newUsername, setNewUsername] = useState(auth.username);
    const [newPassword, setNewPassword] = useState('');
    const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null);
    const [loading, setLoading] = useState(false);

    const handleEditProfile = async (e: React.FormEvent) => {
      e.preventDefault();
      setLoading(true);
      setMessage(null);
      
      if (!auth.userId) {
          setMessage({ type: 'error', text: 'Error: User ID not available for update.' });
          setLoading(false);
          return;
      }
      
      try {
        const response = await axios.post(`${API_URL}/admin/edit_profile/${auth.userId}`, {
          new_username: newUsername,
          new_password: newPassword,
        });
        
        setMessage({ type: 'success', text: response.data.message });
        // Update local state, retaining the user ID
        onLogin('admin', response.data.new_username, auth.userId); 
        setNewPassword(''); // Clear password field
      } catch (err) {
        if (axios.isAxiosError(err) && err.response) {
          setMessage({ type: 'error', text: `Failed: ${err.response.data.detail}` });
        } else {
          setMessage({ type: 'error', text: 'An unexpected error occurred.' });
        }
      } finally {
        setLoading(false);
      }
    };
    
    return (
      <div className="p-6 bg-gray-50 rounded-lg shadow-inner w-full max-w-lg mx-auto">
        <h3 className="text-xl font-semibold mb-4 text-red-800">Edit Your Profile</h3>
        <p className="text-sm text-gray-600 mb-4">Current User: **{auth.username}**</p>
        <form onSubmit={handleEditProfile} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">New Username</label>
            <input
              type="text"
              value={newUsername}
              onChange={(e) => setNewUsername(e.target.value)}
              required
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">New Password</label>
            <input
              type="password"
              value={newPassword}
              onChange={(e) => setNewPassword(e.target.value)}
              required
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
            />
          </div>
          <button
            type="submit"
            disabled={loading || !newUsername || !newPassword}
            className="w-full py-2 px-4 bg-red-600 hover:bg-red-700 text-white rounded-md transition duration-150"
          >
            {loading ? 'Updating...' : 'Update Profile'}
          </button>
        </form>
        {message && (
          <div className={`mt-4 p-3 rounded-md ${message.type === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
            <p>{message.text}</p>
          </div>
        )}
      </div>
    );
};

// --- Main AdminPage Component ---
const AdminPage: React.FC<AdminPageProps> = ({ auth, onLogin }) => {
  const [currentPage, setCurrentPage] = useState<'profile' | 'add-admin' | 'add-employee'>('profile');

  if (auth.role !== 'admin') {
    return (
      <div className="flex justify-center items-center h-full pt-10">
        <LoginForm role="admin" onLogin={onLogin} />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6 bg-white rounded-xl shadow-2xl">
      <h1 className="text-3xl font-extrabold text-red-800 mb-6 border-b pb-2">
        👑 Admin Dashboard
      </h1>
      <div className="flex space-x-4 mb-6">
        <button onClick={() => setCurrentPage('profile')} className={`py-2 px-4 rounded ${currentPage === 'profile' ? 'bg-red-600 text-white' : 'bg-red-100 text-red-700 hover:bg-red-200'}`}>
          Edit Profile
        </button>
        <button onClick={() => setCurrentPage('add-admin')} className={`py-2 px-4 rounded ${currentPage === 'add-admin' ? 'bg-red-600 text-white' : 'bg-red-100 text-red-700 hover:bg-red-200'}`}>
          Add New Admin
        </button>
        <button onClick={() => setCurrentPage('add-employee')} className={`py-2 px-4 rounded ${currentPage === 'add-employee' ? 'bg-red-600 text-white' : 'bg-red-100 text-red-700 hover:bg-red-200'}`}>
          Add New Employee
        </button>
      </div>
      
      <div className="min-h-[400px]">
        {currentPage === 'profile' && <AdminEditProfile auth={auth} onLogin={onLogin} onLogout={() => {}} />}
        {currentPage === 'add-admin' && <AddUserForm userRole="admin" />}
        {currentPage === 'add-employee' && <AddUserForm userRole="employee" />}
      </div>
    </div>
  );
};

export default AdminPage;