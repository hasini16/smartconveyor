// client/src/pages/EmployeePage.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import LoginForm from '../components/LoginForm';

const API_URL = 'http://127.0.0.1:8000';

type AuthRole = 'none' | 'admin' | 'employee';
type AuthState = { role: AuthRole, username: string, userId: number | null };
type RouteDirection = 'straight' | 'left' | 'right';
type ParcelType = 'normal' | 'fast';

interface EmployeePageProps {
  auth: AuthState;
  onLogin: (role: AuthRole, username: string, userId: number | null) => void;
  onLogout: () => void;
}

// --- Data Structures ---
interface City {
  id: number;
  unique_id: number;
  city_name: string;
}

interface DataPoint {
  id: number;
  source_city_id: number;
  destination_city_id: number;
  parcel_type_text: ParcelType;
  route_direction_text: RouteDirection;
  source_city_name: string;
  destination_city_name: string;
}

// --- Helper Component for City Management ---
const CityManagement: React.FC = () => {
  const [cities, setCities] = useState<City[]>([]);
  const [newCityName, setNewCityName] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const fetchCities = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get<City[]>(`${API_URL}/employee/cities`);
      setCities(response.data);
    } catch (err) {
      setError('Failed to fetch cities.');
    } finally {
      setLoading(false);
    }
  };

  const handleAddCity = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage(null);
    if (!newCityName.trim()) return;

    try {
      const response = await axios.post<City>(`${API_URL}/employee/add_city`, { city_name: newCityName });
      setMessage(`City **${response.data.city_name}** added with ID **${response.data.unique_id}**.`);
      setNewCityName('');
      fetchCities(); // Refresh the list
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        setError(`Failed to add city: ${err.response.data.detail}`);
      } else {
        setError('An unexpected error occurred.');
      }
    }
  };

  useEffect(() => {
    fetchCities();
  }, []);

  if (loading) return <p className="text-gray-500">Loading cities...</p>;
  if (error) return <p className="text-red-500">Error: {error}</p>;

  return (
    <div className="space-y-6">
      <h3 className="text-2xl font-bold text-green-700">City Management</h3>
      
      {/* Add New City Form */}
      <form onSubmit={handleAddCity} className="p-4 bg-white rounded-lg shadow space-y-4 max-w-lg">
        <h4 className="text-lg font-semibold">Add New City</h4>
        <div className="flex space-x-2">
          <input
            type="text"
            value={newCityName}
            onChange={(e) => setNewCityName(e.target.value)}
            placeholder="Enter new city name"
            required
            className="flex-grow border border-gray-300 rounded-md p-2"
          />
          <button type="submit" className="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-md transition duration-150">
            Add City
          </button>
        </div>
        {message && <p className="text-green-600 text-sm">{message}</p>}
      </form>

      {/* Existing Cities Table */}
      <div className="bg-white p-4 rounded-lg shadow overflow-x-auto">
        <h4 className="text-lg font-semibold mb-2">Existing Cities ({cities.length})</h4>
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Unique ID</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">City Name</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {cities.map((city) => (
              <tr key={city.id}>
                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{city.unique_id}</td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-700">{city.city_name}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

// --- Helper Component for Data Point Management ---
const DataPointManagement: React.FC = () => {
    const [dataPoints, setDataPoints] = useState<DataPoint[]>([]);
    const [cities, setCities] = useState<City[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const [newDP, setNewDP] = useState<{
        source_city_id: string | number;
        destination_city_id: string | number;
        parcel_type_text: ParcelType | '';
        route_direction_text: RouteDirection | '';
    }>({
        source_city_id: '',
        destination_city_id: '',
        parcel_type_text: '',
        route_direction_text: '',
    });
    
    const [isEditing, setIsEditing] = useState<number | null>(null);
    const [editedRoute, setEditedRoute] = useState<RouteDirection | ''>('');
    const [retrainMessage, setRetrainMessage] = useState<string | null>(null);
    const [retrainLoading, setRetrainLoading] = useState(false);
    
    const fetchCitiesAndDataPoints = async () => {
        setLoading(true);
        setError(null);
        try {
            const [dpResponse, cityResponse] = await Promise.all([
                axios.get<DataPoint[]>(`${API_URL}/employee/datapoints`),
                axios.get<City[]>(`${API_URL}/employee/cities`),
            ]);
            setCities(cityResponse.data);
            setDataPoints(dpResponse.data);
        } catch (err) {
            setError('Failed to fetch data points or cities.');
        } finally {
            setLoading(false);
        }
    };

    const handleNewDPChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
        const { name, value } = e.target;
        setNewDP(prev => ({
            ...prev,
            [name]: value,
        }));
    };
    
    const handleCityNameChange = (e: React.ChangeEvent<HTMLSelectElement>, type: 'source' | 'destination') => {
        const cityName = e.target.value;
        const city = cities.find(c => c.city_name === cityName);
        setNewDP(prev => ({
            ...prev,
            [`${type}_city_id`]: city ? city.unique_id : '',
        }));
    };

    const handleAddDataPoint = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        if (!newDP.source_city_id || !newDP.destination_city_id || !newDP.parcel_type_text || !newDP.route_direction_text) {
            setError('All fields must be selected.');
            return;
        }

        try {
            await axios.post(`${API_URL}/employee/add_datapoint`, {
                source_city_id: Number(newDP.source_city_id),
                destination_city_id: Number(newDP.destination_city_id),
                parcel_type_text: newDP.parcel_type_text,
                route_direction_text: newDP.route_direction_text,
            });
            alert('Data Point added successfully!');
            setNewDP({ source_city_id: '', destination_city_id: '', parcel_type_text: '', route_direction_text: '' });
            fetchCitiesAndDataPoints();
        } catch (err) {
            if (axios.isAxiosError(err) && err.response) {
                setError(`Failed to add data point: ${err.response.data.detail}`);
            } else {
                setError('An unexpected error occurred.');
            }
        }
    };

    const handleEditClick = (dp: DataPoint) => {
        setIsEditing(dp.id);
        setEditedRoute(dp.route_direction_text);
    };

    const handleUpdateRoute = async (datapointId: number) => {
        setError(null);
        if (!editedRoute) return;

        const dpToUpdate = dataPoints.find(dp => dp.id === datapointId);
        if (!dpToUpdate) return;

        try {
            await axios.put(`${API_URL}/employee/update_datapoint/${datapointId}`, {
                source_city_id: dpToUpdate.source_city_id, // Pass minimal required fields for the backend schema
                destination_city_id: dpToUpdate.destination_city_id,
                parcel_type_text: dpToUpdate.parcel_type_text,
                route_direction_text: editedRoute,
            });
            alert('Route updated successfully!');
            setIsEditing(null);
            fetchCitiesAndDataPoints();
        } catch (err) {
            if (axios.isAxiosError(err) && err.response) {
                setError(`Failed to update route: ${err.response.data.detail}`);
            } else {
                setError('An unexpected error occurred.');
            }
        }
    };
    
    const handleRetrain = async () => {
        setRetrainLoading(true);
        setRetrainMessage(null);
        setError(null);
        try {
            const response = await axios.post(`${API_URL}/employee/retrain_model`);
            setRetrainMessage(`Retraining successful! New accuracy: ${response.data.test_accuracy}. Trained on ${response.data.new_data_size} data points.`);
        } catch (err) {
            if (axios.isAxiosError(err) && err.response) {
                setRetrainMessage(`Retraining failed: ${err.response.data.detail}`);
            } else {
                setRetrainMessage('An unexpected error occurred during retraining.');
            }
        } finally {
            setRetrainLoading(false);
        }
    };

    useEffect(() => {
        fetchCitiesAndDataPoints();
    }, []);

    const cityIdMap = cities.reduce((acc, city) => {
        acc[city.unique_id] = city.city_name;
        return acc;
    }, {} as { [key: number]: string });
    
    const uniqueCityNames = cities.map(c => c.city_name).sort();

    if (loading) return <p className="text-gray-500">Loading data...</p>;
    
    return (
        <div className="space-y-8">
            <h3 className="text-2xl font-bold text-green-700">Data Point and Model Management</h3>

            {/* Add New Data Point Feature */}
            <div className="p-6 bg-white rounded-lg shadow-xl">
                <h4 className="text-xl font-semibold mb-4 border-b pb-2">Add New Data Point</h4>
                <form onSubmit={handleAddDataPoint} className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    
                    {/* Source City */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Source City Name (Search & Choose)</label>
                        <select 
                            onChange={(e) => handleCityNameChange(e, 'source')}
                            value={cityIdMap[Number(newDP.source_city_id)] || ''}
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                        >
                            <option value="" disabled>Select Source City</option>
                            {uniqueCityNames.map(name => (
                                <option key={`src-${name}`} value={name}>{name}</option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Source ID (Autofilled): **{newDP.source_city_id || 'N/A'}**</p>
                    </div>

                    {/* Destination City */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Destination City Name (Search & Choose)</label>
                        <select 
                            onChange={(e) => handleCityNameChange(e, 'destination')}
                            value={cityIdMap[Number(newDP.destination_city_id)] || ''}
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                        >
                            <option value="" disabled>Select Destination City</option>
                            {uniqueCityNames.map(name => (
                                <option key={`dest-${name}`} value={name}>{name}</option>
                            ))}
                        </select>
                        <p className="text-xs text-gray-500 mt-1">Destination ID (Autofilled): **{newDP.destination_city_id || 'N/A'}**</p>
                    </div>

                    {/* Parcel Type */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Parcel Type (Normal=0, Fast=1)</label>
                        <select
                            name="parcel_type_text"
                            value={newDP.parcel_type_text}
                            onChange={handleNewDPChange}
                            required
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                        >
                            <option value="" disabled>Select type</option>
                            <option value="normal">Normal (0)</option>
                            <option value="fast">Fast (1)</option>
                        </select>
                    </div>

                    {/* Route Direction */}
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Route Direction (Straight=0, Left=1, Right=2)</label>
                        <select
                            name="route_direction_text"
                            value={newDP.route_direction_text}
                            onChange={handleNewDPChange}
                            required
                            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm p-2"
                        >
                            <option value="" disabled>Select route</option>
                            <option value="straight">Straight (0)</option>
                            <option value="left">Left (1)</option>
                            <option value="right">Right (2)</option>
                        </select>
                    </div>
                    <div className="md:col-span-2">
                        <button type="submit" className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white font-medium rounded-md transition duration-150">
                            Add New Data Point
                        </button>
                    </div>
                </form>
            </div>
            
            {/* Retrain Model Feature */}
            <div className="p-6 bg-white rounded-lg shadow-xl">
                <h4 className="text-xl font-semibold mb-4 border-b pb-2">Retrain Model</h4>
                <button
                    onClick={handleRetrain}
                    disabled={retrainLoading}
                    className="py-2 px-4 bg-yellow-500 hover:bg-yellow-600 text-white font-medium rounded-md transition duration-150 disabled:bg-gray-400"
                >
                    {retrainLoading ? 'Retraining...' : '🔄 Retrain Model on New Data'}
                </button>
                {retrainMessage && (
                    <p className={`mt-3 text-sm ${retrainMessage.startsWith('Retraining failed') ? 'text-red-500' : 'text-green-600'}`}>{retrainMessage}</p>
                )}
            </div>

            {/* Existing Data Points Table */}
            <div className="p-6 bg-white rounded-lg shadow-xl overflow-x-auto">
                <h4 className="text-xl font-semibold mb-4 border-b pb-2">Existing Data Points ({dataPoints.length})</h4>
                {error && <p className="text-red-500 mb-4">Error: {error}</p>}
                
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">ID</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Source (ID)</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Destination (ID)</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Parcel Type</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Route Direction</th>
                            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Action</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {dataPoints.map((dp) => (
                            <tr key={dp.id}>
                                <td className="px-4 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{dp.id}</td>
                                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-700">
                                    {dp.source_city_name} ({dp.source_city_id})
                                </td>
                                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-700">
                                    {dp.destination_city_name} ({dp.destination_city_id})
                                </td>
                                <td className="px-4 py-4 whitespace-nowrap text-sm text-gray-700">
                                    {dp.parcel_type_text.toUpperCase()}
                                </td>
                                <td className="px-4 py-4 whitespace-nowrap text-sm">
                                    {isEditing === dp.id ? (
                                        <select
                                            value={editedRoute}
                                            onChange={(e) => setEditedRoute(e.target.value as RouteDirection)}
                                            className="border border-gray-300 rounded-md p-1 text-gray-700"
                                        >
                                            <option value="straight">Straight (0)</option>
                                            <option value="left">Left (1)</option>
                                            <option value="right">Right (2)</option>
                                        </select>
                                    ) : (
                                        <span className={`font-semibold ${dp.route_direction_text === 'straight' ? 'text-blue-600' : dp.route_direction_text === 'left' ? 'text-orange-600' : 'text-purple-600'}`}>
                                            {dp.route_direction_text.toUpperCase()}
                                        </span>
                                    )}
                                </td>
                                <td className="px-4 py-4 whitespace-nowrap text-sm font-medium space-x-2">
                                    {isEditing === dp.id ? (
                                        <>
                                            <button 
                                                onClick={() => handleUpdateRoute(dp.id)}
                                                className="bg-blue-500 hover:bg-blue-600 text-white px-3 py-1 rounded-md text-xs"
                                            >
                                                Update
                                            </button>
                                            <button 
                                                onClick={() => setIsEditing(null)}
                                                className="bg-gray-300 hover:bg-gray-400 text-gray-800 px-3 py-1 rounded-md text-xs"
                                            >
                                                Cancel
                                            </button>
                                        </>
                                    ) : (
                                        <button 
                                            onClick={() => handleEditClick(dp)}
                                            className="text-indigo-600 hover:text-indigo-900 text-xs"
                                        >
                                            Edit
                                        </button>
                                    )}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
};


// --- Helper Component for Employee Profile Edit ---
const EmployeeEditProfile: React.FC<EmployeePageProps> = ({ auth, onLogin }) => {
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
        const response = await axios.post(`${API_URL}/employee/edit_profile/${auth.userId}`, {
          new_username: newUsername,
          new_password: newPassword,
        });
        
        setMessage({ type: 'success', text: response.data.message });
        onLogin('employee', response.data.new_username, auth.userId); 
        setNewPassword(''); 
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
        <h3 className="text-xl font-semibold mb-4 text-green-800">Edit Your Employee Profile</h3>
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
            className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 text-white rounded-md transition duration-150"
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


// --- Main EmployeePage Component ---
const EmployeePage: React.FC<EmployeePageProps> = ({ auth, onLogin }) => {
  const [currentPage, setCurrentPage] = useState<'cities' | 'data-points' | 'profile'>('data-points');

  if (auth.role !== 'employee') {
    return (
      <div className="flex justify-center items-center h-full pt-10">
        <LoginForm role="employee" onLogin={onLogin} />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-6 bg-white rounded-xl shadow-2xl">
      <h1 className="text-3xl font-extrabold text-green-800 mb-6 border-b pb-2">
        👷 Employee Panel
      </h1>
      <div className="flex space-x-4 mb-6">
        <button onClick={() => setCurrentPage('data-points')} className={`py-2 px-4 rounded ${currentPage === 'data-points' ? 'bg-green-600 text-white' : 'bg-green-100 text-green-700 hover:bg-green-200'}`}>
          Data & Model Management
        </button>
        <button onClick={() => setCurrentPage('cities')} className={`py-2 px-4 rounded ${currentPage === 'cities' ? 'bg-green-600 text-white' : 'bg-green-100 text-green-700 hover:bg-green-200'}`}>
          City Management
        </button>
        <button onClick={() => setCurrentPage('profile')} className={`py-2 px-4 rounded ${currentPage === 'profile' ? 'bg-green-600 text-white' : 'bg-green-100 text-green-700 hover:bg-green-200'}`}>
          Edit Profile
        </button>
      </div>
      
      <div className="min-h-[600px]">
        {currentPage === 'cities' && <CityManagement />}
        {currentPage === 'data-points' && <DataPointManagement />}
        {currentPage === 'profile' && <EmployeeEditProfile auth={auth} onLogin={onLogin} onLogout={() => {}} />}
      </div>
    </div>
  );
};

export default EmployeePage;