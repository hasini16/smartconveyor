// client/src/pages/UserPage.tsx
import React, { useState, ChangeEvent } from 'react';
import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';

interface PredictionResult {
  source_id: number;
  destination_id: number;
  source_city_name: string;
  destination_city_name: string;
  parcel_type: string;
  predicted_route: string;
}

const UserPage: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  
  // Simulated State for data extracted from the image
  // This replaces complex OCR/CV logic for the prototype
  const [extractedData, setExtractedData] = useState({
    source_city_id: 1677, // Example ID from parcels_10000.csv
    destination_city_id: 4015,
    parcel_type_text: 'normal',
  });

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setImageFile(e.target.files[0]);
      
      // --- SIMULATION of Computer Vision/OCR ---
      // Randomly select some IDs and type to simulate extraction
      const randomDataPoints = [
        { src: 1677, dst: 4015, type: 'normal' },
        { src: 3906, dst: 4503, type: 'fast' },
        { src: 4937, dst: 3938, type: 'normal' },
        { src: 3509, dst: 4905, type: 'normal' },
        { src: 3288, dst: 2514, type: 'fast' },
      ];
      const randomData = randomDataPoints[Math.floor(Math.random() * randomDataPoints.length)];

      setExtractedData({
        source_city_id: randomData.src,
        destination_city_id: randomData.dst,
        parcel_type_text: randomData.type as 'normal' | 'fast',
      });
      // --- END SIMULATION ---
    }
  };

  const handlePredict = async () => {
    if (!imageFile) {
      setError("Please upload a package image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Send the extracted data to the backend's prediction endpoint
      const response = await axios.post<PredictionResult>(
        `${API_URL}/user/predict_route`, 
        extractedData
      );
      setResult(response.data);
      setError(null);
    } catch (err) {
      if (axios.isAxiosError(err) && err.response) {
        setError(`Prediction Failed: ${err.response.data.detail}`);
      } else {
        setError("An unexpected error occurred during prediction.");
      }
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-xl shadow-2xl">
      <h1 className="text-3xl font-extrabold text-gray-900 mb-6">
        📦 Smart Conveyor Route Predictor
      </h1>
      
      <p className="text-gray-600 mb-6">
        Upload a package label image to predict the required route. 
        
      </p>

      <div className="space-y-4">
        <label className="block text-sm font-medium text-gray-700">
          1. Upload Package Image (Simulated)
        </label>
        <input 
          type="file" 
          accept="image/*" 
          onChange={handleFileChange}
          className="block w-full text-sm text-gray-500
            file:mr-4 file:py-2 file:px-4
            file:rounded-full file:border-0
            file:text-sm file:font-semibold
            file:bg-indigo-50 file:text-indigo-700
            hover:file:bg-indigo-100"
        />

        {imageFile && (
          <div className="p-4 bg-gray-50 border border-dashed border-gray-300 rounded-lg">
            <p className="text-sm font-medium text-gray-700">
              **Simulated Extracted Data** (from OCR/CV):
            </p>
            <ul className="text-sm text-gray-600 mt-1 list-disc ml-5">
              <li>Source City ID: **{extractedData.source_city_id}**</li>
              <li>Destination City ID: **{extractedData.destination_city_id}**</li>
              <li>Parcel Type: **{extractedData.parcel_type_text.toUpperCase()}** (0=Normal, 1=Fast)</li>
            </ul>
          </div>
        )}
        
        <button
          onClick={handlePredict}
          disabled={loading || !imageFile}
          className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white transition duration-150 
            ${loading || !imageFile ? 'bg-indigo-300' : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'}`}
        >
          {loading ? 'Predicting...' : '✨ Predict Route'}
        </button>
      </div>
      
      {error && (
        <div className="mt-4 p-3 rounded-md bg-red-100 text-red-700">
          **Error**: {error}
        </div>
      )}

      {result && (
        <div className="mt-6 p-6 rounded-xl bg-green-50 border border-green-300 shadow-inner">
          <h2 className="text-xl font-bold text-green-800 mb-2">
            ✅ Prediction Result:
          </h2>
          <p className="text-md text-gray-700">
            Package from **{result.source_city_name}** to **{result.destination_city_name}** ({result.parcel_type.toUpperCase()})
          </p>
          <p className="text-3xl font-extrabold mt-2 text-green-900">
            Route: **{result.predicted_route.toUpperCase()}**
          </p>
        </div>
      )}
    </div>
  );
};

export default UserPage;