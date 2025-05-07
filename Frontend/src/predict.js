import React, { useState } from 'react';
import axios from 'axios';

function PredictForm() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return alert('Please upload a DICOM (.dcm) file');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setResult(response.data);
    } catch (error) {
      console.error(error);
      alert('Prediction failed!');
    }
  };

  return (
    <div>
      <h2>DICOM Prediction</h2>
      <form onSubmit={handleSubmit}>
        <input type="file" accept=".dcm" onChange={handleFileChange} />
        <button type="submit">Predict</button>
      </form>

      {result && (
        <div>
          <h3>Prediction Results:</h3>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default PredictForm;