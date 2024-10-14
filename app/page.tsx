"use client";

import { useState, ChangeEvent, FormEvent } from "react";

export default function Home() {
  const [file, setFile] = useState<File | null>(null); // State to store the selected file
  const [prediction, setPrediction] = useState<string>(""); // State to store prediction result
  const [loading, setLoading] = useState<boolean>(false); // State to manage loading status
  const [error, setError] = useState<string>(""); // State to handle errors

  // Handle file selection
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0] || null; // Check if file is selected
    setFile(selectedFile);
    setPrediction("");
    setError("");
  };

  // Handle form submission
  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    if (!file) {
      setError("Please select an image to upload.");
      return;
    }

    setLoading(true);
    setError("");
    setPrediction("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch("/api/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Network response was not ok.");
      }

      const result = await response.json();
      setPrediction(result.prediction_label || "Unknown Category");
    } catch (err: unknown) {
      if (err instanceof Error) {
        console.error(err.message);
        setError(err.message || "An error occurred while processing the image.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto py-8 text-center">
      <h1 className="text-3xl font-semibold mb-6">Extremism Detection Demo</h1>
      <form onSubmit={handleSubmit} className="mb-6">
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="mb-4"
        />
        <button
          type="submit"
          className="px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          disabled={loading}
        >
          {loading ? "Processing..." : "Upload & Predict"}
        </button>
      </form>

      {prediction && (
        <div className="mt-4 text-green-600">
          <h3>Prediction Result: {prediction}</h3>
        </div>
      )}

      {error && (
        <div className="mt-4 text-red-600">
          <h3>Error: {error}</h3>
        </div>
      )}
    </div>
  );
}
