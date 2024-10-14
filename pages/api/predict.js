// pages/api/predict.js

import formidable from "formidable";
import fs from "fs";
import path from "path";
import { predictImage } from "../../utils/model";

// Disable the default body parser to handle file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};

// Handler for POST requests
const handler = async (req, res) => {
  if (req.method === "POST") {
    const form = new formidable.IncomingForm();
    form.keepExtensions = true; // Keep file extensions

    form.parse(req, async (err, fields, files) => {
      if (err) {
        console.error("Error parsing the file:", err);
        return res.status(500).json({ error: "Error parsing the file." });
      }

      const file = files.file;
      if (!file) {
        return res.status(400).json({ error: "No file uploaded." });
      }

      // Optional: Validate file type
      const allowedTypes = ["image/jpeg", "image/png", "image/jpg"];
      if (!allowedTypes.includes(file.mimetype)) {
        return res.status(400).json({ error: "Unsupported file type." });
      }

      try {
        const imagePath = file.filepath; // Temporary path where the file is stored
        const prediction = await predictImage(imagePath);

        // Clean up: Remove the uploaded file after prediction
        fs.unlink(imagePath, (unlinkErr) => {
          if (unlinkErr) {
            console.error("Error deleting the file:", unlinkErr);
          }
        });

        return res.status(200).json({ prediction_label: prediction });
      } catch (predictionErr) {
        console.error("Prediction error:", predictionErr);
        return res.status(500).json({ error: "Error processing the image." });
      }
    });
  } else {
    res.status(405).json({ message: "Method not allowed." });
  }
};

export default handler;
