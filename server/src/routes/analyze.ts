import { Router } from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";

const router = Router();
const upload = multer({ dest: "uploads/" });

router.post("/analyze-media", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  try {
    const form = new FormData();
    form.append(
      "file",
      fs.createReadStream(req.file.path),
      req.file.originalname
    );

    const aiResponse = await axios.post(
      "http://127.0.0.1:8000/liveness/level2",
      form,
      { headers: form.getHeaders() }
    );

    fs.unlinkSync(req.file.path);

    return res.json({
      source: "protocolaura-ai",
      analysis: aiResponse.data,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "AI service failed" });
  }
});

export default router;
