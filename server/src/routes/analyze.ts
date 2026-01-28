import { Router } from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { extractFrames } from "../utils/extractFrames";

const router = Router();
const upload = multer({ dest: "uploads/" });

router.post("/analyze-media", upload.single("file"), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  const framesDir = path.join("uploads", `frames-${Date.now()}`);

  try {
    // 1️⃣ Extract frames
    const frames = await extractFrames(req.file.path, framesDir);

    const results = [];

    // 2️⃣ Analyze each frame
    for (const framePath of frames) {
      const form = new FormData();
      form.append("file", fs.createReadStream(framePath));

      const response = await axios.post(
        "http://127.0.0.1:8000/liveness/level2",
        form,
        { headers: form.getHeaders() }
      );

      results.push(response.data.signals);
    }

    // 3️⃣ Aggregate (REAL math)
    const avg = (key: string) =>
      results.reduce((a, b) => a + b[key], 0) / results.length;

    const aggregated = {
      humanityScore: Number(avg("humanityScore").toFixed(2)),
      confidence: Number(avg("confidence").toFixed(2)),
      textureVariance: Number(avg("textureVariance").toFixed(2)),
      edgeDensity: Number(avg("edgeDensity").toFixed(4)),
    };

    const verdict =
      aggregated.humanityScore > 40 && aggregated.confidence > 0.55
        ? "authentic"
        : "suspicious";

    // Cleanup
    fs.rmSync(framesDir, { recursive: true, force: true });
    fs.unlinkSync(req.file.path);

    return res.json({
      type: "video",
      framesAnalyzed: results.length,
      verdict,
      signals: aggregated,
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Video analysis failed" });
  }
});

export default router;
