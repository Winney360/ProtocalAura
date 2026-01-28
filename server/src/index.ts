// server/index.ts
import express, { Request, Response, NextFunction } from "express";
import { createServer } from "http";
import { Server as SocketServer } from "socket.io";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";

const app = express();
const httpServer = createServer(app);

// Multer memory storage for uploads
const upload = multer({ storage: multer.memoryStorage() });

// Map to store analysis results
const analysisResults = new Map<
  string,
  { status: "processing" | "complete" | "error"; results?: any }
>();

// Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Logging helper
function log(message: string) {
  const time = new Date().toLocaleTimeString();
  console.log(`[${time}] ${message}`);
}

// Health endpoint
app.get("/api/health", (_req, res) => {
  res.json({ status: "healthy", service: "Protocol Aura", version: "1.0.0" });
});

// --- Level 2: File upload and forward to Python ---
app.post(
  "/api/analyze-media",
  upload.single("file"),
  async (req: Request & { file?:Express.Multer.File }, res) => {
    const file = req.file;
    const analysisId = req.body.analysisId;

    if (!file || !analysisId) {
      return res.status(400).json({ error: "File and analysisId are required" });
    }

    try {
      const form = new FormData();
      form.append("file", file.buffer, { filename: file.originalname });

      // Forward to Python AI service
      const response = await axios.post("http://localhost:8000/analyze", form, {
        headers: form.getHeaders(),
      });

      // Store the result
      analysisResults.set(analysisId, { status: "complete", results: response.data });

      return res.json({ success: true, analysisId, message: "Analysis started" });
    } catch (err: any) {
      console.error("Python service error:", err.message || err);
      analysisResults.set(analysisId, { status: "error" });
      return res.status(500).json({ error: "Python service request failed" });
    }
  }
);

// Get analysis result
app.get("/api/analysis-result/:id", (req, res) => {
  const analysisId = req.params.id;
  const result = analysisResults.get(analysisId);
  if (!result) return res.status(404).json({ error: "Analysis not found" });
  res.json(result);
});

// --- Socket.IO Liveness streaming ---
const io = new SocketServer(httpServer, { cors: { origin: "*", methods: ["GET", "POST"] } });

// Store intervals for live streaming
const livenessIntervals = new Map<string, NodeJS.Timeout>();

io.on("connection", (socket) => {
  log(`Client connected: ${socket.id}`);

  socket.on("start-liveness", async (data: { frame?: Buffer }) => {
    if (livenessIntervals.has(socket.id)) {
      clearInterval(livenessIntervals.get(socket.id)!);
    }

    const sendSignals = async () => {
      try {
        if (!data.frame) return;

        const form = new FormData();
        form.append("file", data.frame, { filename: "frame.jpg" });

        // Call Python liveness endpoint
        const response = await axios.post("http://localhost:8000/liveness", form, {
          headers: form.getHeaders(),
        });

        socket.emit("liveness-signals", response.data);
      } catch (err) {
        console.error("Error fetching liveness from Python:", err);
      }
    };

    // Send immediately and then every 500ms
    await sendSignals();
    const interval = setInterval(sendSignals, 500);
    livenessIntervals.set(socket.id, interval);
  });

  socket.on("stop-liveness", () => {
    if (livenessIntervals.has(socket.id)) {
      clearInterval(livenessIntervals.get(socket.id)!);
      livenessIntervals.delete(socket.id);
    }
  });

  socket.on("disconnect", () => {
    if (livenessIntervals.has(socket.id)) {
      clearInterval(livenessIntervals.get(socket.id)!);
      livenessIntervals.delete(socket.id);
    }
    log(`Client disconnected: ${socket.id}`);
  });
});

// --- Start server ---
const PORT = parseInt(process.env.PORT || "5000", 10);
httpServer.listen(PORT, () => log(`Node.js server running on http://localhost:${PORT}`));
