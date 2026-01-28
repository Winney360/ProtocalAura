// File: server/index.ts (REPLACED VERSION - WITH PORT 5000)
import express from "express";
import { createServer } from "http";
import { Server as SocketServer } from "socket.io";
import cors from "cors";
import analyzeRoutes from "../src/routes/analyze";

const app = express();
const httpServer = createServer(app);

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

// Logging middleware
app.use((req, res, next) => {
  const time = new Date().toLocaleTimeString();
  console.log(`[${time}] ${req.method} ${req.url}`);
  next();
});

// Health endpoint
app.get("/api/health", (_req, res) => {
  res.json({
    status: "healthy",
    service: "Protocol Aura Analysis Service",
    version: "2.0.0",
    features: ["video_analysis", "temporal_analysis", "audio_analysis"],
    timestamp: new Date().toISOString()
  });
});

// Use the new analyze routes
app.use("/api", analyzeRoutes);

// Socket.IO for liveness (optional - for future Step 5)
const io = new SocketServer(httpServer, {
  cors: { origin: "*", methods: ["GET", "POST"] }
});

const livenessIntervals = new Map<string, NodeJS.Timeout>();

io.on("connection", (socket) => {
  console.log(`[Socket] Client connected: ${socket.id}`);

  socket.on("start-liveness", async (data: { frame?: Buffer }) => {
    // TODO: Implement Step 5 - Liveness streaming
    console.log(`[Socket] Start liveness requested from ${socket.id}`);
  });

  socket.on("stop-liveness", () => {
    if (livenessIntervals.has(socket.id)) {
      clearInterval(livenessIntervals.get(socket.id)!);
      livenessIntervals.delete(socket.id);
    }
    console.log(`[Socket] Liveness stopped for ${socket.id}`);
  });

  socket.on("disconnect", () => {
    if (livenessIntervals.has(socket.id)) {
      clearInterval(livenessIntervals.get(socket.id)!);
      livenessIntervals.delete(socket.id);
    }
    console.log(`[Socket] Client disconnected: ${socket.id}`);
  });
});

// Root endpoint
app.get("/", (_req, res) => {
  res.json({
    message: "Protocol Aura Analysis API",
    version: "2.0.0",
    endpoints: {
      "POST /api/analyze-media": "Analyze video with temporal & audio analysis",
      "GET /api/health": "Health check",
      "GET /": "This information"
    }
  });
});

// Error handling
app.use((err: any, _req: any, res: any, _next: any) => {
  console.error("Server error:", err);
  res.status(500).json({
    error: "Internal server error",
    message: err.message
  });
});

// Start server - PORT 5000 as you're used to
const PORT = parseInt(process.env.PORT || "5000", 10);
httpServer.listen(PORT, () => {
  console.log(`=================================`);
  console.log(`🚀 Protocol Aura Server v2.0.0`);
  console.log(`📡 Port: ${PORT}`);
  console.log(`🔗 Endpoints:`);
  console.log(`   POST /api/analyze-media - Video analysis`);
  console.log(`   GET  /api/health        - Health check`);
  console.log(`   GET  /                  - API info`);
  console.log(`🔬 Features:`);
  console.log(`   ✓ Visual authenticity detection`);
  console.log(`   ✓ Temporal consistency analysis`);
  console.log(`   ✓ Audio authenticity verification`);
  console.log(`=================================`);
});