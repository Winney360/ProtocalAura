import type { Express, Request } from "express";
import type { Server } from "http";
import { Server as SocketServer } from "socket.io";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import type { LivenessSignals, PostHocAnalysis } from "@shared/schema";

/* ================================
   Multer (memory only, no disk)
================================ */
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 50 * 1024 * 1024 }
});

/* ================================
   Routes Registration
================================ */
export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {

  /* ================================
     Socket.IO — REAL LIVENESS
  ================================ */
  const io = new SocketServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    }
  });

  io.on("connection", (socket) => {
    console.log("Socket connected:", socket.id);

    /**
     * Client sends a webcam frame (JPEG/PNG buffer)
     * No random values here — Python computes everything
     */
    socket.on("liveness-frame", async (frame: Buffer) => {
      try {
        const form = new FormData();
        form.append("frame", frame, {
          filename: "frame.jpg",
          contentType: "image/jpeg"
        });

        const response = await axios.post<LivenessSignals>(
          "http://localhost:8000/liveness",
          form,
          { headers: form.getHeaders() }
        );

        socket.emit("liveness-signals", response.data);
      } catch (error) {
        console.error("Liveness error:", error);
        socket.emit("liveness-error", {
          message: "Liveness analysis failed"
        });
      }
    });

    socket.on("disconnect", () => {
      console.log("Socket disconnected:", socket.id);
    });
  });

  /* ================================
     Media Analysis (Video / Audio)
  ================================ */
  app.post(
    "/api/analyze-media",
    upload.single("file"),
    async (req: Request & { file?: Express.Multer.File }, res) => {

      if (!req.file) {
        return res.status(400).json({ error: "File is required" });
      }

      try {
        const form = new FormData();
        form.append("file", req.file.buffer, {
          filename: req.file.originalname,
          contentType: req.file.mimetype
        });

        const response = await axios.post<PostHocAnalysis>(
          "http://localhost:8000/analyze",
          form,
          { headers: form.getHeaders() }
        );

        res.json(response.data);
      } catch (error) {
        console.error("Analysis error:", error);
        res.status(500).json({
          error: "Media analysis failed"
        });
      }
    }
  );

  /* ================================
     Health Check
  ================================ */
  app.get("/api/health", (_req, res) => {
    res.json({
      status: "healthy",
      level: 2,
      service: "Protocol Aura",
      engine: "Python AI (CV + Audio)"
    });
  });

  return httpServer;
}
