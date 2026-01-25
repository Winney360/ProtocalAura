import type { Express, Request } from "express";
import { createServer, type Server } from "http";
import { Server as SocketServer } from "socket.io";
import multer, { type Multer } from "multer";
import type { LivenessSignals, PostHocAnalysis, Anomaly } from "@shared/schema";

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 100 * 1024 * 1024 }
});

const analysisResults = new Map<string, {
  status: "processing" | "complete" | "error";
  results?: PostHocAnalysis["results"];
}>();

function generateSimulatedLivenessSignals(): LivenessSignals {
  const baseHumanity = 75 + Math.random() * 20;
  const pulseVariation = Math.sin(Date.now() / 1000) * 5;
  
  return {
    biologicalPulse: Math.round(68 + pulseVariation + Math.random() * 8),
    microExpressionScore: Math.round(70 + Math.random() * 25),
    humanityScore: Math.round(Math.min(100, Math.max(0, baseHumanity + Math.random() * 10 - 5))),
    verificationStatus: baseHumanity > 80 ? "verified" : baseHumanity > 60 ? "verifying" : "suspicious",
    timestamp: Date.now()
  };
}

function generateSimulatedAnomalies(isVideo: boolean): Anomaly[] {
  const videoAnomalyTypes: Anomaly["type"][] = ["face_inconsistency", "lip_sync", "frame_artifact"];
  const audioAnomalyTypes: Anomaly["type"][] = ["audio_pattern", "pitch_anomaly"];
  
  const types = isVideo ? videoAnomalyTypes : audioAnomalyTypes;
  const anomalyCount = Math.floor(Math.random() * 4);
  const anomalies: Anomaly[] = [];
  
  const descriptions: Record<Anomaly["type"], string[]> = {
    face_inconsistency: [
      "Facial boundary artifacts detected",
      "Unnatural skin texture pattern",
      "Eye reflection inconsistency"
    ],
    lip_sync: [
      "Lip movement timing mismatch",
      "Audio-visual sync deviation",
      "Mouth shape anomaly"
    ],
    frame_artifact: [
      "Temporal consistency error",
      "Blending artifact at frame boundary",
      "Compression pattern anomaly"
    ],
    audio_pattern: [
      "Synthetic voice signature detected",
      "Unnatural frequency pattern",
      "Voice cloning artifact"
    ],
    pitch_anomaly: [
      "Pitch variation irregularity",
      "Formant structure anomaly",
      "Prosody pattern deviation"
    ]
  };
  
  for (let i = 0; i < anomalyCount; i++) {
    const type = types[Math.floor(Math.random() * types.length)];
    const severity: Anomaly["severity"] = Math.random() > 0.7 ? "high" : Math.random() > 0.4 ? "medium" : "low";
    
    anomalies.push({
      timestamp: Math.round(Math.random() * 60 * 10) / 10,
      type,
      severity,
      description: descriptions[type][Math.floor(Math.random() * descriptions[type].length)]
    });
  }
  
  return anomalies.sort((a, b) => a.timestamp - b.timestamp);
}

function simulateAnalysis(analysisId: string, isVideo: boolean): void {
  analysisResults.set(analysisId, { status: "processing" });
  
  const processingTime = 3000 + Math.random() * 4000;
  
  setTimeout(() => {
    const anomalies = generateSimulatedAnomalies(isVideo);
    const anomalyImpact = anomalies.reduce((acc, a) => {
      return acc + (a.severity === "high" ? 15 : a.severity === "medium" ? 8 : 3);
    }, 0);
    
    const baseScore = 85 + Math.random() * 10;
    const finalScore = Math.max(10, Math.min(100, Math.round(baseScore - anomalyImpact)));
    
    let verdict: "authentic" | "synthetic" | "inconclusive";
    if (finalScore >= 75) {
      verdict = "authentic";
    } else if (finalScore <= 40) {
      verdict = "synthetic";
    } else {
      verdict = "inconclusive";
    }
    
    analysisResults.set(analysisId, {
      status: "complete",
      results: {
        humanityScore: finalScore,
        anomalies,
        verdict,
        processingTime: Math.round(processingTime / 100) / 10
      }
    });
  }, processingTime);
}

export async function registerRoutes(
  httpServer: Server,
  app: Express
): Promise<Server> {
  const io = new SocketServer(httpServer, {
    cors: {
      origin: "*",
      methods: ["GET", "POST"]
    }
  });

  const livenessIntervals = new Map<string, NodeJS.Timeout>();

  io.on("connection", (socket) => {
    console.log("Client connected:", socket.id);

    socket.on("start-liveness", () => {
      if (livenessIntervals.has(socket.id)) {
        clearInterval(livenessIntervals.get(socket.id));
      }
      
      const sendSignals = () => {
        const signals = generateSimulatedLivenessSignals();
        socket.emit("liveness-signals", signals);
      };
      
      sendSignals();
      const interval = setInterval(sendSignals, 500);
      livenessIntervals.set(socket.id, interval);
    });

    socket.on("stop-liveness", () => {
      if (livenessIntervals.has(socket.id)) {
        clearInterval(livenessIntervals.get(socket.id));
        livenessIntervals.delete(socket.id);
      }
    });

    socket.on("disconnect", () => {
      if (livenessIntervals.has(socket.id)) {
        clearInterval(livenessIntervals.get(socket.id));
        livenessIntervals.delete(socket.id);
      }
      console.log("Client disconnected:", socket.id);
    });
  });

  app.post("/api/analyze-media", upload.single("file"), (req: Request & { file?: Multer.File }, res) => {
    const file = req.file;
    const analysisId = req.body.analysisId as string;
    
    if (!file || !analysisId) {
      return res.status(400).json({ error: "File and analysisId are required" });
    }
    
    const isVideo = file.mimetype.startsWith("video/");
    simulateAnalysis(analysisId, isVideo);
    
    res.json({ 
      success: true, 
      analysisId,
      message: "Analysis started"
    });
  });

  app.get("/api/analysis-result/:id", (req, res) => {
    const analysisId = req.params.id;
    const result = analysisResults.get(analysisId);
    
    if (!result) {
      return res.status(404).json({ error: "Analysis not found" });
    }
    
    res.json(result);
  });

  app.get("/api/health", (_, res) => {
    res.json({ 
      status: "healthy",
      service: "Protocol Aura",
      version: "1.0.0-prototype"
    });
  });

  return httpServer;
}
