// File: server/src/routes/analyze.ts (UPDATED with Temporal Analysis)
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

  const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const framesDir = path.join("uploads", `frames-${sessionId}`);

  try {
    console.log(`[${sessionId}] Processing: ${req.file.originalname}`);
    
    // 1️⃣ Extract frames (30 frames for temporal analysis)
    const frames = await extractFrames(req.file.path, framesDir, 30);
    console.log(`[${sessionId}] Extracted ${frames.length} frames`);

    // Prepare form data for AI service (send all frames at once for temporal analysis)
    const form = new FormData();
    
    // Add frames in order (important for temporal analysis)
    frames.forEach((framePath, index) => {
      form.append("files", fs.createReadStream(framePath), {
        filename: `frame_${index.toString().padStart(3, '0')}.jpg`,
        contentType: "image/jpeg"
      });
    });

    // 2️⃣ Send to AI service with temporal analysis
    console.log(`[${sessionId}] Sending to AI service for temporal analysis...`);
    const aiResponse = await axios.post(
      "http://127.0.0.1:8000/liveness/level2",
      form,
      { 
        headers: form.getHeaders(),
        timeout: 120000 // 2 minute timeout for temporal analysis
      }
    );

    const aiData = aiResponse.data;
    
    // 3️⃣ Process temporal analysis results
    const temporalAnalysis = aiData.temporal_analysis || {
      temporal_consistency_score: 0,
      stability: 'unknown',
      anomaly_count: 0
    };

    // 4️⃣ Enhanced aggregation with temporal factors
    const frameResults = aiData.frame_details || [];
    
    // Calculate averages from all frames
    const avg = (key: string) => {
      if (frameResults.length === 0) return 0;
      return frameResults.reduce((sum: number, frame: any) => sum + (frame[key] || 0), 0) / frameResults.length;
    };

    const aggregated = {
      humanityScore: Number(avg("humanity_score") || 0).toFixed(2),
      confidence: Number(avg("confidence") || 0).toFixed(2),
      textureVariance: Number(avg("visual_metrics?.feature_variance") || 0).toFixed(2),
      edgeDensity: Number(avg("visual_metrics?.feature_entropy") || 0).toFixed(4),
    };

    // 5️⃣ Enhanced verdict with temporal analysis
    let verdict = "suspicious";
    let confidenceLevel = "low";
    
    const baseHumanity = parseFloat(aggregated.humanityScore);
    const baseConfidence = parseFloat(aggregated.confidence);
    const temporalScore = temporalAnalysis.temporal_consistency_score || 0;
    const anomalyCount = temporalAnalysis.anomaly_count || 0;
    
    // Combined scoring: 70% visual, 30% temporal
    const combinedScore = (baseHumanity * 0.7) + (temporalScore * 0.3);
    const combinedConfidence = (baseConfidence * 0.7) + (temporalScore * 0.3);
    
    // Determine verdict with temporal considerations
    if (combinedScore > 0.75 && anomalyCount === 0 && temporalScore > 0.7) {
      verdict = "authentic";
      confidenceLevel = "high";
    } else if (combinedScore > 0.6 && anomalyCount <= 1 && temporalScore > 0.5) {
      verdict = "likely_authentic";
      confidenceLevel = "medium";
    } else if (combinedScore > 0.4) {
      verdict = "needs_review";
      confidenceLevel = "low";
    } else {
      verdict = "suspicious";
      confidenceLevel = "low";
    }

    // 6️⃣ Generate detailed insights
    const insights = {
      visual: {
        score: baseHumanity,
        assessment: baseHumanity > 0.7 ? "strong" : baseHumanity > 0.5 ? "moderate" : "weak"
      },
      temporal: {
        consistency_score: temporalScore,
        stability: temporalAnalysis.stability || 'unknown',
        anomalies_detected: anomalyCount,
        assessment: temporalScore > 0.7 ? "stable" : temporalScore > 0.5 ? "moderate" : "unstable"
      },
      combined: {
        final_score: combinedScore,
        confidence: combinedConfidence,
        confidence_level: confidenceLevel
      }
    };

    // 7️⃣ Anomaly details (if any)
    const anomalies = aiData.temporal_analysis?.anomaly_timeline || [];
    const anomalyDetails = anomalies.map((anomaly: any) => ({
      frame: anomaly.frame_index || 0,
      reason: anomaly.reason || 'unknown',
      score: anomaly.anomaly_score || 0
    }));

    // 8️⃣ Response with temporal insights
    const response = {
      success: true,
      session_id: sessionId,
      analysis: {
        type: "video",
        frames_analyzed: frameResults.length,
        verdict: verdict,
        confidence_level: confidenceLevel,
        temporal_analysis_included: true,
        
        // Core signals
        signals: {
          humanityScore: aggregated.humanityScore,
          confidence: aggregated.confidence,
          textureVariance: aggregated.textureVariance,
          edgeDensity: aggregated.edgeDensity,
          temporalConsistency: temporalScore.toFixed(2),
          anomalyCount: anomalyCount
        },
        
        // Detailed insights
        insights: insights,
        
        // Temporal details
        temporal_details: {
          consistency_score: temporalScore.toFixed(2),
          stability: temporalAnalysis.stability,
          anomaly_count: anomalyCount,
          anomaly_details: anomalyDetails.slice(0, 5), // First 5 anomalies
          has_abrupt_changes: frameResults.some((f: any) => 
            f.temporal_metrics?.is_abrupt_change
          )
        },
        
        // Explainability
        reasons: aiData.explainability?.key_factors || [
          `Visual authenticity: ${insights.visual.assessment}`,
          `Temporal stability: ${insights.temporal.assessment}`,
          anomalyCount > 0 ? `${anomalyCount} anomalies detected` : "No anomalies detected"
        ],
        
        // Raw data (truncated for performance)
        frame_samples: frameResults.slice(0, 3).map((frame: any) => ({
          frame: frame.frame_index,
          score: frame.humanity_score?.toFixed(2),
          verdict: frame.verdict
        }))
      }
    };

    console.log(`[${sessionId}] Analysis complete: ${verdict} (${confidenceLevel} confidence)`);

    // 9️⃣ Cleanup
    try {
      // Clean frames
      frames.forEach(framePath => {
        if (fs.existsSync(framePath)) {
          fs.unlinkSync(framePath);
        }
      });
      
      // Clean frames directory
      if (fs.existsSync(framesDir)) {
        fs.rmSync(framesDir, { recursive: true, force: true });
      }
      
      // Clean uploaded file
      if (fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
      
      console.log(`[${sessionId}] Cleanup completed`);
    } catch (cleanupError) {
      console.warn(`[${sessionId}] Cleanup warning:`, cleanupError);
    }

    return res.json(response);

  } catch (err: any) {
    console.error(`[${sessionId}] Analysis error:`, err);
    
    // Fallback analysis if AI service fails
    const fallbackResponse = {
      success: false,
      error: "Video analysis failed",
      error_details: err.message,
      fallback_analysis: {
        verdict: "needs_review",
        confidence_level: "low",
        note: "Using fallback analysis due to AI service error"
      }
    };

    // Cleanup even on error
    try {
      if (fs.existsSync(framesDir)) {
        fs.rmSync(framesDir, { recursive: true, force: true });
      }
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
    } catch (cleanupError) {
      console.warn("Cleanup failed after error:", cleanupError);
    }

    return res.status(500).json(fallbackResponse);
  }
});

export default router;