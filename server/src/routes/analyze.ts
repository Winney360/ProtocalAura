import { Router } from "express";
import multer from "multer";
import axios from "axios";
import FormData from "form-data";
import fs from "fs";
import path from "path";
import { extractFrames } from "../utils/extractFrames";
import { extractAudio, hasAudioStream } from "../utils/extractAudio";

const router = Router();
const upload = multer({ dest: "uploads/" });

// Helper function to combine verdicts - SIMPLIFIED
const combineVerdicts = (videoVerdict: string, audioAnalysis: any): string => {
  if (!audioAnalysis || !audioAnalysis.success) {
    return videoVerdict;
  }
  
  const audioVerdict = audioAnalysis.final_audio_verdict || 'unknown';
  const audioConfidence = audioAnalysis.confidence || 0;
  
  // SIMPLIFIED LOGIC
  // Authentic: Good video + audio not strongly suspicious
  // Synthetic: Bad video + audio suspicious OR audio very confident
  // Needs review: Everything else
  
  // Good video + audio OK
  if ((videoVerdict === "authentic" || videoVerdict === "likely_real") && 
      audioConfidence < 0.45) {
    return "likely_authentic";
  }
  
  // Both bad
  if (videoVerdict === "suspicious" && audioVerdict === "suspicious") {
    return "likely_synthetic";
  }
  
  // Audio very confident about synthesis
  if (audioVerdict === "suspicious" && audioConfidence > 0.55) {
    return "likely_synthetic";
  }
  
  // Default: needs review (be honest about limits)
  return "needs_review";
};

// Helper to calculate overall confidence
const calculateOverallConfidence = (videoConfidence: number, audioConfidence: number | null, temporalScore: number): number => {
  if (audioConfidence === null) {
    // Video only: 70% visual confidence, 30% temporal
    return (videoConfidence * 0.7) + (temporalScore * 0.3);
  }
  
  // Combined: 50% video, 20% audio, 30% temporal
  return (videoConfidence * 0.5) + (audioConfidence * 0.2) + (temporalScore * 0.3);
};

// Main analysis endpoint
router.post("/analyze-media", upload.single("file"), async (req, res) => {
  const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const framesDir = path.join("uploads", `frames-${sessionId}`);
  
  if (!req.file) {
    return res.status(400).json({ 
      success: false,
      error: "No file uploaded",
      session_id: sessionId 
    });
  }

  try {
    console.log(`[${sessionId}] Processing: ${req.file.originalname}`);
    
    // 1️⃣ Extract frames for video analysis
    console.log(`[${sessionId}] Extracting frames...`);
    const frames = await extractFrames(req.file.path, framesDir, 30);
    
    if (frames.length === 0) {
      throw new Error("No frames could be extracted from video");
    }
    
    console.log(`[${sessionId}] Extracted ${frames.length} frames`);

    // 2️⃣ Prepare video analysis request
    const videoForm = new FormData();
    
    // Add frames in order for temporal analysis
    frames.forEach((framePath, index) => {
      videoForm.append("files", fs.createReadStream(framePath), {
        filename: `frame_${index.toString().padStart(3, '0')}.jpg`,
        contentType: "image/jpeg"
      });
    });

    // 3️⃣ Analyze video with temporal analysis
    console.log(`[${sessionId}] Analyzing video with temporal analysis...`);
    let videoAnalysis;
    try {
      const videoResponse = await axios.post(
        "http://localhost:8000/liveness/level2",
        videoForm,
        { 
          headers: videoForm.getHeaders(),
          timeout: 120000
        }
      );
      videoAnalysis = videoResponse.data;
    } catch (videoError: any) {
      console.error(`[${sessionId}] Video analysis failed:`, videoError.message);
      videoAnalysis = {
        success: false,
        error: videoError.message,
        aggregated_metrics: {
          final_humanity_score: 0,
          final_confidence: 0,
          final_verdict: "error"
        },
        temporal_analysis: {
          temporal_consistency_score: 0,
          stability: 'unknown',
          anomaly_count: 0
        }
      };
    }

    // 4️⃣ Check and extract audio if available
    let audioAnalysis = null;
    let hasAudio = false;
    
    try {
      hasAudio = await hasAudioStream(req.file.path);
      console.log(`[${sessionId}] Audio stream detected: ${hasAudio}`);
      
      if (hasAudio) {
        console.log(`[${sessionId}] Extracting audio...`);
        const audioPath = await extractAudio(req.file.path, framesDir);
        
        // Prepare audio analysis request
        const audioForm = new FormData();
        audioForm.append("file", fs.createReadStream(audioPath));
        
        // Analyze audio
        console.log(`[${sessionId}] Analyzing audio...`);
        try {
          const audioResponse = await axios.post(
            "http://localhost:8000/audio/analyze",
            audioForm,
            { 
              headers: audioForm.getHeaders(),
              timeout: 30000
            }
          );
          audioAnalysis = audioResponse.data;
        } catch (audioError: any) {
          console.warn(`[${sessionId}] Audio analysis failed:`, audioError.message);
          audioAnalysis = {
            success: false,
            error: audioError.message,
            final_audio_verdict: "analysis_error",
            confidence: 0
          };
        }
        
        // Cleanup audio file
        fs.unlinkSync(audioPath);
      }
    } catch (audioExtractError: any) {
      console.warn(`[${sessionId}] Audio extraction failed:`, audioExtractError.message);
      hasAudio = false;
    }

    // 5️⃣ Process video analysis results
    const videoSuccess = videoAnalysis?.success === true;
    const videoMetrics = videoAnalysis?.aggregated_metrics || {};
    const temporalAnalysis = videoAnalysis?.temporal_analysis || {};
    const frameDetails = videoAnalysis?.frame_details || [];
    
    // Calculate video confidence
    const videoHumanityScore = videoMetrics.final_humanity_score || 0;
    const videoConfidence = videoMetrics.final_confidence || 0;
    const temporalScore = temporalAnalysis.temporal_consistency_score || 0;
    const anomalyCount = temporalAnalysis.anomaly_count || 0;
    const stability = temporalAnalysis.stability || 'unknown';
    
    // Determine video verdict
    let videoVerdict = "suspicious";
    if (videoSuccess) {
      const tempVideoVerdict = videoMetrics.final_verdict;
      if (tempVideoVerdict === "likely_real" && anomalyCount === 0 && temporalScore > 0.7) {
        videoVerdict = "authentic";
      } else if (tempVideoVerdict === "likely_real" && anomalyCount <= 1) {
        videoVerdict = "likely_authentic";
      } else if (tempVideoVerdict === "needs_review") {
        videoVerdict = "needs_review";
      } else {
        videoVerdict = "suspicious";
      }
    }

    // 6️⃣ Process audio analysis results
    const audioSuccess = audioAnalysis?.success === true;
    const audioVerdict = audioAnalysis?.final_audio_verdict || 'unknown';
    const audioConfidence = audioAnalysis?.confidence || 0;
    const audioInsights = audioAnalysis?.insights || { strengths: [], anomalies: [] };

    // 7️⃣ Calculate combined results
    const combinedVerdict = combineVerdicts(videoVerdict, audioAnalysis);
    const overallConfidence = calculateOverallConfidence(
      videoConfidence,
      audioSuccess ? audioConfidence : null,
      temporalScore
    );

    // 8️⃣ Generate insights
    const insights = {
      visual: {
        score: videoHumanityScore.toFixed(2),
        assessment: videoHumanityScore > 0.7 ? "strong" : 
                   videoHumanityScore > 0.5 ? "moderate" : "weak",
        confidence: videoConfidence.toFixed(2)
      },
      temporal: {
        consistency_score: temporalScore.toFixed(2),
        stability: stability,
        anomalies_detected: anomalyCount,
        assessment: temporalScore > 0.7 ? "stable" : 
                   temporalScore > 0.5 ? "moderate" : "unstable"
      },
      audio: {
        available: hasAudio,
        verdict: audioVerdict,
        confidence: audioConfidence.toFixed(2),
        assessment: audioSuccess ? (audioConfidence > 0.7 ? "authentic" : 
                   audioConfidence > 0.5 ? "likely_authentic" : "questionable") : "not_analyzed"
      }
    };

    // 9️⃣ Prepare anomaly timeline
    const anomalyTimeline = (videoAnalysis.temporal_analysis?.anomaly_timeline || []).map((anomaly: any) => ({
      frame: anomaly.frame_index || 0,
      reason: anomaly.reason || 'unknown',
      score: anomaly.anomaly_score?.toFixed(2) || '0.00',
      timestamp: `${((anomaly.frame_index || 0) / frames.length * 100).toFixed(1)}%`
    }));

    // 🔟 Prepare frame samples
    const frameSamples = frameDetails.slice(0, 5).map((frame: any) => ({
      frame: frame.frame_index || 0,
      score: frame.humanity_score?.toFixed(2) || '0.00',
      verdict: frame.verdict || 'unknown',
      has_anomaly: frame.temporal_metrics?.has_anomaly || false
    }));

    // 1️⃣1️⃣ Build final response
    const response = {
      success: true,
      session_id: sessionId,
      processing_info: {
        file_name: req.file.originalname,
        file_size: req.file.size,
        file_type: req.file.mimetype,
        frames_analyzed: frames.length,
        audio_analyzed: hasAudio,
        analysis_modes: ['visual', 'temporal'].concat(hasAudio ? ['audio'] : [])
      },
      analysis: {
        final_verdict: combinedVerdict,
        overall_confidence: overallConfidence.toFixed(2),
        confidence_level: overallConfidence > 0.8 ? "high" : 
                        overallConfidence > 0.6 ? "medium" : "low",
        
        // Component scores
        component_scores: {
          visual_authenticity: videoHumanityScore.toFixed(2),
          temporal_consistency: temporalScore.toFixed(2),
          audio_authenticity: audioSuccess ? audioConfidence.toFixed(2) : "N/A",
          anomaly_count: anomalyCount
        },
        
        // Detailed insights
        insights: insights,
        
        // Signals (backward compatibility)
        signals: {
          humanityScore: videoHumanityScore.toFixed(2),
          confidence: overallConfidence.toFixed(2),
          textureVariance: (frameDetails[0]?.visual_metrics?.feature_variance || 0).toFixed(2),
          edgeDensity: (frameDetails[0]?.visual_metrics?.feature_entropy || 0).toFixed(4),
          temporalConsistency: temporalScore.toFixed(2),
          audioQuality: audioSuccess ? audioConfidence.toFixed(2) : "N/A"
        },
        
        // Temporal details
        temporal_details: {
          consistency_score: temporalScore.toFixed(2),
          stability: stability,
          anomaly_count: anomalyCount,
          anomaly_timeline: anomalyTimeline,
          has_abrupt_changes: frameDetails.some((f: any) => 
            f.temporal_metrics?.is_abrupt_change
          )
        },
        
        // Audio details
        audio_details: hasAudio ? {
          verdict: audioVerdict,
          confidence: audioConfidence.toFixed(2),
          strengths: audioInsights.strengths || [],
          anomalies: audioInsights.anomalies || [],
          available: true
        } : {
          available: false,
          note: "No audio stream detected or extraction failed"
        },
        
        // Explainability
        explanations: videoAnalysis.explainability?.key_factors || [
          `Visual authenticity: ${insights.visual.assessment}`,
          `Temporal stability: ${insights.temporal.assessment}`,
          anomalyCount > 0 ? `${anomalyCount} temporal anomalies detected` : "No temporal anomalies",
          hasAudio ? `Audio analysis: ${insights.audio.assessment}` : "Audio: Not available"
        ],
        
        // Raw samples
        samples: {
          frame_samples: frameSamples,
          anomaly_samples: anomalyTimeline.slice(0, 3)
        }
      },
      metadata: {
        analysis_version: "2.0.0",
        features: ['temporal_analysis', 'audio_analysis'],
        timestamp: new Date().toISOString()
      }
    };

    console.log(`[${sessionId}] Analysis complete: ${combinedVerdict} (${overallConfidence.toFixed(2)} confidence)`);

    // 1️⃣2️⃣ Cleanup
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
    } catch (cleanupError: any) {
      console.warn(`[${sessionId}] Cleanup warning:`, cleanupError.message);
    }

    return res.json(response);

  } catch (error: any) {
    console.error(`[${sessionId}] Analysis error:`, error);
    
    // Error response
    const errorResponse = {
      success: false,
      session_id: sessionId,
      error: "Video analysis failed",
      error_details: error.message,
      timestamp: new Date().toISOString(),
      fallback_analysis: {
        verdict: "needs_review",
        confidence_level: "low",
        note: "Using fallback analysis due to processing error"
      }
    };

    // Cleanup on error
    try {
      if (fs.existsSync(framesDir)) {
        fs.rmSync(framesDir, { recursive: true, force: true });
      }
      if (req.file && fs.existsSync(req.file.path)) {
        fs.unlinkSync(req.file.path);
      }
    } catch (cleanupError: any) {
      console.warn(`[${sessionId}] Cleanup failed after error:`, cleanupError.message);
    }

    return res.status(500).json(errorResponse);
  }
});

// Health check endpoint
router.get("/health", (req, res) => {
  res.json({
    status: "healthy",
    service: "Protocol Aura Analysis Service",
    version: "2.0.0",
    features: ["video_analysis", "temporal_analysis", "audio_analysis"],
    timestamp: new Date().toISOString()
  });
});

export default router;