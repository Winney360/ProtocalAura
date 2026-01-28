// File: shared/schema.ts
import { z } from "zod";

// ========== LIVENESS TYPES (EXISTING) ==========
export interface LivenessSignals {
  biologicalPulse: number;
  microExpressionScore: number;
  humanityScore: number;
  verificationStatus: "verifying" | "verified" | "unverified" | "suspicious";
  timestamp: number;
}

// ========== ANOMALY TYPES (UPDATED) ==========
export interface Anomaly {
  timestamp: number;
  type: "face_inconsistency" | "lip_sync" | "frame_artifact" | "audio_pattern" | "pitch_anomaly";
  severity: "low" | "medium" | "high";
  description: string;
}

// ========== ENHANCED ANALYSIS TYPES (NEW - FOR V2.0 API) ==========
// Temporal Analysis Types
export interface TemporalAnomaly {
  frame: number;
  reason: string;
  score: string;
  timestamp: string;
}

export interface FrameSample {
  frame: number;
  score: string;
  verdict: string;
  has_anomaly: boolean;
}

// Insight Metrics
export interface InsightMetric {
  score: string;
  assessment: string;
  confidence: string;
}

export interface TemporalInsight extends InsightMetric {
  stability: string;
  anomalies_detected: number;
}

export interface AudioInsight {
  available: boolean;
  verdict: string;
  confidence: string;
  assessment: string;
}

export interface AnalysisSignals {
  humanityScore: string;
  confidence: string;
  textureVariance: string;
  edgeDensity: string;
  temporalConsistency: string;
  audioQuality: string;
}

// Main Enhanced Analysis Interface
export interface EnhancedAnalysis {
  final_verdict: "highly_authentic" | "authentic" | "likely_authentic" | "needs_review" | "suspicious" | "inconclusive";
  overall_confidence: string;
  confidence_level: "high" | "medium" | "low";
  
  component_scores: {
    visual_authenticity: string;
    temporal_consistency: string;
    audio_authenticity: string;
    anomaly_count: number;
  };
  
  temporal_details: {
    consistency_score: string;
    stability: "high" | "medium" | "low" | "unstable";
    anomaly_count: number;
    anomaly_timeline: TemporalAnomaly[];
    has_abrupt_changes: boolean;
  };
  
  audio_details: {
    available: boolean;
    verdict?: "likely_authentic" | "probably_authentic" | "needs_review" | "suspicious" | "analysis_error";
    confidence?: string;
    strengths?: string[];
    anomalies?: string[];
    note?: string;
  };
  
  explanations: string[];
  
  samples: {
    frame_samples: FrameSample[];
    anomaly_samples: TemporalAnomaly[];
  };
  
  insights: {
    visual: InsightMetric;
    temporal: TemporalInsight;
    audio: AudioInsight;
  };
  
  // Backward compatibility signals
  signals?: AnalysisSignals;
}

// Full Enhanced API Response
export interface EnhancedAnalysisResults {
  success: boolean;
  session_id: string;
  processing_info: {
    file_name: string;
    file_size: number;
    file_type: string;
    frames_analyzed: number;
    audio_analyzed: boolean;
    analysis_modes: string[];
  };
  analysis: EnhancedAnalysis;
  metadata: {
    analysis_version: string;
    features: string[];
    timestamp: string;
  };
}

// ========== LEGACY TYPES (FOR BACKWARD COMPATIBILITY) ==========
export interface LegacyAnalysisResult {
  humanityScore: number;
  anomalies: Anomaly[];
  verdict: "inconclusive" | "authentic" | "synthetic";
  processingTime: number;
  // Optional legacy fields
  confidence?: number;
  textureVariance?: number;
  edgeDensity?: number;
}

// ========== MAIN POST-HOC ANALYSIS TYPE (UPDATED) ==========
export interface PostHocAnalysis {
  id: string;
  fileName: string;
  fileType: "video" | "audio";
  status: "uploading" | "processing" | "complete" | "error";
  progress: number;
  
  // Can store EITHER legacy OR enhanced results
  results?: EnhancedAnalysis | LegacyAnalysisResult;
  
  // Store the full enhanced response (optional, for debugging)
  enhancedResults?: EnhancedAnalysisResults;
  
  // Error handling
  error?: string;
}

// ========== HELPER TYPES ==========
export type VerdictType = 
  | "highly_authentic" 
  | "authentic" 
  | "likely_authentic" 
  | "needs_review" 
  | "suspicious" 
  | "inconclusive" 
  | "synthetic";

// ========== ZOD SCHEMAS (EXISTING) ==========
export const uploadFileSchema = z.object({
  file: z.any(),
});

export type UploadFileInput = z.infer<typeof uploadFileSchema>;

export const insertUserSchema = z.object({
  username: z.string(),
  password: z.string(),
});

export type InsertUser = z.infer<typeof insertUserSchema>;

// ========== UI/OTHER TYPES (EXISTING) ==========
export interface UseCase {
  id: string;
  title: string;
  description: string;
  icon: string;
}

export interface TechStackItem {
  name: string;
  description: string;
  category: "detection" | "analysis" | "security";
}

export interface User {
  id: string;
  username: string;
  password: string;
}

// ========== EXPORTS (EXISTING) ==========
export const users = {} as any;