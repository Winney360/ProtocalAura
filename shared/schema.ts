import { z } from "zod";

export interface LivenessSignals {
  biologicalPulse: number;
  microExpressionScore: number;
  humanityScore: number;
  verificationStatus: "verifying" | "verified" | "unverified" | "suspicious";
  timestamp: number;
}

export interface PostHocAnalysis {
  id: string;
  fileName: string;
  fileType: "video" | "audio";
  status: "uploading" | "processing" | "complete" | "error";
  progress: number;
  results?: {
    humanityScore: number;
    anomalies: Anomaly[];
    verdict: "authentic" | "synthetic" | "inconclusive";
    processingTime: number;
  };
}

export interface Anomaly {
  timestamp: number;
  type: "face_inconsistency" | "lip_sync" | "frame_artifact" | "audio_pattern" | "pitch_anomaly";
  severity: "low" | "medium" | "high";
  description: string;
}

export const uploadFileSchema = z.object({
  file: z.any(),
});

export type UploadFileInput = z.infer<typeof uploadFileSchema>;

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

export const users = {} as any;
export const insertUserSchema = z.object({
  username: z.string(),
  password: z.string(),
});
export type InsertUser = z.infer<typeof insertUserSchema>;
export type User = { id: string; username: string; password: string };
