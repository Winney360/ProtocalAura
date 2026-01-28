// File: client/src/components/PostHocDetection.tsx (UPDATED with Temporal, Audio, Explainability)
import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Upload, FileVideo, FileAudio, AlertCircle, CheckCircle2, 
  XCircle, Clock, Trash2, BarChart3, Activity, Volume2,
   AlertTriangle, Info, ChevronDown, ChevronUp,
  Shield, Cpu, Eye, Headphones, Zap
} from "lucide-react";
import { Button } from "../components/ui/button";
import { Card } from "../components/ui/card";
import { Badge } from "../components/ui/badge";
import { Progress } from "../components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "../components/ui/collapsible";
import type { PostHocAnalysis, Anomaly, EnhancedAnalysis,  LegacyAnalysisResult  } from "../../../shared/schema";

// Extended type for new features
interface EnhancedAnalysisResults {
  success: boolean;
  session_id: string;
  analysis: {
    final_verdict: string;
    overall_confidence: number;
    confidence_level: 'high' | 'medium' | 'low';
    component_scores: {
      visual_authenticity: string;
      temporal_consistency: string;
      audio_authenticity: string;
      anomaly_count: number;
    };
    temporal_details: {
      consistency_score: string;
      stability: 'high' | 'medium' | 'low' | 'unstable';
      anomaly_count: number;
      anomaly_timeline: Array<{
        frame: number;
        reason: string;
        score: string;
        timestamp: string;
      }>;
      has_abrupt_changes: boolean;
    };
    audio_details: {
      available: boolean;
      verdict?: string;
      confidence?: string;
      strengths?: string[];
      anomalies?: string[];
      note?: string;
    };
    explanations: string[];
    samples: {
      frame_samples: Array<{
        frame: number;
        score: string;
        verdict: string;
        has_anomaly: boolean;
      }>;
      anomaly_samples: Array<{
        frame: number;
        reason: string;
        score: string;
        timestamp: string;
      }>;
    };
    insights: {
      visual: {
        score: string;
        assessment: string;
        confidence: string;
      };
      temporal: {
        consistency_score: string;
        stability: string;
        anomalies_detected: number;
        assessment: string;
      };
      audio: {
        available: boolean;
        verdict: string;
        confidence: string;
        assessment: string;
      };
    };
    signals?: {
      humanityScore: string;
      confidence: string;
      textureVariance: string;
      edgeDensity: string;
      temporalConsistency: string;
      audioQuality: string;
    };
  };
  metadata: {
    analysis_version: string;
    features: string[];
    timestamp: string;
  };
}

export default function PostHocDetection() {
  const [analyses, setAnalyses] = useState<PostHocAnalysis[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [expandedAnalysis, setExpandedAnalysis] = useState<string | null>(null);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = Array.from(e.dataTransfer.files);
    await processFiles(files);
  }, []);

  const handleFileSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    await processFiles(files);
  }, []);

  const processFiles = async (files: File[]) => {
    for (const file of files) {
      const isVideo = file.type.startsWith("video/");
      const isAudio = file.type.startsWith("audio/");
      
      if (!isVideo && !isAudio) continue;

      const id = `analysis-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      const newAnalysis: PostHocAnalysis = {
        id,
        fileName: file.name,
        fileType: isVideo ? "video" : "audio",
        status: "uploading",
        progress: 0
      };

      setAnalyses(prev => [...prev, newAnalysis]);

      try {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("analysisId", id);

        setAnalyses(prev => 
          prev.map(a => a.id === id ? { ...a, status: "processing", progress: 20 } : a)
        );

        // Simulate processing with enhanced progress
        const progressInterval = setInterval(() => {
          setAnalyses(prev => 
            prev.map(a => {
              if (a.id === id && a.status === "processing" && a.progress < 95) {
                const increment = 5 + Math.random() * 15;
                return { ...a, progress: Math.min(a.progress + increment, 95) };
              }
              return a;
            })
          );
        }, 800);

        // REAL API call to your Node.js server
        try {
          const response = await fetch("/api/analyze-media", {
            method: "POST",
            body: formData
          });

          if (!response.ok) throw new Error("Upload failed");

          const result = await response.json();
          
          clearInterval(progressInterval);
          
          if (result.success) {
            setAnalyses(prev => 
              prev.map(a => a.id === id ? {
                ...a,
                status: "complete",
                progress: 100,
                results: result.analysis, // Updated to match new response structure
                enhancedResults: result // Store enhanced results
              } : a)
            );
          } else {
            throw new Error(result.error || "Analysis failed");
          }
        } catch (apiError: any) {
          clearInterval(progressInterval);
          throw apiError;
        }

      } catch (error: any) {
        setAnalyses(prev => 
          prev.map(a => a.id === id ? { 
            ...a, 
            status: "error", 
            progress: 0,
            error: error.message 
          } : a)
        );
      }
    }
  };

  const removeAnalysis = (id: string) => {
    setAnalyses(prev => prev.filter(a => a.id !== id));
    if (expandedAnalysis === id) {
      setExpandedAnalysis(null);
    }
  };

  const toggleExpand = (id: string) => {
    setExpandedAnalysis(expandedAnalysis === id ? null : id);
  };

  const getSeverityColor = (severity: Anomaly["severity"]) => {
    switch (severity) {
      case "high": return "text-red-400 bg-red-400/10";
      case "medium": return "text-yellow-400 bg-yellow-400/10";
      default: return "text-blue-400 bg-blue-400/10";
    }
  };

  const getVerdictDisplay = (verdict: string | undefined) => {
  const safeVerdict = verdict || "inconclusive";
  
  switch (safeVerdict) {
    case "highly_authentic":
    case "authentic":
    case "likely_real":
    case "likely_authentic":
      return { 
        icon: <CheckCircle2 className="w-5 h-5" />, 
        color: "text-green-400", 
        label: "Authentic",
        bg: "bg-green-400/10",
        border: "border-green-400/20"
      };
    case "suspicious":
    case "synthetic":
      return { 
        icon: <XCircle className="w-5 h-5" />, 
        color: "text-red-400", 
        label: "Synthetic Detected",
        bg: "bg-red-400/10",
        border: "border-red-400/20"
      };
    case "needs_review":
    case "inconclusive":
      return { 
        icon: <AlertCircle className="w-5 h-5" />, 
        color: "text-yellow-400", 
        label: "Needs Review",
        bg: "bg-yellow-400/10",
        border: "border-yellow-400/20"
      };
    default:
      return { 
        icon: <AlertCircle className="w-5 h-5" />, 
        color: "text-yellow-400", 
        label: "Analysis Required",
        bg: "bg-yellow-400/10",
        border: "border-yellow-400/20"
      };
  }
};

  const getStabilityColor = (stability: string) => {
    switch (stability) {
      case "high": return "text-green-400";
      case "medium": return "text-yellow-400";
      case "low": return "text-orange-400";
      case "unstable": return "text-red-400";
      default: return "text-gray-400";
    }
  };

  const getConfidenceColor = (level: string) => {
    switch (level) {
      case "high": return "text-green-400";
      case "medium": return "text-yellow-400";
      case "low": return "text-red-400";
      default: return "text-gray-400";
    }
  };

  const renderAnalysisDetails = (analysis: any) => {
    const results = analysis.results as EnhancedAnalysisResults['analysis'];
    const verdict = getVerdictDisplay(results?.final_verdict || "inconclusive");
    
    return (
      <div className="space-y-6">
        {/* Overall Verdict Card */}
        <Card className={`p-4 border-2 ${verdict.border} ${verdict.bg}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-md bg-white/10">
                {verdict.icon}
              </div>
              <div>
                <h4 className="font-bold text-lg">{verdict.label}</h4>
                <p className="text-sm text-muted-foreground">
                  Overall Confidence: <span className={getConfidenceColor(results?.confidence_level || 'medium')}>
                    {results?.confidence_level?.toUpperCase() || 'MEDIUM'}
                  </span> ({parseFloat(String (results?.overall_confidence || 0)).toFixed(1)}%)
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-sm text-muted-foreground">Analysis ID</p>
              <p className="text-xs font-mono text-muted-foreground">{analysis.id.slice(0, 8)}...</p>
            </div>
          </div>
        </Card>

        {/* Analysis Tabs */}
        <Tabs defaultValue="overview" className="w-full">
          <TabsList className="grid grid-cols-4 w-full">
            <TabsTrigger value="overview" className="flex items-center gap-2">
              <Eye className="w-4 h-4" /> Overview
            </TabsTrigger>
            <TabsTrigger value="temporal" className="flex items-center gap-2">
              <Activity className="w-4 h-4" /> Temporal
            </TabsTrigger>
            <TabsTrigger value="audio" className="flex items-center gap-2">
              <Headphones className="w-4 h-4" /> Audio
            </TabsTrigger>
            <TabsTrigger value="explainability" className="flex items-center gap-2">
              <Info className="w-4 h-4" /> Why?
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4">
            {/* Component Scores */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Eye className="w-4 h-4 text-blue-400" />
                  <span className="text-xs text-muted-foreground">Visual</span>
                </div>
                <div className="text-xl font-bold">{results?.component_scores?.visual_authenticity || "0.0"}</div>
                <div className="text-xs text-muted-foreground">
                  {results?.insights?.visual?.assessment || "Not analyzed"}
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-purple-400" />
                  <span className="text-xs text-muted-foreground">Temporal</span>
                </div>
                <div className="text-xl font-bold">{results?.component_scores?.temporal_consistency || "0.0"}</div>
                <div className="text-xs text-muted-foreground">
                  {results?.insights?.temporal?.assessment || "Not analyzed"}
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Headphones className="w-4 h-4 text-green-400" />
                  <span className="text-xs text-muted-foreground">Audio</span>
                </div>
                <div className="text-xl font-bold">
                  {results?.audio_details?.available ? 
                    (results?.component_scores?.audio_authenticity || "0.0") : "N/A"}
                </div>
                <div className="text-xs text-muted-foreground">
                  {results?.audio_details?.available ? 
                    (results?.insights?.audio?.assessment || "Not analyzed") : "No audio"}
                </div>
              </Card>

              <Card className="p-4">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-4 h-4 text-red-400" />
                  <span className="text-xs text-muted-foreground">Anomalies</span>
                </div>
                <div className="text-xl font-bold">{results?.component_scores?.anomaly_count || 0}</div>
                <div className="text-xs text-muted-foreground">
                  {results?.component_scores?.anomaly_count > 0 ? "Issues detected" : "No anomalies"}
                </div>
              </Card>
            </div>

            {/* Signal Metrics */}
            {results?.signals && (
              <Card className="p-4">
                <h5 className="font-medium mb-3 flex items-center gap-2">
                  <Cpu className="w-4 h-4" /> Signal Metrics
                </h5>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <span className="text-xs text-muted-foreground">Humanity Score</span>
                    <div className="text-lg font-semibold">{results.signals.humanityScore}</div>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Confidence</span>
                    <div className="text-lg font-semibold">{results.signals.confidence}</div>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground">Temporal Consistency</span>
                    <div className="text-lg font-semibold">{results.signals.temporalConsistency}</div>
                  </div>
                </div>
              </Card>
            )}

            {/* Frame Samples */}
            {results?.samples?.frame_samples && results.samples.frame_samples.length > 0 && (
              <Card className="p-4">
                <h5 className="font-medium mb-3 flex items-center gap-2">
                  <BarChart3 className="w-4 h-4" /> Frame Analysis Samples
                </h5>
                <div className="space-y-2">
                  {results.samples.frame_samples.slice(0, 3).map((frame, idx) => (
                    <div key={idx} className="flex items-center justify-between p-2 rounded-md bg-muted/30">
                      <div className="flex items-center gap-3">
                        <Badge variant={frame.has_anomaly ? "destructive" : "secondary"}>
                          Frame {frame.frame}
                        </Badge>
                        <span className="text-sm">Score: {frame.score}</span>
                      </div>
                      <Badge variant={frame.verdict === "likely_real" ? "default" : "outline"}>
                        {frame.verdict}
                      </Badge>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="temporal" className="space-y-4">
            <Card className="p-4">
              <h5 className="font-medium mb-3 flex items-center gap-2">
                <Activity className="w-4 h-4 text-purple-400" /> Temporal Consistency Analysis
              </h5>
              
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <span className="text-xs text-muted-foreground">Consistency Score</span>
                  <div className={`text-2xl font-bold ${getStabilityColor(results?.temporal_details?.stability || '')}`}>
                    {results?.temporal_details?.consistency_score || "0.0"}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    Stability: <span className={getStabilityColor(results?.temporal_details?.stability || '')}>
                      {results?.temporal_details?.stability?.toUpperCase() || "UNKNOWN"}
                    </span>
                  </div>
                </div>
                <div>
                  <span className="text-xs text-muted-foreground">Anomalies Detected</span>
                  <div className={`text-2xl font-bold ${results?.temporal_details?.anomaly_count > 0 ? 'text-red-400' : 'text-green-400'}`}>
                    {results?.temporal_details?.anomaly_count || 0}
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {results?.temporal_details?.has_abrupt_changes ? "Abrupt changes detected" : "No abrupt changes"}
                  </div>
                </div>
              </div>

              {/* Anomaly Timeline */}
              {results?.temporal_details?.anomaly_timeline && results.temporal_details.anomaly_timeline.length > 0 ? (
                <div>
                  <h6 className="text-sm font-medium mb-2">Anomaly Timeline</h6>
                  <div className="space-y-2 max-h-60 overflow-y-auto">
                    {results.temporal_details.anomaly_timeline.map((anomaly, idx) => (
                      <div key={idx} className="flex items-center justify-between p-2 rounded-md bg-red-400/10">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="w-4 h-4 text-red-400" />
                          <div>
                            <div className="text-sm">Frame {anomaly.frame}</div>
                            <div className="text-xs text-muted-foreground">{anomaly.reason}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-mono">Score: {anomaly.score}</div>
                          <div className="text-xs text-muted-foreground">{anomaly.timestamp}</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  <CheckCircle2 className="w-8 h-8 mx-auto mb-2 text-green-400" />
                  No temporal anomalies detected
                </div>
              )}
            </Card>
          </TabsContent>

          <TabsContent value="audio" className="space-y-4">
            <Card className="p-4">
              <h5 className="font-medium mb-3 flex items-center gap-2">
                <Headphones className="w-4 h-4 text-green-400" /> Audio Authenticity Analysis
              </h5>
              
              {results?.audio_details?.available ? (
                <>
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div>
                      <span className="text-xs text-muted-foreground">Audio Verdict</span>
                      <div className={`text-2xl font-bold ${results.audio_details.verdict === 'likely_authentic' ? 'text-green-400' : 'text-red-400'}`}>
                        {results.audio_details.verdict?.replace('_', ' ').toUpperCase() || "UNKNOWN"}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        Confidence: {results.audio_details.confidence || "0.0"}
                      </div>
                    </div>
                    <div>
                      <span className="text-xs text-muted-foreground">Analysis</span>
                      <div className="text-lg font-semibold">
                        {results.insights?.audio?.assessment?.toUpperCase() || "NOT ANALYZED"}
                      </div>
                      <div className="text-xs text-muted-foreground">
                        {results.audio_details.anomalies?.length || 0} anomalies found
                      </div>
                    </div>
                  </div>

                  {/* Audio Strengths */}
                  {results.audio_details.strengths && results.audio_details.strengths.length > 0 && (
                    <div className="mb-3">
                      <h6 className="text-sm font-medium mb-2">Audio Strengths</h6>
                      <div className="flex flex-wrap gap-2">
                        {results.audio_details.strengths.map((strength, idx) => (
                          <Badge key={idx} variant="secondary" className="bg-green-400/10 text-green-400">
                            <CheckCircle2 className="w-3 h-3 mr-1" /> {strength}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Audio Anomalies */}
                  {results.audio_details.anomalies && results.audio_details.anomalies.length > 0 && (
                    <div>
                      <h6 className="text-sm font-medium mb-2">Audio Anomalies</h6>
                      <div className="space-y-2">
                        {results.audio_details.anomalies.map((anomaly, idx) => (
                          <div key={idx} className="flex items-start gap-2 p-2 rounded-md bg-yellow-400/10">
                            <AlertTriangle className="w-4 h-4 text-yellow-400 mt-0.5 shrink-0" />
                            <span className="text-sm">{anomaly}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-8">
                  <Headphones className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                  <p className="text-muted-foreground">No audio stream detected in this file</p>
                  <p className="text-sm text-muted-foreground mt-1">
                    {results?.audio_details?.note || "Audio analysis not available"}
                  </p>
                </div>
              )}
            </Card>
          </TabsContent>

          <TabsContent value="explainability" className="space-y-4">
            <Card className="p-4">
              <h5 className="font-medium mb-3 flex items-center gap-2">
                <Info className="w-4 h-4" /> Why This Verdict?
              </h5>
              
              {results?.explanations && results.explanations.length > 0 ? (
                <div className="space-y-3">
                  {results.explanations.map((explanation, idx) => {
                    const isPositive = explanation.includes("✅") || explanation.includes("High") || explanation.includes("Excellent");
                    const isWarning = explanation.includes("⚠️") || explanation.includes("Moderate") || explanation.includes("Good");
                    const isNegative = explanation.includes("❌") || explanation.includes("Low") || explanation.includes("Poor");
                    
                    let Icon = CheckCircle2;
                    let bgColor = "bg-green-400/10";
                    let textColor = "text-green-400";
                    
                    if (isWarning) {
                      Icon = AlertTriangle;
                      bgColor = "bg-yellow-400/10";
                      textColor = "text-yellow-400";
                    } else if (isNegative) {
                      Icon = XCircle;
                      bgColor = "bg-red-400/10";
                      textColor = "text-red-400";
                    }
                    
                    return (
                      <div key={idx} className={`flex items-start gap-3 p-3 rounded-md ${bgColor}`}>
                        <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${textColor}`} />
                        <span className="text-sm">{explanation.replace(/[✅⚠️❌]/g, '').trim()}</span>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <div className="text-center py-4 text-muted-foreground">
                  No detailed explanations available
                </div>
              )}

              {/* Features Used */}
              {analysis.enhancedResults?.metadata?.features && (
                <div className="mt-4 pt-4 border-t">
                  <h6 className="text-sm font-medium mb-2 flex items-center gap-2">
                    <Zap className="w-4 h-4" /> Analysis Features Used
                  </h6>
                  <div className="flex flex-wrap gap-2">
                    {analysis.enhancedResults.metadata.features.map((feature: string, idx: number) => (
                      <Badge key={idx} variant="outline" className="text-xs">
                        {feature.replace('_', ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    );
  };

  return (
    <section id="post-hoc" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-primary font-mono text-sm tracking-wider mb-4 block">
            PROTOCOL AURA v2.0
          </span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4">
            AI-Powered Media Authenticity Verification
          </h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Advanced deepfake detection with temporal analysis, audio verification, 
            and explainable AI. Upload videos for comprehensive authenticity analysis.
          </p>
          <div className="flex flex-wrap justify-center gap-4 mt-6">
            <Badge variant="secondary" className="gap-1">
              <Activity className="w-3 h-3" /> Temporal Analysis
            </Badge>
            <Badge variant="secondary" className="gap-1">
              <Headphones className="w-3 h-3" /> Audio Verification
            </Badge>
            <Badge variant="secondary" className="gap-1">
              <Info className="w-3 h-3" /> Explainable AI
            </Badge>
            <Badge variant="secondary" className="gap-1">
              <Shield className="w-3 h-3" /> Real Analysis
            </Badge>
          </div>
        </motion.div>

        {/* Upload Area */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <Card 
            className={`p-8 border-2 border-dashed transition-colors ${
              isDragging ? "border-primary bg-primary/5" : "border-border"
            }`}
            onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <div className="text-center">
              <div className="w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center mx-auto mb-6">
                <Upload className={`w-10 h-10 transition-colors ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <h3 className="font-semibold text-xl mb-3">Upload Media for Analysis</h3>
              <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                Drag and drop video files for comprehensive authenticity analysis with 
                temporal consistency checks and audio verification.
              </p>
              <div className="flex items-center justify-center gap-4 mb-6">
                <Badge variant="secondary" className="gap-2 py-2">
                  <FileVideo className="w-4 h-4" /> Video Files
                </Badge>
                <Badge variant="secondary" className="gap-2 py-2">
                  <BarChart3 className="w-4 h-4" /> Frame Analysis
                </Badge>
                <Badge variant="secondary" className="gap-2 py-2">
                  <Volume2 className="w-4 h-4" /> Audio Analysis
                </Badge>
              </div>
              <input
                type="file"
                accept="video/*"
                multiple
                className="hidden"
                id="file-upload"
                onChange={handleFileSelect}
              />
              <Button size="lg" asChild>
                <label htmlFor="file-upload" className="cursor-pointer gap-2">
                  <Upload className="w-4 h-4" />
                  Browse Video Files
                </label>
              </Button>
              <p className="text-xs text-muted-foreground mt-4">
                Supports MP4, MOV, AVI, WEBM formats with audio
              </p>
            </div>
          </Card>
        </motion.div>

        {/* Analysis Results */}
        <AnimatePresence mode="popLayout">
          {analyses.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-8 space-y-4"
            >
              <h3 className="text-xl font-bold">Analysis Results ({analyses.length})</h3>
              
              {analyses.map((analysis) => (
                <motion.div
                  key={analysis.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  layout
                >
                  <Collapsible 
                    open={expandedAnalysis === analysis.id}
                    onOpenChange={() => toggleExpand(analysis.id)}
                    className="w-full"
                  >
                    <Card className="overflow-hidden">
                      {/* Analysis Header */}
                      <div className="p-4 bg-card/50 border-b">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center text-primary">
                              {analysis.fileType === "video" ? (
                                <FileVideo className="w-6 h-6" />
                              ) : (
                                <FileAudio className="w-6 h-6" />
                              )}
                            </div>
                            <div>
                              <h4 className="font-semibold truncate max-w-xs">{analysis.fileName}</h4>
                              <p className="text-sm text-muted-foreground capitalize">
                                {analysis.fileType} • {analysis.status}
                              </p>
                            </div>
                          </div>
                          
                          <div className="flex items-center gap-3">
                           {analysis.status === "complete" && analysis.results && (
                                <div className={`flex items-center gap-2 px-3 py-1 rounded-full ${
                                  getVerdictDisplay(
                                    // Safely access final_verdict from enhanced results OR verdict from legacy
                                    (analysis.results as EnhancedAnalysis)?.final_verdict || 
                                    (analysis.results as LegacyAnalysisResult)?.verdict
                                  ).bg
                                }`}>
                                  {getVerdictDisplay(
                                    (analysis.results as EnhancedAnalysis)?.final_verdict || 
                                    (analysis.results as LegacyAnalysisResult)?.verdict
                                  ).icon}
                                  <span className={`font-medium ${
                                    getVerdictDisplay(
                                      (analysis.results as EnhancedAnalysis)?.final_verdict || 
                                      (analysis.results as LegacyAnalysisResult)?.verdict
                                    ).color
                                  }`}>
                                    {getVerdictDisplay(
                                      (analysis.results as EnhancedAnalysis)?.final_verdict || 
                                      (analysis.results as LegacyAnalysisResult)?.verdict
                                    ).label}
                                  </span>
                              </div>
                            )}
                            
                            <CollapsibleTrigger asChild>
                              <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                                {expandedAnalysis === analysis.id ? (
                                  <ChevronUp className="h-4 w-4" />
                                ) : (
                                  <ChevronDown className="h-4 w-4" />
                                )}
                              </Button>
                            </CollapsibleTrigger>
                            
                            <Button
                              size="sm"
                              variant="ghost"
                              className="text-muted-foreground h-8 w-8 p-0"
                              onClick={() => removeAnalysis(analysis.id)}
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>

                        {/* Progress Bar */}
                        {(analysis.status === "uploading" || analysis.status === "processing") && (
                          <div className="mt-4">
                            <div className="flex items-center justify-between mb-2">
                              <span className="text-sm text-muted-foreground">
                                {analysis.status === "uploading" ? "Uploading..." : "Analyzing..."}
                              </span>
                              <span className="text-sm font-medium">{Math.round(analysis.progress)}%</span>
                            </div>
                            <Progress value={analysis.progress} className="h-2" />
                          </div>
                        )}
                      </div>

                      {/* Analysis Details */}
                      <CollapsibleContent>
                        <div className="p-4">
                          {analysis.status === "complete" && analysis.results ? (
                            renderAnalysisDetails(analysis)
                          ) : analysis.status === "error" ? (
                            <div className="flex items-center gap-3 p-4 rounded-md bg-red-400/10 text-red-400">
                              <AlertCircle className="w-5 h-5 shrink-0" />
                              <div>
                                <p className="font-medium">Analysis Failed</p>
                                <p className="text-sm">{analysis.error || "Please try again"}</p>
                              </div>
                            </div>
                          ) : (
                            <div className="text-center py-8 text-muted-foreground">
                              <Clock className="w-8 h-8 mx-auto mb-3" />
                              <p>Analysis in progress...</p>
                            </div>
                          )}
                        </div>
                      </CollapsibleContent>
                    </Card>
                  </Collapsible>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}