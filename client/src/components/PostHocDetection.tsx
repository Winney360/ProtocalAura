import { useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Upload, FileVideo, FileAudio, AlertCircle, CheckCircle2, XCircle, Clock, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import type { PostHocAnalysis, Anomaly } from "@shared/schema";

export default function PostHocDetection() {
  const [analyses, setAnalyses] = useState<PostHocAnalysis[]>([]);
  const [isDragging, setIsDragging] = useState(false);

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

        const response = await fetch("/api/analyze-media", {
          method: "POST",
          body: formData
        });

        if (!response.ok) throw new Error("Upload failed");

        setAnalyses(prev => 
          prev.map(a => a.id === id ? { ...a, status: "processing", progress: 20 } : a)
        );

        const progressInterval = setInterval(() => {
          setAnalyses(prev => 
            prev.map(a => {
              if (a.id === id && a.status === "processing" && a.progress < 90) {
                return { ...a, progress: a.progress + Math.random() * 15 };
              }
              return a;
            })
          );
        }, 500);

        const pollResult = async () => {
          try {
            const response = await fetch(`/api/analysis-result/${id}`);
            if (!response.ok) {
              setTimeout(pollResult, 1000);
              return;
            }
            const result = await response.json();
            
            if (result.status === "complete") {
              clearInterval(progressInterval);
              setAnalyses(prev => 
                prev.map(a => a.id === id ? {
                  ...a,
                  status: "complete",
                  progress: 100,
                  results: result.results
                } : a)
              );
            } else if (result.status === "error") {
              clearInterval(progressInterval);
              setAnalyses(prev => 
                prev.map(a => a.id === id ? { ...a, status: "error", progress: 0 } : a)
              );
            } else {
              setTimeout(pollResult, 1000);
            }
          } catch {
            setTimeout(pollResult, 1000);
          }
        };

        setTimeout(pollResult, 2000);

      } catch {
        setAnalyses(prev => 
          prev.map(a => a.id === id ? { ...a, status: "error", progress: 0 } : a)
        );
      }
    }
  };

  const removeAnalysis = (id: string) => {
    setAnalyses(prev => prev.filter(a => a.id !== id));
  };

  const getSeverityColor = (severity: Anomaly["severity"]) => {
    switch (severity) {
      case "high": return "text-red-400 bg-red-400/10";
      case "medium": return "text-yellow-400 bg-yellow-400/10";
      default: return "text-blue-400 bg-blue-400/10";
    }
  };

  const getVerdictDisplay = (verdict: "authentic" | "synthetic" | "inconclusive") => {
    switch (verdict) {
      case "authentic":
        return { icon: <CheckCircle2 className="w-5 h-5" />, color: "text-green-400", label: "Authentic" };
      case "synthetic":
        return { icon: <XCircle className="w-5 h-5" />, color: "text-red-400", label: "Synthetic Detected" };
      default:
        return { icon: <AlertCircle className="w-5 h-5" />, color: "text-yellow-400", label: "Inconclusive" };
    }
  };

  return (
    <section id="post-hoc" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-primary font-mono text-sm tracking-wider mb-4 block">POST-HOC ANALYSIS</span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-post-hoc-title">AI Detection</h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Upload video or audio files for deepfake and synthetic media analysis. 
            <span className="text-primary font-medium"> Simulated results</span> for demonstration.
          </p>
        </motion.div>

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
              <div className="w-16 h-16 rounded-md bg-primary/10 flex items-center justify-center mx-auto mb-4">
                <Upload className={`w-8 h-8 transition-colors ${isDragging ? "text-primary" : "text-muted-foreground"}`} />
              </div>
              <h3 className="font-semibold mb-2">Upload Media for Analysis</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Drag and drop video or audio files, or click to browse
              </p>
              <div className="flex items-center justify-center gap-4 mb-4">
                <Badge variant="secondary" className="gap-1">
                  <FileVideo className="w-3 h-3" /> Video
                </Badge>
                <Badge variant="secondary" className="gap-1">
                  <FileAudio className="w-3 h-3" /> Audio
                </Badge>
              </div>
              <input
                type="file"
                accept="video/*,audio/*"
                multiple
                className="hidden"
                id="file-upload"
                onChange={handleFileSelect}
                data-testid="input-file-upload"
              />
              <Button asChild data-testid="button-browse-files">
                <label htmlFor="file-upload" className="cursor-pointer">
                  Browse Files
                </label>
              </Button>
            </div>
          </Card>
        </motion.div>

        <AnimatePresence mode="popLayout">
          {analyses.length > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: "auto" }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-8 space-y-4"
            >
              {analyses.map((analysis) => (
                <motion.div
                  key={analysis.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  layout
                >
                  <Card className="p-4 bg-card/50 backdrop-blur-sm" data-testid={`card-analysis-${analysis.id}`}>
                    <div className="flex items-start justify-between mb-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center text-primary">
                          {analysis.fileType === "video" ? (
                            <FileVideo className="w-5 h-5" />
                          ) : (
                            <FileAudio className="w-5 h-5" />
                          )}
                        </div>
                        <div>
                          <h4 className="font-medium text-sm truncate max-w-[200px]">{analysis.fileName}</h4>
                          <p className="text-xs text-muted-foreground capitalize">{analysis.fileType} file</p>
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        {analysis.status === "complete" && analysis.results && (
                          <div className={`flex items-center gap-1 ${getVerdictDisplay(analysis.results.verdict).color}`}>
                            {getVerdictDisplay(analysis.results.verdict).icon}
                            <span className="text-sm font-medium">{getVerdictDisplay(analysis.results.verdict).label}</span>
                          </div>
                        )}
                        <Button
                          size="icon"
                          variant="ghost"
                          className="text-muted-foreground"
                          onClick={() => removeAnalysis(analysis.id)}
                          data-testid={`button-remove-${analysis.id}`}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </div>
                    </div>

                    {(analysis.status === "uploading" || analysis.status === "processing") && (
                      <div className="mb-4">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs text-muted-foreground">
                            {analysis.status === "uploading" ? "Uploading..." : "Analyzing..."}
                          </span>
                          <span className="text-xs text-primary">{Math.round(analysis.progress)}%</span>
                        </div>
                        <Progress value={analysis.progress} className="h-2" />
                      </div>
                    )}

                    {analysis.status === "complete" && analysis.results && (
                      <div className="space-y-4">
                        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                          <div className="glass-effect p-3 rounded-md">
                            <span className="text-xs text-muted-foreground block mb-1">Authenticity Score</span>
                            <span className="text-xl font-bold gradient-text" data-testid={`score-${analysis.id}`}>
                              {analysis.results.humanityScore}%
                            </span>
                          </div>
                          <div className="glass-effect p-3 rounded-md">
                            <span className="text-xs text-muted-foreground block mb-1">Anomalies Found</span>
                            <span className="text-xl font-bold text-foreground">
                              {analysis.results.anomalies.length}
                            </span>
                          </div>
                          <div className="glass-effect p-3 rounded-md">
                            <span className="text-xs text-muted-foreground block mb-1 flex items-center gap-1">
                              <Clock className="w-3 h-3" /> Processing Time
                            </span>
                            <span className="text-xl font-bold text-foreground">
                              {analysis.results.processingTime}s
                            </span>
                          </div>
                        </div>

                        {analysis.results.anomalies.length > 0 && (
                          <div>
                            <h5 className="text-sm font-medium mb-2">Detected Anomalies</h5>
                            <div className="space-y-2 max-h-40 overflow-y-auto">
                              {analysis.results.anomalies.map((anomaly, i) => (
                                <div 
                                  key={i} 
                                  className="flex items-center justify-between p-2 rounded-md bg-muted/30"
                                  data-testid={`anomaly-${analysis.id}-${i}`}
                                >
                                  <div className="flex items-center gap-2">
                                    <Badge variant="secondary" className={getSeverityColor(anomaly.severity)}>
                                      {anomaly.severity}
                                    </Badge>
                                    <span className="text-sm">{anomaly.description}</span>
                                  </div>
                                  <span className="text-xs text-muted-foreground font-mono">
                                    {anomaly.timestamp.toFixed(1)}s
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {analysis.status === "error" && (
                      <div className="flex items-center gap-2 text-red-400">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-sm">Analysis failed. Please try again.</span>
                      </div>
                    )}

                    <Badge 
                      variant="secondary" 
                      className="mt-4 bg-primary/10 text-primary border-primary/20"
                    >
                      SIMULATED ANALYSIS
                    </Badge>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
