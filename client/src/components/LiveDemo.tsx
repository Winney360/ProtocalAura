import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Camera, CameraOff, Activity, Heart, ShieldCheck, AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import type { LivenessSignals } from "@shared/schema";
import { io, Socket } from "socket.io-client";

interface LiveDemoProps {
  isActive: boolean;
  onClose: () => void;
}

export default function LiveDemo({ isActive, onClose }: LiveDemoProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [socket, setSocket] = useState<Socket | null>(null);
  const [signals, setSignals] = useState<LivenessSignals>({
    biologicalPulse: 0,
    microExpressionScore: 0,
    humanityScore: 0,
    verificationStatus: "verifying",
    timestamp: Date.now()
  });

  const startCamera = useCallback(async () => {
    setIsConnecting(true);
    setCameraError(null);
    
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setStream(mediaStream);
      
      const newSocket = io(window.location.origin);
      setSocket(newSocket);
      
      newSocket.on("connect", () => {
        newSocket.emit("start-liveness");
      });
      
      newSocket.on("liveness-signals", (data: LivenessSignals) => {
        setSignals(data);
      });
      
    } catch (err) {
      setCameraError("Camera access denied. Please enable camera permissions.");
    } finally {
      setIsConnecting(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (socket) {
      socket.emit("stop-liveness");
      socket.disconnect();
      setSocket(null);
    }
    setSignals({
      biologicalPulse: 0,
      microExpressionScore: 0,
      humanityScore: 0,
      verificationStatus: "verifying",
      timestamp: Date.now()
    });
  }, [stream, socket]);

  useEffect(() => {
    if (isActive && !stream) {
      startCamera();
    }
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      if (socket) {
        socket.disconnect();
      }
    };
  }, [isActive]);

  const getStatusColor = (status: LivenessSignals["verificationStatus"]) => {
    switch (status) {
      case "verified": return "text-green-400";
      case "suspicious": return "text-red-400";
      case "unverified": return "text-yellow-400";
      default: return "text-primary";
    }
  };

  const getStatusIcon = (status: LivenessSignals["verificationStatus"]) => {
    switch (status) {
      case "verified": return <ShieldCheck className="w-5 h-5" />;
      case "suspicious": return <AlertTriangle className="w-5 h-5" />;
      default: return <RefreshCw className="w-5 h-5 animate-spin" />;
    }
  };

  if (!isActive) return null;

  return (
    <section id="live-demo" className="py-24 px-6">
      <div className="max-w-5xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <span className="text-primary font-mono text-sm tracking-wider mb-4 block">LIVE DEMONSTRATION</span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-live-demo-title">Real-Time Verification</h2>
          <p className="text-muted-foreground max-w-xl mx-auto">
            Experience Protocol Aura's liveness detection. This is a <span className="text-primary font-medium">simulated prototype</span> for demonstration purposes.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
          >
            <Card className="overflow-hidden bg-card/50 backdrop-blur-sm">
              <div className="relative aspect-[4/3] bg-black/50 flex items-center justify-center">
                {cameraError ? (
                  <div className="text-center p-6">
                    <CameraOff className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                    <p className="text-sm text-muted-foreground mb-4">{cameraError}</p>
                    <Button onClick={startCamera} data-testid="button-retry-camera">
                      <Camera className="w-4 h-4 mr-2" />
                      Retry
                    </Button>
                  </div>
                ) : (
                  <>
                    <video
                      ref={videoRef}
                      autoPlay
                      playsInline
                      muted
                      className="w-full h-full object-cover"
                      data-testid="video-webcam-feed"
                    />
                    
                    <AnimatePresence>
                      {stream && (
                        <motion.div
                          initial={{ opacity: 0 }}
                          animate={{ opacity: 1 }}
                          exit={{ opacity: 0 }}
                          className="absolute inset-0 flex items-center justify-center pointer-events-none"
                        >
                          <div className="relative w-48 h-48 md:w-64 md:h-64">
                            <motion.div
                              className="absolute inset-0 rounded-full border-2 border-primary/40"
                              animate={{
                                scale: [1, 1.05, 1],
                                opacity: [0.4, 0.8, 0.4]
                              }}
                              transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: "easeInOut"
                              }}
                            />
                            <motion.div
                              className="absolute inset-2 rounded-full border-2 border-primary/60"
                              animate={{
                                scale: [1, 1.03, 1],
                                opacity: [0.6, 1, 0.6]
                              }}
                              transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: "easeInOut",
                                delay: 0.2
                              }}
                            />
                            <motion.div
                              className="absolute inset-4 rounded-full border-2 border-primary animate-glow"
                              animate={{
                                scale: [1, 1.02, 1]
                              }}
                              transition={{
                                duration: 2,
                                repeat: Infinity,
                                ease: "easeInOut",
                                delay: 0.4
                              }}
                            />
                            
                            <motion.div
                              className="absolute inset-0 overflow-hidden rounded-full"
                              style={{ clipPath: "inset(0 0 50% 0)" }}
                            >
                              <motion.div
                                className="absolute inset-0 bg-gradient-to-b from-primary/30 to-transparent"
                                animate={{ y: ["-100%", "200%"] }}
                                transition={{
                                  duration: 3,
                                  repeat: Infinity,
                                  ease: "linear"
                                }}
                              />
                            </motion.div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    {isConnecting && (
                      <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                        <div className="text-center">
                          <RefreshCw className="w-8 h-8 text-primary animate-spin mx-auto mb-2" />
                          <p className="text-sm text-muted-foreground">Initializing camera...</p>
                        </div>
                      </div>
                    )}
                  </>
                )}

                <Badge 
                  variant="secondary" 
                  className="absolute top-4 left-4 bg-black/60 backdrop-blur-sm border-primary/30"
                >
                  SIMULATION
                </Badge>
              </div>

              <div className="p-4 flex justify-between items-center border-t border-border">
                <div className="flex items-center gap-2">
                  {stream ? (
                    <>
                      <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                      <span className="text-sm text-muted-foreground">Camera Active</span>
                    </>
                  ) : (
                    <>
                      <span className="w-2 h-2 bg-muted rounded-full" />
                      <span className="text-sm text-muted-foreground">Camera Inactive</span>
                    </>
                  )}
                </div>
                <div className="flex gap-2">
                  {stream ? (
                    <Button size="sm" variant="outline" onClick={stopCamera} data-testid="button-stop-camera">
                      <CameraOff className="w-4 h-4 mr-2" />
                      Stop
                    </Button>
                  ) : (
                    <Button size="sm" onClick={startCamera} data-testid="button-start-camera">
                      <Camera className="w-4 h-4 mr-2" />
                      Start
                    </Button>
                  )}
                </div>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="space-y-4"
          >
            <Card className="p-6 bg-card/50 backdrop-blur-sm">
              <div className="flex items-center justify-between mb-6">
                <h3 className="font-semibold text-lg">Verification Status</h3>
                <div className={`flex items-center gap-2 ${getStatusColor(signals.verificationStatus)}`}>
                  {getStatusIcon(signals.verificationStatus)}
                  <span className="font-medium capitalize">{signals.verificationStatus}</span>
                </div>
              </div>

              <div className="mb-6">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-muted-foreground">Humanity Score</span>
                  <span className="text-2xl font-bold gradient-text" data-testid="text-humanity-score">
                    {signals.humanityScore}%
                  </span>
                </div>
                <div className="h-3 bg-muted rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-primary to-[hsl(200,100%,60%)]"
                    initial={{ width: 0 }}
                    animate={{ width: `${signals.humanityScore}%` }}
                    transition={{ duration: 0.5, ease: "easeOut" }}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <MetricCard
                  icon={<Heart className="w-5 h-5" />}
                  label="Biological Pulse"
                  value={`${signals.biologicalPulse} BPM`}
                  testId="metric-biological-pulse"
                />
                <MetricCard
                  icon={<Activity className="w-5 h-5" />}
                  label="Micro-Expression"
                  value={`${signals.microExpressionScore}%`}
                  testId="metric-micro-expression"
                />
              </div>
            </Card>

            <Card className="p-4 bg-primary/5 border-primary/20">
              <p className="text-sm text-muted-foreground">
                <span className="text-primary font-medium">Note:</span> This demonstration uses simulated data to showcase 
                the Protocol Aura interface. Production systems analyze actual biometric signals in real-time.
              </p>
            </Card>

            <Button variant="outline" className="w-full" onClick={onClose} data-testid="button-close-demo">
              Close Demo
            </Button>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function MetricCard({ icon, label, value, testId }: { icon: React.ReactNode; label: string; value: string; testId: string }) {
  return (
    <div className="glass-effect p-4 rounded-md" data-testid={testId}>
      <div className="flex items-center gap-2 mb-2 text-primary">
        {icon}
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <span className="text-lg font-semibold">{value}</span>
    </div>
  );
}
