import { motion } from "framer-motion";
import { Shield, Zap, Eye } from "lucide-react";
import { Button } from "@/components/ui/button";

interface HeroProps {
  onStartDemo: () => void;
}

export default function Hero({ onStartDemo }: HeroProps) {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-br from-background via-background to-[hsl(260,30%,10%)]" />
      
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-primary/5 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-[hsl(280,80%,60%)]/5 rounded-full blur-3xl" />
      </div>

      <div className="absolute inset-0 opacity-10">
        <svg className="w-full h-full" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern id="grid" width="60" height="60" patternUnits="userSpaceOnUse">
              <path d="M 60 0 L 0 0 0 60" fill="none" stroke="currentColor" strokeWidth="0.5" className="text-primary" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>

      <div className="relative z-10 max-w-6xl mx-auto px-6 text-center">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 mb-8 rounded-md bg-primary/10 border border-primary/20">
            <span className="w-2 h-2 bg-primary rounded-full animate-pulse" />
            <span className="text-sm font-medium text-primary" data-testid="text-status-badge">Enterprise Security Protocol</span>
          </div>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.1, ease: "easeOut" }}
          className="text-5xl md:text-7xl font-bold mb-6 tracking-tight"
          data-testid="text-hero-title"
        >
          <span className="text-foreground">Humanity</span>
          <br />
          <span className="gradient-text">Verified</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: "easeOut" }}
          className="text-lg md:text-xl text-muted-foreground max-w-2xl mx-auto mb-10 leading-relaxed"
          data-testid="text-hero-description"
        >
          Advanced liveness verification and AI detection technology. 
          Real-time biological signal analysis meets sophisticated deepfake detection 
          for enterprise-grade authentication.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3, ease: "easeOut" }}
          className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-16"
        >
          <Button 
            size="lg" 
            onClick={onStartDemo}
            className="min-w-[200px]"
            data-testid="button-start-demo"
          >
            <Eye className="w-4 h-4 mr-2" />
            Start Live Demo
          </Button>
          <Button 
            variant="outline" 
            size="lg"
            className="min-w-[200px]"
            data-testid="button-learn-more"
            onClick={() => {
              document.getElementById("how-it-works")?.scrollIntoView({ behavior: "smooth" });
            }}
          >
            Learn More
          </Button>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
          className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-4xl mx-auto"
        >
          <FeatureCard
            icon={<Shield className="w-6 h-6" />}
            title="Real-Time Verification"
            description="Biological pulse and micro-expression analysis in milliseconds"
          />
          <FeatureCard
            icon={<Zap className="w-6 h-6" />}
            title="AI Detection"
            description="Post-hoc analysis identifies synthetic media artifacts"
          />
          <FeatureCard
            icon={<Eye className="w-6 h-6" />}
            title="Humanity Score"
            description="Comprehensive authenticity assessment from 0-100%"
          />
        </motion.div>
      </div>

      <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
        <motion.div
          animate={{ y: [0, 8, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          className="w-6 h-10 rounded-full border-2 border-muted-foreground/30 flex items-start justify-center p-2"
        >
          <div className="w-1 h-2 bg-muted-foreground/50 rounded-full" />
        </motion.div>
      </div>
    </section>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="glass-effect p-6 rounded-md hover-elevate transition-all duration-300" data-testid={`card-feature-${title.toLowerCase().replace(/\s+/g, '-')}`}>
      <div className="w-12 h-12 rounded-md bg-primary/10 flex items-center justify-center mb-4 text-primary">
        {icon}
      </div>
      <h3 className="font-semibold text-foreground mb-2">{title}</h3>
      <p className="text-sm text-muted-foreground">{description}</p>
    </div>
  );
}
