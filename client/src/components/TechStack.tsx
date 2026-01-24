import { motion } from "framer-motion";
import { Fingerprint, Activity, ScanFace, Mic, Shield, Lock } from "lucide-react";

const technologies = [
  {
    category: "detection",
    items: [
      { icon: <ScanFace className="w-5 h-5" />, name: "Facial Analysis", description: "Real-time micro-expression tracking" },
      { icon: <Fingerprint className="w-5 h-5" />, name: "Biometric Pulse", description: "Remote photoplethysmography signals" },
      { icon: <Activity className="w-5 h-5" />, name: "Motion Patterns", description: "Natural movement detection" }
    ]
  },
  {
    category: "analysis",
    items: [
      { icon: <Mic className="w-5 h-5" />, name: "Audio Forensics", description: "Synthetic voice detection" },
      { icon: <ScanFace className="w-5 h-5" />, name: "Deepfake Detection", description: "GAN artifact identification" },
      { icon: <Activity className="w-5 h-5" />, name: "Temporal Analysis", description: "Frame consistency verification" }
    ]
  },
  {
    category: "security",
    items: [
      { icon: <Shield className="w-5 h-5" />, name: "End-to-End Encryption", description: "AES-256 data protection" },
      { icon: <Lock className="w-5 h-5" />, name: "Zero-Knowledge", description: "Privacy-preserving analysis" },
      { icon: <Shield className="w-5 h-5" />, name: "SOC 2 Compliant", description: "Enterprise security standards" }
    ]
  }
];

const categoryLabels: Record<string, string> = {
  detection: "Liveness Detection",
  analysis: "AI Analysis",
  security: "Security & Compliance"
};

export default function TechStack() {
  return (
    <section id="tech-stack" className="py-24 px-6 bg-gradient-to-b from-[hsl(260,25%,8%)] to-background">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-primary font-mono text-sm tracking-wider mb-4 block" data-testid="text-section-label-tech">TECHNOLOGY</span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-tech-stack-title">Technical Stack</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Built on cutting-edge technology for reliable, secure, and scalable verification.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-3 gap-8">
          {technologies.map((category, categoryIndex) => (
            <motion.div
              key={category.category}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: categoryIndex * 0.15 }}
              data-testid={`tech-category-${category.category}`}
            >
              <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
                <span className="w-8 h-px bg-primary" />
                {categoryLabels[category.category]}
              </h3>
              
              <div className="space-y-4">
                {category.items.map((item, itemIndex) => (
                  <motion.div
                    key={item.name}
                    initial={{ opacity: 0, x: -10 }}
                    whileInView={{ opacity: 1, x: 0 }}
                    viewport={{ once: true }}
                    transition={{ duration: 0.4, delay: categoryIndex * 0.15 + itemIndex * 0.1 }}
                    className="glass-effect p-4 rounded-md hover-elevate"
                    data-testid={`tech-item-${item.name.toLowerCase().replace(/\s+/g, '-')}`}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                        {item.icon}
                      </div>
                      <div>
                        <h4 className="font-medium text-sm">{item.name}</h4>
                        <p className="text-xs text-muted-foreground">{item.description}</p>
                      </div>
                    </div>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
