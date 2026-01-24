import { motion } from "framer-motion";
import { Scan, Brain, ShieldCheck, FileSearch } from "lucide-react";

const steps = [
  {
    icon: <Scan className="w-8 h-8" />,
    title: "Capture",
    description: "Real-time video feed captures facial movements, micro-expressions, and biological signals through standard webcam technology.",
    phase: "01"
  },
  {
    icon: <Brain className="w-8 h-8" />,
    title: "Analyze",
    description: "Advanced algorithms process biological pulse patterns, facial muscle movements, and temporal consistency markers.",
    phase: "02"
  },
  {
    icon: <FileSearch className="w-8 h-8" />,
    title: "Detect",
    description: "AI detection systems identify synthetic artifacts, deepfake signatures, and manipulation patterns in media content.",
    phase: "03"
  },
  {
    icon: <ShieldCheck className="w-8 h-8" />,
    title: "Verify",
    description: "Comprehensive Humanity Score provides confidence assessment for authentication decisions across enterprise systems.",
    phase: "04"
  }
];

export default function HowItWorks() {
  return (
    <section id="how-it-works" className="py-24 px-6 bg-gradient-to-b from-background to-[hsl(260,25%,8%)]">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-primary font-mono text-sm tracking-wider mb-4 block" data-testid="text-section-label">PROCESS</span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-how-it-works-title">How It Works</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Protocol Aura combines real-time biometric analysis with advanced AI detection 
            to deliver comprehensive authenticity verification.
          </p>
        </motion.div>

        <div className="relative">
          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-gradient-to-b from-primary/50 via-primary/20 to-transparent hidden lg:block" />
          
          <div className="space-y-12 lg:space-y-0 lg:grid lg:grid-cols-2 lg:gap-8">
            {steps.map((step, index) => (
              <motion.div
                key={step.phase}
                initial={{ opacity: 0, x: index % 2 === 0 ? -30 : 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                className={`relative ${index % 2 === 0 ? 'lg:pr-12 lg:text-right' : 'lg:pl-12 lg:col-start-2'}`}
                data-testid={`step-${step.phase}`}
              >
                <div className={`glass-effect p-6 rounded-md hover-elevate ${index % 2 === 0 ? 'lg:ml-auto' : ''} max-w-md`}>
                  <div className={`flex items-center gap-4 mb-4 ${index % 2 === 0 ? 'lg:flex-row-reverse' : ''}`}>
                    <div className="w-14 h-14 rounded-md bg-primary/10 flex items-center justify-center text-primary flex-shrink-0">
                      {step.icon}
                    </div>
                    <div>
                      <span className="text-primary font-mono text-xs tracking-wider">{step.phase}</span>
                      <h3 className="text-xl font-semibold">{step.title}</h3>
                    </div>
                  </div>
                  <p className={`text-muted-foreground text-sm leading-relaxed ${index % 2 === 0 ? 'lg:text-right' : ''}`}>
                    {step.description}
                  </p>
                </div>
                
                <div className="hidden lg:block absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full bg-primary border-4 border-background"
                  style={{ [index % 2 === 0 ? 'right' : 'left']: '-8px' }}
                />
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
