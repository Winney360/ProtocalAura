import { motion } from "framer-motion";
import { Landmark, Scale, Building2, Tv } from "lucide-react";

const useCases = [
  {
    id: "financial",
    icon: <Landmark className="w-8 h-8" />,
    title: "Financial Services",
    description: "Remote identity verification for banking, insurance, and investment platforms. Prevent fraud in video-based KYC processes and secure high-value transactions.",
    applications: ["KYC/AML Compliance", "Remote Account Opening", "Transaction Authorization"]
  },
  {
    id: "legal",
    icon: <Scale className="w-8 h-8" />,
    title: "Legal & Notarization",
    description: "Authenticate participants in remote online notarization and virtual court proceedings. Ensure witness and signatory identity with confidence.",
    applications: ["Remote Notarization", "Virtual Depositions", "E-Signing Verification"]
  },
  {
    id: "government",
    icon: <Building2 className="w-8 h-8" />,
    title: "Government & Defense",
    description: "Secure access control for sensitive systems and verify identities in remote clearance procedures. Critical infrastructure protection.",
    applications: ["Clearance Verification", "Secure Communications", "Border Control"]
  },
  {
    id: "media",
    icon: <Tv className="w-8 h-8" />,
    title: "Media Verification",
    description: "Detect deepfakes and synthetic media in news content. Authenticate source material and protect against misinformation campaigns.",
    applications: ["Content Authenticity", "Source Verification", "Deepfake Detection"]
  }
];

export default function UseCases() {
  return (
    <section id="use-cases" className="py-24 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-16"
        >
          <span className="text-primary font-mono text-sm tracking-wider mb-4 block" data-testid="text-section-label-usecases">APPLICATIONS</span>
          <h2 className="text-3xl md:text-4xl font-bold mb-4" data-testid="text-use-cases-title">Use Cases</h2>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Enterprise-grade verification for industries where authenticity is non-negotiable.
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-6">
          {useCases.map((useCase, index) => (
            <motion.div
              key={useCase.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
              className="glass-effect rounded-md p-6 hover-elevate group"
              data-testid={`card-usecase-${useCase.id}`}
            >
              <div className="flex items-start gap-4 mb-4">
                <div className="w-14 h-14 rounded-md bg-primary/10 flex items-center justify-center text-primary shrink-0 group-hover:bg-primary/20 transition-colors">
                  {useCase.icon}
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">{useCase.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">{useCase.description}</p>
                </div>
              </div>
              
              <div className="flex flex-wrap gap-2 mt-4 pl-18">
                {useCase.applications.map((app) => (
                  <span
                    key={app}
                    className="px-3 py-1 text-xs font-medium bg-secondary/50 text-secondary-foreground rounded-md"
                  >
                    {app}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
