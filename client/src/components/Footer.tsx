import { Shield } from "lucide-react";

export default function Footer() {
  return (
    <footer className="py-12 px-6 border-t border-border">
      <div className="max-w-6xl mx-auto">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-md bg-primary/10 flex items-center justify-center">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <div>
              <span className="font-semibold text-foreground">Protocol Aura</span>
              <p className="text-xs text-muted-foreground">Humanity Verified</p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-6 text-sm text-muted-foreground">
            <a href="#how-it-works" className="hover-elevate px-2 py-1 rounded-md" data-testid="link-footer-how-it-works">
              How It Works
            </a>
            <a href="#use-cases" className="hover-elevate px-2 py-1 rounded-md" data-testid="link-footer-use-cases">
              Use Cases
            </a>
            <a href="#tech-stack" className="hover-elevate px-2 py-1 rounded-md" data-testid="link-footer-tech-stack">
              Technology
            </a>
            <a href="#post-hoc" className="hover-elevate px-2 py-1 rounded-md" data-testid="link-footer-detection">
              AI Detection
            </a>
          </div>

          <div className="text-center md:text-right">
            <p className="text-xs text-muted-foreground">
              Prototype Demonstration
            </p>
            <p className="text-xs text-muted-foreground">
              All analysis results are simulated
            </p>
          </div>
        </div>

        <div className="mt-8 pt-6 border-t border-border/50 text-center">
          <p className="text-xs text-muted-foreground">
            Protocol Aura is a technology demonstration. Production implementations require proper licensing and integration.
          </p>
        </div>
      </div>
    </footer>
  );
}
