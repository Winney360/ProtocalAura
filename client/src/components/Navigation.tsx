import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Menu, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface NavigationProps {
  onStartDemo: () => void;
}

export default function Navigation({ onStartDemo }: NavigationProps) {
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const navLinks = [
    { href: "#how-it-works", label: "How It Works" },
    { href: "#use-cases", label: "Use Cases" },
    { href: "#tech-stack", label: "Technology" },
    { href: "#post-hoc", label: "AI Detection" }
  ];

  return (
    <motion.header
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        isScrolled ? "bg-background/80 backdrop-blur-lg border-b border-border" : "bg-transparent"
      }`}
    >
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <a href="#" className="flex items-center gap-3" data-testid="link-logo">
          <div className="w-9 h-9 rounded-md bg-primary/10 flex items-center justify-center">
            <Shield className="w-5 h-5 text-primary" />
          </div>
          <span className="font-semibold text-foreground hidden sm:block">Protocol Aura</span>
        </a>

        <nav className="hidden md:flex items-center gap-6">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="text-sm text-muted-foreground hover:text-foreground transition-colors"
              data-testid={`link-nav-${link.label.toLowerCase().replace(/\s+/g, '-')}`}
            >
              {link.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-4">
          <Button
            size="sm"
            onClick={onStartDemo}
            className="hidden sm:flex"
            data-testid="button-nav-demo"
          >
            Live Demo
          </Button>

          <Button
            size="icon"
            variant="ghost"
            className="md:hidden"
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            data-testid="button-mobile-menu"
          >
            {isMobileMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
          </Button>
        </div>
      </div>

      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="md:hidden bg-background/95 backdrop-blur-lg border-b border-border"
          >
            <nav className="flex flex-col p-4 gap-2">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  className="text-sm text-muted-foreground hover:text-foreground transition-colors py-2 px-4 rounded-md hover-elevate"
                  onClick={() => setIsMobileMenuOpen(false)}
                  data-testid={`link-mobile-${link.label.toLowerCase().replace(/\s+/g, '-')}`}
                >
                  {link.label}
                </a>
              ))}
              <Button
                size="sm"
                onClick={() => {
                  onStartDemo();
                  setIsMobileMenuOpen(false);
                }}
                className="mt-2"
                data-testid="button-mobile-demo"
              >
                Live Demo
              </Button>
            </nav>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.header>
  );
}
