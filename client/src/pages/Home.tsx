import { useState, useCallback } from "react";
import Navigation from "../components/Navigation";
import Hero from "../components/Hero";
import HowItWorks from "../components/HowItWorks";
import UseCases from "../components/UseCases";
import TechStack from "../components/TechStack";
import LiveDemo from "../components/LiveDemo";
import PostHocDetection from "../components/PostHocDetection";
import Footer from "../components/Footer";

export default function Home() {
  const [isDemoActive, setIsDemoActive] = useState(false);

  const handleStartDemo = useCallback(() => {
    setIsDemoActive(true);
    setTimeout(() => {
      document.getElementById("live-demo")?.scrollIntoView({ behavior: "smooth" });
    }, 100);
  }, []);

  const handleCloseDemo = useCallback(() => {
    setIsDemoActive(false);
  }, []);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Navigation onStartDemo={handleStartDemo} />
      
      <main>
        <Hero onStartDemo={handleStartDemo} />
        <HowItWorks />
        <UseCases />
        <TechStack />
        <LiveDemo isActive={isDemoActive} onClose={handleCloseDemo} />
        <PostHocDetection />
      </main>
      
      <Footer />
    </div>
  );
}
