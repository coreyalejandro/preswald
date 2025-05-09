import React from "react";
import { Storyboard, StoryCard } from "@/components/ui/storyboard";

export default function StoryboardDemo() {
  return (
    <main className="min-h-screen bg-gray-50 py-10">
      <Storyboard title="Preswald Project Storyboard">
        <StoryCard
          title="Kickoff"
          description="Initial project meeting and planning."
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8"><circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" /></svg>
          }
        />
        <StoryCard
          title="Design"
          description="Wireframes and UI design phase."
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M3 9h18M9 21V9" /></svg>
          }
        />
        <StoryCard
          title="Development"
          description="Implementation of features."
          icon={
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="w-8 h-8"><path d="M16 18v-6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v6" /><rect x="2" y="12" width="20" height="8" rx="2" /></svg>
          }
        />
        <StoryCard
          title="Launch"
          description="Go live and monitor performance."
        />
      </Storyboard>
    </main>
  );
} 