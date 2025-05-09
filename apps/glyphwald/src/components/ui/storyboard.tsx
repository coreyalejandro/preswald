// Example usage:
// <Storyboard title="Project Timeline">
//   <StoryCard title="Kickoff" description="Initial project meeting and planning." />
//   <StoryCard title="Design" description="Wireframes and UI design phase." icon={<svg ... />} />
//   <StoryCard title="Development" description="Implementation of features." />
// </Storyboard>

import React, { ReactNode } from "react";

interface StoryboardProps {
  title: string;
  children: ReactNode;
  className?: string;
}

export function Storyboard({ title, children, className = "" }: StoryboardProps) {
  return (
    <section
      aria-label={title}
      className={`rounded-xl bg-white/80 shadow-lg p-6 max-w-3xl mx-auto my-8 border border-gray-200 ${className}`}
    >
      <h2 className="text-2xl font-bold text-primary mb-6 tracking-tight flex items-center gap-2">
        <svg className="w-7 h-7 text-primary" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M16 3v4M8 3v4"/></svg>
        {title}
      </h2>
      <div className="grid gap-4 md:grid-cols-2">
        {children}
      </div>
    </section>
  );
}

interface StoryCardProps {
  title: string;
  description: string;
  icon?: ReactNode;
  className?: string;
}

export function StoryCard({ title, description, icon, className = "" }: StoryCardProps) {
  return (
    <article
      className={`rounded-lg border border-gray-200 bg-gradient-to-br from-gray-50 to-white p-4 shadow-sm hover:shadow-md transition-shadow flex items-start gap-4 ${className}`}
      tabIndex={0}
      aria-label={title}
    >
      {icon && <span className="text-primary w-8 h-8 flex-shrink-0">{icon}</span>}
      <div>
        <h3 className="text-lg font-semibold mb-1 text-gray-900">{title}</h3>
        <p className="text-gray-600 text-sm leading-relaxed">{description}</p>
      </div>
    </article>
  );
} 