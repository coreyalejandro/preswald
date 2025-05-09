import React, { createContext, useContext, useState, ReactNode } from "react";

interface SidebarContextType {
  isOpen: boolean;
  setIsOpen: (open: boolean) => void;
}

const SidebarContext = createContext<SidebarContextType | undefined>(undefined);

export function SidebarProvider({ children }: { children: ReactNode }) {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <SidebarContext.Provider value={{ isOpen, setIsOpen }}>
      {children}
    </SidebarContext.Provider>
  );
}

export function SidebarInset({ children }: { children: ReactNode }) {
  const { isOpen } = useContext(SidebarContext)!;
  return (
    <aside className={`transition-all duration-300 ${isOpen ? 'w-64' : 'w-16'} bg-gray-100 h-full`}>
      {children}
    </aside>
  );
}

export function SidebarTrigger() {
  const { isOpen, setIsOpen } = useContext(SidebarContext)!;
  return (
    <button
      className="p-2 m-2 rounded bg-gray-200 hover:bg-gray-300 focus:outline-none"
      aria-label={isOpen ? "Close sidebar" : "Open sidebar"}
      onClick={() => setIsOpen(!isOpen)}
    >
      {isOpen ? "Close" : "Open"}
    </button>
  );
} 