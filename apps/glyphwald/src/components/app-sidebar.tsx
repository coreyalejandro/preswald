import React from "react";
import { Link } from "react-router-dom";

export function AppSidebar() {
  return (
    <nav className="h-full w-full p-4 bg-gray-200 border-r">
      <ul className="space-y-2">
        <li>
          <Link to="/" className="block p-2 rounded hover:bg-gray-300">Home</Link>
        </li>
        <li>
          <a href="#" className="block p-2 rounded hover:bg-gray-300">Dashboard</a>
        </li>
        <li>
          <a href="#" className="block p-2 rounded hover:bg-gray-300">Settings</a>
        </li>
        <li>
          <Link to="/storyboard-demo" className="block p-2 rounded hover:bg-gray-300">Storyboard Demo</Link>
        </li>
      </ul>
    </nav>
  );
} 