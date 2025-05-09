import React from "react";

export function Separator({ className = "my-4 border-t border-gray-300", ...props }: React.HTMLAttributes<HTMLHRElement>) {
  return <hr className={className} {...props} />;
} 