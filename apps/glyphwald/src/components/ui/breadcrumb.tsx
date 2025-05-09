import React from "react";

export function Breadcrumb({ children, ...props }: React.HTMLAttributes<HTMLElement>) {
  return (
    <nav aria-label="Breadcrumb" {...props}>
      <ol className="flex items-center space-x-2">{children}</ol>
    </nav>
  );
}

export function BreadcrumbList({ children, ...props }: React.HTMLAttributes<HTMLOListElement>) {
  return (
    <ol className="flex items-center space-x-2" {...props}>
      {children}
    </ol>
  );
}

export function BreadcrumbItem({ children, className = "", ...props }: React.LiHTMLAttributes<HTMLLIElement>) {
  return (
    <li className={className} {...props}>
      {children}
    </li>
  );
}

export function BreadcrumbLink({ children, href = "#", ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement>) {
  return (
    <a
      href={href}
      className="text-sm font-medium text-gray-600 hover:text-gray-900 transition-colors"
      {...props}
    >
      {children}
    </a>
  );
}

export function BreadcrumbPage({ children, ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span className="text-sm font-semibold text-gray-900" aria-current="page" {...props}>
      {children}
    </span>
  );
}

export function BreadcrumbSeparator({ className = "mx-2 text-gray-400", ...props }: React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span className={className} role="presentation" {...props}>
      /
    </span>
  );
} 