# Preswald Monorepo

This monorepo consolidates all Preswald-related applications and shared packages.

## Apps
- apps/preswald        — The main Preswald project (Python backend, docs, frontend)
- apps/glyphwald       — CMS for managing content, used by Preswald
- apps/supawald        — Serverless Supabase-based CMS
- apps/my_first_preswald_app
- apps/my_preswald_homepage
- apps/my_emerging_llm_architectures
- apps/my_roo_code_preswald
- apps/my_superstore_documentary
- apps/mystery_dashboard_chtgpt4o

## Packages
- packages/ui          — Shared ShadCN UI component library

## Getting Started
Install dependencies and start all development servers:
```bash
pnpm install
pnpm dev
```

Build for production:
```bash
pnpm build
```