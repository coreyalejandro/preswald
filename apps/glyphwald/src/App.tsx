import {
  Routes,
  Route,
  Outlet,
  Link as RouterLink // Import Link as RouterLink to avoid conflict
} from "react-router-dom";
import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  // BreadcrumbSeparator, // Removed as it's unused
} from "@/components/ui/breadcrumb"
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"

import { AppSidebar } from "@/components/app-sidebar"
import { Separator } from "@/components/ui/separator"
import StoryboardDemo from "@/pages/storyboard-demo"; // Import the demo page

// Define a layout component
function AppLayout() {
  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b">
          <div className="flex items-center gap-2 px-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="mr-2 h-4" />
            <Breadcrumb>
              <BreadcrumbList>
                <BreadcrumbItem className="hidden md:block">
                  <BreadcrumbLink asChild> {/* Use asChild for RouterLink integration */}
                    <RouterLink to="/">Building Your Application</RouterLink>
                  </BreadcrumbLink>
                </BreadcrumbItem>
                {/* You can add more breadcrumbs based on the route */}
              </BreadcrumbList>
            </Breadcrumb>
          </div>
        </header>
        <main className="flex flex-1 flex-col gap-4 p-4">
          <Outlet /> {/* Child routes will render here */}
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}

// Define your main content for the home page
function HomePageContent() {
  return (
    <>
      <BreadcrumbPage>Home</BreadcrumbPage> {/* Example dynamic breadcrumb */}
      <div className="grid auto-rows-min gap-4 md:grid-cols-3">
        <div className="aspect-video rounded-xl bg-muted/50" />
        <div className="aspect-video rounded-xl bg-muted/50" />
        <div className="aspect-video rounded-xl bg-muted/50" />
      </div>
      <div className="min-h-[100vh] flex-1 rounded-xl bg-muted/50 md:min-h-min" />
    </>
  );
}

function App() {
  return (
    <Routes>
      <Route path="/" element={<AppLayout />}>
        <Route index element={<HomePageContent />} />
        <Route path="storyboard-demo" element={<StoryboardDemo />} />
      </Route>
    </Routes>
  );
}

export default App
