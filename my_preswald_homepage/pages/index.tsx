import type { NextPage } from 'next';
import Head from 'next/head';
import SuperstoreSalesDashboard from '../Dashboard';

const Home: NextPage = () => {
  return (
    <div className="min-h-screen bg-background">
      <Head>
        <title>Superstore Sales Dashboard</title>
        <meta name="description" content="Analytics dashboard for Superstore sales data" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="container mx-auto px-4 py-8">
        <SuperstoreSalesDashboard />
        
      </main>

      <footer className="border-t">
        <div className="container mx-auto px-4 py-6 text-center text-sm text-muted-foreground">
          © {new Date().getFullYear()} Superstore Sales Dashboard. All rights reserved.
        </div>
      </footer>
    </div>
  );
};

export default Home; 