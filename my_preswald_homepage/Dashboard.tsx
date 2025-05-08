import { Separator } from '@radix-ui/react-separator';
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import { DollarSign, TrendingUp, Users, Package } from 'lucide-react';

// Sample data for the chart
const sampleData = [
    { category: 'Furniture', sales: 4000 },
    { category: 'Technology', sales: 3000 },
    { category: 'Office Supplies', sales: 2000 },
    { category: 'Services', sales: 2780 },
];

// Stats card component
const StatsCard = ({ title, value, icon: Icon, trend }: { title: string; value: string; icon: any; trend: string }) => (
    <motion.div
        whileHover={{ scale: 1.02 }}
        className="bg-white p-6 rounded-lg shadow-md"
    >
        <div className="flex items-center justify-between">
            <div>
                <p className="text-gray-500 text-sm">{title}</p>
                <h3 className="text-2xl font-bold mt-1">{value}</h3>
                <p className="text-green-500 text-sm mt-2">{trend}</p>
            </div>
            <div className="bg-blue-100 p-3 rounded-full">
                <Icon className="w-6 h-6 text-blue-600" />
            </div>
        </div>
    </motion.div>
);

export default function SuperstoreSalesDashboard() {
    return (
        <div className="p-6 max-w-7xl mx-auto">
            <div className="flex items-center justify-between mb-8">
                <h1 className="text-3xl font-bold">Sales Dashboard</h1>
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md"
                >
                    Download Report
                </motion.button>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                <StatsCard
                    title="Total Sales"
                    value="$54,239"
                    icon={DollarSign}
                    trend="+12.5% from last month"
                />
                <StatsCard
                    title="Revenue Growth"
                    value="23.5%"
                    icon={TrendingUp}
                    trend="+4.1% from last month"
                />
                <StatsCard
                    title="Total Sales"
                    value="$54,239"
                    icon={DollarSign}
                    trend="+12.5% from last month"
                />

                <StatsCard
                    title="Total Customers"
                    value="1,429"
                    icon={Users}
                    trend="+8.2% from last month"
                />
                <StatsCard
                    title="Products Sold"
                    value="2,941"
                    icon={Package}
                    trend="+15.3% from last month"
                />
            </div>

            {/* Chart Section */}
            <div className="bg-white p-6 rounded-lg shadow-md mb-8">
                <h2 className="text-xl font-semibold mb-4">Sales by Category</h2>
                <BarChart
                    width={800}
                    height={400}
                    data={sampleData}
                    margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="category" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Bar dataKey="sales" fill="#3b82f6" />
                </BarChart>
            </div>

            {/* Recent Orders Section */}
            <div className="bg-white p-6 rounded-lg shadow-md">
                <h2 className="text-xl font-semibold mb-4">Recent Orders</h2>
                <div className="overflow-x-auto">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead>
                            <tr>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Order ID</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Customer</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Product</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Amount</th>
                                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                            </tr>
                        </thead>
                        <tbody className="bg-white divide-y divide-gray-200">
                            {[1, 2, 3].map((_, index) => (
                                <tr key={index}>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">#ORD-{2024001 + index}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">Customer {index + 1}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">Product {index + 1}</td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">${(1000 + index * 500).toLocaleString()}</td>
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                                            Completed
                                        </span>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
