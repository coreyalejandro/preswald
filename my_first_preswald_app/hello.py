import pandas as pd
import plotly.express as px
from preswald import text, connect, slider, selectbox, plotly
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

try:
    connect()

    # --- Data Loading and Preprocessing ---
    def load_data(filepath="data/my_sample_superstore.csv"):
        df = pd.read_csv(filepath)
        df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
        df["Profit Margin"] = df["Profit"] / df["Sales"]

        # Simulate Inventory Data (Replace with your actual inventory data if available)
        np.random.seed(42)
        df['Quantity in Stock'] = np.random.randint(10, 100, size=len(df))

        # Calculate Total Sales and Profit (using the full dataset before potential filtering)
        total_sales = df["Sales"].sum()
        total_profit = df["Profit"].sum()

        # Calculate Contribution Percentages (handle potential division by zero)
        df['Sales Contribution %'] = (df['Sales'] / total_sales * 100) if total_sales != 0 else 0
        df['Profit Contribution %'] = (df['Profit'] / total_profit * 100) if total_profit != 0 else 0
        return df

    df = load_data()
    logger.info(f"Data loaded successfully. Shape: {df.shape}")

    # --- Sidebar / Controls ---
    text("## Filters & Parameters")
    metric_options = ["Sales", "Profit"]
    selected_metric_size = selectbox(
        "Select Size Metric for Scatter Plot:",
        options=metric_options,
        default="Sales"
    )

    # --- Dashboard Title ---
    text("# 🛒 Merchandising Assortment Overview")
    text("This dashboard shows assortment depth, SKU counts, price distribution, and inventory turnover.")

    # --- 1. Profit Contrib % vs. Sales Contrib % by Sub-Category (Scatter Plot) ---
    text("## 1. Contribution Analysis by Sub-Category")
    text("*(Size based on selected metric)*")

    # Aggregate data for the scatter plot (one point per sub-category/segment)
    scatter_data = df.groupby(['Sub-Category', 'Segment']).agg(
        Total_Sales=('Sales', 'sum'),
        Total_Profit=('Profit', 'sum'),
        Sales_Contrib_Pct=('Sales Contribution %', 'sum'),
        Profit_Contrib_Pct=('Profit Contribution %', 'sum')
    ).reset_index()

    # Determine size column based on parameter
    size_col_agg = 'Total_Sales' if selected_metric_size == 'Sales' else 'Total_Profit'

    fig1 = px.scatter(
        scatter_data,
        x='Profit_Contrib_Pct',
        y='Sales_Contrib_Pct',
        color='Segment',
        size=size_col_agg,
        hover_data=['Sub-Category', 'Total_Sales', 'Total_Profit'],
        title='Profit Contribution % vs. Sales Contribution % by Sub-Category & Segment',
        labels={
            'Profit_Contrib_Pct': 'Margin Contrib %',
            'Sales_Contrib_Pct': 'Net Sales Contrib %',
            size_col_agg: f'Total {selected_metric_size} (Size)'
        }
    )
    fig1.update_layout(template="plotly_white")
    plotly(fig1)

    # --- 2. SKU Count by Category & Segment (Table) ---
    text("## 2. SKU Count by Category & Segment")
    sku_counts = df.groupby(['Category', 'Segment'])['Product ID'].nunique().reset_index(name='SKU Count')
    try:
        dataframe(sku_counts)  # Attempt to use dataframe if available
    except NameError:
        logger.warning("dataframe component not found, using text/HTML fallback for SKU table.")
        text(sku_counts.to_html(index=False))

    # --- 3. Price Distribution by Segment (Histogram) ---
    text("## 3. Order Value Distribution by Segment")
    fig3 = px.histogram(
        df,
        x='Sales',
        color='Segment',
        facet_row='Segment',
        nbins=30,
        title='Order Value Distribution by Segment',
        labels={'Sales': 'Order Value (Sales)'},
        height=600
    )
    fig3.update_layout(template="plotly_white", bargap=0.1)
    plotly(fig3)

    # --- 4. Inventory Turnover by Segment (Line Chart) ---
    text("## 4. Inventory Turnover by Segment")
    df['Order Month'] = df['Order Date'].dt.to_period('M').astype(str)

    # Group by month and segment to calculate turnover
    turnover_data = df.groupby(['Order Month', 'Segment']).agg(
        Total_Quantity=('Quantity', 'sum'),
        Average_Inventory=('Quantity in Stock', 'mean')
    ).reset_index()

    # Calculate Turnover (handle potential division by zero)
    turnover_data['Inventory Turnover'] = turnover_data.apply(
        lambda row: row['Total_Quantity'] / row['Average_Inventory'] if row['Average_Inventory'] != 0 else 0, axis=1
    )
    # Sort by month for proper line plotting
    turnover_data = turnover_data.sort_values(by='Order Month')

    fig4 = px.line(
        turnover_data,
        x='Order Month',
        y='Inventory Turnover',
        color='Segment',
        title='Monthly Inventory Turnover by Segment',
        labels={'Inventory Turnover': 'Turnover Rate', 'Order Month': 'Month'},
        markers=True
    )
    fig4.update_layout(template="plotly_white")
    plotly(fig4)

except Exception as e:
    logger.error(f"Error in main function: {e}")