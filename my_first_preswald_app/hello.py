import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from preswald import text, connect, slider, selectbox, plotly
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io  # For handling word cloud image buffer
import logging
import base64
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Initialize preswald
    connect()

    # --- Helper Functions ---
    def load_data(filepath="data/my_sample_superstore.csv"):
        """Loads and preprocesses the Superstore data, ensuring hashable types."""
        logger.info("Attempting to load and process data...")
        try:
            df = pd.read_csv(filepath, encoding='latin1')
            logger.info(f"CSV loaded. Initial shape: {df.shape}")
            
            # --- Basic Preprocessing ---
            df["Order Date"] = pd.to_datetime(df["Order Date"], format="%m/%d/%Y")
            df['Order Year'] = df['Order Date'].dt.year
            df['Order Month'] = df['Order Date'].dt.to_period('M')
            df['Order Week'] = df['Order Date'].dt.strftime('W%U')
            df["Profit Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"], 0)

            # --- Simulate Missing Data ---
            np.random.seed(42)
            # Ensure simulated inventory is standard int
            qty_max = df['Quantity'].max() if not df['Quantity'].empty else 10  # Handle empty Quantity
            df['Quantity in Stock'] = np.random.randint(qty_max, qty_max * 5 + 10, size=len(df)).astype(int)

            # Brand Type
            national_brands_subcats = ['Bookcases', 'Chairs', 'Tables', 'Appliances', 'Machines', 'Copiers', 'Phones']
            df['Brand Type'] = df['Sub-Category'].apply(lambda x: 'National' if x in national_brands_subcats else 'Private').astype(str)

            # Size
            def assign_size(name):
                name_lower = str(name).lower()
                if 'small' in name_lower: return 'S'
                if 'medium' in name_lower: return 'M'
                if 'large' in name_lower or 'executive' in name_lower: return 'L'
                if 'letter' in name_lower: return 'Letter'
                if 'legal' in name_lower: return 'Legal'
                if len(name_lower) % 3 == 0: return 'W28'
                if len(name_lower) % 3 == 1: return 'W30'
                return 'W32'
            df['Simulated Size'] = df['Product Name'].apply(assign_size).astype(str)

            # Color
            def assign_color(row):
                if row['Profit'] < 0: return 'Red'
                if row['Segment'] == 'Consumer': return 'Blue'
                if row['Segment'] == 'Corporate': return 'Green'
                return 'Gray'
            df['Simulated Color'] = df.apply(assign_color, axis=1).astype(str)

            # Calculate Contributions
            total_sales = df["Sales"].sum()
            total_profit = df["Profit"].sum()
            df['Sales Contribution %'] = (df['Sales'] / total_sales * 100) if total_sales != 0 else 0
            df['Profit Contribution %'] = (df['Profit'] / total_profit * 100) if total_profit != 0 else 0

            # --- Aggressive Type Conversion before Return ---
            logger.info("Starting aggressive type conversion before returning...")
            logger.info(f"dtypes BEFORE conversion:\n{df.dtypes}")

            for col in df.columns:
                # Convert specific numpy integer types if they exist
                if df[col].dtype == np.int32 or df[col].dtype == np.int64:
                    logger.info(f"Column '{col}' is numpy int type ({df[col].dtype}). Attempting conversion.")
                    # Check for NaNs before converting to standard int
                    if df[col].isnull().sum() == 0:
                        try:
                            # Try converting to standard Python int first
                            df[col] = df[col].astype(int)
                            logger.info(f"Successfully converted '{col}' to standard int.")
                        except OverflowError:
                            # If standard int overflows, fall back to float
                            logger.warning(f"OverflowError converting '{col}' to standard int. Converting to float64 instead.")
                            df[col] = df[col].astype(np.float64)
                        except Exception as e:
                            logger.error(f"Could not convert '{col}' to standard int: {e}. Trying float64.")
                            try:
                                df[col] = df[col].astype(np.float64)
                            except Exception as e2:
                                logger.error(f"Could not convert '{col}' to float64 either: {e2}.")
                    else:
                        # If column has NaNs, convert to float64 as standard int can't handle NaNs
                        logger.warning(f"Column '{col}' has NaNs. Converting to float64.")
                        df[col] = df[col].astype(np.float64)
                # Convert Period objects to string (can sometimes cause issues)
                elif pd.api.types.is_period_dtype(df[col]):
                    logger.info(f"Converting PeriodDtype column '{col}' to string.")
                    df[col] = df[col].astype(str)
                # Convert potential Pandas IntegerArray types (like Int64) to float if they have NaNs
                elif pd.api.types.is_integer_dtype(df[col]) and not (df[col].dtype == np.int32 or df[col].dtype == np.int64):
                    if df[col].isnull().sum() > 0:
                        logger.warning(f"Column '{col}' ({df[col].dtype}) has NaNs. Converting to float64.")
                        df[col] = df[col].astype(np.float64)
                    else:
                        # If no NaNs, try converting to standard int
                        try:
                            logger.info(f"Column '{col}' ({df[col].dtype}) has no NaNs. Converting to standard int.")
                            df[col] = df[col].astype(int)
                        except Exception as e:
                            logger.error(f"Could not convert nullable int '{col}' to standard int: {e}. Trying float64.")
                            df[col] = df[col].astype(np.float64)

            logger.info(f"dtypes AFTER conversion:\n{df.dtypes}")
            logger.info("Data processing complete.")
            return df

        except FileNotFoundError:
            logger.error(f"FileNotFoundError: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Error in load_data: {e}", exc_info=True)  # Log full traceback
            return None

    def format_currency(value):
        """Formats a number as currency."""
        return f"${float(value):,.2f}"

    def format_percent(value):
        """Formats a number as percentage."""
        return f"{float(value):.1f}%"

    # --- Load Data ---
    df_full = load_data()
    if df_full is None:
        text("⚠️ Dashboard cannot be displayed because data failed to load.")
        raise Exception("Data loading failed")

    # --- Title and Introduction ---
    text("# 🛒 Merchandising Assortment Overview")
    text("This dashboard shows assortment depth, SKU counts, price distribution, and inventory turnover.")

    # --- Controls ---
    text("## ⚙️ Controls & Filters")
    
    # Year Selection
    available_years = sorted([int(year) for year in df_full['Order Year'].unique()], reverse=True)
    selected_year = selectbox("Select Year:", options=available_years)

    # Department Filter (Using Category as proxy)
    available_depts = ["All"] + sorted(df_full['Category'].unique().tolist())
    selected_dept = selectbox("Select Department (Category):", options=available_depts)

    # Class Filter (Using Segment as proxy)
    available_classes = ["All"] + sorted(df_full['Segment'].unique().tolist())
    selected_class = selectbox("Select Class (Segment):", options=available_classes)

    # Parameter for scatter plot size
    metric_options = ["Sales", "Profit"]
    selected_metric_size = selectbox(
        "Select Size Metric for Scatter Plot:",
        options=metric_options,
        default="Sales"
    )

    # Apply Filters
    df_filtered = df_full.copy()
    if selected_year:
        df_filtered = df_filtered[df_filtered['Order Year'] == selected_year]
    if selected_dept != "All":
        df_filtered = df_filtered[df_filtered['Category'] == selected_dept]
    if selected_class != "All":
        df_filtered = df_filtered[df_filtered['Segment'] == selected_class]

    # Calculate KPIs
    net_sales_filtered = float(df_filtered['Sales'].sum())
    total_profit_filtered = float(df_filtered['Profit'].sum())
    count_brands_filtered = int(df_filtered['Sub-Category'].nunique())
    margin_percent_filtered = float(total_profit_filtered / net_sales_filtered * 100) if net_sales_filtered != 0 else 0.0

    # --- KPI Display ---
    text("## 📊 Key Performance Indicators")
    text(f"Net Sales: {format_currency(net_sales_filtered)}")
    text(f"Count of Brands: {count_brands_filtered}")
    text(f"Margin %: {format_percent(margin_percent_filtered)}")

    # --- Panel 1: Sales & Margin Performance ---
    text("## 1. Margin vs Sales Contribution %")
    text("*(Size based on selected metric)*")

    # Convert aggregated data to native types
    scatter_data = df_filtered.groupby(['Sub-Category', 'Segment']).agg({
        'Sales': lambda x: float(sum(x)),
        'Profit': lambda x: float(sum(x)),
        'Sales Contribution %': lambda x: float(sum(x)),
        'Profit Contribution %': lambda x: float(sum(x))
    }).reset_index()
    scatter_data.columns = ['Sub-Category', 'Segment', 'Total_Sales', 'Total_Profit', 'Sales_Contrib_Pct', 'Profit_Contrib_Pct']

    if not scatter_data.empty:
        size_col = 'Total_Sales' if selected_metric_size == 'Sales' else 'Total_Profit'
        fig1 = px.scatter(
            scatter_data,
            x='Profit_Contrib_Pct',
            y='Sales_Contrib_Pct',
            color='Segment',
            size=size_col,
            hover_data=['Sub-Category', 'Total_Sales', 'Total_Profit'],
            title='Contribution Analysis by Sub-Category & Segment',
            labels={
                'Profit_Contrib_Pct': 'Margin Contrib %',
                'Sales_Contrib_Pct': 'Net Sales Contrib %',
                size_col: f'Total {selected_metric_size} (Size)'
            }
        )
        fig1.update_layout(template="plotly_white")
        plotly(fig1)

    # --- Panel 2: SKU Count Analysis ---
    text("## 2. SKU Count Analysis")
    
    # SKU Count by Category & Segment
    sku_counts = df_filtered.groupby(['Category', 'Segment']).agg({
        'Product ID': lambda x: int(x.nunique())
    }).reset_index()
    sku_counts.columns = ['Category', 'Segment', 'SKU Count']
    
    # Create a stacked bar chart for SKU counts
    fig2 = px.bar(
        sku_counts,
        x='Category',
        y='SKU Count',
        color='Segment',
        title='SKU Count by Category & Segment',
        barmode='stack'
    )
    fig2.update_layout(template="plotly_white")
    plotly(fig2)

    # Display the detailed table below the chart
    text("### Detailed SKU Count Table")
    text("```")
    text(sku_counts.to_string(index=False))
    text("```")

    # --- Panel 3: Price Distribution Analysis ---
    text("## 3. Price Distribution Analysis")
    
    # Create price bins
    df_filtered['Price_Bin'] = pd.qcut(df_filtered['Sales'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Distribution by Segment
    fig3 = px.histogram(
        df_filtered,
        x='Sales',
        color='Segment',
        nbins=30,
        title='Price Distribution by Segment',
        marginal='box'
    )
    fig3.update_layout(template="plotly_white")
    plotly(fig3)

    # Price Range Summary
    price_summary = df_filtered.groupby('Price_Bin').agg({
        'Sales': [
            ('count', lambda x: int(len(x))),
            ('mean', lambda x: float(x.mean())),
            ('min', lambda x: float(x.min())),
            ('max', lambda x: float(x.max()))
        ]
    })
    price_summary.columns = ['count', 'mean', 'min', 'max']
    
    text("### Price Range Summary")
    text("```")
    text(price_summary.to_string())
    text("```")

    # --- Panel 4: Inventory Analysis ---
    text("## 4. Inventory Analysis")
    
    # Calculate Inventory Metrics
    df_filtered['Inventory_Turnover'] = df_filtered.apply(
        lambda row: float(row['Quantity'] / row['Quantity in Stock']) if row['Quantity in Stock'] != 0 else 0.0,
        axis=1
    )
    df_filtered['Days_of_Supply'] = df_filtered.apply(
        lambda row: float(365 / row['Inventory_Turnover']) if row['Inventory_Turnover'] != 0 else 0.0,
        axis=1
    )
    
    # Inventory Turnover by Segment Over Time
    turnover_data = df_filtered.groupby(['Order Month', 'Segment']).agg({
        'Inventory_Turnover': lambda x: float(x.mean())
    }).reset_index()

    fig4 = px.line(
        turnover_data,
        x='Order Month',
        y='Inventory_Turnover',
        color='Segment',
        title='Average Inventory Turnover by Segment Over Time'
    )
    fig4.update_layout(template="plotly_white")
    plotly(fig4)

    # Days of Supply Summary
    dos_summary = df_filtered.groupby('Segment').agg({
        'Days_of_Supply': [
            ('mean', lambda x: float(x.mean())),
            ('min', lambda x: float(x.min())),
            ('max', lambda x: float(x.max()))
        ]
    })
    dos_summary.columns = ['mean', 'min', 'max']
    
    text("### Days of Supply Summary")
    text("```")
    text(dos_summary.to_string())
    text("```")

    # --- Panel 5: Brand Analysis ---
    text("## 5. Brand Analysis")
    
    # Brand Type Distribution
    brand_dist = df_filtered.groupby(['Brand Type', 'Category']).agg({
        'Sales': lambda x: float(sum(x)),
        'Profit': lambda x: float(sum(x)),
        'Product ID': lambda x: int(x.nunique())
    }).reset_index()

    fig5 = px.treemap(
        brand_dist,
        path=[px.Constant("All"), 'Brand Type', 'Category'],
        values='Sales',
        color='Profit',
        title='Brand Distribution by Category',
        color_continuous_scale='RdYlBu'
    )
    plotly(fig5)

    # --- Panel 6: Size Analysis ---
    text("## 6. Size Analysis")
    
    # Size Distribution
    size_dist = df_filtered.groupby(['Simulated Size', 'Category']).agg({
        'Sales': lambda x: float(sum(x)),
        'Product ID': lambda x: int(x.nunique())
    }).reset_index()

    fig6 = px.bar(
        size_dist,
        x='Category',
        y='Sales',
        color='Simulated Size',
        title='Sales by Size and Category',
        barmode='group'
    )
    fig6.update_layout(template="plotly_white")
    plotly(fig6)

    # --- Panel 7: Word Cloud of Product Names ---
    text("## 7. Product Name Word Cloud")
    
    # Prepare text for word cloud
    text_data = ' '.join(df_filtered['Product Name'].astype(str))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        max_words=100
    ).generate(text_data)
    
    # Convert matplotlib plot to image buffer
    buf = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Display the word cloud image
    text("### Most Common Product Terms")
    text(f"![Word Cloud](data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()})")

except Exception as e:
    logger.error(f"Error in main function: {e}")
    text(f"⚠️ An error occurred: {str(e)}")