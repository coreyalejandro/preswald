from preswald import text, connect, plotly, table, sidebar, get_df, button, slider, selectbox
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Initialize Preswald
connect()

# --- Data Loading and Preprocessing ---
df = get_df('superstore')
if df is None:
    print('ERROR: get_df returned None. Data source may be missing or misconfigured.')
    raise SystemExit(1)
print('DEBUG: Columns:', getattr(df, 'columns', 'NO DF'))
print('DEBUG: First row:', df.iloc[0] if hasattr(df, 'iloc') and len(df) > 0 else 'EMPTY DF')
df["Order Date"] = pd.to_datetime(df["Order Date"])
# Convert numeric columns
df["Sales"] = pd.to_numeric(df["Sales"])
df["Profit"] = pd.to_numeric(df["Profit"])
df["Profit Margin"] = df["Profit"] / df["Sales"]
# Add absolute profit for sizing
df["Absolute Profit"] = np.abs(df["Profit"])

# Create region dataframes for regional analysis
east_df = df[df["Region"] == "East"]
west_df = df[df["Region"] == "West"]
central_df = df[df["Region"] == "Central"]
south_df = df[df["Region"] == "South"]

def main():
    # --- Sidebar ---
    with sidebar():
        text("# Superstore Saga")
        text("## The Visualization Documentary")
        text("---")
        text("Explore the dramatic story of sales and profits through interactive visualizations.")
        
        text("### Navigation Guide")
        text("- Each Act represents a different analytical perspective")
        text("- Navigate between chapters within each Act")
        text("- Look for 'Key Insights' to understand the narrative")
        
        # Add chapter quick links in sidebar
        text("### Jump to Act")
        if button("Act 1: Core Metrics"):
            # Handle navigation to Act 1
            pass
        if button("Act 2: Relationships"):
            # Handle navigation to Act 2
            pass
        if button("Act 3: Complex Interplay"):
            # Handle navigation to Act 3
            pass
        if button("Final Act: Path Forward"):
            # Handle navigation to Final Act
            pass
    
    # --- Main Content ---
    # Section Navigation Tabs
    current_act = selectbox("Select Act", [
        "Welcome",
        "Act 1: Understanding Our Core Metrics",
        "Act 2: Exploring Relationships",
        "Act 3: The Complex Interplay",
        "Final Act: The Path Forward"
    ])
    
    # Display content based on selected tab
    if current_act == "Welcome":
        display_welcome()
    elif current_act == "Act 1: Understanding Our Core Metrics":
        display_act1()
    elif current_act == "Act 2: Exploring Relationships":
        display_act2()
    elif current_act == "Act 3: The Complex Interplay":
        display_act3()
    elif current_act == "Final Act: The Path Forward":
        display_final_act()

def display_welcome():
    text("# Welcome to the Superstore Saga")
    text("## Your Visualization Documentary")
    
    text("""
    This document tells the story of how to derive insights from Superstore sample data. 
    Think of each as a narrative documentary. We'll follow the Superstore team as they 
    navigate the data and learn lessons on the way.
    """)
    
    text("### How to Navigate")
    text("""
    - Tap the Section signifier above to browse all of the data.
    - All Story chapters display immediately
    - Go to next or previous chapter from within the chapter
    """)
    
    # Display preview of visualizations
    text("#### Sales Distribution")
    fig = px.histogram(df, x="Sales", nbins=20, height=200)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    plotly.show(fig)
    
    text("#### Profit Analysis")
    fig = px.histogram(df, x="Profit", nbins=20, height=200)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    plotly.show(fig)
    
    text("#### Strategic Matrix")
    fig = px.scatter(df.sample(100), x="Sales", y="Profit", height=200)
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    plotly.show(fig)

def display_act1():
    text("# Act 1: Understanding Our Core Metrics")
    
    # Chapter navigation within Act 1
    chapter = selectbox("Select Chapter", [
        "The Distribution of Sales",
        "The Story of Profits",
        "Product Categories: The Main Characters"
    ])
    
    if chapter == "The Distribution of Sales":
        text("## The Distribution of Sales")
        
        # Regional sales distribution
        text("### Distribution of Sales by Region")
        
        # East region
        fig_east = px.histogram(east_df, x="Sales", nbins=30,
                                title="East Region Sales Distribution",
                                labels={"Sales": "Sales ($)", "count": "Frequency"})
        fig_east.update_layout(template="plotly_white")
        plotly.show(fig_east)
        
        # West region
        fig_west = px.histogram(west_df, x="Sales", nbins=30,
                                title="West Region Sales Distribution",
                                labels={"Sales": "Sales ($)", "count": "Frequency"})
        fig_west.update_layout(template="plotly_white")
        plotly.show(fig_west)
        
        # Central region
        fig_central = px.histogram(central_df, x="Sales", nbins=30,
                                title="Central Region Sales Distribution",
                                labels={"Sales": "Sales ($)", "count": "Frequency"})
        fig_central.update_layout(template="plotly_white")
        plotly.show(fig_central)
        
        # South region
        fig_south = px.histogram(south_df, x="Sales", nbins=30,
                                title="South Region Sales Distribution",
                                labels={"Sales": "Sales ($)", "count": "Frequency"})
        fig_south.update_layout(template="plotly_white")
        plotly.show(fig_south)
        
        text("### Key Insights")
        text("""
        - Most sales values are within a closer range
        - Smaller regions show a smaller range for sales
        - East's distribution suggests opportunities for increasing average transaction value
        - South's distribution, while small, looks similar to West's
        """)
        
        # Chapter navigation buttons
        if button("← Welcome", variant="outline"):
            # Navigation logic
            pass
        if button("Next: The Story of Profits →"):
            # Navigation logic
            pass
    
    elif chapter == "The Story of Profits":
        text("## The Story of Profits")
        
        text("### Distribution of Profits")
        # Create a more detailed profit distribution visualization
        fig_profit = px.histogram(df, x="Profit", nbins=40, color_discrete_sequence=["#3366CC"],
                                title="Distribution of Profit Values",
                                labels={"Profit": "Profit ($)", "count": "Frequency"})
        fig_profit.update_layout(template="plotly_white")
        plotly.show(fig_profit)
        
        # Add regional profit boxplots for comparison
        fig_region_profit = px.box(df, x="Region", y="Profit", 
                                title="Profit Distribution by Region",
                                labels={"Region": "Region", "Profit": "Profit ($)"})
        fig_region_profit.update_layout(template="plotly_white")
        plotly.show(fig_region_profit)
        
        text("### Key Insights")
        text("""
        - East region leads in positive profit values
        - West and Central regions have lower positive profits
        - Average transaction values in the East are high
        """)
        
        # Chapter navigation buttons
        if button("← The Distribution of Sales", variant="outline"):
            # Navigation logic
            pass
        if button("Next: Product Categories →"):
            # Navigation logic
            pass
    
    elif chapter == "Product Categories: The Main Characters":
        text("## Product Categories: The Main Characters")
        
        text("### Profits Distribution Across Categories")
        # Category profit distribution
        fig_category = px.box(df, x="Category", y="Profit",
                            title="Profit Distribution by Product Category",
                            labels={"Category": "Category", "Profit": "Profit ($)"})
        fig_category.update_layout(template="plotly_white")
        plotly.show(fig_category)
        
        # Category profit totals
        category_profit = df.groupby("Category").agg({"Profit": "sum"}).reset_index()
        fig_category_total = px.bar(category_profit, x="Category", y="Profit",
                                title="Total Profit by Category",
                                labels={"Category": "Category", "Profit": "Total Profit ($)"})
        fig_category_total.update_layout(template="plotly_white")
        plotly.show(fig_category_total)
        
        text("### Key Insights")
        text("""
        - Overall, we get positive profits across categories
        - Technology shows a significant number of sales
        - Furniture shows the lowest profits
        """)
        
        # Chapter navigation buttons
        if button("← The Story of Profits", variant="outline"):
            # Navigation logic
            pass
        if button("Next: Act 2 →"):
            # Navigation logic
            pass

def display_act2():
    text("# Act 2: Exploring Relationships")
    
    # Chapter navigation within Act 2
    chapter = selectbox("Select Chapter", [
        "The Sales-Profit Dynamic",
        "Regional Performance Analysis",
        "Category Performance Snap Shot"
    ])
    
    if chapter == "The Sales-Profit Dynamic":
        text("## The Sales-Profit Dynamic")
        
        # Create a more detailed sales vs profit visualization
        fig_sales_profit = px.scatter(df, x="Sales", y="Profit", color="Category", size="Absolute Profit",
                                    opacity=0.7, title="Sales-Profit Relationship",
                                    labels={"Sales": "Sales ($)", "Profit": "Profit ($)"})
        
        # Add trend line
        fig_sales_profit.update_layout(template="plotly_white")
        plotly.show(fig_sales_profit)
        
        text("### Key Insights")
        text("""
        - Sales and profits correlate but not linearly
        - Very high sales do not guarantee a large profit
        - Profits support sales performance
        """)
        
        # Chapter navigation buttons
        if button("← Act 1", variant="outline"):
            # Navigation logic
            pass
        if button("Next: Regional Performance →"):
            # Navigation logic
            pass
    
    elif chapter == "Regional Performance Analysis":
        text("## Regional Performance Analysis")
        
        # Create region-based visualizations
        region_metrics = df.groupby("Region").agg({
            "Sales": "sum",
            "Profit": "sum",
            "Profit Margin": "mean"
        }).reset_index()
        
        # Region performance bar chart
        fig_region = px.bar(region_metrics, x="Region", y=["Sales", "Profit"],
                            barmode="group", title="Sales and Profit by Region",
                            labels={"value": "Amount ($)", "Region": "Region", "variable": "Metric"})
        fig_region.update_layout(template="plotly_white")
        plotly.show(fig_region)
        
        # Regional profit margin
        fig_region_margin = px.bar(region_metrics, x="Region", y="Profit Margin",
                                title="Average Profit Margin by Region",
                                labels={"Region": "Region", "Profit Margin": "Profit Margin (%)"})
        fig_region_margin.update_layout(template="plotly_white")
        fig_region_margin.update_yaxes(tickformat=".1%")
        plotly.show(fig_region_margin)
        
        text("### Key Insights")
        text("""
        - Eastern region leads in positive profit values
        - West and Central regions have lower positive profits
        - Central region sees the widest variance in profits
        """)
        
        # Chapter navigation buttons
        if button("← Sales-Profit Dynamic", variant="outline"):
            # Navigation logic
            pass
        if button("Next: Category Performance →"):
            # Navigation logic
            pass
    
    elif chapter == "Category Performance Snap Shot":
        text("## Category Performance Snap Shot")
        
        # Create category snapshots
        # Category by sales and profit bubble chart
        category_metrics = df.groupby(["Category", "Region"]).agg({
            "Sales": "sum",
            "Profit": "sum"
        }).reset_index()
        
        fig_category = px.scatter(category_metrics, x="Sales", y="Profit", 
                                size="Sales", color="Category", facet_col="Region",
                                title="Category Performance by Region",
                                labels={"Sales": "Sales ($)", "Profit": "Profit ($)"})
        fig_category.update_layout(template="plotly_white")
        plotly.show(fig_category)
        
        text("### Key Insights")
        text("""
        - Applicants updated in Superstore before
        - More categories created with Superstore alike
        - More applicants with service
        """)
        
        # Chapter navigation buttons
        if button("← Regional Performance", variant="outline"):
            # Navigation logic
            pass
        if button("Next: Act 3 →"):
            # Navigation logic
            pass

def display_act3():
    text("# Act 3: The Complex Interplay")
    
    # For Act 3, we'll focus on temporal analysis
    text("## Temporal Performance by Category")
    
    # Create time-based visualizations
    # Prepare time series data
    df['Year-Month'] = df['Order Date'].dt.to_period('M')
    time_series = df.groupby(['Year-Month', 'Category']).agg({
        "Sales": "sum",
        "Profit": "sum"
    }).reset_index()
    time_series['Year-Month'] = time_series['Year-Month'].astype(str)
    
    # Plot time series of sales by category
    fig_time_sales = px.line(time_series, x="Year-Month", y="Sales", color="Category",
                            title="Sales Trends by Category Over Time",
                            labels={"Year-Month": "Month", "Sales": "Sales ($)"})
    fig_time_sales.update_layout(template="plotly_white", xaxis_tickangle=-45)
    plotly.show(fig_time_sales)
    
    # Plot time series of profit by category
    fig_time_profit = px.line(time_series, x="Year-Month", y="Profit", color="Category",
                            title="Profit Trends by Category Over Time",
                            labels={"Year-Month": "Month", "Profit": "Profit ($)"})
    fig_time_profit.update_layout(template="plotly_white", xaxis_tickangle=-45)
    plotly.show(fig_time_profit)
    
    text("### Key Insights")
    text("""
    - Track category valuation status
    - Ensure growth opportunities to user category
    - Ensure applicants does not violate the law
    - Maximize benefits for category user
    """)
    
    # Chapter navigation buttons
    if button("← Act 2", variant="outline"):
        # Navigation logic
        pass
    if button("Next: Final Act →"):
        # Navigation logic
        pass

def display_final_act():
    text("# Final Act: The Path Forward")
    
    text("## Strategic Opportunities Matrix")
    
    # Create a complex multi-faceted visualization
    # Prepare aggregated data
    strategic_df = df.groupby(["Category", "Region"]).agg({
        "Sales": "sum",
        "Profit": "sum",
        "Order ID": "count"  # Number of orders
    }).reset_index()
    strategic_df["Profit Margin"] = strategic_df["Profit"] / strategic_df["Sales"]
    strategic_df["Avg Order Value"] = strategic_df["Sales"] / strategic_df["Order ID"]
    
    # Create a bubble chart with multiple dimensions
    fig_strategy = px.scatter(strategic_df, x="Sales", y="Profit Margin", 
                            size="Profit", color="Category", facet_col="Region",
                            hover_data=["Avg Order Value"],
                            title="Strategic Opportunities Matrix",
                            labels={
                                "Sales": "Total Sales ($)",
                                "Profit Margin": "Profit Margin",
                                "Profit": "Total Profit ($)",
                                "Avg Order Value": "Avg. Order Value ($)"
                            })
    fig_strategy.update_layout(template="plotly_white")
    fig_strategy.update_yaxes(tickformat=".1%")
    plotly.show(fig_strategy)
    
    text("### No Brainer Insights")
    text("""
    **Revenue Growth Driver**
    - Technology shows steady growth with ample sales volume
    - High value furniture generally generates growth output
    - Office Supplies has minimal sales growth
    """)
    
    text("### Actionable Insight Scope")
    text("""
    - Technology's strategic importance clearly warrants action
    - High value furniture generally generates growth output
    - Office Supplies has minimal sales growth
    """)
    
    text("### Challenges to Address")
    text("""
    - Sales regions have inconsistent profit margins
    - Regional allocation budget support roles and responsibilities
    """)
    
    text("### Recommended Activity")
    text("""
    - Optimize pricing strategies in each region
    - Review inventory storage requirements
    - Explore alternative category strategies in other regions
    - Regional allocation budget support roles and responsibilities
    """)
    
    text("### Action Opportunities")
    text("""
    - Create cost-effective high quality marketing campaigns
    - Regional resources for the unique opportunities
    - Thorough assessment of budget constraints
    - Review category requirements with resources
    """)
    
    text("## Your Next Steps")
    text("""
    1. Complete the checklist above to guide your strategy.
    2. Use this approach to derive insights with your own data.
    3. Share these insights with your team.

    Remember: Data driven insight will drive your revenue within a week or two.
    """)
    
    # Chapter navigation buttons
    if button("← Act 3", variant="outline"):
        # Navigation logic
        pass
    if button("Back to Welcome"):
        # Navigation logic
        pass

# Run main function
if __name__ == "__main__":
    main()

# Make data and functions available in REPL
__all__ = [
    'df', 
    'display_welcome', 
    'display_act1',
    'display_act2',
    'display_act3',
    'display_final_act',
    'main'
]
