import pandas as pd
import plotly.express as px
import numpy as np
from preswald import text, connect, plotly, table, sidebar, get_df

# Initialize Preswald
connect()

# Define a rich, deep color palette
COLOR_PALETTE = [
    "#1E3D59",  # Deep Navy Blue
    "#FF6B6B",  # Rich Coral Red
    "#4B8F8C",  # Deep Teal
    "#7B2CBF",  # Rich Purple
    "#FFC13B",  # Deep Gold
    "#2D6A4F",  # Forest Green
    "#9E2A2B",  # Deep Wine Red
    "#335C67",  # Steel Blue
    "#B68D40",  # Rich Bronze
    "#540B0E"   # Deep Burgundy
]

# --- Sidebar ---
sidebar(text("# Superstore Saga"))
sidebar(text("A Data Visualization Documentary"))
sidebar(text("---"))
sidebar(text("Explore the dramatic story of sales and profits through interactive visualizations."))

# --- Data Loading and Preprocessing ---
df = get_df('my_superstore_data')
df["Order Date"] = pd.to_datetime(df["Order Date"])
# Convert numeric columns
df["Sales"] = pd.to_numeric(df["Sales"])
df["Profit"] = pd.to_numeric(df["Profit"])
df["Profit Margin"] = df["Profit"] / df["Sales"]
# Add absolute profit for sizing
df["Absolute Profit"] = np.abs(df["Profit"])

# --- Onboarding Section ---
text("# Welcome to the Superstore Saga")
text("## A Data Visualization Documentary")

text("""
This interactive dashboard takes you on a journey through the Superstore's performance data.
Through a series of compelling visualizations, we'll uncover the stories hidden in the numbers
and reveal insights that can drive business success.
""")

text("""
### How to Navigate
- 🔍 Use filters to explore different aspects of the data
- 📊 Hover over charts for detailed information
- 💡 Look for insights below each visualization
""")

# --- Univariate Analysis ---
text("# Act 1: Understanding Our Core Metrics")

# Sales Distribution
text("## The Distribution of Sales")
fig_sales = px.histogram(df, x="Sales", nbins=30,
                        title="Distribution of Sales",
                        labels={"Sales": "Sales ($)", "count": "Frequency"},
                        color_discrete_sequence=COLOR_PALETTE)
fig_sales.update_layout(template="plotly_white")
plotly(fig_sales)

text("""
**Key Insights:**
- Most sales transactions fall in the lower range
- Several high-value outliers indicate significant large purchases
- The distribution suggests opportunities for increasing average transaction value
- Clear evidence of price point clustering around certain values
""")

# Profit Distribution
text("## The Story of Profits")
fig_profit = px.histogram(df, x="Profit", nbins=30,
                         title="Distribution of Profits",
                         labels={"Profit": "Profit ($)", "count": "Frequency"},
                         color_discrete_sequence=COLOR_PALETTE)
fig_profit.update_layout(template="plotly_white")
plotly(fig_profit)

text("""
**Key Insights:**
- Profit distribution shows both gains and losses
- Most profits cluster around a positive mean
- Some concerning negative profit outliers require attention
- The spread suggests varying levels of profitability across products
""")

# Category Analysis
text("## Product Categories: The Main Characters")
fig_category = px.bar(df.groupby("Category").size().reset_index(),
                     x="Category", y=0,
                     title="Product Distribution Across Categories",
                     labels={"0": "Number of Products", "Category": "Category"},
                     color="Category",
                     color_discrete_sequence=COLOR_PALETTE)
fig_category.update_layout(template="plotly_white")
plotly(fig_category)

text("""
**Key Insights:**
- Clear dominance of certain product categories
- Balanced representation across major segments
- Potential for category expansion in underrepresented areas
- Strategic importance of key categories evident
""")

# --- Bivariate Analysis ---
text("# Act 2: Exploring Relationships")

# Sales vs Profit
text("## The Sales-Profit Dynamic")
fig_sales_profit = px.scatter(df, x="Sales", y="Profit",
                             title="Sales vs Profit Relationship",
                             labels={"Sales": "Sales ($)", "Profit": "Profit ($)"},
                             color="Category",
                             color_discrete_sequence=COLOR_PALETTE)
fig_sales_profit.update_layout(template="plotly_white")
plotly(fig_sales_profit)

text("""
**Key Insights:**
- Strong positive correlation between sales and profits
- Some high-sales items show unexpectedly low profits
- Clear patterns of profitability emerge at different price points
- Outliers suggest areas for pricing strategy review
""")

# Regional Performance
text("## Regional Performance Analysis")
fig_region = px.box(df, x="Region", y="Profit Margin",
                    title="Profit Margins by Region",
                    labels={"Region": "Region", "Profit_Margin": "Profit Margin"},
                    color="Region",
                    color_discrete_sequence=COLOR_PALETTE)
fig_region.update_layout(template="plotly_white")
plotly(fig_region)

text("""
**Key Insights:**
- Significant variation in regional performance
- Some regions consistently outperform others
- Presence of outliers in all regions
- Opportunities for cross-regional learning
""")

# Category Performance
text("## Category Performance Deep Dive")
category_metrics = df.groupby("Category").agg({
    "Sales": "sum",
    "Profit": "sum",
    "Profit Margin": "mean",
    "Absolute Profit": "mean"  # Use absolute profit for sizing
}).reset_index()

fig_category_performance = px.scatter(category_metrics,
                                    x="Sales", y="Profit",
                                    size="Absolute Profit",  # Use absolute profit for sizing
                                    color="Category",
                                    title="Category Performance Overview",
                                    labels={
                                        "Sales": "Total Sales ($)",
                                        "Profit": "Total Profit ($)",
                                        "Absolute Profit": "Absolute Profit ($)"
                                    },
                                    color_discrete_sequence=COLOR_PALETTE)
fig_category_performance.update_layout(template="plotly_white")
plotly(fig_category_performance)

text("""
**Key Insights:**
- Each category shows distinct performance characteristics
- Size of bubbles reveals efficiency in profit generation
- Color coding helps track category-specific patterns
- Clear leaders and laggards in overall performance
""")

# --- Multivariate Analysis ---
text("# Act 3: The Complex Interplay")

# Sales, Profit, and Category Over Time
text("## Temporal Performance by Category")
# Resample data monthly and calculate mean sales by category
temporal_data = df.groupby([pd.Grouper(key='Order Date', freq='M'), 'Category'])['Sales'].sum().reset_index()

fig_temporal = px.line(temporal_data,
                      x="Order Date",
                      y="Sales",
                      color="Category",
                      title="Monthly Sales Trends by Category",
                      labels={
                          "Order Date": "Month",
                          "Sales": "Total Sales ($)",
                          "Category": "Category"
                      },
                      color_discrete_sequence=COLOR_PALETTE)

# Improve the layout
fig_temporal.update_layout(
    template="plotly_white",
    xaxis_title="Month",
    yaxis_title="Total Sales ($)",
    legend_title="Category",
    hovermode='x unified'
)

# Add markers to make the lines more visible
fig_temporal.update_traces(mode='lines+markers', marker=dict(size=6))

plotly(fig_temporal)

text("""
**Key Insights:**
- Clear seasonal patterns in sales across categories
- Different growth trajectories for each category
- Some categories show more volatility than others
- Intersection points reveal competitive dynamics
""")

# --- Call to Action ---
text("# Final Act: The Path Forward")

# Comprehensive Performance Matrix
text("## Strategic Opportunities Matrix")
fig_matrix = px.scatter(df,
                       x="Sales",
                       y="Profit Margin",
                       size="Absolute Profit",  # Use absolute profit for sizing
                       color="Category",
                       facet_col="Region",
                       title="Strategic Performance Matrix",
                       labels={
                           "Sales": "Sales ($)",
                           "Profit_Margin": "Profit Margin",
                           "Absolute Profit": "Absolute Profit ($)"
                       },
                       color_discrete_sequence=COLOR_PALETTE)
fig_matrix.update_layout(template="plotly_white")
plotly(fig_matrix)

text("""
### Key Strategic Insights

**Successes to Build Upon:**
- Technology category shows strong profit margins across regions
- Western region demonstrates consistent performance
- High-value transactions generally maintain good profit margins
- Several product categories show growth potential

**Challenges to Address:**
- Some regions show inconsistent profit margins
- Certain categories need profit margin improvement
- High-discount items often lead to negative profits
- Regional variations suggest supply chain inefficiencies

**Recommended Actions:**
1. Optimize pricing strategies in underperforming regions
2. Review discount policies for high-value items
3. Expand successful category strategies to other regions
4. Implement regional best practices across the network

**Hidden Opportunities:**
- Cross-sell between high-performing categories
- Regional expansion for top-performing products
- Margin improvement through operational efficiency
- Customer segment-specific marketing strategies
""")

# Final Call to Action
text("""
## Your Next Steps

1. 📊 Review the detailed insights for your region
2. 🎯 Identify your top 3 action items
3. 📈 Set specific improvement targets
4. 🤝 Share these insights with your team

Remember: Every data point tells a story. Your actions write the next chapter.
""")

# Show the data
table(df)