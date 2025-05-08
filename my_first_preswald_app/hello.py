from preswald import text, slider, selectbox, sidebar, card, get_df, connect
import pandas as pd
import plotly.express as px

# State-to-code mapping for choropleth
STATE_CODES = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
}

def main():
    connect()
    df = get_df('my_sample_superstore')

    # Preprocess
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    df['Profit Margin'] = df['Profit'] / df['Sales']

    # Sidebar controls
    with sidebar():
        num_rows = slider(
            "Number of Rows to Display",
            min_value=5,
            max_value=len(df),
            value=10
        )
        segments = ['All Segments'] + sorted(df['Segment'].unique().tolist())
        segment = selectbox(
            "Customer Segment",
            options=segments,
            index=0
        )

    # Filter
    data = df.head(num_rows)
    if segment != 'All Segments':
        data = data[data['Segment'] == segment]

    # 1️⃣ Scatter plot
    with card("Sales vs Profit Scatter"):
        fig1 = px.scatter(
            data,
            x='Sales',
            y='Profit',
            color='Segment',
            title='Sales vs Profit by Segment'
        )
        card.plotly(fig1)

    # 2️⃣ Bar chart
    with card("Sales by Category"):
        sales_cat = data.groupby('Category')['Sales'].sum().reset_index()
        fig2 = px.bar(
            sales_cat,
            x='Category',
            y='Sales',
            title='Sales by Category'
        )
        card.plotly(fig2)

    # 3️⃣ Profit margin histogram
    with card("Profit Margin Distribution"):
        fig3 = px.histogram(
            data,
            x='Profit Margin',
            nbins=20,
            title='Profit Margin Distribution'
        )
        card.plotly(fig3)

    # 4️⃣ Orders over time
    with card("Monthly Order Count"):
        orders_time = (
            data
            .groupby(pd.Grouper(key='Order Date', freq='M'))
            ['Order ID']
            .count()
            .reset_index()
        )
        fig4 = px.line(
            orders_time,
            x='Order Date',
            y='Order ID',
            title='Orders Over Time'
        )
        card.plotly(fig4)

    # 5️⃣ Choropleth map
    with card("Profit by State"):
        map_df = data.copy()
        map_df['State Code'] = map_df['State'].map(STATE_CODES)
        profit_state = (
            map_df
            .groupby('State Code')['Profit']
            .sum()
            .reset_index()
        )
        fig5 = px.choropleth(
            profit_state,
            locations='State Code',
            locationmode='USA-states',
            color='Profit',
            scope='usa',
            title='Profit by State'
        )
        card.plotly(fig5)

if __name__ == "__main__":
    main()
