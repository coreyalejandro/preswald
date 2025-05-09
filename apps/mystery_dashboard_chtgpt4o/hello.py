from preswald import Workflow, text, sidebar, get_df, selectbox, plotly
import pandas as pd
import plotly.express as px

workflow = Workflow()

@workflow.atom()
def load_data():
    # Load merged data CSV directly
    df = pd.read_csv("data/merged_data.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Profit Margin"] = df["Profit"] / df["Sales"]
    df["Month"] = df["Order Date"].dt.to_period("M").astype(str)
    return df

@workflow.atom()
def intro():
    text("# 🕵️ Magnum B.I. Presents: *The Margin that Vanished*")
    text("""
    *It started like any good mystery: quietly.*
    Sales soared. Headlines celebrated. Bonuses flowed.

    But beneath the surface, something didn't add up. Margins—those thin lifelines of profit—were vanishing. Not dipped. Not declined. *Gone.*

    Welcome to Superstore's strangest quarter yet.
    You're the analyst called in to solve the case.

    You'll follow clues across regions, customers, and categories. You'll interrogate discounts and revisit past orders. You'll uncover blind spots no one dared to examine.

    Each screen in this dashboard is a *chapter*. Each chart: a clue.
    This isn't just data—it's a crime scene.
    """)

@workflow.atom()
def ch1_explanation():
    text("## 📘 Chapter 1: A Pattern in Time")
    text("""
    First, you trace the timeline. You plot profit margin by month—looking for the break.
    A dip in Q3? Or was it earlier?

    Scroll the months. Watch the line. The moment of collapse might be hiding in plain sight.
    """)

@workflow.atom()
def ch1_visual(df=load_data()):
    monthly = df.groupby("Month").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    monthly["Profit Margin"] = monthly["Profit"] / monthly["Sales"]
    fig = px.line(monthly, x="Month", y="Profit Margin", title="Monthly Profit Margin")
    plotly(fig)

@workflow.atom()
def ch2_explanation():
    text("## 📘 Chapter 2: The Leaky Categories")
    text("""
    Some products sold like wildfire—but barely brought profit.
    Could it be the flashy Tech gear? Or cheap Office Supplies?

    This bar chart breaks down profit margin by category.
    Which category should we interrogate next?
    """)

@workflow.atom()
def ch2_visual(df=load_data()):
    cat = df.groupby("Category").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    cat["Profit Margin"] = cat["Profit"] / cat["Sales"]
    fig = px.bar(cat, x="Category", y="Profit Margin", title="Profit Margin by Category")
    plotly(fig)

@workflow.atom()
def ch3_explanation():
    text("## 📘 Chapter 3: The Usual Suspects")
    text("""
    Not all customers are equal. Some segments are cash cows; others, black holes.

    Magnum B.I. lines up the suspects: Consumer, Corporate, Home Office.
    Who's draining the budget—and who's paying the bills?
    """)

@workflow.atom()
def ch3_visual(df=load_data()):
    seg = df.groupby("Segment").agg({"Profit": "sum"}).reset_index()
    fig = px.pie(seg, names="Segment", values="Profit", title="Profit Share by Segment")
    plotly(fig)
