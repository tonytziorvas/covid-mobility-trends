import psycopg2
import os
import calendar
from datetime import datetime

from dotenv import load_dotenv

import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


TITLES = [
    "Retail and recreation",
    "Grocery and pharmacy stores",
    "Parks and outdoor spaces",
    "Public transport stations",
    "Workplace visitors",
    "Time spent at home",
]


def make_connection(table, column=None, value=None):
    """Create a connection to PostgreSQL database

    Returns
    -------
    connection
        Load DB credential from .env file and connect to PostgreSQL
    """
    load_dotenv()

    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")

    try:
        print("Connecting to the PostgreSQL database...")
        with psycopg2.connect(
            host=HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=PORT
        ) as conn:
            print("Connection successful")
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            db_version = cursor.fetchone()

            print(f"Version: {db_version}\nFetching Data from database...")

            if column is not None and value is not None:
                query = f"SELECT * FROM {table} WHERE {column} = '{value}'"
            else:
                query = f"SELECT * FROM {table}"

            df = pd.read_sql_query(sql=query, con=conn, parse_dates=["date"])
            print(f"Data Fetched\n============")

        return df

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
        return -1


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Check DataFrame for null rows and/or columns

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    DataFrame
        cleaned df
    """
    df.info()

    # Cleaning null columns
    print("\nChecking for null columns...")

    for column in df.columns:
        if df[column].isnull().all():
            df.drop(column, axis=1, inplace=True)
            print(f"-- Dropped null column: {column}")

    # Clearing null rows
    print("\nClearing null rows...")
    df.dropna(axis=0, how="all").sort_values(by="date", inplace=True)
    return df


def plot_corr_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Generate a correlation matrix

    Parameters
    ----------
    df : DataFrame
    """
    correlation_matrix = df.corr()
    labels = [" ".join(column.split("_")[:-4]).title() for column in df.columns[3:]]
    fig = px.imshow(correlation_matrix, x=labels, y=labels)
    fig.update_xaxes(side="top")
    fig.show()


def aggregate_by_group(df: pd.DataFrame, group: str = "season"):
    """Aggregate data specified by the group parameter (can be set either month or season)

    Parameters
    ----------
    df : DataFrame
    group : str, optional
        group to aggregate by, by default "season"

    Returns
    -------
    DataFrame   
    """

    # Extracting the month from the date column
    df["month"] = df["date"].dt.month.astype(int)

    if group == "season":
        # Grouping months into seasons
        df["season"] = df["month"] % 12 // 3 + 1
        df["season"][df["season"] == 1] = "Winter"
        df["season"][df["season"] == 2] = "Spring"
        df["season"][df["season"] == 3] = "Summer"
        df["season"][df["season"] == 4] = "Autumn"

    start_idx = df.columns.get_loc("retail_and_recreation_percent_change_from_baseline")
    stats = (
        df[df.columns[start_idx:]]
        .groupby(group)
        .agg(["mean", "median", "min", "max"])
        .reset_index()
    )

    return df, stats, start_idx


def season_histplots(df: pd.DataFrame, start_idx: int):

    for column in df.columns[start_idx : start_idx + 6]:
        fig = px.histogram(
            data_frame=df,
            x=column,
            marginal="box",
            width=1200,
            height=600,
            color="season",
        )
        fig.update_layout(bargap=0.05)
        fig.show()


def plot_exponential_moving_average(
    df: pd.DataFrame, column: str, alpha: int, title: str
):

    EMA = df[column].ewm(alpha=alpha, adjust=True).mean()
    x = df["date"]
    y = df[column]

    fig = go.Figure(layout_title_text=title)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="daily", opacity=0.5))
    fig.add_trace(
        go.Scatter(
            x=x, y=EMA, mode="lines", name="7-day moving average", fill="tozeroy"
        )
    )

    fig.show()


def day_by_day_trends(df: pd.DataFrame):
    """Plot daily trends 

    Parameters
    ----------
    df : source dataframe
    """

    for column, title in zip(df.columns[1:-1], TITLES):
        percent_from_baseline = df[column]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=df["day_name"], y=percent_from_baseline, width=0.5))
        fig.update_layout(title=title)

        fig.show()


def monthly_changes(df: pd.DataFrame):
    monthly_report = (
        df.resample("MS", on="date").mean().reset_index()
    )  # MS = Month Start

    # Range starts from 1 in order to skip Jan-2020
    month_range = [
        calendar.month_abbr[i % 12 + 1] for i in range(1, monthly_report.shape[0] + 1)
    ]
    year_range = [2020] * 11 + [2021] * 12

    # Create labels with format: MMM-YYYY
    x_labels = [
        month_range[i] + "-" + str(year_range[i]) for i in range(len(month_range))
    ]

    # Exclude month and date columns
    y_labels = [
        label.replace("_", " ").title() for label in monthly_report.columns[1:-1]
    ]

    for elem, label, title in zip(monthly_report.columns[1:-1], y_labels, TITLES):
        fig = px.line(
            data_frame=monthly_report,
            x=monthly_report.index.values,
            y=elem,
            title=title,
            labels={"x": "Month", elem: label},
            markers=True,
        )
        fig.update_layout(
            xaxis={
                "tickvals": tuple(range(monthly_report.shape[0])),
                "ticktext": x_labels,
                "tickangle": -45,
            }
        )
        fig.show()


def select_daterange(
    df: pd.DataFrame, start_date: datetime, end_date: datetime
) -> pd.DataFrame:

    fmt = "%Y-%m-%d"
    start_date = datetime.strptime(start_date, fmt)
    end_date = datetime.strptime(end_date, fmt)

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    return df.loc[mask]


def corr_heatmap(df: pd.DataFrame, x, y):
    x_index = df.columns.get_loc(x) - 3
    x_label = TITLES[x_index]

    y_index = df.columns.get_loc(y) - 3
    y_label = TITLES[y_index]

    fig = px.density_heatmap(
        data_frame=df,
        x=x,
        y=y,
        labels={x: x_label, y: y_label},
        marginal_x="violin",
        marginal_y="violin",
        width=800,
        height=800,
    )

    fig.show()
