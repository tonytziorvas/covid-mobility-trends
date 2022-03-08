import calendar
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from tqdm import tqdm

TITLES = [
    "Retail and recreation",
    "Grocery and pharmacy stores",
    "Parks and outdoor spaces",
    "Public transport stations",
    "Workplace visitors",
    "Time spent at home",
]


def init():
    if not os.path.exists("../misc") or os.path.exists("misc"):
        print("No misc folder located\n Creating misc folder in parent directory...")
        os.mkdir("../misc")
    else:
        print("Misc Folder located!")
    if not os.path.exists("../plots") or os.path.exists("plots"):
        print("No plots folder located\n Creating plots folder in parent directory...")
        os.mkdir("../plots")
    else:
        print("Plots Folder located!")


def download_file(url):
    with requests.get(url, stream=True) as res:
        file_name = url.split("/")[-1]
        file_path = f"../../Data/{file_name}"
        total_size = int(res.headers.get("content-length", 0))
        block_size = 4096

        with open(file_path, "wb") as file:
            with tqdm(
                total=total_size, unit_scale=True, desc=file_name, initial=0, ascii=True
            ) as progress_bar:
                for data in res.iter_content(chunk_size=block_size):
                    progress_bar.update(len(data))
                    file.write(data)

        return file_path


def _make_connection(dialect: str = "psycopg2") -> Engine:
    """Make a connection to the database using the psycopg2 dialect if no other is sp

    Parameters
    ----------
    dialect : str, optional

    Returns
    -------
        A connection to the database.

    """
    dotenv_path = Path("../misc/.env")
    load_dotenv(dotenv_path=dotenv_path)

    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")
    DB_NAME = os.getenv("DB_NAME")

    print(
        "Connecting to the PostgreSQL database...\n========================================"
    )

    connection_string = (
        f"postgresql+{dialect}://{DB_USER}:{DB_PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
    )
    return create_engine(connection_string, echo=True)


def fetch_data_from_database(
    table,
    where_column: Optional[str] = None,
    where_value: Optional[str] = None,
    order_by: bool = False,
    order_col: Optional[str] = None,
    limit: bool = False,
    limit_amount: Optional[str] = None,
    chunk_size: int = 500000,
) -> pd.DataFrame:

    engine = _make_connection()
    with engine.connect() as connection:
        with connection.begin():
            if where_column is not None and where_value is not None:
                query = f"SELECT * FROM {table} WHERE {where_column} = '{where_value}'"
            else:
                query = f"SELECT * FROM {table}"

            if order_by and order_col is not None:
                query += f" ORDER BY {order_col}"

            if limit and limit_amount is not None:
                query += f" LIMIT {limit_amount}"

            print(f"Data Fetched\n============")

            chunks = [
                c for c in pd.read_sql(sql=query, con=connection, chunksize=chunk_size)
            ]
            return pd.concat(chunks, ignore_index=True).to_frame()


def rearrange_df(df: pd.DataFrame) -> pd.DataFrame:

    """
    1. Convert columns to rows
    2. Convert rows to columns
    3. Drop level to get one level of columns
    4. Rename axis to remove transportation_type which is our previous column name
    5. Reset index to convert our indices to columns
    """

    inter_df = pd.melt(
        frame=df,
        id_vars=["region", "transportation_type"],
        value_vars=tuple(df.columns[2:]),
        var_name="date",
        value_name="percent_change_from_baseline",
    )
    inter_df = pd.pivot_table(
        inter_df, columns="transportation_type", index=["region", "date"]
    )

    inter_df.columns = inter_df.columns.droplevel()
    inter_df = inter_df.rename_axis(None, axis=1)
    inter_df = inter_df.reset_index()

    return inter_df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    It takes a dataframe as an argument, and returns a cleaned dataframe.

    Args:
      df (pd.DataFrame): The dataframe to clean

    Returns:
      A cleaned dataframe.
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


def plot_corr_matrix(df: pd.DataFrame):
    """
    Plot a correlation matrix of a pandas dataframe.

    Args:
      df (pd.DataFrame): The dataframe to be plotted
    """

    correlation_matrix = df.corr()
    labels = [" ".join(column.split("_")[:-4]).title() for column in df.columns[3:]]
    fig = px.imshow(correlation_matrix, x=labels, y=labels)
    fig.update_xaxes(side="top")
    fig.show()


def aggregate_by_group(df: pd.DataFrame, group: str = "season"):
    """
    It takes a dataframe and a group name as input.
    It then extracts the month from the date column and groups the months into seasons.
    It then calculates the mean, median, min and max for each season.

    Args:
      df (pd.DataFrame): The dataframe to aggregate
      group (str): The column to group by. Defaults to season

    Returns:
      The dataframe, the statistics, and the index of the first column of the statistics.
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
    """
    It plots the daily values of a column in a dataframe, and plots the exponential moving average of
    the column.

    Args:
      df (pd.DataFrame): The dataframe containing the data to be plotted.
      column (str): the column in the dataframe to plot
      alpha (int): The smoothing factor.
      title (str): The title of the plot
    """

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
    """
    It takes a dataframe, resamples it by month, and then plots the mean of each column.

    Args:
      df (pd.DataFrame): pd.DataFrame
    """
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


def corr_heatmap(df: pd.DataFrame, x_column, y_column):
    x_index = df.columns.get_loc(x_column) - 3
    x_label = TITLES[x_index]

    y_index = df.columns.get_loc(y_column) - 3
    y_label = TITLES[y_index]

    fig = px.density_heatmap(
        data_frame=df,
        x=x_column,
        y=y_column,
        labels={x_column: x_label, y_column: y_label},
        marginal_x="violin",
        marginal_y="violin",
        width=800,
        height=800,
    )

    fig.show()
