import calendar
import os
import re
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dotenv import load_dotenv
from matplotlib import offsetbox
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

TITLES = [
    "Retail and recreation",
    "Grocery and pharmacy stores",
    "Parks and outdoor spaces",
    "Public transport stations",
    "Workplace visitors",
    "Time spent at home",
]


def __init():
    if not os.path.exists("../misc") or os.path.exists("misc"):
        print(
            "No misc folder located at current or parent directory.\n Creating misc folder in parent directory..."
        )
        os.mkdir("../misc")
    else:
        print("Misc Folder located!")
    if not os.path.exists("../plots") or os.path.exists("plots"):
        print(
            "No plots folder located at current or parent directory.\n Creating plots folder in parent directory..."
        )
        os.mkdir("../plots")
    else:
        print("Plots Folder located!")


def download_file(url: str, block_size: int = 4096) -> str:
    """It downloads a file from a URL and saves it to a specified location
    
    Parameters
    ----------
    url : str
        The URL of the file you want to download.
    block_size : int, optional
        The amount of data to read at a time.
    
    Returns
    -------
        The file path to the downloaded file.
    
    """
    with requests.get(url, stream=True) as res:
        file_name = url.split("/")[-1]
        file_path = f"../../Data/{file_name}"
        total_size = int(res.headers.get("content-length", 0))

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
        "=================================================\n"
        + "Creating connection to the PostgreSQL database...\n"
        + "================================================="
    )

    connection_string = (
        f"postgresql+{dialect}://{DB_USER}:{DB_PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
    )
    return create_engine(connection_string, echo=False)


def fetch_data_from_database(
    table,
    where_column: Optional[str] = None,
    where_value: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: Optional[str] = None,
    date_column: str = "date",
    chunk_size: int = 500000,
):
    """It fetches data from a database table and returns it as a pandas dataframe
    
    Parameters
    ----------
    table
        The name of the table you want to fetch from the database.
    where_column : Optional[str]
        The column to filter on.
    where_value : Optional[str]
        The value of the column that you want to filter by.
    order_by : Optional[str]
        This is the column name to order the data by.
    limit : Optional[str]
        The number of rows to fetch from the database.
    date_column : Optional[str]
        This is a list of columns that should be parsed as datetime.
    chunk_size : int, optional
        The number of rows to read in at a time.
    
    Returns
    -------
        A dataframe
    
    """
    with _make_connection().connect() as connection:

        print("Opening connection...")
        with connection.begin():
            if where_column is not None and where_value is not None:
                query = f"SELECT * FROM {table} WHERE {where_column} = '{where_value}'"
            else:
                query = f"SELECT * FROM {table}"

            if order_by is not None:
                query += f" ORDER BY {order_by}"

            if limit is not None:
                query += f" LIMIT {limit}"

            print(f"Data Fetched\n============")

            chunks = [
                c
                for c in pd.read_sql_query(
                    sql=query,
                    con=connection,
                    chunksize=chunk_size,
                    parse_dates=date_column,
                )
            ]

            return pd.concat(chunks, ignore_index=True, axis=0)


def rearrange_df(df: pd.DataFrame) -> pd.DataFrame:
    """The function takes in a dataframe, and returns a dataframe with the following transformations:

    - The dataframe is melted, so that the date and the transportation type become the two columns.
    - The dataframe is pivoted, so that the transportation types become the columns.
    - The dataframe is indexed by region and date.
    - The dataframe is renamed so that the transportation types are the column names.
    - The dataframe is reset, so that the index is the only index

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to modify.

    Returns
    -------
        A dataframe with the following columns:
        - region
        - date
        - walking
        - driving
        - transit

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
    inter_df = inter_df.rename_axis(None, axis=1).reset_index()

    return inter_df


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """The function takes in a dataframe and returns a cleaned dataframe

    Parameters
    ----------
    df : pd.DataFrame
        the dataframe to be cleaned

    Returns
    -------
        A dataframe with the cleaned data.

    """
    print("\nChecking for null columns...")

    for column in df.columns:
        if df[column].isnull().all():
            df.drop(column, axis=1, inplace=True)
            print(f"-- Dropped null column: {column}")

    print("\nClearing null rows...")
    df.dropna(axis=0, how="all").sort_values(by="date", inplace=True)
    return df


def plot_ema(df: pd.DataFrame, column: str, alpha: int, title: str):
    """It plots the daily values of a column in a dataframe, and plots the exponential moving average of
    the column
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot
    column : str
        The column to plot.
    alpha : int
        The smoothing factor for the exponential moving average.
    title : str
        The title of the plot.
    
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

    # MS = Month Start
    monthly_report = df.resample("MS", on="date").mean().reset_index()

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


def test_stationarity(
    timeseries: pd.Series, nlags: int = 0, confidence: float = 0.05
) -> bool:
    """The Augmented Dickey-Fuller test is a statistical test for testing 
    whether a time series has a unit root, e.g. has a trend or more generally is autoregressive
    
    Parameters
    ----------
    timeseries : pd.Series
        the time series to test
    nlags : int, optional
        The number of lags to use in the ADF regression. If 0, the method selects the lag length using the
    maxlag parameter.
    confidence : float
        float, optional
    
    Returns
    -------
        a boolean value.
    
    """
    print("===========================")
    print(f" > Is the {nlags}-lag data stationary ?")
    reject = False
    result = adfuller(timeseries, autolag="AIC")

    print(f"Test statistic = {result[0]:.3f}")
    print(f"P-value = {result[1]:.3f}")
    print("Critical values :")
    for key, value in result[4].items():  # type: ignore
        if int(key[:-1]) / 100 == confidence:
            reject = result[0] > value
            print(
                f"\t{key}: {value} - The data is {'not' if reject else ''} stationary with {100-int(key[:-1])}% confidence"
            )

    return reject


def select_daterange(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:

    fmt = "%Y-%m-%d"
    start_date = datetime.strptime(start, fmt)
    end_date = datetime.strptime(end, fmt)

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)

    return df.loc[mask]


def corr_heatmap(df: pd.DataFrame, x_column, y_column):
    x_index = df.columns.get_loc(x_column) - 4
    x_label = TITLES[x_index]

    y_index = df.columns.get_loc(y_column) - 45
    y_label = TITLES[y_index]

    fig = px.density_heatmap(
        data_frame=df,
        x=x_column,
        y=y_column,
        labels=dict(x_column=x_label, y_column=y_label),
        width=800,
        height=800,
    )

    fig.show()
