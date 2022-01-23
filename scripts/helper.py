import requests
import psycopg2
import os
from io import StringIO

import calendar
from datetime import datetime

from tqdm import tqdm  # Progress Bar
from dotenv import load_dotenv
from pathlib import Path
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


def download_file(url):
    """
    Download the requested file to the same directory
    """

    file_name = url.split("/")[-1]

    req = requests.get(url, stream=True, allow_redirects=True)
    total_size = int(req.headers.get("content-length"))
    initial_pos = 0
    file_path = f"../../Data/{file_name}"

    # Progress Bar to monitor download
    with open(file_path, "wb") as obj:
        with tqdm(
            total=total_size,
            unit_scale=True,
            desc=file_name,
            initial=initial_pos,
            ascii=True,
        ) as pbar:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    obj.write(chunk)
                    pbar.update(len(chunk))

    return file_path


def _make_connection():
    """Create a connection to PostgreSQL database

    Returns
    -------
    connection
        Load DB credential from .env file and connect to PostgreSQL
    """
    dotenv_path = Path("../misc/.env")
    load_dotenv(dotenv_path=dotenv_path)

    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    HOST = os.getenv("HOST")
    PORT = os.getenv("PORT")

    try:
        print("Connecting to the PostgreSQL database...")
        conn = psycopg2.connect(
            host=HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD, port=PORT
        )

        print("Connection successful")
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        db_version = cursor.fetchone()

        print(f"Version: {db_version}\nFetching Data from database...")

        return conn

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Error: {error}")
        return -1


def fetch_data_from_database(table, column=None, value=None) -> pd.DataFrame:

    conn = _make_connection()

    if column is not None and value is not None:
        query = f"SELECT * FROM {table} WHERE {column} = '{value}'"
    else:
        query = f"SELECT * FROM {table}"

    df = pd.read_sql_query(sql=query, con=conn, parse_dates=["date"])
    print(f"Data Fetched\n============")

    return df


def create_tables_google(conn):

    queries = (
        """ 
        CREATE TABLE IF NOT EXISTS countries_google (
            country_region_code varchar(2) PRIMARY KEY,
            country_region varchar(30)    
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS location_util_google (
            country_region_code varchar(2),
            iso_3166_2_code varchar(6),
            census_fips_code numeric(5,0),
            place_id varchar(27),
            
            FOREIGN KEY (country_region_code) 
                REFERENCES countries_google (country_region_code)
        )
        """,
        """ 
        CREATE TABLE IF NOT EXISTS mobility_stats_google (
            country_region_code varchar(2),
            sub_region_1 varchar(100),
            sub_region_2 varchar(100),
            metro_area varchar(50),
            date date,
            retail_and_recreation_percent_change_from_baseline numeric(4,0),
            grocery_and_pharmacy_percent_change_from_baseline numeric(4,0),
            parks_percent_change_from_baseline numeric(4,0),
            transit_stations_percent_change_from_baseline numeric(4,0),
            workplaces_percent_change_from_baseline numeric(4,0),
            residential_percent_change_from_baseline numeric(4,0),
            
            FOREIGN KEY (country_region_code) 
                REFERENCES countries_google (country_region_code)
        )
        """,
    )

    try:
        with conn.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
            conn.commit()
            print("Tables created successfully!\n----------------------------")
            return 1
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while creating tables!\nRolling back changes...\n", error)
        conn.rollback()
        return -1


def create_tables_apple(conn):

    queries = (
        """ 
        CREATE TABLE IF NOT EXISTS countries_apple (
            region varchar(48),
            geo_type varchar(14),
            alternative_name varchar(85),
            sub_region varchar(33),
            country varchar(20)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS mobility_stats_apple (
            region varchar(48),
            date date,
            driving numeric(6,2),
            transit numeric(6,2),
            walking numeric(6,2),
            
            PRIMARY KEY (region, date)
        )
        """,
    )

    try:
        with conn.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
            conn.commit()
            print("Tables created successfully!\n----------------------------")

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while creating tables!\nRolling back changes...\n", error)
        conn.rollback()
        return -1


def create_tables_json(conn):

    queries = """ 
        CREATE TABLE IF NOT EXISTS covid_data_greece (
            date date PRIMARY KEY,
            confirmed numeric(7, 0),
            recovered numeric(5, 0),
            deaths numeric(5, 0)
        )
        """

    try:
        with conn.cursor() as cursor:
            for query in queries:
                cursor.execute(query)
            conn.commit()

            print(
                f"Table `covid_data_greece` created successfully!\n-----------------------------------------------"
            )

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error while creating tables!\nRolling back changes...\n", error)
        conn.rollback()


def import_data(conn, table_name, df) -> int:

    buffer = StringIO()
    df.to_csv(buffer, header=False, index=False)
    buffer.seek(0)

    with conn.cursor() as cursor:
        try:
            cursor.execute(f"TRUNCATE {table_name} CASCADE;")
            print(f"Truncated {table_name}")

            df.where(pd.notnull(df), None)

            cursor.copy_expert(f"COPY {table_name} from STDIN CSV QUOTE '\"'", buffer)
            conn.commit()
            print("Done!\n-------------------------------")
            return 1

        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            return -1


def rearrange_df(df) -> pd.DataFrame:

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
        value_vars=df.columns[2:],
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
