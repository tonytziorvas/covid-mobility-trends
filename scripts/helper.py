import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from statsmodels.tsa.statespace.sarimax import SARIMAX
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


def _init():
    """
    > If the misc and plots folders don't exist in the current or parent directory, create them

    """
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
    """
    > Download a file from a URL and display a progress bar while downloading

    Parameters
    ----------
    url : str
        The URL of the file to download.
    block_size : int, optional
        The size of each chunk of data that is downloaded.

    Returns
    -------
        The file path of the downloaded file.

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
    """
    > Create a connection to the PostgreSQL database

    Parameters
    ----------
    dialect : str, optional
        The dialect of SQL to use.

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
    date_column: Optional[str] = None,
    chunk_size: int = 500000,
):
    """
    > It fetches data from a database table and returns it as a pandas dataframe

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
            # Returns a Dataframe
            return pd.concat(chunks, ignore_index=True, axis=0)


def last_entry(table_name: str):
    """
    > This function returns the last date in the table

    Parameters
    ----------
    table_name : str
        the name of the table you want to query

    Returns
    -------
        A dataframe with the max date from the table

    """

    query = f"select max(date) as date from {table_name}"

    with _make_connection().connect() as connection:
        with connection.begin():
            return pd.read_sql_query(sql=query, con=connection)


def rearrange_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    > The function takes in a dataframe, and returns a dataframe with the following transformations:

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
    inter_df = (
        pd.pivot_table(inter_df, index=["region", "date"], columns="transportation_type")
        .droplevel(level=0, axis=1)
        .rename_axis(None, axis=1)
        .reset_index()
    )

    return inter_df


def preproccess_pipeline(
    df: pd.DataFrame,
    numeric_columns: List[str],
    group_subset: Union[str, List[str]],
    threshold: float = None,
) -> pd.DataFrame:
    """
    > This function takes in a dataframe, a list of numeric columns, a list of columns to group by, and
    a threshold for dropping samples with missing values. It then checks for null columns, duplicate
    entries, and samples with missing values. It then fills the remaining missing values with forward
    fill

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to be preprocessed
    numeric_columns : List[str]
        A list of columns that are numeric. This is used to fill null values with forward fill method.
    group_subset : Union[str, List[str]]
        The columns that we want to group by when aggregating duplicate entries.
    threshold : float
        The threshold for dropping samples with missing values.

    Returns
    -------
        A dataframe with the following preprocessing steps:
    1. Drop columns that are all null
    2. Aggregate duplicate entries
    3. Drop samples that have more than 20% missing values
    4. Fill remaining null values with forward fill method

    """

    print("Step 1: Checking for null columns...")
    cols_to_drop = [column for column in df.columns if df[column].isnull().all()]
    if len(cols_to_drop) == 0:
        print("--- No null columns found")
    else:
        for column in cols_to_drop:
            print(f"--- Found null column: {column}")

    if "country_region_code" in df.columns:
        cols_to_drop.append("country_region_code")

    df.drop(columns=cols_to_drop, inplace=True)
    numeric_columns = list(set(numeric_columns) - set(cols_to_drop))

    print("Step 2: Checking for duplicate entries...")
    duplicates = (
        df[df.duplicated(subset=group_subset, keep=False)].groupby(by=group_subset).mean()
    )

    if duplicates.shape[0] != 0:
        print("--- Found duplicate entries... Aggregating result...")
        df.drop_duplicates(keep=False, inplace=True)
        df = pd.concat([df, duplicates], axis=0).sort_values(by="date")
    else:
        print("--- No duplicate entries found")

    print("Step 3: Dropping samples that have more than 20% missing values")
    df.dropna(thresh=int(df.shape[1] * threshold), axis=0, inplace=True)

    print("Step 4: Fill remaining null values with forward fill method")
    df.loc[:, numeric_columns].ffill(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def test_stationarity(timeseries: pd.Series, nlags: int = 0, confidence: float = 0.05) -> bool:
    """
    > The Augmented Dickey-Fuller test is a statistical test for testing
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
        True if the time series is stationary.

    """
    print("===========================")
    print(f" > Is the {nlags}-lag data stationary ?")
    reject = False
    result = adfuller(timeseries, autolag="AIC")

    print(f"Test statistic = {result[0]:.3f}")
    print(f"P-value = {result[1]:.3f}")
    print(f"Num of Lags = {result[2]:.3f}")
    print("Critical values :")
    for key, value in result[4].items():  # type: ignore
        if int(key[:-1]) / 100 == confidence:
            reject = result[0] > value
            print(
                f"\t{key}: {value} - The data is {'not' if reject else ''} stationary with {100-int(key[:-1])}% confidence"
            )

    return reject


def optimize_SARIMA(parameters_list: List[int], d: int, D: int, exog: str) -> pd.DataFrame:
    """
    > Given a list of parameters, it will try to fit a SARIMA model for each parameter and return the one
    with the lowest AIC

    Parameters
    ----------
    parameters_list - list with (p, q, P, Q) tuples
    d : int
        The number of times that the previous observation is used to forecast the next value.
    D : int
        Non-seasonal differencing (by how many times the original series was differenced)
    exog : str
        The univariate time series as a pandas Series object.

    Returns
    -------
        A dataframe with the parameters and the AIC value

    """

    results = []

    for param in tqdm(parameters_list):
        try:
            model = SARIMAX(
                exog,
                order=(param[0], d, param[1]),
                seasonal_order=(param[2], D, param[3], param[4]),
            ).fit(disp=False, method="powell")
        except:
            continue

        aic = model.aic
        results.append([param, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q)x(P,Q)", "AIC"]

    result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True, inplace=True)

    return result_df


def walk_forward_validation(data: pd.DataFrame, column: str, n_train: int):
    """
    > The function takes in a dataframe, a column name, and the number of training days. It then creates a
    SARIMA model, forecasts the next week, and adds that week to the training data. It repeats this
    process until all the data has been forecasted

    Parameters
    ----------
    data : pd.DataFrame
        the dataframe containing the data to be used for the model
    column : str
        the column of the dataframe to be forecasted
    n_train : int
        The number of days to use for training.

    """

    predictions = np.array([])
    forecast_ci = np.zeros((0, 2))
    mape_list = []
    train = data[:n_train]
    test = data[column][n_train:]
    day_list = list(range(0, test.shape[0], 7))  # weeks 1, 2, 3, 4...

    for i in day_list:
        model = SARIMAX(train, order=(3, 1, 1), seasonal_order=(3, 1, 1, 7)).fit(
            method="powell"
        )

        # Forecast daily loads for week i
        forecast = model.get_forecast(steps=7)
        forecast_ci = np.concatenate((forecast_ci, forecast.conf_int()), axis=0)
        predictions = np.concatenate((predictions, forecast.predicted_mean), axis=None)
        j = i - 7

        mape_score = (abs(test[j:i] - predictions[j:i]) / test[j:i]) * 100
        mape_mean = mape_score.mean()
        mape_list.append(mape_mean)  # Add week i to training data for next loop
        train = np.concatenate((train, test[j:i]), axis=None)

    return predictions, mape_list, forecast_ci
