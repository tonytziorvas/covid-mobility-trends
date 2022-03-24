import os
from pathlib import Path
from typing import List, Optional

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
    """
    It downloads a file from a URL and saves it to a specified location
    
    Parameters
    ----------
    url : str
        The URL of the file you want to download.
    block_size : int, optional
        The amount of data to read into memory at once.
    
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
    """
    Create a connection to the PostgreSQL database
    
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

    connection_string = f"postgresql+{dialect}://{DB_USER}:{DB_PASSWORD}@{HOST}:{PORT}/{DB_NAME}"
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
    It fetches data from a database table and returns it as a pandas dataframe
    
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
                    sql=query, con=connection, chunksize=chunk_size, parse_dates=date_column,
                )
            ]
            # Returns a Dataframe
            return pd.concat(chunks, ignore_index=True, axis=0)


def rearrange_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    The function takes in a dataframe, and returns a dataframe with the following transformations:

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

    inter_df = pd.pivot_table(inter_df, columns="transportation_type", index=["region", "date"])

    inter_df.columns = inter_df.columns.droplevel()
    inter_df.rename_axis(None, axis=1).reset_index(inplace=True)

    return inter_df


def preproccess_pipeline(df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
    """- Drop columns that have all null values
    - Drop duplicate rows
    - Drop rows where more than 20% of the values are null
    - Forward fill missing numeric values
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to preprocess
    numeric_columns : List[str]
        The columns that we want to fill
    
    Returns
    -------
        The preproccess_pipeline function returns a dataframe with the following characteristics:
        1. All null columns have been dropped
        2. All duplicate rows have been dropped
        3. All rows with more than 20% of the values missing have been dropped
        4. All rows with null values have been dropped
        5. All numeric columns have been forward filled
    
    """

    print("Step 1: Checking for null columns...")
    cols_to_drop = [column for column in df.columns if df[column].isnull().all()]
    if len(cols_to_drop) == 0:
        print("-- No null columns found")
    else:
        for column in cols_to_drop:
            print(f"-- Found null column: {column}")

    cols_to_drop.append("country_region_code")
    df.drop(columns=cols_to_drop, inplace=True)

    print("Step 2: Removing duplicate entries...")
    df.drop_duplicates(keep="first", inplace=True)

    print("Step 3: Dropping samples that have more than 20% missing values")
    df.dropna(thresh=int(df.shape[1] * 0.8), axis=0, inplace=True)

    print("Step 4: Fill remaining null values with forward fill method")
    df.loc[:, numeric_columns].ffill(inplace=True)
    df.reset_index(inplace=True, drop=True)

    return df


def test_stationarity(timeseries: pd.Series, nlags: int = 0, confidence: float = 0.05) -> bool:
    """
    The Augmented Dickey-Fuller test is a statistical test for testing 
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
    print(f"Num of Lags = {result[2]:.3f}")
    print("Critical values :")
    for key, value in result[4].items():  # type: ignore
        if int(key[:-1]) / 100 == confidence:
            reject = result[0] > value
            print(
                f"\t{key}: {value} - The data is {'not' if reject else ''} stationary with {100-int(key[:-1])}% confidence"
            )

    return reject


def optimize_SARIMA(parameters_list: List[int], d: int, D: int, endog) -> pd.DataFrame:
    """
    Given a list of parameters, it will try to fit a SARIMA model for each parameter and return the one
    with the lowest AIC
    
    Parameters
    ----------
    parameters_list - list with (p, q, P, Q) tuples
    d : int
        The number of times that the previous observation is used to forecast the next value.
    D : int
        Non-seasonal differencing (by how many times the original series was differenced)
    endog
        The univariate time series as a pandas Series object.
    
    Returns
    -------
        A dataframe with the parameters and the AIC value
    
    """

    results = []

    for param in tqdm(parameters_list):
        try:
            model = SARIMAX(
                endog, order=(param[0], d, param[1]), seasonal_order=(param[2], D, param[3], param[4])
            ).fit(disp=False, method="powell")
        except:
            continue

        aic = model.aic
        results.append([param, aic])

    result_df = pd.DataFrame(results)
    result_df.columns = ["(p,q)x(P,Q)", "AIC"]

    return result_df.sort_values(by="AIC", ascending=True).reset_index(drop=True)


def walk_forward_validation(data, n_test):
    """
    The function takes in a time series, splits it into training and testing data, and then fits a
    SARIMA model to the training data
    
    Parameters
    ----------
    data
        the data to be used for training and testing
    n_test
        Number of days to forecast
    
    Returns
    -------
        the predictions and the mape_list.
    
    """

    predictions = np.array([])
    mape_list = []
    train, test = data[:n_test], data[n_test:]
    day_list = [7, 14, 21, 28]  # weeks 1,2,3,4
    for i in day_list:
        model = SARIMAX(train, order=(3, 1, 1), seasonal_order=(3, 1, 1, 7)).fit(method="powell")

        # Forecast daily loads for week i
        forecast = model.get_forecast(steps=7)
        predictions = np.concatenate((predictions, forecast), axis=None)
        j = i - 7
        mape_score = (abs(test[j:i] - predictions[j:i]) / test[j:i]) * 100
        mape_mean = mape_score.mean()
        mape_list.append(mape_mean)  # Add week i to training data for next loop
        train = np.concatenate((train, test[j:i]), axis=None)

        return predictions, mape_list
