import pandas as pd
from pathlib import Path

import helper
import postgres

file_path = [
    filename
    for filename in Path("../data").glob("*.csv")
    if filename.stem.startswith("applemobilitytrends")
][0]

postgres.create_tables_apple()


df = pd.read_csv(filepath_or_buffer=file_path.as_posix(), low_memory=False)

dates = df[df.columns[6:]]
nulls = dates[dates.isna().all(axis=1)].index
df.drop(index=nulls, inplace=True)

countries_util = (
    df.loc[:, ("region", "geo_type", "alternative_name", "sub-region", "country")]
    .drop_duplicates()
    .sort_values(by="region")
    .reset_index(drop=True)
)

df.drop(columns=["geo_type", "alternative_name", "country", "sub-region"], inplace=True)

res = helper.last_entry("mobility_stats_apple")

inter_df = helper.rearrange_df(df)
inter_df = inter_df[inter_df["date"] > str(res["date"].values[0])]


query = "select * from countries_apple"
existing_countries = (
    pd.read_sql_query(sql=query, con=helper._make_connection().connect())
    .sort_values(by="region")
    .reset_index(drop=True)
)

existing_countries.head()

countries_util = pd.concat([countries_util, existing_countries]).drop_duplicates(
    keep=False
)

postgres.import_data(countries_util, "countries_apple")
postgres.import_data(inter_df, "mobility_stats_apple")

file_path.unlink()
