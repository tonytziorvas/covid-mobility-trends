from pathlib import Path
import json
import pandas as pd
import helper
import postgres


root = "../data/covid19-data-greece/data/greece"
file = Path(root, "general/timeseries_greece.json")

with open(file, "r") as f:
    data = json.loads(f.read())

    postgres.create_tables_json()
    df = pd.json_normalize(data, record_path="Greece")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

        res = helper.last_entry("covid_data_greece")
        df = df[df["date"] > str(res["date"].values[0])]
        postgres.import_data(df, "covid_data_greece")
