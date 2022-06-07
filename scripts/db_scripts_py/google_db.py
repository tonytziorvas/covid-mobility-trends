import pandas as pd

import helper
import postgres


file_path = helper.download_file(
    url="https://www.gstatic.com/covid19/mobility/Global_Mobility_Report.csv"
)


postgres.create_tables_google()

df = pd.read_csv(filepath_or_buffer=file_path, parse_dates=True, low_memory=False)


res = helper.last_entry("mobility_stats_google")
df = df[df["date"] > str(res["date"].values[0])]


countries = df.loc[:, ("country_region_code", "country_region")].drop_duplicates()
countries["country_region_code"][countries["country_region"] == "Namibia"] = "NA"
countries = countries.sort_values(by="country_region_code").reset_index(drop=True)

stats = df.drop(
    columns=["country_region", "iso_3166_2_code", "census_fips_code", "place_id"]
)

location_util = (
    df.loc[
        :, ("country_region_code", "iso_3166_2_code", "census_fips_code", "place_id")
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


query = "select * from countries_google"
existing_countries = (
    pd.read_sql_query(sql=query, con=helper._make_connection().connect())
    .sort_values(by="country_region_code", axis=0)
    .reset_index(drop=True)
)


countries = [
    x
    for x in countries.values
    if x[0] not in existing_countries["country_region_code"].values
]
countries = pd.DataFrame(countries, columns=["country_region_code", "country_region"])


query = "select * from location_util_google"
existing_util = pd.read_sql_query(sql=query, con=helper._make_connection().connect())


location_util = [
    x
    for x in location_util.values
    if x[0] not in existing_util.country_region_code.values
]
location_util = pd.DataFrame(
    location_util,
    columns=["country_region_code", "iso_3166_2_code", "census_fips_code", "place_id"],
)


postgres.import_data(countries, "countries_google")
postgres.import_data(stats, "mobility_stats_google")
postgres.import_data(location_util, "location_util_google")
