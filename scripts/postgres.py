from sqlalchemy import (
    Column,
    Date,
    ForeignKey,
    MetaData,
    Numeric,
    String,
    Table,
    inspect,
)

import helper


def create_tables_google():
    engine = helper._make_connection()
    
    with engine.connect() as connection:
        with connection.begin():
            meta = MetaData(engine)

            countries_google = Table(
                'countries_google',
                meta,
                Column('country_region_code', String(2), primary_key=True),
                Column('country_region', String(30)),
            )
            location_util_google = Table(
                'location_util_google',
                meta,
                Column(
                    'country_region_code',
                    String(2),
                    ForeignKey('countries_google.country_region_code'),
                ),
                Column('iso_3166_2_code', String(6)),
                Column('census_fips_code', Numeric(precision=5)),
                Column('place_id', String(27)),
            )
            mobility_stats_google = Table(
                'mobility_stats_google',
                meta,
                Column(
                    'country_region_code',
                    String(2),
                    ForeignKey('countries_google.country_region_code'),
                ),
                Column('sub_region_1', String(100)),
                Column('sub_region_2', String(100)),
                Column('metro_area', String(50)),
                Column('date', Date),
                Column(
                    'retail_and_recreation_percent_change_from_baseline',
                    Numeric(precision=4, scale=0),
                ),
                Column(
                    'grocery_and_pharmacy_percent_change_from_baseline',
                    Numeric(precision=4, scale=0),
                ),
                Column(
                    'parks_percent_change_from_baseline', Numeric(precision=4, scale=0)
                ),
                Column(
                    'transit_stations_percent_change_from_baseline',
                    Numeric(precision=4, scale=0),
                ),
                Column(
                    'workplaces_percent_change_from_baseline',
                    Numeric(precision=4, scale=0),
                ),
                Column(
                    'residential_percent_change_from_baseline',
                    Numeric(precision=4, scale=0),
                ),
            )

    for table in [countries_google, location_util_google, mobility_stats_google]:
        if inspect(engine).has_table(table.name):
            print(f'=======\nTable {table.name} already exists! Skipping...')
        else:
            print(f'=======\nCreating table {table.name}...')
            table.create(engine)


def create_tables_apple():
    engine = helper._make_connection()
    
    with engine.connect() as connection:
        with connection.begin():
            meta = MetaData(engine)

            countries_apple = Table(
                'countries_apple',
                meta,
                Column('region', String(48)),
                Column('geo_type', String(14)),
                Column('alternative_name', String(85)),
                Column('sub-region', String(33)),
                Column('country', String(20)),
            )
            mobility_stats_apple = Table(
                'mobility_stats_apple',
                meta,
                Column('region', String(48), primary_key=True),
                Column('date', Date, primary_key=True),
                Column('driving', Numeric(precision=6, scale=2)),
                Column('transit', Numeric(precision=6, scale=2)),
                Column('walking', Numeric(precision=6, scale=2)),
            )

            for table in [countries_apple, mobility_stats_apple]:
                if inspect(engine).has_table(table.name):
                    print(f'=======\nTable {table.name} already exists! Skipping...')
                else:
                    print(f'=======\nCreating table {table.name}...')
                    table.create(engine)

        for table in [countries_apple, mobility_stats_apple]:
            if inspect(engine).has_table(table.name):
                print(f'=======\nTable {table.name} already exists! Skipping...')
            else:
                print(f'=======\nCreating table {table.name}...')
                table.create(engine)


def create_tables_json():
    engine = helper._make_connection()
    
    with engine.connect() as connection:
        with connection.begin():
            meta = MetaData(engine)
            
            covid_data_greece = Table(
                'covid_data_greece', 
                meta,
                Column('date', Date, primary_key=True),
                Column('confirmed', Numeric(precision=7, scale=0)),
                Column('recovered', Numeric(precision=7, scale=0)),
                Column('deaths', Numeric(precision=7, scale=0)),
            )

            if inspect(engine).has_table(covid_data_greece.name):
                print(f'=======\nTable {covid_data_greece.name} already exists! Skipping...')
            else:
                print(f'=======\nCreating table {covid_data_greece.name}...')
                covid_data_greece.create(engine)


def import_data(df, table_name):
    engine = helper._make_connection()
    df.to_sql(table_name, engine, index=False, if_exists='append')
