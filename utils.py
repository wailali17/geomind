import os
import json
import numpy as np
import pandas as pd
import duckdb

def extract_all_the_places_data(min_lon, max_lon, min_lat, max_lat):
    atp_df = pd.DataFrame(columns=["longitude", "latitude", "spider_id", "shop", "full_address", "address", "city", "postcode", "business_name", "branch", "website", "brand"])
    for file in os.listdir("data/raw/alltheplaces"):
        if "_gb.geojson" in file:
            if os.stat(f"data/raw/alltheplaces/{file}").st_size == 0:
                continue
            else:
                with open(f"data/raw/alltheplaces/{file}") as f:
                    atp = json.load(f)
                for features in atp["features"]:
                    if features["geometry"]:
                        x, y = features["geometry"]["coordinates"]
                        if (x >= min_lon) & (x <= max_lon) & (y >= min_lat) & (y <= max_lat):
                            relevant_data = [
                                features["geometry"]["coordinates"][0],
                                features["geometry"]["coordinates"][1],
                                features["properties"].get("@spider", np.nan),
                                features["properties"].get("shop", np.nan),
                                features["properties"].get("addr:full", np.nan),
                                features["properties"].get("addr:street_address", np.nan),
                                features["properties"].get("addr:city", np.nan),
                                features["properties"].get("addr:postcode", np.nan),
                                features["properties"].get("name", np.nan),
                                features["properties"].get("branch", np.nan),
                                features["properties"].get("website", np.nan),
                                features["properties"].get("brand", np.nan),
                            ]
                            atp_df.loc[len(atp_df)] = relevant_data
                        else:
                            continue
        else:
            continue
    atp_df["longitude"] = round(atp_df["longitude"], 6)
    atp_df["latitude"] = round(atp_df["latitude"], 6)

    return atp_df

def extract_overture_data(min_lon, max_lon, min_lat, max_lat):
    con = duckdb.connect("data/raw/overture_maps.ddb")

    con.sql("install spatial")
    con.sql("install httpfs")
    con.sql("load spatial")
    con.sql("load httpfs")

    con.sql(f"""
        CREATE TABLE IF NOT EXISTS places AS 
        SELECT 
            id, 
            upper(names['primary']) as names_primary, 
            upper(addresses[1]['freeform']) as address, 
            upper(addresses[1]['locality']) as city, 
            upper(addresses[1]['region']) as state, 
            left(addresses[1]['postcode'], 5) as postcode, 
            geometry, 
            ST_X(geometry) as longitude,
            ST_Y(geometry) as latitude,
            categories 
        FROM (
            SELECT * 
            FROM read_parquet('s3://overturemaps-us-west-2/release/2024-09-18.0/theme=places/type=place/*', filename=true, hive_partitioning=1),
            WHERE 
                bbox.xmin BETWEEN {min_lon} AND {max_lon} AND
                bbox.ymin BETWEEN {min_lat} AND {max_lat} AND
                confidence > 0.5
        );
    """)
    overture_df = con.sql("""SELECT * FROM places""").df()
    overture_df["longitude"] = round(overture_df["longitude"], 6)
    overture_df["latitude"] = round(overture_df["latitude"], 6)
    return overture_df


def using_ons_datasets(retail_df, postcode):
    ons_lookup = pd.read_csv("data/raw/ons/PCD_OA_LSOA_MSOA_LAD_MAY22_UK_LU.csv", encoding="latin-1")
    retail_df = retail_df.merge(ons_lookup[["pcds", "lsoa11cd"]], left_on=postcode, right_on="pcds", how="left").drop("pcds", axis=1).rename(columns={"lsoa11cd":"lsoa"})
    ons_gva = pd.read_excel("data/raw/ons/uksmallareagvaestimates1998to2022.xlsx", sheet_name="Table 1", skiprows=1)
    retail_df = retail_df.merge(ons_gva[["LSOA code", "2022"]], left_on="lsoa", right_on="LSOA code", how="left").drop("LSOA code", axis=1).rename(columns={"2022":"gva_millions"})
    ons_pop = pd.read_excel("data/raw/ons/correcteddataforpublication.xlsx", skiprows=3, sheet_name="Table 1")
    retail_df = retail_df.merge(ons_pop[["LSOA", "Census Count"]], left_on="lsoa", right_on="LSOA", how="left").drop("LSOA", axis=1).rename(columns={"Census Count":"cencus_pop"})

    ons_hi_def = pd.read_excel("data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx", sheet_name="Table 15", skiprows=1)
    retail_df = retail_df.merge(
        ons_hi_def[[
            "LSOA 2011 code", "Travel to Work Area code", "Built-Up Area Sub Divisions code", "Westminster Parliamentary Constituency code",
            "Ward_Code", "Clinical Comissioning Group 21 Code"
            ]], 
        left_on="lsoa", right_on="LSOA 2011 code", how="left"
        ).drop("LSOA 2011 code", axis=1).rename(
            columns={
                "Travel to Work Area code":"ttwa_code",
                "Built-Up Area Sub Divisions code":"town_code",
                "Westminster Parliamentary Constituency code":"pc_code",
                "Ward_Code": "ward_code",
                "Clinical Comissioning Group 21 Code":"ccg_code",
                }
            )
    ons_ttwa_gdhi = pd.read_excel("data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx", sheet_name="Table 1", skiprows=1)
    retail_df = retail_df.merge(
        ons_ttwa_gdhi[ons_ttwa_gdhi["Transaction"] == "Gross Disposable Household Income"][["TTWA code", 2021]]
        , left_on="ttwa_code", right_on="TTWA code", how="left").drop("TTWA code", axis=1).rename(columns={2021:"ttwa_gdhi_millions"}
        )

    ons_town_gdhi = pd.read_excel("data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx", sheet_name="Table 2", skiprows=1)
    retail_df = retail_df.merge(
        ons_town_gdhi[ons_town_gdhi["Transaction"] == "Gross Disposable Household Income"][["Town code", 2021]]
        , left_on="town_code", right_on="Town code", how="left").drop("Town code", axis=1).rename(columns={2021:"town_gdhi_millions"}
        )

    ons_pc_gdhi = pd.read_excel("data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx", sheet_name="Table 3", skiprows=1)
    retail_df = retail_df.merge(
        ons_pc_gdhi[ons_pc_gdhi["Transaction"] == "Gross Disposable Household Income"][["PC code", 2021]]
        , left_on="pc_code", right_on="PC code", how="left").drop("PC code", axis=1).rename(columns={2021:"wgpc_gdhi_millions"}
        )

    ons_ward_gdhi = pd.read_excel("data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx", sheet_name="Table 6", skiprows=1)
    retail_df = retail_df.merge(
        ons_ward_gdhi[ons_ward_gdhi["Transaction"] == "Gross Disposable Household Income"][["Ward code", 2021]]
        , left_on="ward_code", right_on="Ward code", how="left").drop("Ward code", axis=1).rename(columns={2021:"ward_gdhi_millions"}
        )

    ons_ccg_gdhi = pd.read_excel("data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx", sheet_name="Table 7", skiprows=1)
    retail_df = retail_df.merge(
        ons_ccg_gdhi[ons_ccg_gdhi["Transaction"] == "Gross Disposable Household Income"][["CCG code", 2021]]
        , left_on="ccg_code", right_on="CCG code", how="left").drop("CCG code", axis=1).rename(columns={2021:"ccg_gdhi_millions"}
        )
    return retail_df
