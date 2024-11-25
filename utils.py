from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from langchain.schema import SystemMessage, HumanMessage
from sklearn.model_selection import train_test_split
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from sklearn.neighbors import BallTree
from rapidfuzz import process, fuzz
from geopy.distance import geodesic
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from typing import List, Dict
from typing import Optional
import geopandas as gpd
import xgboost as xgb
import pandas as pd
import numpy as np
import os


sector_multipliers = {
    'Offices (Inc Computer Centres)': (3, 5),
    'Shops': (2, 4),
    'Factories, Workshops and Warehouses (Incl Bakeries & Dairies)': (2, 3),
    'Car Spaces': (1, 2),
    'Stores': (2, 3),
    'Advertising Right': (1, 2),
    'Showrooms': (3, 6),
    'Public Houses/Pub Restaurants (National Scheme)': (6, 8),
    'Communication Stations (National Scheme)': (1, 2),
    'Restaurants': (3, 5),
    'Hairdressing/Beauty Salons': (2, 4),
    'Pitches for Stalls, Sales or Promotions': (2, 3),
    'Land Used For Storage': (1, 2),
    'Vehicle Repair Workshops & Garages': (2, 4),
    'Local Authority Schools (National Scheme)': (1, 1),  # Minimal turnover
    'Independent Distribution Network Operators (INDOs)': (1, 2),
    'Car Parks (Surfaced Open)': (1, 2),
    'Cafes': (3, 5),
    'Day Nurseries/Play Schools': (2, 4),
    'Surgeries, Clinics, Health Centres (Rental Valuation)': (2, 3),
    'Community Day Centres': (1, 2),
    'Other': (5,5)
}

def xgb_model(df, drop_cols, target, model_type, evaluation_df):
    X = df.drop([*drop_cols, target], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    # print(f"RMSE: {rmse} in comparison to STD: {np.std(y_test)}\n- The difference between to STD and RMSE is {round(np.std(y_test) - rmse, 2)}")
    # print(f"MAPE: {mape}")
    # print(f"R2 score: {r2}")


    X_test["actual"] = y_test
    X_test["preds"] = y_pred

    underzero = len(X_test[X_test.preds < 0]) / len(X_test)
    # print(f"{round(underzero*100, 2)}% of the test set are negative predictions")
    evaluation_df.loc[model_type] = [rmse, np.std(y_test), mape, r2, underzero]

    plt.barh(
        model.feature_names_in_,
        model.feature_importances_,
    )
    plt.title(f"Feature improtance for {model_type}")
    plt.show()
    return evaluation_df

def using_openstreetmap_data(df):
    loc_df = df.to_crs(epsg=3857)
    locations_coords = np.array(list(zip(loc_df.geometry.x, loc_df.geometry.y)))
    waterways_gdf = gpd.read_file("data/raw/openstreetmap/gis_osm_waterways_free_1.shp")
    waterways_gdf = waterways_gdf.to_crs(epsg=3857)
    waterways_gdf["centroid"] = waterways_gdf["geometry"].centroid
    waterways_gdf = waterways_gdf.set_geometry("centroid")

    waterways_coords = np.array(list(zip(waterways_gdf.geometry.x, waterways_gdf.geometry.y)))
    water_tree = cKDTree(waterways_coords)
    water_distances, water_indices = water_tree.query(locations_coords, k=1)
    loc_df["distance_to_water"] = water_distances
    loc_df["nearest_waterway_index"] = water_indices

    transport_gdf = gpd.read_file("data/raw/openstreetmap/gis_osm_transport_free_1.shp")
    transport_gdf = transport_gdf.to_crs(epsg=3857)
    transport_coords = np.array(list(zip(transport_gdf.geometry.x, transport_gdf.geometry.y)))
    transport_tree = cKDTree(transport_coords)
    transport_distances, transport_indices = transport_tree.query(locations_coords, k=1)
    loc_df["distance_to_transport"] = transport_distances
    loc_df["nearest_transport_index"] = transport_indices

    return loc_df

def preprocess_address(address):
    # Convert to lowercase, remove common noise terms, and strip punctuation
    noise_terms = ["unit", "ground floor", "part", "floor", "units"]
    if address is None:
        return ""
    address = address.lower()
    for term in noise_terms:
        address = address.replace(term, "")
    address = address.replace(",", "").strip()
    return address

def combining_geolytix_openlocal(geolytix_df, openlocal_df):
    geolytix_df["combined_address"] = geolytix_df.apply(
        lambda row: f"{row['add_one']} {"" if pd.isna(row["add_two"]) else row["add_two"]} {row["postcode"]}".strip().lower(), axis=1
    ).apply(preprocess_address)
    openlocal_df["normalized_address"] = openlocal_df["voapropertyaddress"].apply(preprocess_address)

    openlocal_coords = np.radians(openlocal_df[["latitude", "longitude"]])
    geolytix_coords = np.radians(geolytix_df[["latitude", "longitude"]])
    # Haversine distance (km)
    tree = BallTree(openlocal_coords, metric="haversine")  
    
    # Convert 500 meters to radians (Earth radius = 6371 km)
    radius = 0.5 / 6371  
    indices_list = tree.query_radius(geolytix_coords, r=radius)
    geolytix_df["retailer"] = np.where(geolytix_df["fascia"].str.contains("Extra"), geolytix_df["retailer"].str.replace("Extra", "Superstore"), geolytix_df["retailer"])

    matches = []
    for idx, nearby_indices in enumerate(indices_list):
        geo_row = geolytix_df.iloc[idx]
        candidates = openlocal_df.iloc[nearby_indices]

        # Iterate through all candidates to find the best match
        best_match = None
        best_score = 0
        best_distance = float("inf")
        for _, open_row in candidates.iterrows():
            sim_score = fuzz.partial_ratio(geo_row["combined_address"], open_row["normalized_address"])
            geo_point = (geo_row["latitude"], geo_row["longitude"])
            open_point = (open_row["latitude"], open_row["longitude"])
            distance = geodesic(geo_point, open_point).meters

            # Check if retailer name is present in the OpenLocal address
            retailer_weight = 10 if geo_row["retailer"].lower() in open_row["normalized_address"] else 0
            weighted_score = sim_score + retailer_weight

            # Update the best match based on the weighted score and distance
            if weighted_score > best_score or (weighted_score == best_score and distance < best_distance):
                best_match = open_row
                best_score = weighted_score
                best_distance = distance

        # Append the best match after checking all candidates
        if best_match is not None:
            matches.append({
                "geolytix_id": geo_row["id"],
                "geolytix_address": geo_row["combined_address"],
                "openlocal_id": best_match["ogc_fid"],
                "openlocal_address": best_match["voapropertyaddress"],
                "similarity": best_score,
                "distance_meters": best_distance
            })


    matches_df = pd.DataFrame(matches)
    return matches_df

LLM = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    request_timeout=45,
    max_retries=3,
)

load_dotenv()

class StoreInfo(BaseModel):
    company_turnover_uk: Optional[int] = Field(description="The annual turnover for the given company in the UK")
    number_locations_uk: Optional[int] = Field(description="The number of retail locations for the given company in the UK")
    number_locations_manchester: Optional[int] = Field(description="The number of retail locations for the given company in Greater Manchester")

class Company(BaseModel):
    __root__: Dict[str, StoreInfo]

def llm_for_statistical_figures(company_name):
    structured_llm = LLM.with_structured_output(Company, method="json_mode")

    system_role = """
    You are an excellent researcher. Your task is to provide data in selected retail companies.
    Ensure that your output is formatted as JSON to match the expected schema.
    """
    human_prompt = """
    Provide the following details for the company "{company_name}":
    - Company's annual turnover in the UK for retail stores ONLY
    - Number of retail stores in the UK
    - Number of retail stores in Greater Manchester, UK
    Do not include explanations or additional text.
    Ensure that you convert the output into an integer e.g. if 1 million then convert that to 1000000
    """
    prompt = ChatPromptTemplate(
        messages=[
            SystemMessage(content=system_role),
            HumanMessagePromptTemplate.from_template(human_prompt),
        ],
    )

    _input = prompt.format_prompt(
        company_name=company_name,
    )
    base_prompt = f"""
    Using the following commands, respond in JSON format with the following structure:
    - 'company_name' as key.
    - The values should be a JSON object containing the keys 'company_turnover_uk', 'number_locations_uk', and 'number_locations_manchester'.
    {_input.to_messages()}
    """
    output = structured_llm.invoke(base_prompt)
    return output