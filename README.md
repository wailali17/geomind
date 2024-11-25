# Turnover Estimation for Businesses in Manchester, UK

## Overview

This repository contains code and methodology to estimate the turnover of businesses in Manchester, UK. The primary aim is to provide a baseline for estimating potential revenue loss for businesses in case of disruptions like flooding. The project utilises a combination of open-source datasets and machine learning models to estimate turnover and evaluate performance.

### Key Features
- **Data Integration**: OpenLocal, GeoLytix, ONS, and OpenStreetMap data were combined to enrich the dataset with rateable values, geospatial features, and firmographics.
- **Modelling Approaches**:
    - **Model 1**: Baseline turnover estimation using sector-specific multipliers.
    - **Model 2**: Incorporates geospatial features like distance to transport and water.
    - **Model 3**: Estimates turnover using company-level data such as total turnover and floor area.

- **Machine Learning**: Uses XGBoost for model training and evaluation.

## Prerequisites

### Environment Setup

1. Install Dependencies:
    - Ensure you have Python 3.10+ installed.
    - Install required packages using:
        ```pip install -r requirements.txt```
2. API Key for OpenAI:
    - Obtain an OpenAI API Key.
    - Create a .env file in the project root directory and add:
        ```OPENAI_API_KEY=<your_openai_api_key>```

### Datasets
Ensure the following datasets are available in the data/raw directory:
- OpenLocal Dataset: data/raw/openlocal/GeoTAM_Hackathon_OpenLocal.gpkg
- GeoLytix Retail Points: data/raw/geolytix/geolytix_retailpoints_v33_202408.csv
- ONS Datasets:
    - data/raw/ons/correcteddataforpublication.xlsx
    - data/raw/ons/PCD_OA_LSOA_MSOA_LAD_MAY22_UK_LU.csv
    - data/raw/ons/ukgranulargdhiothergeographies2002to2021.xlsx
    - data/raw/ons/uksmallareagvaestimates1998to2022.xlsx
- OpenStreetMaps: Extract all files in the downloaded zip in the following directory data/raw/openstreemap/

See below for all the sources for each dataset.

## How to Run the Code
### Step-by-Step Instructions
1. Load the Jupyter Notebook:
    - Open estimating_turnover_geoatm.ipynb.
2. Import Helper Scripts:
    - The notebook begins by importing:
        - utils: Contains utility functions like sector_multipliers, XGBoost training, and LLM integrations.
        - utils_data_prep: Functions for dataset preparation.
3. Read and Clean OpenLocal Data:
    - The openlocal_dataset function from utils_data_prep:
        - Loads the OpenLocal dataset.
        - Filters out records with missing Rateable Value.
4. Merge ONS Data:
    - Use using_ons_datasets from utils_data_prep to enrich the dataset with population, GVA, and income data.
5. Model 1: Baseline Turnover Estimation:
    - Calculate turnover:
        - Multiply Rateable Value by sector-specific multipliers.
        - Multipliers are stored in utils.py.
    - Visualize correlations:
        - Plot a heatmap for feature correlations.
    - Run XGBoost:
        - Use xgb_model from utils.py to train and evaluate the baseline model.
6. Model 2: Adding Geospatial Features:
    - Calculate distances to transport links and waterways using OpenStreetMap data.
    - Train and evaluate the model:
        - Incorporate geospatial features alongside rateable values.
7. Model 3: Turnover Based on Company Data:
    - Data Matching:
        - Merge GeoLytix data with OpenLocal using fuzzy matching.
    - LLM Integration:
        - Use OpenAI API to extract company-level statistics (e.g., total turnover).
    - Turnover Distribution:
        - Distribute Manchester turnover across stores based on floor area.
    - Train and evaluate:
        - Run XGBoost to predict turnover using enriched features.
8. Save/Visualise Results:
    - Evaluation results are stored in a DataFrame for comparison.

## Code Structure
```
📂 GEOMIND
├── 📂 data
│   ├── 📂 proc
│   ├── 📂 raw
│   │   ├── 📂 alltheplaces
│   │   ├── 📂 geolytix
│   │   ├── 📂 ons
│   │   ├── 📂 openlocal
│   │   └── 📂 openstreetmap
│   │       ├── BasicCompanyDataAsOneFile-2024-11-01.csv
│   │       └── overture_maps.dbb
├── .env
├── .gitignore
├── estimating_turnover_geoatm.ipynb
├── README.md
├── requirements.txt
├── utils_data_prep.py
└── utils.py
```

## Data Sources
- ONS Postcode LSOA look up - https://geoportal.statistics.gov.uk/datasets/e7824b1475604212a2325cd373946235/about
- ONS small area GVA estimates - https://www.ons.gov.uk/economy/grossvalueaddedgva/datasets/uksmallareagvaestimates
- ONS Population - https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationandmigrationstatisticstransformationlowerlayersuperoutputareapopulationdifferenceenglandandwales
- ONS Household income - https://www.ons.gov.uk/economy/regionalaccounts/grossdisposablehouseholdincome/datasets/ukgrossdisposablehouseholdincomegdhiforothergeographicareas
- Geolytix - https://drive.google.com/file/d/1B8M7m86rQg2sx2TsHhFa2d-x-dZ1DbSy/view
- OpenLocal - Provided by GeoTam Hackathon team
- OpenStreetMap - https://download.geofabrik.de/europe/united-kingdom/england/greater-manchester.html

