import os
from dotenv import load_dotenv
import pandas as pd
from mp_api.client import MPRester

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
MP_API_KEY = os.getenv("MP_API_KEY")

# Check if the API key was loaded successfully
if MP_API_KEY is None:
    raise ValueError("MP_API_KEY not found in environment variables. Please set it in your .env file.")

# Define the path to save the data
DATA_DIR = "../data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
FILE_PATH = os.path.join(DATA_DIR, "materials_composition_data.csv")

# Define the fields we want to retrieve
fields_to_retrieve = [
    "material_id",
    "formula_pretty",
    "elements",
    "nelements",
    "composition",
    "band_gap",
    "formation_energy_per_atom",
    "density",
]

def fetch_materials_data(total_materials_to_fetch=20000, chunk_size=1000):
    print("Connecting to Materials Project API...")
    with MPRester(MP_API_KEY) as mpr:
        all_materials_data = []
        
        num_chunks = (total_materials_to_fetch + chunk_size - 1) // chunk_size

        print(f"Attempting to fetch {total_materials_to_fetch} materials in {num_chunks} chunks of {chunk_size}...")

        try:
            docs_iterator = mpr.materials.summary.search(
                fields=fields_to_retrieve,
                num_chunks=num_chunks, # This will now be 20 for 20000/1000
                chunk_size=chunk_size
            )

            fetched_count = 0
            for doc in docs_iterator:
                material_info = {}
                for field in fields_to_retrieve:
                    material_info[field] = getattr(doc, field, None)
                
                all_materials_data.append(material_info)
                fetched_count += 1
                
                # Update progress more frequently for smaller chunks
                if fetched_count % chunk_size == 0 or fetched_count == total_materials_to_fetch:
                    print(f"Fetched {fetched_count} materials...")
                
                if fetched_count >= total_materials_to_fetch:
                    break # Stop once we've collected enough materials

        except Exception as e:
            print(f"An error occurred during data fetching: {e}")
            print("This could be due to API rate limits, network issues, or a very large request.")
            print("Consider reducing 'total_materials_to_fetch' or 'chunk_size' further if issues persist.")

    print(f"Finished fetching {len(all_materials_data)} materials.")
    return pd.DataFrame(all_materials_data)

if __name__ == "__main__":
    df = fetch_materials_data(total_materials_to_fetch=20000, chunk_size=1000)
    
    if not df.empty:
        df.to_csv(FILE_PATH, index=False)
        print(f"Data saved to {FILE_PATH}")

        print("\nFirst 5 rows of the DataFrame:")
        print(df.head())
        print("\nDataFrame Info:")
        df.info()
    else:
        print("No data fetched. DataFrame is empty.")