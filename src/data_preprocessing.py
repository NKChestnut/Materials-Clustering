import pandas as pd
import numpy as np
import os
import ast
import re

# Define paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_DATA_FILE = os.path.join(DATA_DIR, "materials_composition_data.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_materials_data.csv")

print(f"Loading data from {RAW_DATA_FILE}...")
try:
    df = pd.read_csv(RAW_DATA_FILE)
    print("Data loaded successfully.")
    print(f"Number of materials loaded: {len(df)}")
    # Ensure 'composition' column is treated as strings for parsing
    df['composition'] = df['composition'].astype(str)
except FileNotFoundError:
    print(f"Error: {RAW_DATA_FILE} not found. Please ensure data fetching was successful in Step 1.")
    print("Expected path:", os.path.abspath(RAW_DATA_FILE))
    exit()

print("\n Step 2: Data Preprocessing and Feature Engineering")

# Parse 'composition' column into actual dictionaries
def parse_composition_string(comp_str):
    """
    Parses a composition string from Materials Project into a dictionary.
    Handles both simple 'ElementCount' (e.g., 'O10') and dictionary strings.
    """
    if comp_str.startswith("{") and comp_str.endswith("}"):
        try:
            # Safely evaluate dictionary string
            return ast.literal_eval(comp_str)
        except (ValueError, SyntaxError):
            return None # Handle malformed dictionary strings
    else:
        # Handle simple ElementCount format like 'O10' or 'C100'
        match = re.match(r'([A-Za-z]+)([0-9.]+)', comp_str)
        if match:
            element = match.group(1)
            count = float(match.group(2))
            return {element: count}
        return None # Could not parse

print("Parsing 'composition' column...")
df['parsed_composition'] = df['composition'].apply(parse_composition_string)

# Drop rows where parsing failed
df.dropna(subset=['parsed_composition'], inplace=True)
print(f"Number of materials after parsing composition and dropping NaNs: {len(df)}")

# Identify all unique elements and create a master list
# Extract all elements from all parsed compositions
all_elements = set()
for comp_dict in df['parsed_composition']:
    if comp_dict: # Ensure it's not None
        all_elements.update(comp_dict.keys())

all_elements_sorted = sorted(list(all_elements)) # Sort for consistent column order
print(f"Found {len(all_elements_sorted)} unique elements.")


# Create compositional features
print("Creating elemental atomic percentage features...")
composition_features = []
for index, row in df.iterrows():
    comp_dict = row['parsed_composition']
    if comp_dict:
        total_atoms = sum(comp_dict.values())
        if total_atoms == 0: # Avoid division by zero
            element_percentages = {elem: 0.0 for elem in all_elements_sorted}
        else:
            element_percentages = {elem: comp_dict.get(elem, 0.0) / total_atoms for elem in all_elements_sorted}
        composition_features.append(element_percentages)
    else:
        # If parsing failed or comp_dict is None, append a row of zeros
        composition_features.append({elem: 0.0 for elem in all_elements_sorted})

# Convert list of dictionaries to a DataFrame
composition_df = pd.DataFrame(composition_features, index=df.index)

# Join the new compositional features with the original DataFrame
df = pd.concat([df, composition_df], axis=1)

print("\nDataFrame Head with new compositional features:")
print(df[all_elements_sorted].head())

print("\nDataFrame Info after feature engineering:")
df.info()

# Handle other numerical columns and potential NaNs
# Check for NaNs in numerical columns that will be used for clustering
numerical_cols = ['nelements', 'band_gap', 'formation_energy_per_atom', 'density']
print("\nChecking for NaNs in numerical columns before final processing:")
print(df[numerical_cols].isnull().sum())

for col in numerical_cols:
    if df[col].isnull().any():
        # Only fill if there are NaNs to avoid unnecessary calculations
        df[col].fillna(df[col].mean(), inplace=True)
        print(f"Filled NaNs in '{col}' with its mean.")

print("\nNaNs after filling:")
print(df[numerical_cols].isnull().sum())

# Select final features for clustering
features_for_clustering = all_elements_sorted + numerical_cols

# Create the final DataFrame for clustering
df_processed = df[features_for_clustering].copy()

print(f"\nProcessed DataFrame for Clustering (first 5 rows and info):")
print(df_processed.head())
df_processed.info()

# Save the processed data
print(f"\nSaving processed data to {PROCESSED_DATA_FILE}...")
df_processed.to_csv(PROCESSED_DATA_FILE, index=False)
print("Processed data saved successfully.")