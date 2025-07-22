# Materials Composition Clustering Project

## Overview

This project demonstrates the application of unsupervised machine learning techniques, specifically K-Means clustering, to identify natural groupings within a large dataset of materials based on their elemental compositions and other fundamental properties. The goal is to uncover hidden patterns and classify materials into distinct families without prior knowledge of their categories, providing valuable insights for materials discovery and design.

## Features

* **Data Acquisition:** Programmatic fetching of materials data (composition, band gap, formation energy, density) from the Materials Project API.
* **Data Preprocessing:** Cleaning raw data, handling missing values, and transforming string-based composition data into structured numerical formats.
* **Feature Engineering:** Creation of elemental atomic percentage features for each material, representing its unique chemical makeup.
* **Dimensionality Reduction (PCA):** Application of Principal Component Analysis to reduce high-dimensional compositional data for visualization and to understand variance distribution.
* **Optimal Cluster Determination:** Utilization of the Elbow Method and Silhouette Score to identify an appropriate number of clusters (`k`) for K-Means.
* **K-Means Clustering:** Implementation of the K-Means algorithm to group similar materials.
* **Cluster Interpretation:** Analysis of the average properties and elemental compositions within each identified cluster to characterize their unique traits.
* **Visualization:** Generation of plots to visualize PCA explained variance, Elbow Method, Silhouette Scores, and the final clusters in 2D PCA space.

## Results & Insights

The project successfully identified 10 distinct clusters of materials. A significant majority of materials fell into a large "general" cluster, while several smaller, highly specialized clusters emerged. These specialized clusters were often dominated by a single element (e.g., Silicon, Iron, Titanium, Plutonium) and exhibited characteristic average properties (e.g., high band gap for silicon-rich materials, high density for plutonium-rich materials). This demonstrates the ability to segment a complex materials space into chemically meaningful groups.

## Technologies Used

* **Python 3.x**
* **Libraries:**
    * `pandas` (for data manipulation and analysis)
    * `numpy` (for numerical operations)
    * `scikit-learn` (for PCA, StandardScaler, KMeans)
    * `matplotlib` (for plotting)
    * `seaborn` (for enhanced visualizations)
    * `mp-api` (for Materials Project API interaction)
    * `python-dotenv` (for environment variable management)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/materials-clustering-project.git](https://github.com/your-username/materials-clustering-project.git)
    cd materials-clustering-project
    ```
    (Replace `your-username` and `materials-clustering-project` with your actual GitHub details)

2.  **Create and activate a virtual environment:**
    * **Using `conda` (recommended):**
        ```bash
        conda create -n materials_env python=3.9
        conda activate materials_env
        ```
    * **Using `venv`:**
        ```bash
        python -m venv venv
        # On Windows:
        .\venv\Scripts\activate
        # On macOS/Linux:
        source venv/bin/activate
        ```

3.  **Install required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```
    **(Note:** You'll need to create `requirements.txt` first. See "Creating `requirements.txt`" below.)

4.  **Obtain Materials Project API Key:**
    * Go to <https://materialsproject.org/> and sign up for a free account.
    * Find your API key on your dashboard/profile page.

5.  **Set up `.env` file:**
    * In the root directory of your project (`materials-clustering-project/`), create a file named `.env`.
    * Add your API key to this file in the following format:
        ```
        MP_API_KEY="YOUR_ACTUAL_API_KEY_HERE"
        ```
        (Replace `"YOUR_ACTUAL_API_KEY_HERE"` with the key you obtained).

### Creating `requirements.txt`

Before installing packages, create a `requirements.txt` file in your project root with the following content:
* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* mp-api
* python-dotenv

## Usage

Navigate to the root of your project directory in your activated virtual environment.

1.  **Fetch Raw Data:**
    ```bash
    python src/get_mp_data.py
    ```
    This will download `materials_composition_data.csv` into the `data/` folder.

2.  **Preprocess Data & Feature Engineering:**
    ```bash
    python src/data_preprocessing.py
    ```
    This will generate `processed_materials_data.csv` in the `data/` folder.

3.  **Perform Clustering Analysis & Visualization:**
    ```bash
    python src/clustering_analysis.py
    ```
    This script will:
    * Load the processed data.
    * Perform data scaling and PCA.
    * Display plots for PCA explained variance, Elbow Method, and Silhouette Score (close each plot to proceed).
    * Apply K-Means clustering.
    * Display a 2D PCA scatter plot of the clusters.
    * Print a detailed analysis of the characteristics of each cluster.
    * Save `clustered_materials_data.csv` to the `data/` folder.

## Project Structure
materials-clustering-project/
├── data/
│   ├── materials_composition_data.csv  # Raw data fetched from MP
│   ├── processed_materials_data.csv    # Cleaned and feature-engineered data
│   └── clustered_materials_data.csv    # Processed data with cluster assignments
├── src/
│   ├── get_mp_data.py                  # Script for data acquisition
│   ├── data_preprocessing.py           # Script for data cleaning and feature engineering
│   └── clustering_analysis.py          # Script for PCA, K-Means, and interpretation
├── .env                                # Stores API key (not committed to Git)
├── .gitignore                          # Specifies files/folders to ignore in Git
└── README.md                           # This file

## Possible Future Work

* **Advanced Feature Engineering:** Incorporate more sophisticated materials descriptors (e.g., Magpie features, SOAP descriptors) using libraries like `matminer` or `pymatgen` to potentially improve clustering quality.
* **Alternative Clustering Algorithms:** Explore other unsupervised learning methods such as DBSCAN, Hierarchical Clustering, or Gaussian Mixture Models to compare results.
* **Interactive Visualization:** Create an interactive visualization of the clusters (e.g., using Plotly or Dash) to allow for dynamic exploration of material properties within clusters.
* **Domain-Specific Validation:** Collaborate with a materials scientist to validate the chemical meaningfulness of the identified clusters.
* **Predictive Modeling:** Use the generated clusters as labels to train a supervised classification model, predicting the cluster membership of new, unseen materials.
