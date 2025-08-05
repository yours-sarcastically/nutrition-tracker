# Personalized Evidence-Based Nutrition Tracker

This project is a complete, two-part system for generating a personalized, evidence-based nutrition plan. It consists of a data processing pipeline and an interactive user application.

1.  USDA Data Client (Jupyter Notebook): A powerful tool for querying the USDA FoodData Central API. It searches for foods, applies advanced filters, intelligently selects serving sizes, and exports clean, standardized nutritional data to a CSV file.
2.  Nutrition Tracker (Streamlit App): An interactive web application that uses the data from the notebook. It allows users to input their personal metrics, calculates their unique nutritional targets, and lets them log food intake to track their progress.

## Project Structure

The project is organized into a modular structure that separates the data pipeline from the user-facing application.

```
nutrition_tracker/
├── 📜 usda_food_data_client.ipynb
├── 📂 core/
│   ├── __init__.py
│   ├── calculations.py
│   ├── data.py
│   └── models.py
├── 📂 data/
│   └── nutrition_results.csv
├── 📂 ui/
│   ├── __init__.py
│   ├── components.py
│   └── sidebar.py
├── app.py
├── config.py
└── README.md
```

## Component 1: USDA Data Client (Jupyter Notebook)

The `usda_food_data_client.ipynb` notebook is the starting point of the project. Its primary purpose is to search, filter, and process food nutrition data from the official USDA database, ultimately generating the `nutrition_results.csv` file that the Streamlit application depends on.

### Key Features

*   Interactive Food Search: Query multiple food items against the USDA database with support for both generic (Survey FNDDS) and branded food products.
*   Advanced Filtering: Apply a chain of filters to refine results, including keyword matching, requiring search terms to be at the start of the description, and a strict filter for exact or comma-separated phrases.
*   Serving Size Analysis: Automatically finds and selects the most logical household serving size for each food item (e.g., "1 cup" for milk, "1 medium" for an apple) and includes data validation to warn against unrealistic values.
*   Detailed Nutrition Analysis & Export: Fetches key nutritional data (calories, protein, fat, carbs) for the selected serving size, cleans the food names for readability, and exports the final, processed data to `nutrition_results.csv`.

## Component 2: Personalized Nutrition Tracker (Streamlit App)

The `app.py` is the interactive web application that users will run. It provides a user-friendly interface for personalized nutrition planning and tracking.

### Key Improvements (Architectural)

1.  Multi-File Structure: The application's logic is separated into distinct modules for better organization and maintainability.
    *   `app.py`: The main application entry point.
    *   `config.py`: Contains all static configuration and constants.
    *   `core/`: Contains the core "business logic" and data models.
    *   `ui/`: Contains modules for building the user interface.
    *   `data/`: Directory for data files.

2.  Formalized Data Structures: Dictionaries have been replaced with Python `dataclasses` (`core/models.py`) for type safety, better IDE support, and more self-documenting code.

## How to Run the Full System

Follow these two steps to get the entire system running.

### Step 1: Generate the Food Database (Jupyter Notebook)

First, you must run the data client to create the food database.

1.  Get a free API Key from the [USDA FoodData Central](https://fdc.nal.usda.gov/api-guide.html).
2.  Open the Notebook: Launch `usda_food_data_client.ipynb` in a Jupyter Notebook environment.
3.  Add Your API Key: In Cell 1, replace the placeholder string in the `api_key` variable with your actual key.
4.  Configure Food List: In Cell 10, update the `EXAMPLE_ANALYSIS_FOOD_LIST` with the exact food descriptions you want in your database. You can use the interactive search tool in the notebook to find these descriptions.
5.  Run the Analysis: Execute the `main_nutrition_analysis()` function (located in Cell 11). This will process your food list and save the results.
6.  Move the File: A file named `nutrition_results.csv` will be created in the root `nutrition_tracker/` directory. You must move this file into the `data/` sub-directory.

### Step 2: Run the Nutrition Tracker App (Streamlit)

Once the `nutrition_results.csv` is in the `data/` folder, you can run the main application.

1.  Navigate to the root `nutrition_tracker/` directory in your terminal.
2.  Run the Streamlit command:

    ```sh
    streamlit run app.py
    ```

The application will open in your web browser, pre-loaded with the food data you generated.
