# USDA Food Data Search & Nutrition Tracker

Welcome! This repository contains a complete, two-part system for evidence-based nutrition planning. It pairs a powerful food database tool with a personalized meal tracker to give you everything you need for a successful and sustainable nutrition journey.

## System Overview

This project is built around two core components that work together seamlessly to provide a comprehensive nutrition planning solution:

1.  **USDA Data Client** (Jupyter Notebook): Your personal data chef for fetching and preparing food information.
2.  **Nutrition Tracker** (Streamlit App): An interactive dashboard for planning and tracking your daily meals.

## Project Structure

```
nutrition_tracker/
├── 📜 usda_food_data_client.ipynb    # USDA API client and data processor
├── 📜 nutrition_tracker.py           # Streamlit nutrition tracking app
├── 📂 data/
│   └── nutrition_results.csv         # Your processed, ready-to-use food database
└── README.md
```

## Component 1: USDA Food Data Central Client

Think of the `usda_food_data_client.ipynb` notebook as your powerhouse for data. It’s a sophisticated tool designed to connect to the official USDA FoodData Central API, where it intelligently queries, filters, and processes raw nutrition data into a clean, easy-to-use format.

### Key Features

**🔍 Advanced Food Search**
*   Search for both generic foods and your favorite branded products.
*   Zero in on the right items with smart keyword filtering and strict matching options.
*   Use the interactive interface to explore the database and refine your queries.

**📏 Smart Serving Size Analysis**
*   Automatically selects the most sensible household serving sizes instead of obscure gram amounts.
*   Uses category-based preferences to pick the most logical units for different foods.
*   Includes data validation warnings to help you spot unrealistic portion sizes.

**🧹 Intelligent Food Name Cleaning**
*   Strips away technical USDA descriptors and preparation methods for cleaner names.
*   Standardizes food names so they're consistent and easy to read.

**📊 Comprehensive Data Export**
*   Processes and exports your chosen foods into a tidy `nutrition_results.csv` file.
*   Calculates calories, protein, fat, and carbohydrates for each serving.
*   Categorizes foods by their primary nutrient source, making them easy to sort in the app.

## Component 2: Personalized Nutrition Tracker

The `nutrition_tracker.py` app is your interactive dashboard for personalized nutrition. It takes the prepared data from the notebook and turns it into a user-friendly platform for planning your meals and tracking your daily intake.

### Key Features

**🧮 Evidence-Based Calculations**
*   Calculates your Basal Metabolic Rate (BMR) using the trusted Mifflin-St Jeor equation.
*   Estimates your Total Daily Energy Expenditure (TDEE) with scientifically validated activity multipliers.
*   Sets goal-specific calorie targets for weight loss, maintenance, or muscle gain.
*   Employs a protein-first strategy for distributing your macronutrients.

**🎯 Personalized Goal Setting**
*   Set custom daily nutrition targets based on your unique body metrics and goals.
*   Full support for both metric and imperial units.
*   Fine-tune your protein and fat intake with advanced settings.
*   Calculates your ideal daily water intake based on weight and activity level.

**📱 Interactive Food Logging**
*   Browse your custom food database, complete with helpful categories and emoji indicators.
*   Quickly log foods with pre-selected servings or enter your own custom portions.
*   Watch your progress in real-time as you track against your daily targets.
*   Find foods in a snap with a smart and speedy search bar.

**📊 Comprehensive Progress Tracking**
*   Dive deeper with interactive charts that compare your intake to your targets.
*   Get personalized recommendations to help you meet your nutritional goals.
*   Export your data to PDF reports or CSV files for sharing or further analysis.

**💡 Educational Content & Guidance**
*   Learn as you go with evidence-based nutrition tips and strategies.
*   Get clear explanations on activity levels and your metabolism.
*   Receive helpful advice for breaking through plateaus and staying motivated.
*   Includes special micronutrient considerations for vegetarian diets.

## Getting Started

### Prerequisites

First, make sure you have the necessary libraries installed.

```bash
pip install streamlit pandas plotly reportlab requests ipywidgets
```

### Step 1: Create Your Custom Food Database

1.  **Get a USDA API Key**: Register for a free key at the [USDA FoodData Central website](https://fdc.nal.usda.gov/api-guide.html). It's quick and easy!

2.  **Configure the Notebook**:
    *   Open `usda_food_data_client.ipynb` in a Jupyter environment.
    *   In Cell 1, paste your API key: `api_key = "your_key_here"`.
    *   In Cell 10, update the `EXAMPLE_ANALYSIS_FOOD_LIST` with the foods you want in your database.

3.  **Process Your Foods**:
    *   Use the interactive search interface in the notebook to find the exact food descriptions you need.
    *   Run the `main_nutrition_analysis()` function to process your list. The notebook will work its magic and generate a `nutrition_results.csv` file for you.

### Step 2: Run the Nutrition Tracker

With your food database ready, just run the following command in your terminal:

```bash
streamlit run nutrition_tracker.py
```

The app will open right in your browser, ready to go with your personalized food database!

## Usage Examples

### Interactive Food Search

```python
# In Cell 10, configure your search terms
INTERACTIVE_FOOD_LIST_INPUT = ["banana", "lentils", "spinach"]

# Then, use the interactive widgets to:
# - Apply keyword and strict filtering
# - Export the results as a ready-to-use Python list
```

### Nutrition Analysis

```python
# In Cell 10, define the exact foods for your database
EXAMPLE_ANALYSIS_FOOD_LIST = [
    'Banana, raw',
    'Lentils, from canned',
    'Spinach, frozen, cooked, no added fat'
]

# Run the analysis to generate your CSV
main_nutrition_analysis()
```

## Scientific Foundation

This system isn't based on guesswork. It's built on a foundation of established nutritional science and peer-reviewed research to ensure you get reliable, evidence-based guidance:

*   **BMR Calculation**: Mifflin-St Jeor equation, endorsed by the Academy of Nutrition and Dietetics.
*   **Activity Multipliers**: Validated coefficients from exercise physiology research.
*   **Macronutrient Targets**: Based on guidelines from the International Society of Sports Nutrition.
*   **Caloric Adjustments**: Uses conservative, sustainable rates for healthy body composition changes.

## Key Improvements

*   **Modular Design**: A clear separation between the data-processing backend and the user-facing app.
*   **Smart Serving Sizes**: No more guessing—the system intelligently selects realistic, household portions.
*   **Data Validation**: Built-in warnings help you catch and review potentially unrealistic nutritional values.
*   **Flexible Exports**: Get your data in multiple formats for analysis, reporting, or sharing.
*   **Integrated Education**: Learn about the science behind your nutrition plan with helpful tips and guidance.

## Contributing

Contributions are always welcome! If you have ideas for improvements, feel free to fork the repo and submit a pull request. Areas for enhancement could include:
*   Adding more food database sources.
*   Creating new and insightful data visualizations.
*   Improving the mobile-responsive design.
*   Integrating with popular fitness tracking APIs.

---

Built with scientific rigor for sustainable success. Happy tracking! 🍎
