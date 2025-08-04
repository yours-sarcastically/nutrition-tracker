# Personalized Evidence-Based Nutrition Tracker

A comprehensive Streamlit application for calculating personalized daily nutritional targets and tracking food intake for healthy weight gain.

## Features

- **Personalized Calculations**: BMR and TDEE calculations using the Mifflin-St Jeor equation
- **Customizable Targets**: Adjustable caloric surplus, protein intake, and macronutrient ratios
- **Food Database**: Comprehensive database with nutritional rankings and emoji indicators
- **Progress Tracking**: Real-time progress bars and personalized recommendations
- **Clean Architecture**: Modular design with separated business logic and UI components

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nutrition-tracker
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure `nutrition_database_final.csv` is in the project root directory

4. Run the application:
   ```bash
   streamlit run main.py
   ```

## Project Structure

```
nutrition-tracker/
├── main.py                          # Main application entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── nutrition_database_final.csv     # Food database (not included)
└── src/                            # Source code package
    ├── __init__.py                 # Package initialization
    ├── config.py                   # Configuration management and constants
    ├── data_models.py              # Data classes for type safety and structure
    ├── nutrition_calculator.py     # Business logic for nutritional calculations
    ├── food_database.py            # Food data loading and processing
    └── ui_components.py            # Streamlit UI components
```

## Usage

1. **Enter Personal Information**: Use the sidebar to input your age, height, weight, sex, and activity level
2. **Adjust Advanced Settings**: Optionally modify caloric surplus, protein intake, and fat percentage
3. **View Nutritional Targets**: See your personalized daily targets for calories and macronutrients
4. **Select Foods**: Choose foods from categorized tabs and specify serving amounts
5. **Monitor Progress**: Track your daily intake against targets with progress bars
6. **Follow Recommendations**: Get personalized suggestions based on your current intake

## Architecture

The application follows clean architecture principles with clear separation of concerns:

### Core Components

- **`main.py`**: Application orchestration and entry point
- **`src/config.py`**: Centralized configuration management
- **`src/data_models.py`**: Type-safe data structures with validation
- **`src/nutrition_calculator.py`**: Mathematical formulas and business logic
- **`src/food_database.py`**: Data loading, processing, and ranking algorithms
- **`src/ui_components.py`**: Reusable Streamlit interface components

### Key Benefits

- **Modularity**: Each component has a single responsibility
- **Type Safety**: Data classes prevent errors and improve IDE support
- **Maintainability**: Business logic is separated from presentation
- **Testability**: Components can be tested independently
- **Configurability**: Settings are centralized and easily modifiable

## Food Database Format

The application expects a CSV file (`nutrition_database_final.csv`) with the following columns:

- `name`: Food item name
- `calories`: Calories per serving
- `protein`: Protein in grams per serving
- `carbs`: Carbohydrates in grams per serving
- `fat`: Fat in grams per serving
- `category`: Food category (e.g., "PRIMARY PROTEIN SOURCES")
- `serving_unit`: Optional serving unit description

## Customization

### Adding New Food Categories

1. Add foods to your CSV with new category names
2. Update `NUTRIENT_CATEGORY_MAP` in `src/config.py` if needed
3. The application will automatically create tabs for new categories

### Modifying Calculation Parameters

Edit values in `src/config.py`:

- `DEFAULT_CALORIC_SURPLUS`: Default calorie surplus for weight gain
- `DEFAULT_PROTEIN_PER_KG`: Default protein per kg of body weight
- `ACTIVITY_MULTIPLIERS`: TDEE calculation multipliers
- `TARGET_WEEKLY_GAIN_RATE`: Target weekly weight gain percentage

### Customizing UI Elements

- Modify `src/ui_components.py` to change interface elements
- Update emoji rankings in `config.py` to change food prioritization
- Adjust progress bar thresholds in recommendation logic

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes following the existing architecture
4. Test your changes thoroughly
5. Submit a pull request with a clear description

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool provides general nutritional guidance based on established formulas and should not replace professional medical advice. Consult with a healthcare provider or registered dietitian for personalized medical guidance, especially if you have specific health conditions or dietary requirements.
```

## Setup Instructions

To set up the project with this advanced directory structure:

1. **Create the directory structure**:
   ```bash
   mkdir nutrition-tracker
   cd nutrition-tracker
   mkdir src
   ```

2. **Create all the files** as shown above in their respective locations

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Add your food database**:
   - Place `nutrition_database_final.csv` in the root directory (same level as `main.py`)

5. **Run the application**:
   ```bash
   streamlit run main.py
   ```

This structure provides better organization while maintaining all the original functionality. The modular design makes it easier to maintain, test, and extend the application in the future! 🚀
