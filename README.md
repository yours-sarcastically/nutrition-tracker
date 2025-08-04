# Personalized Evidence-Based Nutrition Tracker 🍽️

A comprehensive nutrition tracking application for healthy weight gain using vegetarian food sources with personalized daily targets.

## Features

- **Personalized Calculations**: Uses the Mifflin-St Jeor equation for BMR and activity-based TDEE
- **Interactive Food Logging**: Easy-to-use interface for tracking daily food intake
- **Smart Food Ranking**: Emoji-based system highlighting nutritionally dense foods
- **Progress Tracking**: Real-time progress bars and personalized recommendations
- **Comprehensive Database**: Extensive vegetarian food database with detailed nutritional information

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd nutrition_tracker
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure your data file is in place:
```bash
mkdir -p data
# Place your nutrition_database_final.csv in the data directory
```

## Usage

Run the application:
```bash
streamlit run main.py
```

Or use Python directly:
```bash
python main.py
```

## Project Structure

```
nutrition_tracker/
├── src/
│   ├── config.py          # Configuration settings
│   ├── models.py          # Data models and classes
│   ├── calculator.py      # Nutrition calculation engine
│   ├── database.py        # Food database management
│   ├── session_manager.py # Streamlit session management
│   └── ui/
│       ├── components.py  # Reusable UI components
│       └── main.py       # Main UI controller
├── tests/                 # Unit tests
├── data/                  # Data files
├── main.py               # Application entry point
└── requirements.txt      # Python dependencies
```

## Configuration

The application can be configured through environment variables:

- `DEBUG`: Enable debug mode (default: False)
- `LOG_LEVEL`: Set logging level (default: INFO)
- `NUTRITION_DATA_PATH`: Path to nutrition database CSV

## Testing

Run tests with pytest:
```bash
pytest tests/
```

## Data Format

The nutrition database should be a CSV file with the following columns:
- `name`: Food name
- `category`: Food category
- `calories`: Calories per serving
- `protein`: Protein grams per serving
- `carbs`: Carbohydrate grams per serving
- `fat`: Fat grams per serving
- `serving_unit`: Serving unit description

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Nutritional Disclaimer

This application is for educational and informational purposes only. Always consult with a healthcare professional or registered dietitian before making significant changes to your diet or nutrition plan.
