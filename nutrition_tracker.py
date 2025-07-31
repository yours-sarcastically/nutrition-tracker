# -----------------------------------------------------------------------------
# Streamlit Vegetarian Nutrition Tracker for Healthy Weight Gain
# -----------------------------------------------------------------------------

"""
Interactive nutrition tracking application for vegetarian weight gain planning.

This Streamlit application provides a comprehensive tool for tracking daily
nutritional intake using vegetarian food sources. The application calculates
total caloric and macronutrient consumption based on user-selected foods and
serving sizes, then compares results against established daily targets for
healthy weight gain.

The application features a categorized food database organized by nutritional
focus including primary protein sources, carbohydrate sources, fat sources,
and micronutrient sources. Users can select foods through quick-select buttons
or custom serving inputs, with real-time calculation of nutritional totals.

Key Features:
- Daily nutritional targets with minimum and maximum ranges for calories,
  protein, carbohydrates, and fat
- Categorized vegetarian food database with detailed nutritional information
- Interactive food selection interface with quick-select buttons and custom
  serving inputs
- Real-time calculation and display of total nutritional intake
- Progress tracking against daily targets with visual progress bars
- Personalized recommendations for meeting nutritional goals
- Detailed food log with tabular display of selected items

Usage:
1. Run the Streamlit application using 'streamlit run nutrition_tracker.py'
2. View daily nutritional targets displayed at the top of the interface
3. Navigate through food category tabs to select desired foods
4. Use quick-select buttons (1x-5x) or custom serving inputs for portions
5. Click 'Calculate Daily Intake' to view comprehensive nutritional summary
6. Review progress bars and personalized recommendations
7. Use 'Clear All Selections' to reset all food choices

Daily Targets:
- Calories: 2800-2900 kcal for healthy weight gain
- Protein: 110-120g for muscle building and recovery
- Carbohydrates: 410-430g for energy and performance
- Fat: 75-85g for hormone production and absorption

The application maintains session state to preserve food selections across
user interactions and provides comprehensive feedback on nutritional adequacy
through progress tracking and targeted recommendations.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import requests
import time
from typing import Dict, List, Optional
from fractions import Fraction

# -----------------------------------------------------------------------------
# Cell 2: USDA API Logic Integration (from Program 2)
# -----------------------------------------------------------------------------

class USDANutritionAPI:
    """
    A class to interact with the USDA FoodData Central API
    and retrieve nutritional information for foods.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()

    def _parse_measure(self, measure: str) -> Optional[tuple]:
        if not measure:
            return None
        parts = measure.strip().lower().split()
        if len(parts) < 2:
            # Handle cases like "100g" where there is no space
            if len(parts) == 1:
                qty_str = ''.join(filter(str.isdigit, parts[0]))
                unit = ''.join(filter(str.isalpha, parts[0]))
                if qty_str and unit:
                    parts = [qty_str, unit]
                else:
                    return None
            else:
                return None
        
        qty_str = parts[0]
        unit = " ".join(parts[1:]).strip()
        try:
            quantity = float(Fraction(qty_str))
        except (ValueError, ZeroDivisionError):
            try:
                quantity = float(qty_str)
            except ValueError:
                return None
        return quantity, unit

    def _find_portion_grams(self, food_details: Dict, unit: str) -> Optional[float]:
        """
        Look through foodPortions for a matching household measure and return
        the gram weight for that portion. Recognizes common synonyms.
        """
        unit = unit.lower()
        alias_map = {
            'tbsp': ['tbsp', 'tablespoon', 'tablespoons', 'tbl', 'tbs'],
            'tablespoon': ['tablespoon', 'tablespoons', 'tbsp', 'tbl', 'tbs'],
            'tbsps': ['tablespoon', 'tablespoons', 'tbsp', 'tbl', 'tbs'],
            'tsp': ['tsp', 'teaspoon', 'teaspoons'],
            'teaspoon': ['teaspoon', 'teaspoons', 'tsp'],
            'cup': ['cup', 'cups'], 'cups': ['cup', 'cups'],
            'oz': ['oz', 'ounce', 'ounces'],
            'fl oz': ['fl oz', 'fluid ounce', 'fluid ounces'],
            'g': ['g', 'gram', 'grams'], 'grams': ['g', 'gram', 'grams'],
            'kg': ['kg', 'kilogram', 'kilograms'],
            'slice': ['slice', 'slices'], 'slices': ['slice', 'slices'],
            'piece': ['piece', 'pieces'], 'pieces': ['piece', 'pieces'],
            'medium': ['medium'], 'large': ['large'], 'small': ['small']
        }
        search_terms = alias_map.get(unit, [unit])
        
        # Prioritize exact matches in portionDescription
        for portion in food_details.get("foodPortions", []):
            desc = str(portion.get("portionDescription", "")).lower()
            if any(f" {term} " in f" {desc} " or desc == term for term in search_terms):
                if portion.get("gramWeight"):
                    return portion.get("gramWeight")

        # Fallback to broader search and measureUnit abbreviation
        for portion in food_details.get("foodPortions", []):
            desc = str(portion.get("portionDescription", "")).lower()
            abbr = str(portion.get("measureUnit", {}).get("abbreviation", "")).lower().replace('.', '')
            if any(term in desc or term == abbr for term in search_terms):
                if portion.get("gramWeight"):
                    return portion.get("gramWeight")
        return None

    def search_food(self, query: str, data_type: List[str] = None, page_size: int = 5) -> Dict:
        url = f"{self.base_url}/foods/search"
        params = {'api_key': self.api_key, 'query': query, 'pageSize': page_size}
        if data_type:
            params['dataType'] = data_type
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error searching for {query}: {e}")
            return {}

    def get_food_details(self, fdc_id: int) -> Dict:
        url = f"{self.base_url}/food/{fdc_id}"
        params = {'api_key': self.api_key}
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting details for FDC ID {fdc_id}: {e}")
            return {}

    def extract_nutrition_data(self, food_data: Dict) -> Dict:
        nutrition = {
            'name': food_data.get('description', 'Unknown'),
            'fdc_id': food_data.get('fdcId'), 'calories': 0, 'protein': 0,
            'carbs': 0, 'fat': 0
        }
        nutrient_map = {1008: 'calories', 1003: 'protein', 1005: 'carbs', 1004: 'fat'}
        for nutrient in food_data.get('foodNutrients', []):
            nid = nutrient.get('nutrient', {}).get('id')
            if nid in nutrient_map:
                nutrition[nutrient_map[nid]] = round(nutrient.get('amount', 0), 2)
        return nutrition

    def get_nutrition_for_food_by_measure(self, food_name: str, measure: str) -> Optional[Dict]:
        parsed = self._parse_measure(measure)
        if not parsed:
            print(f"Could not parse measure '{measure}'.")
            return None
            
        quantity, unit = parsed
        search_results = self.search_food(food_name, data_type=["SR Legacy", "Foundation", "Branded"])
        if not search_results or 'foods' not in search_results or not search_results['foods']:
            print(f"No results found for {food_name}")
            return None
            
        fdc_id = search_results['foods'][0].get('fdcId')
        if not fdc_id: return None
        
        details = self.get_food_details(fdc_id)
        if not details: return None
        
        base_nutrition_per_100g = self.extract_nutrition_data(details)
        total_grams = None
        
        if unit in ['g', 'grams']:
            total_grams = quantity
        else:
            grams_per_portion = self._find_portion_grams(details, unit)
            if grams_per_portion:
                total_grams = grams_per_portion * quantity
            else:
                print(f"No household measure '{unit}' found for {food_name}. Cannot calculate.")
                return None
        
        if total_grams is None: return None

        scale = total_grams / 100.0
        for k in ['calories', 'protein', 'carbs', 'fat']:
            base_nutrition_per_100g[k] = round(base_nutrition_per_100g[k] * scale, 2)
        
        return base_nutrition_per_100g


# -----------------------------------------------------------------------------
# Cell 2: Page Configuration and Initial Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Nutrition Tracker",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# Cell 3: Daily Nutritional Targets and Food Database Configuration
# -----------------------------------------------------------------------------

# ------ Calculate Daily Nutritional Targets Based on Article Logic ------

# Default user profile parameters from the article for dynamic calculation.
# These values are not displayed in the UI.
_WEIGHT_KG = 57.5
_HEIGHT_CM = 180
_AGE_YEARS = 26
_ACTIVITY_MULTIPLIER = 1.55  # Moderately Active

# Calculate BMR using the Mifflin-St Jeor equation for men.
_bmr = (10 * _WEIGHT_KG) + (6.25 * _HEIGHT_CM) - (5 * _AGE_YEARS) + 5

# Calculate TDEE by applying the activity multiplier.
_tdee = _bmr * _ACTIVITY_MULTIPLIER

# Set caloric target based on TDEE, a surplus, and rounding per the article.
# Article: (TDEE ~2441 + Surplus 400) = ~2841, rounded to 2850.
_calorie_target = 2850

# Calculate macronutrient targets based on the final calorie goal.
_protein_target_g = 2.0 * _WEIGHT_KG  # Protein at 2.0 g/kg
_calories_from_fat = _calorie_target * 0.25  # Fat at 25% of calories
_fat_target_g = _calories_from_fat / 9
_calories_from_protein = _protein_target_g * 4
_calories_from_carbs = _calorie_target - _calories_from_protein - _calories_from_fat
_carbs_target_g = _calories_from_carbs / 4

# ------ Daily Nutritional Targets for Weight Gain ------

# Define daily targets with min/max ranges around the calculated values.
# This dictionary replaces the original hardcoded values.
daily_targets = {
    'calories': {'min': _calorie_target - 50, 'max': _calorie_target + 50},
    'protein': {'min': int(_protein_target_g) - 5, 'max': int(_protein_target_g) + 5},
    'carbs': {'min': int(round(_carbs_target_g)) - 10, 'max': int(round(_carbs_target_g)) + 10},
    'fat': {'min': int(round(_fat_target_g)) - 5, 'max': int(round(_fat_target_g)) + 5}
}

# ------ Dynamic Food Database Generation ------

@st.cache_data
def get_food_data(_api_client):
    """
    Dynamically fetches nutritional data for a predefined list of foods
    using the USDA FoodData Central API and caches the result.
    """
    print("Fetching nutritional data from USDA API... (This runs only once)")
    
    food_items_to_fetch = {
        'PRIMARY PROTEIN SOURCES': [
            {'food': 'Greek Yogurt, plain, non-fat', 'measure': '1 Cup'},
            {'food': 'Cottage Cheese, full fat', 'measure': '1 Cup'},
            {'food': 'Whey Protein powder', 'measure': '1 Tablespoon'},
            {'food': 'Chickpeas, cooked', 'measure': '1 Cup'},
            {'food': 'Black Beans, cooked', 'measure': '1 Cup'},
            {'food': 'Tofu, firm', 'measure': '200 Grams'},
            {'food': 'Lentils, cooked', 'measure': '1 Cup'},
            {'food': 'Pinto Beans, cooked', 'measure': '1 Cup'},
            {'food': 'Mozzarella Cheese, whole milk', 'measure': '100g'},
            {'food': 'Hummus', 'measure': '1 Cup'},
            {'food': 'Nuts, mixed', 'measure': '1 Cup'},
            {'food': 'Egg, whole, cooked', 'measure': '1 Medium'},
            {'food': 'Cheese, cheddar, slice', 'measure': '1 Slice'},
            {'food': 'Peas, green, cooked', 'measure': '1 Cup'},
        ],
        'PRIMARY CARBOHYDRATE SOURCES': [
            {'food': 'Oats, cooked', 'measure': '1 Cup'},
            {'food': 'Rice, white, cooked', 'measure': '1 Cup'},
            {'food': 'Pasta, cooked', 'measure': '1 Cup'},
            {'food': 'Whole Wheat Bread', 'measure': '2 Slices'},
            {'food': 'Potatoes, boiled', 'measure': '1 Medium'},
            {'food': 'Bananas', 'measure': '1 Medium'},
            {'food': 'Corn, sweet, cooked', 'measure': '1 Cup'},
            {'food': 'Muesli', 'measure': '1 Cup'},
            {'food': 'Couscous, cooked', 'measure': '1 Cup'},
            {'food': 'Dates, medjool', 'measure': '5 Pieces'},
            {'food': 'Dried fruits, mixed', 'measure': '1 Cup'},
            {'food': 'Trail Mix', 'measure': '1 Cup'},
        ],
        'PRIMARY FAT SOURCES': [
            {'food': 'Peanut Butter, smooth', 'measure': '1 Tablespoon'},
            {'food': 'Olive Oil', 'measure': '1 Tablespoon'},
            {'food': 'Avocados', 'measure': '1 Medium'},
            {'food': 'Coconut Milk, canned, full fat', 'measure': '1 Cup'},
            {'food': 'Whole Milk, 3.25%', 'measure': '1 Cup'},
            {'food': 'Chia Seeds, dried', 'measure': '1 Tablespoon'},
            {'food': 'Cream Cheese', 'measure': '1 Tablespoon'},
        ],
        'PRIMARY MICRONUTRIENT SOURCES': [
            {'food': 'Spinach, raw', 'measure': '1 Cup'},
            {'food': 'Broccoli, cooked', 'measure': '1 Cup'},
            {'food': 'Berries, mixed', 'measure': '1 Cup'},
            {'food': 'Carrots, cooked', 'measure': '1 Cup'},
            {'food': 'Mushrooms, white, cooked', 'measure': '1 Cup'},
            {'food': 'Apples, with skin', 'measure': '1 Medium'},
            {'food': 'Tomato Puree', 'measure': '1 Cup'},
            {'food': 'Oranges', 'measure': '1 Medium'},
            {'food': 'Cauliflower, cooked', 'measure': '1 Cup'},
        ]
    }
    
    dynamic_foods_db = {category: [] for category in food_items_to_fetch}
    
    for category, items in food_items_to_fetch.items():
        for item in items:
            food_name = item['food']
            measure_str = item['measure']
            
            nutrition_data = _api_client.get_nutrition_for_food_by_measure(food_name, measure_str)
            time.sleep(0.2)  # Short delay to be respectful to the API
            
            display_name = f"{food_name.split(',')[0]} ({measure_str})"
            if nutrition_data:
                food_entry = {
                    'name': display_name,
                    'calories': nutrition_data.get('calories', 0),
                    'protein': nutrition_data.get('protein', 0),
                    'carbs': nutrition_data.get('carbs', 0),
                    'fat': nutrition_data.get('fat', 0)
                }
                dynamic_foods_db[category].append(food_entry)
            else:
                # Add a placeholder if API call fails
                dynamic_foods_db[category].append({
                    'name': f"{display_name} - DATA NOT FOUND",
                    'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0
                })

    return dynamic_foods_db

# Initialize API and fetch data
API_KEY = "PodqZM9xrI5ByN5sS8zlEMf2haudDydBMCzt3U4N"  # Public DEMO_KEY
api = USDANutritionAPI(API_KEY)
foods = get_food_data(api)


# -----------------------------------------------------------------------------
# Cell 4: Session State Initialization and Custom Styling
# -----------------------------------------------------------------------------

# ------ Initialize Session State for Food Selections ------

if 'food_selections' not in st.session_state:
    st.session_state.food_selections = {}

# ------ Custom CSS for Enhanced Button Styling ------

st.markdown("""
<style>
/* Active button styling */
.active-button {
    background-color: #ff6b6b !important;
    color: white !important;
    border: 2px solid #ff6b6b !important;
}

/* Custom button container */
.button-container {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 5: Application Title and Introduction Display
# -----------------------------------------------------------------------------

st.title("🥗 Nutrition Tracker")
st.markdown("""
Welcome to your interactive nutrition tracking application! This tool helps
you plan and monitor your daily food intake for healthy weight gain using
vegetarian food sources.
""")

# -----------------------------------------------------------------------------
# Cell 6: Daily Nutritional Targets Display Section
# -----------------------------------------------------------------------------

st.header("🎯 Daily Nutritional Targets")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Calories", f"{daily_targets['calories']['min']}-"
                        f"{daily_targets['calories']['max']} kcal")
with col2:
    st.metric("Protein", f"{daily_targets['protein']['min']}-"
                       f"{daily_targets['protein']['max']}g")
with col3:
    st.metric("Carbohydrates", f"{daily_targets['carbs']['min']}-"
                             f"{daily_targets['carbs']['max']}g")
with col4:
    st.metric("Fat", f"{daily_targets['fat']['min']}-"
                   f"{daily_targets['fat']['max']}g")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 7: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("📝 Select Your Foods")
st.markdown("Choose foods using the buttons (1-5 servings) or enter custom "
            "serving amounts.")

# ------ Create Category Tabs for Food Organization ------

# Create tabs for different food categories
tabs = st.tabs(list(foods.keys()))

for i, (category, items) in enumerate(foods.items()):
    with tabs[i]:
        # Create columns for better layout
        for j in range(0, len(items), 2):
            col1, col2 = st.columns(2)

            # ------ First Item in Row ------
            if j < len(items):
                with col1:
                    food = items[j]
                    st.subheader(food['name'])

                    # Create unique key for this food item
                    key = f"{category}_{food['name']}"

                    # Get current serving value
                    current_serving = st.session_state.food_selections.get(
                        food['name'], 0.0)

                    # Quick select buttons with active state styling
                    button_col1, button_col2, button_col3, button_col4, \
                        button_col5 = st.columns(5)

                    with button_col1:
                        button_type = ("primary" if current_serving == 1.0
                                       else "secondary")
                        if st.button("1×", key=f"{key}_1", type=button_type):
                            st.session_state.food_selections[food['name']] = 1.0
                            st.rerun()

                    with button_col2:
                        button_type = ("primary" if current_serving == 2.0
                                       else "secondary")
                        if st.button("2×", key=f"{key}_2", type=button_type):
                            st.session_state.food_selections[food['name']] = 2.0
                            st.rerun()

                    with button_col3:
                        button_type = ("primary" if current_serving == 3.0
                                       else "secondary")
                        if st.button("3×", key=f"{key}_3", type=button_type):
                            st.session_state.food_selections[food['name']] = 3.0
                            st.rerun()

                    with button_col4:
                        button_type = ("primary" if current_serving == 4.0
                                       else "secondary")
                        if st.button("4×", key=f"{key}_4", type=button_type):
                            st.session_state.food_selections[food['name']] = 4.0
                            st.rerun()

                    with button_col5:
                        button_type = ("primary" if current_serving == 5.0
                                       else "secondary")
                        if st.button("5×", key=f"{key}_5", type=button_type):
                            st.session_state.food_selections[food['name']] = 5.0
                            st.rerun()

                    # Custom serving input - ensure value is always float
                    custom_serving = st.number_input(
                        "Custom servings:",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(current_serving),
                        step=0.1,
                        key=f"{key}_custom"
                    )

                    if custom_serving > 0:
                        st.session_state.food_selections[
                            food['name']] = custom_serving
                    elif (food['name'] in st.session_state.food_selections
                          and custom_serving == 0):
                        del st.session_state.food_selections[food['name']]

                    # Display nutritional info
                    st.caption(f"Per Serving: {food['calories']:.0f} kcal | "
                               f"{food['protein']:.1f} g Protein | "
                               f"{food['carbs']:.1f} g Carbohydrates | "
                               f"{food['fat']:.1f} g Fat")

            # ------ Second Item in Row ------
            if j + 1 < len(items):
                with col2:
                    food = items[j + 1]
                    st.subheader(food['name'])

                    # Create unique key for this food item
                    key = f"{category}_{food['name']}"

                    # Get current serving value
                    current_serving = st.session_state.food_selections.get(
                        food['name'], 0.0)

                    # Quick select buttons with active state styling
                    button_col1, button_col2, button_col3, button_col4, \
                        button_col5 = st.columns(5)

                    with button_col1:
                        button_type = ("primary" if current_serving == 1.0
                                       else "secondary")
                        if st.button("1×", key=f"{key}_1", type=button_type):
                            st.session_state.food_selections[food['name']] = 1.0
                            st.rerun()

                    with button_col2:
                        button_type = ("primary" if current_serving == 2.0
                                       else "secondary")
                        if st.button("2×", key=f"{key}_2", type=button_type):
                            st.session_state.food_selections[food['name']] = 2.0
                            st.rerun()

                    with button_col3:
                        button_type = ("primary" if current_serving == 3.0
                                       else "secondary")
                        if st.button("3×", key=f"{key}_3", type=button_type):
                            st.session_state.food_selections[food['name']] = 3.0
                            st.rerun()

                    with button_col4:
                        button_type = ("primary" if current_serving == 4.0
                                       else "secondary")
                        if st.button("4×", key=f"{key}_4", type=button_type):
                            st.session_state.food_selections[food['name']] = 4.0
                            st.rerun()

                    with button_col5:
                        button_type = ("primary" if current_serving == 5.0
                                       else "secondary")
                        if st.button("5×", key=f"{key}_5", type=button_type):
                            st.session_state.food_selections[food['name']] = 5.0
                            st.rerun()

                    # Custom serving input - ensure value is always float
                    custom_serving = st.number_input(
                        "Custom servings:",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(current_serving),
                        step=0.1,
                        key=f"{key}_custom"
                    )

                    if custom_serving > 0:
                        st.session_state.food_selections[
                            food['name']] = custom_serving
                    elif (food['name'] in st.session_state.food_selections
                          and custom_serving == 0):
                        del st.session_state.food_selections[food['name']]

                    # Display nutritional info
                    st.caption(f"Per Serving: {food['calories']:.0f} kcal | "
                               f"{food['protein']:.1f} g Protein | "
                               f"{food['carbs']:.1f} g Carbohydrates | "
                               f"{food['fat']:.1f} g Fat")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 8: Calculation Button and Nutritional Results Display
# -----------------------------------------------------------------------------

if st.button("📊 Calculate Daily Intake", type="primary",
             use_container_width=True):

    # ------ Initialize Nutritional Totals ------

    # Initialize totals
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    selected_foods = []

    # ------ Calculate Total Nutritional Values ------

    # Create a flattened map for easy lookup
    all_foods_map = {food['name']: food for category_items in foods.values() for food in category_items}

    # Calculate totals
    for food_name, servings in st.session_state.food_selections.items():
        if servings > 0 and food_name in all_foods_map:
            food = all_foods_map[food_name]
            total_calories += food['calories'] * servings
            total_protein += food['protein'] * servings
            total_carbs += food['carbs'] * servings
            total_fat += food['fat'] * servings
            selected_foods.append(f"{food['name']} × {servings:.1f}")

    # ------ Display Comprehensive Results ------

    # Display results
    st.header("📊 Daily Intake Summary")

    # Selected foods
    if selected_foods:
        st.subheader("🍽️ Foods Consumed Today:")
        # Create columns for better display
        num_columns = min(3, len(selected_foods))
        if num_columns > 0:
            cols = st.columns(num_columns)
            for i, food_text in enumerate(selected_foods):
                with cols[i % num_columns]:
                    st.write(f"• {food_text}")
    else:
        st.info("No foods selected")

    # ------ Nutritional Totals Display ------

    # Nutritional totals
    st.subheader("📈 Total Nutritional Intake:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calories", f"{total_calories:.0f} kcal")
    with col2:
        st.metric("Protein", f"{total_protein:.1f}g")
    with col3:
        st.metric("Carbohydrates", f"{total_carbs:.1f}g")
    with col4:
        st.metric("Fat", f"{total_fat:.1f}g")

    # ------ Progress Bars for Target Achievement ------

    # Progress bars
    st.subheader("🎯 Daily Target Achievement:")

    # Calories progress
    cal_percent = (min((total_calories / daily_targets['calories']['min'])
                       * 100, 100) if daily_targets['calories']['min'] > 0
                   else 0)
    st.progress(cal_percent / 100,
                text=f"Calories: {cal_percent:.0f}% of minimum target")

    # Protein progress
    prot_percent = (min((total_protein / daily_targets['protein']['min'])
                        * 100, 100) if daily_targets['protein']['min'] > 0
                    else 0)
    st.progress(prot_percent / 100,
                text=f"Protein: {prot_percent:.0f}% of minimum target")

    # Carbs progress
    carb_percent = (min((total_carbs / daily_targets['carbs']['min'])
                        * 100, 100) if daily_targets['carbs']['min'] > 0
                    else 0)
    st.progress(carb_percent / 100,
                text=f"Carbohydrates: {carb_percent:.0f}% of minimum target")

    # Fat progress
    fat_percent = (min((total_fat / daily_targets['fat']['min'])
                       * 100, 100) if daily_targets['fat']['min'] > 0
                   else 0)
    st.progress(fat_percent / 100,
                text=f"Fat: {fat_percent:.0f}% of minimum target")

    # ------ Personalized Recommendations ------

    # Recommendations
    st.subheader("💡 Personalized Recommendations:")
    recommendations = []

    if total_calories < daily_targets['calories']['min']:
        deficit = daily_targets['calories']['min'] - total_calories
        recommendations.append(
            f"• You need {deficit:.0f} more calories to reach your minimum "
            f"daily target")

    if total_protein < daily_targets['protein']['min']:
        deficit = daily_targets['protein']['min'] - total_protein
        recommendations.append(
            f"• You need {deficit:.0f}g more protein for muscle building "
            f"and recovery")

    if total_carbs < daily_targets['carbs']['min']:
        deficit = daily_targets['carbs']['min'] - total_carbs
        recommendations.append(
            f"• You need {deficit:.0f}g more carbohydrates for energy "
            f"and performance")

    if total_fat < daily_targets['fat']['min']:
        deficit = daily_targets['fat']['min'] - total_fat
        recommendations.append(
            f"• You need {deficit:.0f}g more healthy fats for hormone "
            f"production")

    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("🎉 Outstanding work! You have successfully met all your "
                   "minimum daily nutritional targets!")

    # ------ Detailed Food Log Table ------

    # Display selected foods as a table
    if selected_foods:
        st.subheader("📋 Detailed Food Log")
        food_log = []
        for food_name, servings in st.session_state.food_selections.items():
            if servings > 0 and food_name in all_foods_map:
                food = all_foods_map[food_name]
                food_log.append({
                    'Food': food['name'],
                    'Servings': f"{servings:.1f}",
                    'Calories': f"{food['calories'] * servings:.0f}",
                    'Protein (g)': f"{food['protein'] * servings:.1f}",
                    'Carbs (g)': f"{food['carbs'] * servings:.1f}",
                    'Fat (g)': f"{food['fat'] * servings:.1f}"
                })
        
        if food_log:
            df = pd.DataFrame(food_log)
            st.dataframe(df, use_container_width=True)

    # ------ Thank You Message ------

    # Footer message - only appears after calculation
    st.markdown("---")
    st.markdown("""
    🌟 Thanks for using the Nutrition Tracker!
    Keep fueling your body with wholesome plant-based goodness, and remember -
    every healthy choice you make is a step toward your goals. You've got this! 💪
    """)
    print("📊 Daily nutritional intake calculated successfully! 🎯")

# -----------------------------------------------------------------------------
# Cell 9: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("🔄 Clear All Selections", use_container_width=True):
    st.session_state.food_selections = {}
    st.rerun()
    print("🔄 All food selections have been cleared successfully! ✨")
