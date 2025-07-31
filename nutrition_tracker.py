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
import re
from fractions import Fraction
from typing import Dict, List, Optional, Any

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

# ------ Daily Nutritional Targets for Weight Gain ------

# Define daily targets for weight gain with minimum and maximum ranges
daily_targets = {
    'calories': {'min': 2800, 'max': 2900},
    'protein': {'min': 110, 'max': 120},
    'carbs': {'min': 410, 'max': 430},
    'fat': {'min': 75, 'max': 85}
}

# ------ USDA API Integration ------

# Category keywords to a prioritized list of preferred units.
# This is now matched against the official 'wweiaFoodCategoryDescription' from the API.
# --- MODIFIED: Prioritize individual units for most fruits/vegetables ---
CATEGORY_UNITS = {
    # Dairy & Dairy Products
    'yogurt': ['cup', 'container', 'oz'],
    'milk': ['cup', 'oz', 'container'],
    'cheese': ['slice', 'stick', 'cup', 'curd', 'inch'],
    'cream': ['tablespoon', 'cup', 'container'],
    
    # Protein Foods
    'beans': ['cup'],
    'lentils': ['cup'],
    'chickpeas': ['cup', 'pea'],
    'edamame': ['cup', 'pod'],
    'hummus': ['tablespoon', 'container'],
    'almonds': ['cup', 'oz', 'nut', 'package'],
    'mixed nuts': ['cup', 'package', 'oz'], # ADDED
    'seeds': ['cup', 'oz', 'package'],
    'peanut butter': ['tablespoon', 'serving'],
    'almond butter': ['tablespoon'],
    'tahini': ['tablespoon'],
    'egg': ['egg', 'cup'],
    
    # Grains
    'oats': ['cup'],
    'rice': ['cup'],
    'pasta': ['cup', 'oz'],
    'bread': ['slice', 'inch'],
    'bagel': ['bagel', 'large', 'regular', 'small', 'miniature'],
    'quinoa': ['cup'],
    'corn': ['cup', 'ear'],
    'couscous': ['cup', 'oz'],
    
    # Fruits
    'apple': ['medium', 'large', 'small', 'cup', 'slice', 'package'],
    'banana': ['banana', 'cup', 'slice', 'inch'],
    'orange': ['fruit', 'cup', 'section', 'slice'],
    'berries': ['cup', 'berry'],
    'raisins': ['cup', 'box', 'raisin', 'oz'],
    'date': ['date', 'cup'],
    
    # Vegetables
    'spinach': ['cup', 'leaf'],
    'broccoli': ['cup', 'floweret', 'piece'],
    'cauliflower': ['cup', 'floweret', 'piece'],
    'carrots': ['cup', 'carrot', 'slice', 'stick'],
    'mushrooms': ['cup', 'whole', 'slice'],
    'tomatoes': ['tomato', 'whole', 'cup', 'slice', 'cherry', 'grape', 'plum'],
    'potato': ['potato', 'cup'],
    'sweet potato': ['medium', 'large', 'small', 'cup', 'oz'],
    'avocado': ['fruit', 'cup', 'slice'],
    'brussels sprouts': ['sprout', 'cup'],
    'peas': ['cup'],
    'green beans': ['cup', 'bean', 'piece'], # ADDED
    
    # Snacks & Mixtures
    'trail mix': ['cup', 'package'], # ADDED
    
    # Fats, Oils & Sweets
    'oil': ['tablespoon', 'cup'],
    
    # Beverages
    'juice': ['oz', 'box', 'container', 'pouch'],
    'drink': ['cup', 'bottle', 'can', 'oz'], # ADDED
    'shake': ['cup', 'bottle', 'can', 'oz'], # ADDED
}

def clean_food_name(raw_name: str, is_branded: bool = False) -> str:
    """Cleans up food description strings for better readability."""
    if is_branded:
        # For branded items, take the primary name before a comma.
        # This turns "MUESLI, ORIGINAL" into "Muesli".
        cleaned_name = raw_name.split(',')[0].strip()
        return cleaned_name.title()
    else:
        # For generic foods, apply more complex cleaning logic.
        # This list removes common USDA descriptors.
        junk_patterns = [
            r',? ns as to.*',
            r',? nfs',
            r',? from canned',
            r',? cooked',
            r',? raw',
            r',? plain',
            r',? 100%',
            r',? regular',
            r',? unsweetened',
            r',? ready-to-drink',
        ]
        
        # Make a lowercase copy to work with
        cleaned_name = raw_name.lower()
        
        # Remove all the junk patterns using regex substitution
        for pattern in junk_patterns:
            cleaned_name = re.sub(pattern, '', cleaned_name)
        
        # Split by comma, strip whitespace from each part, and remove any empty results
        parts = [part.strip() for part in cleaned_name.split(',') if part.strip()]
        
        # If there are multiple parts (e.g., "bread, whole wheat"), reverse them
        if len(parts) > 1:
            parts.reverse()

        # Join the cleaned parts with a space and convert to title case
        return ' '.join(parts).title()


class USDANutritionAPI:
    """
    A class to get nutrition for single foods from the USDA Survey (FNDDS) database,
    using a smart default serving size for calculations based on the food's official category.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()

    def _find_exact_match_id(self, query: str) -> Optional[int]:
        """Searches only FNDDS for an exact match and returns its FDC ID."""
        url = f"{self.base_url}/foods/search"
        params = {
            'api_key': self.api_key,
            'query': query,
            'dataType': ["Survey (FNDDS)"],
            'pageSize': 5
        }
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            results = response.json()
            for food in results.get('foods', []):
                if food.get('description', '').lower() == query.lower():
                    return food.get('fdcId')
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching for '{query}': {e}")
        return None

    def _get_food_details(self, fdc_id: int) -> Dict:
        """Fetches the full details for a given FDC ID."""
        url = f"{self.base_url}/food/{fdc_id}"
        params = {'api_key': self.api_key, 'format': 'full'}
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Error getting details for FDC ID {fdc_id}: {e}")
        return {}

    def _get_available_measures(self, food_details: Dict) -> List[str]:
        """Extracts all available household measures from the food's portion data."""
        portions = food_details.get('foodPortions', [])
        if not portions:
            return [] # Return empty list if no portions
        
        measures = []
        for portion in portions:
            desc = portion.get('portionDescription')
            grams = portion.get('gramWeight')
            if desc and grams and desc != 'N/A' and "Quantity not specified" not in desc:
                # Special formatting for fluid ounces to show both volume and weight
                if 'fl oz' in desc:
                     measures.append(f"{desc} ({grams:.1f}g)")
                else:
                     measures.append(f"{desc} ({grams:.1f}g)")
        
        return sorted(list(set(measures))) # Use set to remove duplicates

    def _get_default_measure(self, food_category: Optional[str], available_measures: List[str]) -> str:
        """Determines the most logical default measure based on the food's WWEIA category."""
        if food_category:
            # --- Special handling for juice to prefer 'cup' conversion ---
            if 'juice' in food_category.lower() or 'drink' in food_category.lower() or 'shake' in food_category.lower():
                # If a 'cup' measure already exists, use it directly.
                for measure in available_measures:
                    if 'cup' in measure.lower():
                        return 'cup'
                # If not, but 'fl oz' exists, plan for conversion by returning 'cup'.
                for measure in available_measures:
                    if 'fl oz' in measure.lower():
                        return 'cup'
            # --- End Special Handling ---
            category_lower = food_category.lower()
            for category_keyword, units in CATEGORY_UNITS.items():
                if category_keyword in category_lower:
                    for unit in units:
                        for measure in available_measures:
                            if unit in measure.lower():
                                return unit
        
        # Fallback for items without a matching category
        preferred_units = [
            'medium', 'large', 'small', 'piece', 'slice', 'cup', 'container', 
            'tablespoon', 'tbsp', 'oz', 'ounce', 'tsp', 'teaspoon'
        ]
        
        for unit in preferred_units:
            for measure in available_measures:
                if unit in measure.lower():
                    return unit
        
        if available_measures:
            return available_measures[0].split('(')[0].strip()

        return "100g"

    def _find_portion_grams(self, food_details: Dict, unit_to_find: str) -> Optional[float]:
        """Finds the gram weight for a given unit from the food's portions."""
        if unit_to_find == "100g":
            return 100.0
            
        unit_to_find = unit_to_find.lower()
        
        # Create a priority list for matching to handle cases like "large" vs "extra large"
        # We want to match "1 large" before "1 extra large" if our unit is "large"
        exact_match_pattern = re.compile(r'\b' + re.escape(unit_to_find) + r'\b')
        
        best_match = None
        
        for portion in food_details.get('foodPortions', []):
            desc = portion.get('portionDescription', '').lower()
            # Prefer exact word matches first (e.g., 'large' doesn't match 'extra large')
            if exact_match_pattern.search(desc):
                return portion.get('gramWeight')
            # Fallback to substring match if no exact match is found yet
            if unit_to_find in desc and not best_match:
                best_match = portion.get('gramWeight')
        
        return best_match

    def get_food_nutrition(self, query: str) -> Optional[Dict]:
        """Gets nutrition data for a food item and returns it in the format needed by the app."""
        fdc_id = self._find_exact_match_id(query)
        if not fdc_id:
            return None

        food_details = self._get_food_details(fdc_id)
        if not food_details:
            return None
        
        wweia_category_obj = food_details.get('wweiaFoodCategory', {})
        food_category = wweia_category_obj.get('wweiaFoodCategoryDescription')
        
        available_measures = self._get_available_measures(food_details)
        default_unit = self._get_default_measure(food_category, available_measures)
        grams_per_portion = self._find_portion_grams(food_details, default_unit)

        # --- Conversion Logic for units like 'cup' that may not exist as a direct measure ---
        if default_unit == 'cup' and not grams_per_portion:
            # Attempt to convert from 'fl oz' to 'cup'
            base_grams_per_fl_oz = None
            # Prioritize the "1 fl oz" measure for the most accurate conversion
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                # Find a portion that represents a single fluid ounce
                if re.match(r'^1\s+(fl\s*)?oz', desc):
                    base_grams_per_fl_oz = portion.get('gramWeight')
                    break
            
            if base_grams_per_fl_oz:
                # 1 cup = 8 fl oz. Calculate grams for 1 cup.
                grams_per_portion = base_grams_per_fl_oz * 8.0
        # --- END NEW ---

        # --- Data Extraction ---
        nutrients_100g = {'Calories': 0.0, 'Protein': 0.0, 'Fat': 0.0, 'Carbohydrates': 0.0}
        nutrient_map = {1008: 'Calories', 1003: 'Protein', 1004: 'Fat', 1005: 'Carbohydrates'}
        
        for n in food_details.get('foodNutrients', []):
            nutrient_id = n.get('nutrient', {}).get('id')
            if nutrient_id in nutrient_map:
                key = nutrient_map[nutrient_id]
                nutrients_100g[key] = n.get('amount', 0.0)

        description = food_details.get('description', 'N/A')
        cleaned_description = clean_food_name(description, is_branded=False)
        
        # Scale nutrition to the default serving size
        if grams_per_portion:
            scale = grams_per_portion / 100.0
            serving_name = f"{cleaned_description} (1 {default_unit})"
            return {
                'name': serving_name,
                'calories': nutrients_100g['Calories'] * scale,
                'protein': nutrients_100g['Protein'] * scale,
                'carbs': nutrients_100g['Carbohydrates'] * scale,
                'fat': nutrients_100g['Fat'] * scale
            }
        else:
            # Fallback to 100g serving
            serving_name = f"{cleaned_description} (100g)"
            return {
                'name': serving_name,
                'calories': nutrients_100g['Calories'],
                'protein': nutrients_100g['Protein'],
                'carbs': nutrients_100g['Carbohydrates'],
                'fat': nutrients_100g['Fat']
            }

def get_dynamic_foods():
    """Generates the foods dictionary using USDA API data."""
    api_key = "PodqZM9xrI5ByN5sS8zlEMf2haudDydBMCzt3U4N" # Demo key
    api = USDANutritionAPI(api_key)

    food_queries = {
        'PRIMARY PROTEIN SOURCES': [
            "Yogurt, Greek, NS as to type of milk, plain",
            "Cheese, cottage, NFS",
            "Chickpeas, NFS",
            "Black beans, NFS",
            "Lentils, from canned",
            "Lentils, NFS",
            "Cheese, Mozzarella, NFS",
            "Hummus, plain",
            "Almonds, NFS",
            "Edamame, cooked",
            "Cheese, Cheddar",
            "Green peas, NS as to form, cooked",
            "Soy milk, unsweetened",
            "Mixed nuts, NFS",
            "Egg, whole, raw",
            "Peas, NFS",
        ],
        'PRIMARY CARBOHYDRATE SOURCES': [
            "Oats, raw",
            "Rice, white, cooked, NS as to fat",
            "Pasta, cooked",
            "Bread, whole wheat",
            "Potato, NFS",
            "Banana, raw",
            "Corn, raw",
            "Couscous, plain, cooked",
            "Date",
            "Raisins",
            "Trail mix, NFS",
        ],
        'PRIMARY FAT SOURCES': [
            "Peanut butter",
            "Olive oil",
            "Avocado, raw",
            "Coconut milk",
            "Milk, whole",
            "Chia seeds",
            "Cream cheese, regular, plain",
        ],
        'PRIMARY MICRONUTRIENT SOURCES': [
            "Spinach, raw",
            "Broccoli, raw",
            "Blueberries, raw",
            "Carrots, raw",
            "Mushrooms, raw",
            "Apple, raw",
            "Tomatoes, canned, cooked",
            "Orange, raw",
            "Cauliflower, NS as to form, cooked",
        ]
    }
    
    foods = {}
    
    for category, queries in food_queries.items():
        foods[category] = []
        for query in queries:
            nutrition_data = api.get_food_nutrition(query)
            if nutrition_data:
                foods[category].append(nutrition_data)
            time.sleep(0.5)  # Be polite to the API
    
    return foods

# ------ Comprehensive Vegetarian Food Database ------

# Initialize foods database with dynamic data
if 'foods_loaded' not in st.session_state:
    with st.spinner("Loading nutrition data from USDA database..."):
        foods = get_dynamic_foods()
        st.session_state.foods = foods
        st.session_state.foods_loaded = True
else:
    foods = st.session_state.foods

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
st.markdown("Choose foods using the buttons (0-5 servings) or enter custom "
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

    # Calculate totals
    for category, items in foods.items():
        for food in items:
            if food['name'] in st.session_state.food_selections:
                servings = st.session_state.food_selections[food['name']]
                if servings > 0:
                    total_calories += food['calories'] * servings
                    total_protein += food['protein'] * servings
                    total_carbs += food['carbs'] * servings
                    total_fat += food['fat'] * servings
                    selected_foods.append(f"{food['name']} x {servings:.1f}")

    # ------ Display Comprehensive Results ------

    # Display results
    st.header("📊 Daily Intake Summary")

    # Selected foods
    if selected_foods:
        st.subheader("🍽️ Foods Consumed Today:")
        # Create columns for better display
        cols = st.columns(3)
        for i, food in enumerate(selected_foods):
            with cols[i % 3]:
                st.write(f"• {food}")
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
        for category, items in foods.items():
            for food in items:
                if food['name'] in st.session_state.food_selections:
                    servings = st.session_state.food_selections[food['name']]
                    if servings > 0:
                        food_log.append({
                            'Food': food['name'],
                            'Servings': servings,
                            'Calories': food['calories'] * servings,
                            'Protein (g)': food['protein'] * servings,
                            'Carbs (g)': food['carbs'] * servings,
                            'Fat (g)': food['fat'] * servings
                        })

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
