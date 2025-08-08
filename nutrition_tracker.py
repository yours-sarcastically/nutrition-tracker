#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# A Personalized Evidence-Based Nutrition Tracker for Goal-Specific Meal Planning
# ---------------------------------------------------------------------------

"""
This script implements an interactive, evidence-based nutrition tracker using
Streamlit. It is designed to help users achieve personalized nutrition goals,
such as weight loss, maintenance, or gain, with a focus on vegetarian food
sources.

Core Functionality and Scientific Basis:
- Basal Metabolic Rate (BMR) Calculation: The application uses the Mifflin-St
  Jeor equation, which is widely recognized by organizations like the Academy
  of Nutrition and Dietetics for its accuracy.
  - For Males: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
  - For Females: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

- Total Daily Energy Expenditure (TDEE): The BMR is multiplied by a
  scientifically validated activity factor to estimate the total number of
  calories burned in a day, including physical activity.

- Goal-Specific Caloric Adjustments:
  - Weight Loss: A conservative 20 percent caloric deficit from TDEE.
  - Weight Maintenance: Caloric intake is set equal to TDEE.
  - Weight Gain: A controlled 10 percent caloric surplus over TDEE.

- Macronutrient Strategy: The script follows a protein-first approach,
  consistent with modern nutrition science.
  1. Protein intake is determined based on grams per kilogram of body weight.
  2. Fat intake is set as a percentage of total daily calories.
  3. Carbohydrate intake is calculated from the remaining caloric budget.

Implementation Details:
- The user interface is built with Streamlit, providing interactive widgets
  for user input and data visualization.
- The food database is managed using the Pandas library.
- Progress visualizations are created with Streamlit's native components and
  Plotly for generating detailed charts.

Usage Documentation:
1. Prerequisites: Ensure you have the required Python libraries installed.
   You can install them using pip:
   pip install streamlit pandas plotly

2. Running the Application: Save this script as a Python file (for example,
   `nutrition_app.py`) and run it from your terminal using the following
   command:
   streamlit run nutrition_app.py

3. Interacting with the Application:
   - Use the sidebar to enter your personal details, such as age, height,
     weight, sex, activity level, and primary nutrition goal.
   - Click "Calculate & Update Targets" to validate inputs and see your plan.
   - Your personalized daily targets for calories and macronutrients will be
     calculated and displayed.
   - Navigate through the food tabs to select the number of servings for
     each food item you consume.
   - The daily summary section will update in real time to show your
     progress toward your targets.
"""

# ---------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# ---------------------------------------------------------------------------

import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import json
from io import StringIO

# ---------------------------------------------------------------------------
# Cell 2: Page Configuration and Initial Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Your Personal Nutrition Coach 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Cell 3: Unified Configuration Constants
# ---------------------------------------------------------------------------

# ------ Default Parameter Values Based on Published Research ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 57.5,
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': "weight_gain",
    'protein_per_kg': 2.0,
    'fat_percentage': 0.25
}

# ------ Activity Level Multipliers for TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Activity Level Descriptions ------
ACTIVITY_DESCRIPTIONS = {
    'sedentary': "Little to no exercise, desk job",
    'lightly_active': "Light exercise one to three days per week",
    'moderately_active': "Moderate exercise three to five days per week",
    'very_active': "Heavy exercise six to seven days per week",
    'extremely_active': "Very heavy exercise, a physical job, or "
                        "two times per day training"
}

# ------ Goal-Specific Targets Based on an Evidence-Based Guide ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # -20% from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,   # 0% from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # +10% over TDEE
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
}

# ------ Tooltips for Metrics and Emojis ------
TOOLTIPS = {
    'bmr': "Basal Metabolic Rate: Calories your body burns at rest just to stay alive.",
    'tdee': "Total Daily Energy Expenditure: Your total daily calorie needs, including all activities.",
    'caloric_adjustment': "The daily calorie surplus or deficit applied to your TDEE to achieve your goal.",
    'weekly_change': "An estimate of your weekly weight change based on your caloric adjustment.",
    'water': "A general daily hydration recommendation. Needs may vary.",
    '🥇': "Gold Medal: An all-star food! High in its primary nutrient and calorie-dense.",
    '🔥': "High Calorie: One of the most calorie-dense options in its category.",
    '💪': "High Protein: An excellent source of protein.",
    '🍚': "High Carb: A great source of carbohydrates for energy.",
    '🥑': "High Fat: A fantastic source of healthy fats."
}

# ------ Unified Configuration for All App Components ------
CONFIG = {
    'emoji_order': {'🥇': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
    },
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Calories',
                     'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein',
                    'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates',
                  'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    'form_fields': {
        'age': {'type': 'number', 'label': 'Age (years)',
                'min': 16, 'max': 80, 'step': 1,
                'placeholder': 'Enter your age', 'required': True},
        'height': {'type': 'number', 'label': 'Height',
                   'min': 50.0, 'max': 90.0, 'step': 0.5,
                   'placeholder': 'Enter your height', 'required': True},
        'weight': {'type': 'number', 'label': 'Weight',
                   'min': 80.0, 'max': 350.0, 'step': 1.0,
                   'placeholder': 'Enter your weight', 'required': True},
        'sex': {'type': 'selectbox', 'label': 'Biological Sex',
                'options': ["Male", "Female"], 'required': True},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level',
                           'options': [
                               ("Sedentary", "sedentary"),
                               ("Lightly Active", "lightly_active"),
                               ("Moderately Active", "moderately_active"),
                               ("Very Active", "very_active"),
                               ("Extremely Active", "extremely_active")
                           ], 'required': True},
        'goal': {'type': 'selectbox', 'label': 'Your Goal',
                 'options': [
                     ("Weight Loss", "weight_loss"),
                     ("Weight Maintenance", "weight_maintenance"),
                     ("Weight Gain", "weight_gain")
                 ], 'required': True},
        'protein_per_kg': {'type': 'number',
                           'label': 'Protein Goal (g/kg)',
                           'min': 1.2, 'max': 3.0, 'step': 0.1,
                           'help': 'Define your daily protein target in grams per kilogram of body weight.',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number',
                           'label': 'Fat Intake (% of calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Set the share of your daily calories that should come from healthy fats.',
                           'convert': lambda x: x / 100 if x is not None else None,
                           'advanced': True, 'required': False}
    }
}


# ---------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables."""
    # This function now runs only once at the start of a session.
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.food_selections = {}
        st.session_state.submitted = False
        st.session_state.units = "Metric (kg/cm)"
        for field in CONFIG['form_fields'].keys():
            st.session_state[f'user_{field}'] = None


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration."""
    session_key = f'user_{field_name}'

    if field_config['type'] == 'number':
        # Adjust label and value ranges for units
        label = field_config['label']
        min_val, max_val, step = field_config['min'], field_config['max'], field_config['step']
        
        if st.session_state.units == 'Imperial (lbs/in)':
            if field_name == 'height':
                label += ' (in)'
            elif field_name == 'weight':
                label += ' (lbs)'
        else: # Metric
            if field_name == 'height':
                label += ' (cm)'
                min_val, max_val, step = 140, 220, 1
            elif field_name == 'weight':
                label += ' (kg)'
                min_val, max_val, step = 40.0, 150.0, 0.5
        
        value = container.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=st.session_state.get(session_key), # Use .get for safety
            step=step,
            placeholder=field_config.get('placeholder'),
            help=field_config.get('help'),
            key=session_key # Add unique key
        )
    
    elif field_config['type'] == 'selectbox':
        options = field_config['options']
        # Set default index carefully
        current_value = st.session_state.get(session_key)
        if field_name in ['activity_level', 'goal']:
            try:
                index = [opt[1] for opt in options].index(current_value)
            except (ValueError, TypeError):
                index = 0
            
            selection = container.selectbox(
                field_config['label'], options, index=index,
                format_func=lambda x: x[0], key=session_key
            )
            value = selection[1]
        else:
            try:
                index = options.index(current_value)
            except (ValueError, TypeError):
                index = 0
            value = container.selectbox(
                field_config['label'], options, index=index, key=session_key
            )
            
    st.session_state[session_key] = value
    return value


def get_final_values(user_inputs):
    """Processes all user inputs, applies defaults, and handles unit conversions."""
    final_values = {}
    
    # Convert imperial units to metric for calculations
    height = user_inputs.get('height')
    weight = user_inputs.get('weight')
    
    if st.session_state.units == 'Imperial (lbs/in)':
        if height: final_values['height_cm'] = height * 2.54
        if weight: final_values['weight_kg'] = weight * 0.453592
    else:
        if height: final_values['height_cm'] = height
        if weight: final_values['weight_kg'] = weight
        
    # Process other fields, falling back to defaults if necessary
    for field, value in user_inputs.items():
        if field not in ['height', 'weight']:
            final_values[field] = value if value is not None else DEFAULTS.get(field)

    # Apply goal-specific defaults for advanced settings if they are not user-set
    goal = final_values.get('goal')
    if goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[goal]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            # Note: The lambda for fat_percentage conversion runs in create_unified_input
            final_values['fat_percentage'] = goal_config['fat_percentage']

    return final_values


def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculates daily fluid needs based on body weight and activity."""
    if not weight_kg: return 0
    base_needs = weight_kg * 35  # Baseline is 35 milliliters per kilogram

    activity_bonus = {'sedentary': 0, 'lightly_active': 300, 'moderately_active': 500, 'very_active': 700, 'extremely_active': 1000}
    climate_multiplier = {'cold': 0.9, 'temperate': 1.0, 'hot': 1.2, 'very_hot': 1.4}
    total_ml = (base_needs + activity_bonus.get(activity_level, 500)) * climate_multiplier.get(climate, 1.0)
    return round(total_ml)


def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics with tooltips in a configurable column layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            label, value, *rest = metric_info
            delta = rest[0] if rest else None
            help_text = rest[1] if len(rest) > 1 else None
            st.metric(label, value, delta, help=help_text)


def get_progress_bar_color(percent):
    """Returns a color based on the percentage value."""
    if percent < 50: return "#ff4d4d"  # Red
    if percent < 80: return "#ffa500"  # Yellow/Orange
    return "#2ecc71"  # Green

def create_progress_tracking(totals, targets):
    """Creates color-coded progress bars and recommendations for nutritional targets."""
    st.subheader("Your Daily Dashboard 🎯")

    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = min(actual / target * 100, 100) if target > 0 else 0
        color = get_progress_bar_color(percent)

        progress_text = f"{config['label']}: {actual:.0f} / {target:.0f} {config['unit']} ({percent:.0f}%)"
        st.markdown(f"""
        {progress_text}
        <div style="background-color: #e0e0e0; border-radius: 5px; height: 8px; width: 100%;">
            <div style="background-color: {color}; width: {percent}%; border-radius: 5px; height: 100%;"></div>
        </div>
        """, unsafe_allow_html=True)
        st.write("") # Spacer


def calculate_daily_totals(food_selections, foods):
    """Calculates the total daily nutrition from all selected foods."""
    totals = {nutrient: 0 for nutrient in CONFIG['nutrient_configs'].keys()}
    selected_foods = []
    all_foods = [item for sublist in foods.values() for item in sublist]
    
    for food in all_foods:
        servings = food_selections.get(food['name'], 0)
        if servings > 0:
            for nutrient in totals:
                totals[nutrient] += food[nutrient] * servings
            selected_foods.append({'food': food, 'servings': servings})
    return totals, selected_foods


def generate_summary_csv(totals, targets, selected_foods):
    """Generates a CSV string of the daily nutrition summary."""
    summary_data = {
        'Metric': ['Calories (kcal)', 'Protein (g)', 'Carbohydrates (g)', 'Fat (g)'],
        'Your Intake': [totals['calories'], totals['protein'], totals['carbs'], totals['fat']],
        'Your Target': [targets['total_calories'], targets['protein_g'], targets['carb_g'], targets['fat_g']]
    }
    summary_df = pd.DataFrame(summary_data)
    
    food_data = {
        'Food Item': [item['food']['name'] for item in selected_foods],
        'Servings': [item['servings'] for item in selected_foods]
    }
    food_df = pd.DataFrame(food_data)
    
    output = StringIO()
    summary_df.to_csv(output, index=False)
    output.write("\nLogged Foods:\n")
    food_df.to_csv(output, index=False)
    return output.getvalue()


# ---------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# ---------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculates the Basal Metabolic Rate using the Mifflin-St Jeor equation."""
    if not all([age, height_cm, weight_kg, sex]): return 0
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculates Total Daily Energy Expenditure based on activity level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Calculates the estimated weekly weight change from a caloric adjustment."""
    # Based on the approximation that one kilogram of body fat is ~7,700 kcal.
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(age, height_cm, weight_kg, sex='male',
                                   activity_level='moderately_active',
                                   goal='weight_gain', protein_per_kg=None,
                                   fat_percentage=None):
    """Calculates personalized daily nutritional targets."""
    if not all([age, height_cm, weight_kg, sex, activity_level, goal]):
        # Return default/zeroed dictionary if essential inputs are missing
        return {key: 0 for key in ['bmr', 'tdee', 'total_calories', 'caloric_adjustment',
                                   'protein_g', 'protein_calories', 'fat_g', 'fat_calories',
                                   'carb_g', 'carb_calories', 'estimated_weekly_change',
                                   'protein_percent', 'carb_percent', 'fat_percent']} | {'goal': goal}

    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment

    protein_per_kg_final = protein_per_kg if protein_per_kg is not None else goal_config['protein_per_kg']
    fat_percentage_final = fat_percentage if fat_percentage is not None else goal_config['fat_percentage']

    protein_g = protein_per_kg_final * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage_final
    fat_g = fat_calories / 9 if fat_calories > 0 else 0
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4 if carb_calories > 0 else 0
    
    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': calculate_estimated_weekly_change(caloric_adjustment),
        'goal': goal
    }

    # Safely calculate percentages to avoid division by zero
    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
    else:
        targets['protein_percent'] = targets['carb_percent'] = targets['fat_percent'] = 0

    return targets


# ---------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# ---------------------------------------------------------------------------

@st.cache_data
def get_food_data(file_path):
    """Loads food database, assigns emojis, and caches the result."""
    # 1. Load the database
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in df['category'].unique()}
    for _, row in df.iterrows():
        category = row['category']
        if category in foods:
            foods[category].append({
                'name': f"{row['name']} ({row['serving_unit']})",
                'calories': row['calories'], 'protein': row['protein'],
                'carbs': row['carbs'], 'fat': row['fat']
            })

    # 2. Assign emojis (This logic now runs only once per cache hit)
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'calories': {}}
    for category, items in foods.items():
        if not items: continue
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]

        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(items, key=lambda x: x[map_info['sort_by']], reverse=True)
            top_foods[map_info['key']] = [food['name'] for food in sorted_by_nutrient[:3]]

    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]}
    emoji_mapping = {'high_cal_nutrient': '🥇', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑'}

    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])
            emoji = ''
            if is_high_calorie and is_top_nutrient: emoji = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: emoji = emoji_mapping['high_calorie']
            elif food_name in top_foods['protein']: emoji = emoji_mapping['protein']
            elif food_name in top_foods['carbs']: emoji = emoji_mapping['carbs']
            elif food_name in top_foods['fat']: emoji = emoji_mapping['fat']
            food['emoji'] = emoji
    return foods


def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        emoji = food.get('emoji', '')
        tooltip = TOOLTIPS.get(emoji, f"A serving of {food['name']}")
        st.subheader(f"{emoji} {food['name']}", help=tooltip)
        
        food_key_name = food['name'].replace(" ", "_").lower()
        key_prefix = f"{category}_{food_key_name}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)

        col1, col2 = st.columns([2, 1.2])
        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key_prefix}_{k}", type=button_type, help=f"Set to {k} servings", use_container_width=True):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()
        with col2:
            # This number_input allows for custom fractional servings
            custom_serving = st.number_input(
                "Custom Servings",
                min_value=0.0, max_value=20.0, # Increased max servings cap
                value=float(current_serving), step=0.1,
                key=f"{key_prefix}_custom",
                label_visibility="collapsed"
            )

        if custom_serving != current_serving:
            st.session_state.food_selections[food['name']] = custom_serving
            st.rerun()

        caption_text = f"Per Serving: {food['calories']} kcal | {food['protein']}g P | {food['carbs']}g C | {food['fat']}g F"
        st.caption(caption_text)


def render_food_grid(items, category, columns=2):
    """Renders a grid of food items, with an added search filter."""
    search_query = st.text_input("Search foods in this category...", key=f"search_{category}").lower()
    
    if search_query:
        items = [item for item in items if search_query in item['name'].lower()]
        if not items:
            st.caption("No foods match your search.")
            return

    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)


# ---------------------------------------------------------------------------
# Cell 7: Initialize Application
# ---------------------------------------------------------------------------

# ------ Initialize Session State (runs only once) ------
initialize_session_state()

# ------ Load Food Database and Assign Emojis (cached) ------
foods = get_food_data('nutrition_results.csv')

# ------ Apply Custom CSS for Enhanced Styling ------
st.markdown("""
<style>
    /* General Readability & Contrast */
    .stApp { background-color: #ffffff; }
    .stCaption { color: #555555; font-size: 0.9rem; }
    [data-testid="stMetricLabel"] > div { font-weight: 500; }

    /* Button Styling */
    .stButton>button[kind="primary"] { background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b; }
    .stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; color: #ff6b6b; }
    .stButton>button[kind="secondary"]:hover { border-color: #ff4757; color: #ff4757; background-color: #fff0f0; }

    /* Sidebar Styling */
    [data-testid="stSidebar"] { background-color: #f0f2f6; }
    
    /* Hide input instructions which can be redundant */
    [data-testid="InputInstructions"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
Welcome! This isn’t just another calorie counter—it’s your personalized guide, built on rock-solid science to help you smash your goals. Whether you’re aiming to shed a few pounds, hold steady, or bulk up, we’ve crunched the numbers so you can focus on enjoying your food. Let’s get rolling! 🚀
""")

# ------ Sidebar for User Input ------
st.sidebar.header("Let’s Get Personal 📊")

# --- Unit Selection ---
st.sidebar.radio(
    "Choose Your Units",
    ["Metric (kg/cm)", "Imperial (lbs/in)"],
    key='units',
    horizontal=True,
)

# --- Save/Load Progress ---
with st.sidebar.expander("Save or Load Your Progress"):
    # Save button
    if st.session_state.submitted:
        user_data_to_save = {key: val for key, val in st.session_state.items() if key.startswith('user_')}
        progress_data = {
            'user_inputs': user_data_to_save,
            'food_selections': st.session_state.food_selections,
            'units': st.session_state.units
        }
        st.download_button(
            label="💾 Save My Progress",
            data=json.dumps(progress_data, indent=4),
            file_name="my_nutrition_progress.json",
            mime="application/json",
            key="save_button"
        )

    # Load button
    uploaded_file = st.file_uploader("📂 Load Progress File (.json)", type="json", key="load_uploader")
    if uploaded_file is not None:
        try:
            loaded_data = json.load(uploaded_file)
            # Restore user inputs
            for key, val in loaded_data.get('user_inputs', {}).items():
                st.session_state[key] = val
            # Restore food selections
            st.session_state.food_selections = loaded_data.get('food_selections', {})
            st.session_state.units = loaded_data.get('units', 'Metric (kg/cm)')
            st.session_state.submitted = True # Assume loaded data should be displayed
            st.toast("Progress loaded successfully! 👍")
            st.rerun() # Rerun to reflect loaded state
        except json.JSONDecodeError:
            st.error("Invalid file format. Please upload a valid JSON file.")

# --- User Input Form ---
with st.sidebar.form(key="user_input_form"):
    all_inputs = {}
    standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
    advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

    for field_name, field_config in standard_fields.items():
        all_inputs[field_name] = create_unified_input(field_name, field_config)

    with st.expander("Advanced Settings ⚙️"):
        for field_name, field_config in advanced_fields.items():
            value = create_unified_input(field_name, field_config)
            if 'convert' in field_config:
                value = field_config['convert'](value)
            all_inputs[field_name] = value

    # --- Form Submission Button ---
    submitted = st.form_submit_button("Calculate & Update Targets", type="primary", use_container_width=True)
    if submitted:
        st.session_state.submitted = True
        # Validate required fields
        required_fields = {
            'age': all_inputs.get('age'),
            'height': all_inputs.get('height'),
            'weight': all_inputs.get('weight')
        }
        missing_fields = [label for label, val in required_fields.items() if val is None or val == 0]
        if missing_fields:
            st.error(f"Whoops! Please fill in all required fields: {', '.join(missing_fields).replace('_', ' ')}.")
            st.session_state.submitted = False # Prevent content from showing
        else:
            # Show motivational toast on successful calculation
            goal_text = dict(CONFIG['form_fields']['goal']['options']).get(all_inputs['goal'])
            st.toast(f"Targets for {goal_text.lower()} calculated! Let's get to it! 💪", icon="🎉")

# ------ Process Final Values ------
final_values = get_final_values({k: st.session_state.get(f'user_{k}') for k in CONFIG['form_fields'].keys()})
targets = calculate_personalized_targets(**final_values)

# ---------------------------------------------------------------------------
# Cell 9: Main Content Area (Conditional Display)
# ---------------------------------------------------------------------------

if not st.session_state.submitted:
    st.info("👈 Pop your details into the sidebar, then hit **Calculate** to get your personalized nutrition plan!")
    st.image("https://images.unsplash.com/photo-1546069901-ba9599a7e63c?q=80&w=1780", caption="Healthy food is waiting for you!", use_column_width=True)
else:
    # ------ Unified Target Display System ------
    goal_labels = dict(CONFIG['form_fields']['goal']['options'])
    goal_label = goal_labels.get(targets['goal'], "Your Goal")
    st.header(f"Your Custom Nutrition Roadmap for {goal_label} 🎯")

    hydration_ml = calculate_hydration_needs(final_values.get('weight_kg'), final_values.get('activity_level'))

    # Display BMR/TDEE Metrics
    st.subheader("Your Metabolic Profile")
    metrics_tdee = [
        ("BMR", f"{targets['bmr']} kcal", None, TOOLTIPS['bmr']),
        ("TDEE", f"{targets['tdee']} kcal", None, TOOLTIPS['tdee']),
        ("Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal", None, TOOLTIPS['caloric_adjustment']),
        ("Est. Weekly Change", f"{targets['estimated_weekly_change']:+.2f} kg", None, TOOLTIPS['weekly_change']),
    ]
    display_metrics_grid(metrics_tdee, 4)
    st.divider()

    # Display Target Metrics
    st.subheader("Your Daily Nutrition Targets")
    metrics_targets = [
        ("Total Calories", f"{targets['total_calories']} kcal"),
        ("Protein", f"{targets['protein_g']} g", f"{targets['protein_percent']:.0f}%"),
        ("Carbohydrates", f"{targets['carb_g']} g", f"{targets['carb_percent']:.0f}%"),
        ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}%"),
        ("Water", f"~{hydration_ml} ml", f"~{hydration_ml/250:.1f} cups", TOOLTIPS['water'])
    ]
    display_metrics_grid(metrics_targets, 5)
    st.divider()
    
    # ---------------------------------------------------------------------------
    # Cell 10: Collapsible Evidence-Based Tips
    # ---------------------------------------------------------------------------
    with st.expander("Explore Your Evidence-Based Game Plan 📚", expanded=False):
        tab1, tab2, tab3 = st.tabs(["🏆 The Big Three", "🧠 Mindset & Strategy", "🔬 The Science"])
        with tab1:
            st.markdown("""
            ### 💧 Master Your Hydration
            - **Daily Goal**: Aim for ~35 ml per kg of body weight.
            - **Training Bonus**: Add 500-750 ml for every hour of intense exercise.
            - **Fat Loss Hack**: Drinking 500 ml of water before meals can increase fullness and reduce calorie intake.
            
            ### 😴 Sleep For Success
            - **The Science**: Getting less than 7 hours of sleep can significantly hinder fat loss and muscle gain.
            - **Daily Goal**: Shoot for a consistent 7-9 hours per night.
            - **Pro Tip**: Create a sleep sanctuary: dark, cool (18-20°C), and screen-free for an hour before bed.

            ### 📊 Track Your Wins
            - **Weigh-in Ritual**: Weigh yourself in the morning, post-bathroom, pre-food, for consistency.
            - **Focus on Trends**: Look at the weekly average weight, not daily blips. Daily weight can fluctuate by 1-2 kg!
            - **Be Patient**: Don't change your plan at the first sign of a stall. Give it at least two weeks.
            """)
        with tab2:
            st.markdown("""
            ### 🎯 The 80/20 Rule
            Perfection is the enemy of progress. Aim for 80% consistency with your targets. This allows for life's unpredictabilities (parties, cravings) without derailing you. It's a marathon, not a sprint.

            ### 📈 When Progress Stalls...
            **If Losing Weight:**
            1. **Honesty Check**: Are you tracking everything accurately? Sauces, oils, and drinks count!
            2. **Add Movement**: Increase daily steps by 2,000 before cutting calories.
            3. **Diet Break**: Every 8-12 weeks, consider eating at maintenance for 1-2 weeks to reset hormones and psychology.

            **If Gaining Weight:**
            1. **Liquid Calories**: Smoothies and shakes are easier to consume than another whole meal.
            2. **Healthy Fats**: Nuts, seeds, and oils are calorie-dense and easy to add.
            3. **Small Bump**: If stuck for 2+ weeks, add 100-200 calories to your daily target.
            """)
        with tab3:
            st.markdown("""
            ### Understanding Your Metabolism
            - **BMR (Basal Metabolic Rate)** is the energy your body needs just to keep the lights on. We use the Mifflin-St Jeor equation, the gold standard for its accuracy.
            - **TDEE (Total Daily Energy Expenditure)** builds on BMR, factoring in your daily activity. This is your true maintenance calorie level.
            
            ### Nutrient Timing
            - **Protein Pacing**: Spread your protein intake evenly across 3-5 meals (20-40g per meal) to maximize muscle protein synthesis.
            - **Workout Window**: Consuming protein and carbs around your workout can aid recovery and performance, but total daily intake is what matters most.
            
            ### Vegetarian/Vegan Focus
            Pay special attention to these micronutrients: **B₁₂, Iron, Calcium, Zinc, and Omega-3s**. Fortified foods (plant milks, nutritional yeast) and targeted supplementation can be a smart strategy. Always consult a healthcare provider for personalized advice.
            """)

    # ---------------------------------------------------------------------------
    # Cell 12: Food Selection Interface
    # ---------------------------------------------------------------------------
    st.header("Track Your Daily Intake 🥗")
    st.markdown("Select your food servings below. Your dashboard and sidebar summary will update in real-time.")

    if st.button("🔄 Start Fresh: Reset All Food Selections", key="reset_foods"):
        st.session_state.food_selections = {}
        st.rerun()

    available_categories = [cat for cat, items in sorted(foods.items()) if items]
    tabs = st.tabs(available_categories)

    for i, category in enumerate(available_categories):
        with tabs[i]:
            items = foods[category]
            sorted_items_in_category = sorted(items, key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
            render_food_grid(sorted_items_in_category, category, columns=2)
            
    # ---------------------------------------------------------------------------
    # Cell 13: Daily Summary and Progress Tracking
    # ---------------------------------------------------------------------------
    st.header("Today’s Scorecard 📊")
    totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

    if selected_foods:
        create_progress_tracking(totals, targets)
        st.divider()
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Nutrition Snapshot")
            summary_metrics = [
                ("Calories", f"{totals['calories']:.0f} kcal"), ("Protein", f"{totals['protein']:.0f} g"),
                ("Carbs", f"{totals['carbs']:.0f} g"), ("Fat", f"{totals['fat']:.0f} g")
            ]
            display_metrics_grid(summary_metrics, 2)
            
            # Export Button
            csv_data = generate_summary_csv(totals, targets, selected_foods)
            st.download_button(
                label="📄 Export Today's Summary (CSV)",
                data=csv_data,
                file_name=f"nutrition_summary_{pd.Timestamp.now().strftime('%Y-%m-%d')}.csv",
                mime="text/csv",
                key="export_csv"
            )

        with col2:
            st.subheader("Macronutrient Split")
            macro_values = [totals['protein'], totals['carbs'], totals['fat']]
            if sum(macro_values) > 0:
                fig = go.Figure(go.Pie(
                    labels=['Protein', 'Carbs', 'Fat'], values=macro_values, hole=.4,
                    marker_colors=['#ff6b6b', '#feca57', '#48dbfb'],
                    textinfo='label+percent', insidetextorientation='radial'
                ))
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Select foods to see the macronutrient split.")

        with st.expander("View Your Logged Foods"):
            for item in selected_foods:
                food, servings = item['food'], item['servings']
                total_cals = food['calories'] * servings
                st.markdown(f"**{food['name']}** - {servings} serving(s) ({total_cals:.0f} kcal)")
    else:
        st.info("Select some foods above to see your scorecard and progress tracking come to life!")
        create_progress_tracking({'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}, targets)

# ---------------------------------------------------------------------------
# Cell 14: Dynamic Sidebar Summary & Feedback Form
# ---------------------------------------------------------------------------

# ------ Add a dynamic summary to the bottom of the sidebar ------
if st.session_state.submitted:
    with st.sidebar:
        st.divider()
        st.header("Live Summary")
        totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
        
        # Calories
        cal_prog = min(totals['calories'] / targets['total_calories'], 1.0) if targets['total_calories'] > 0 else 0
        st.progress(cal_prog, text=f"🔥 Cals: {totals['calories']:.0f} / {targets['total_calories']:.0f}")

        # Protein
        pro_prog = min(totals['protein'] / targets['protein_g'], 1.0) if targets['protein_g'] > 0 else 0
        st.progress(pro_prog, text=f"💪 Pro: {totals['protein']:.0f}g / {targets['protein_g']:.0f}g")

        # Carbs
        carb_prog = min(totals['carbs'] / targets['carb_g'], 1.0) if targets['carb_g'] > 0 else 0
        st.progress(carb_prog, text=f"🍚 Carbs: {totals['carbs']:.0f}g / {targets['carb_g']:.0f}g")

        # Fat
        fat_prog = min(totals['fat'] / targets['fat_g'], 1.0) if targets['fat_g'] > 0 else 0
        st.progress(fat_prog, text=f"🥑 Fat: {totals['fat']:.0f}g / {targets['fat_g']:.0f}g")

# --- Footer and Feedback ---
st.divider()
with st.form("feedback_form", clear_on_submit=True):
    st.subheader("Got Feedback? 💬")
    st.markdown("Help us improve! Let us know what you think or what features you'd like to see.")
    feedback_text = st.text_area("Your feedback", placeholder="I love this app, but I wish it could...")
    submitted_feedback = st.form_submit_button("Send Feedback")

    if submitted_feedback:
        if feedback_text:
            # In a real app, you would send this to a server/database.
            print(f"--- FEEDBACK RECEIVED ---\n{feedback_text}\n-------------------------")
            st.toast("Thank you for your feedback! We appreciate it. 🙏", icon="💖")
        else:
            st.toast("Please enter some feedback before sending.", icon="⚠️")

st.markdown("""
### The Fine Print ⚠️
This tool is for informational purposes and provides estimates. Nutritional needs are highly individual. Consult a qualified healthcare provider before making significant dietary changes. Listen to your body and adjust based on your performance, energy, and overall well-being.
""")
