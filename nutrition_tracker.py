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
   - Your personalized daily targets for calories and macronutrients will be
     calculated and displayed automatically.
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
        'caloric_adjustment': 0.0,  # 0% from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # +10% over TDEE
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
}

# ------ SUGGESTION 2: TOOLTIPS FOR EMOJIS ------
EMOJI_TOOLTIPS = {
    '🥇': "Nutritional All-Star: High in its primary nutrient and calorie-dense.",
    '🔥': "High Calorie: One of the more calorie-dense options in its group.",
    '💪': "High Protein: An excellent source of protein.",
    '🍚': "High Carb: A great source of carbohydrates.",
    '🥑': "High Fat: A rich source of healthy fats."
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
        'age': {'type': 'number', 'label': 'Age (in years)',
                'min': 16, 'max': 80, 'step': 1,
                'placeholder': 'Enter your age in years', 'required': True},
        'height': {'type': 'number', 'label': 'Height',
                   'min_cm': 140, 'max_cm': 220, 'step_cm': 1,
                   'min_in': 55, 'max_in': 87, 'step_in': 1,
                   'placeholder': 'Enter your height', 'required': True},
        'weight': {'type': 'number', 'label': 'Weight',
                   'min_kg': 40.0, 'max_kg': 150.0, 'step_kg': 0.5,
                   'min_lbs': 88.0, 'max_lbs': 330.0, 'step_lbs': 1.0,
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
                           ],
                           'help': "If torn, pick the lower level. It's better to underestimate.",
                           'required': True},
        'goal': {'type': 'selectbox', 'label': 'Your Goal',
                 'options': [
                     ("Weight Loss", "weight_loss"),
                     ("Weight Maintenance", "weight_maintenance"),
                     ("Weight Gain", "weight_gain")
                 ], 'required': True},
        'protein_per_kg': {'type': 'number',
                           'label': 'Protein Goal (g/kg)',
                           'min': 1.2, 'max': 3.0, 'step': 0.1,
                           'help': 'Define your daily protein target in grams per kilogram of body weight',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number',
                           'label': 'Fat Intake (% of calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Set the share of your daily calories that should come from healthy fats',
                           'convert': lambda x: x / 100 if x else None,
                           'advanced': True, 'required': False}
    }
}


# ---------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables."""
    # SUGGESTION 9: ENSURE ALL NECESSARY SESSION STATE VARS ARE INITIALIZED
    session_vars = (
        ['food_selections', 'form_submitted', 'units'] +
        [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    )

    for var in session_vars:
        if var not in st.session_state:
            if var == 'food_selections':
                st.session_state[var] = {}
            elif var == 'form_submitted':
                st.session_state[var] = False
            elif var == 'units':
                st.session_state['units'] = 'Metric'
            else:
                st.session_state[var] = None

def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration."""
    session_key = f'user_{field_name}'

    if field_config['type'] == 'number':
        # SUGGESTION 11: DYNAMIC UNIT HANDLING FOR INPUTS
        if field_name in ['height', 'weight']:
            units = st.session_state.get('units', 'Metric')
            unit_suffix = '_cm' if field_name == 'height' and units == 'Metric' else \
                          '_in' if field_name == 'height' and units == 'Imperial' else \
                          '_kg' if field_name == 'weight' and units == 'Metric' else '_lbs'
            min_val = field_config[f'min{unit_suffix}']
            max_val = field_config[f'max{unit_suffix}']
            step_val = field_config[f'step{unit_suffix}']
            label = f"{field_config['label']} ({'cm' if unit_suffix == '_cm' else 'in' if unit_suffix == '_in' else 'kg' if unit_suffix == '_kg' else 'lbs'})"
        else:
            min_val = field_config['min']
            max_val = field_config['max']
            step_val = field_config['step']
            label = field_config['label']

        if field_config.get('advanced'):
            default_val = DEFAULTS.get(field_name, 0)
            display_val = int(default_val * 100) if field_name == 'fat_percentage' else default_val
            placeholder = f"Default: {display_val}"
        else:
            placeholder = field_config.get('placeholder')

        value = container.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=st.session_state[session_key],
            step=step_val,
            placeholder=placeholder,
            help=field_config.get('help'),
            key=session_key # SUGGESTION 9: EXPLICIT KEY
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
            default_key = DEFAULTS.get(field_name)
            try:
                index = next(i for i, (_, val) in enumerate(options) if val == current_value)
            except StopIteration:
                index = next((i for i, (_, val) in enumerate(options) if val == default_key), 0)

            selection = container.selectbox(
                field_config['label'],
                options,
                index=index,
                format_func=lambda x: x[0],
                help=field_config.get('help'),
                key=session_key # SUGGESTION 9: EXPLICIT KEY
            )
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(
                field_config['label'],
                options,
                index=index,
                key=session_key # SUGGESTION 9: EXPLICIT KEY
            )
    st.session_state[session_key] = value
    return value


def get_final_values(user_inputs):
    """Processes all user inputs and applies default values where needed."""
    final_values = {}
    # SUGGESTION 11: CONVERT IMPERIAL TO METRIC BEFORE CALCULATIONS
    units = st.session_state.get('units', 'Metric')
    if units == 'Imperial':
        if user_inputs.get('weight') is not None:
            user_inputs['weight_kg'] = user_inputs['weight'] * 0.453592
        if user_inputs.get('height') is not None:
            user_inputs['height_cm'] = user_inputs['height'] * 2.54
    else:
        user_inputs['weight_kg'] = user_inputs.get('weight')
        user_inputs['height_cm'] = user_inputs.get('height')

    # Map original field names to calculation keys
    final_values['weight_kg'] = user_inputs.get('weight_kg')
    final_values['height_cm'] = user_inputs.get('height_cm')

    # Process other fields
    for field in ['age', 'sex', 'activity_level', 'goal', 'protein_per_kg', 'fat_percentage']:
        final_values[field] = user_inputs.get(field) if user_inputs.get(field) is not None else DEFAULTS.get(field)

    goal = final_values['goal']
    if goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[goal]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']

    return final_values


def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculates daily fluid needs based on body weight and activity."""
    if not weight_kg: return 0
    base_needs = weight_kg * 35
    activity_bonus = {'sedentary': 0, 'lightly_active': 300, 'moderately_active': 500, 'very_active': 700, 'extremely_active': 1000}
    climate_multiplier = {'cold': 0.9, 'temperate': 1.0, 'hot': 1.2, 'very_hot': 1.4}
    total_ml = (base_needs + activity_bonus.get(activity_level, 500)) * climate_multiplier.get(climate, 1.0)
    return round(total_ml)


def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics in a configurable column layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            # SUGGESTION 2: ADD TOOLTIPS TO METRICS
            st.metric(
                label=metric_info[0],
                value=metric_info[1],
                delta=metric_info[2] if len(metric_info) > 2 else None,
                help=metric_info[3] if len(metric_info) > 3 else None
            )

def find_best_food_for_nutrient(nutrient, deficit, foods):
    """Finds a food that is a good source for a needed nutrient."""
    best_food = None
    highest_nutrient_val = 0
    all_foods = [item for sublist in foods.values() for item in sublist]
    for food in all_foods:
        if food[nutrient] > highest_nutrient_val:
            highest_nutrient_val = food[nutrient]
            best_food = food
    if best_food and highest_nutrient_val > 0:
        suggestion_servings = 1
        return (f"Adding just {suggestion_servings} serving of {best_food['name']} will give you a solid {best_food[nutrient]:.0f} grams of {nutrient}.")
    return None


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = min(actual / target * 100, 100) if target > 0 else 0
        
        # SUGGESTION 1: COLOR-CODED PROGRESS BARS (VIA HTML/MARKDOWN)
        if percent >= 80: color = "#48dbfb" # Blue
        elif percent >= 50: color = "#feca57" # Yellow
        else: color = "#ff6b6b" # Red
        
        progress_html = f"""
        <div style="margin-bottom: 5px;">
            {config['label']}: {actual:.0f} / {target:.0f} {config['unit']} ({percent:.0f}%)
        </div>
        <div style="background-color: #f0f2f6; border-radius: 5px; height: 10px; width: 100%;">
            <div style="background-color: {color}; border-radius: 5px; height: 100%; width: {percent}%;"></div>
        </div>
        """
        st.markdown(progress_html, unsafe_allow_html=True)
        
        if actual < target:
            deficit = target - actual
            base_rec = f"You need {deficit:.0f} more {config['unit']} of {config['label'].lower()}."
            if nutrient in ['protein', 'carbs', 'fat']:
                food_suggestion = find_best_food_for_nutrient(nutrient, deficit, foods)
                if food_suggestion:
                    base_rec += f" **Suggestion:** {food_suggestion}"
            recommendations.append(base_rec)

    return recommendations


def calculate_daily_totals(food_selections, foods):
    """Calculates the total daily nutrition from all selected foods."""
    totals = {nutrient: 0 for nutrient in CONFIG['nutrient_configs'].keys()}
    selected_foods = []
    for category, items in foods.items():
        for food in items:
            servings = food_selections.get(food['name'], 0)
            if servings > 0:
                for nutrient in totals:
                    totals[nutrient] += food[nutrient] * servings
                selected_foods.append({'food': food, 'servings': servings})
    return totals, selected_foods


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
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(age, height_cm, weight_kg, sex='male',
                                   activity_level='moderately_active',
                                   goal='weight_gain', protein_per_kg=None,
                                   fat_percentage=None):
    """Calculates personalized daily nutritional targets."""
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
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4
    estimated_weekly_change = calculate_estimated_weekly_change(caloric_adjustment)

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 3),
        'goal': goal
    }
    
    # SUGGESTION 8: HANDLE DIVIDE-BY-ZERO EDGE CASE
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

def load_food_database(file_path):
    """Loads the vegetarian food database from a specified CSV file."""
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
    return foods


def assign_food_emojis(foods):
    """Assigns emojis to foods based on a unified ranking system."""
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
            if is_high_calorie and is_top_nutrient: food['emoji'] = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: food['emoji'] = emoji_mapping['high_calorie']
            elif food_name in top_foods['protein']: food['emoji'] = emoji_mapping['protein']
            elif food_name in top_foods['carbs']: food['emoji'] = emoji_mapping['carbs']
            elif food_name in top_foods['fat']: food['emoji'] = emoji_mapping['fat']
            else: food['emoji'] = ''
    return foods

# SUGGESTION 7: CACHE THE COMBINED LOADING AND EMOJI ASSIGNMENT
@st.cache_data
def load_and_process_foods(file_path):
    """Loads food database and assigns emojis, caching the result."""
    foods = load_food_database(file_path)
    foods_with_emojis = assign_food_emojis(foods)
    return foods_with_emojis


def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        emoji = food.get('emoji', '')
        # SUGGESTION 2: Add tooltip to emoji
        tooltip = EMOJI_TOOLTIPS.get(emoji, "")
        st.subheader(f"{emoji} {food['name']}", help=tooltip if tooltip else None)
        
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)
        col1, col2 = st.columns([2, 1.2])

        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key}_{k}", type=button_type, help=f"Set to {k} servings", use_container_width=True):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()

        with col2:
            # SUGGESTION 8: CAP MAX SERVINGS
            custom_serving = st.number_input("Custom", min_value=0.0, max_value=20.0, value=float(current_serving), step=0.1, key=f"{key}_custom", label_visibility="collapsed")
        
        # SUGGESTION 9: Use callbacks to reduce full reruns if logic was more complex, but for now this is fine.
        if custom_serving != current_serving:
            if custom_serving > 0:
                st.session_state.food_selections[food['name']] = custom_serving
            elif food['name'] in st.session_state.food_selections:
                del st.session_state.food_selections[food['name']]
            st.rerun()

        st.caption(f"Per Serving: {food['calories']} kcal | {food['protein']}g P | {food['carbs']}g C | {food['fat']}g F")


def render_food_grid(items, category, columns=2):
    """Renders a grid of food items for a given category."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)


# ---------------------------------------------------------------------------
# Cell 7: Initialize Application
# ---------------------------------------------------------------------------

initialize_session_state()

# Use the new cached function
foods = load_and_process_foods('nutrition_results.csv')

# SUGGESTION 10: IMPROVE COLOR CONTRAST
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
/* A slightly darker, more accessible primary color */
.stButton>button[kind="primary"] { background-color: #D9534F; color: white; border: 1px solid #D9534F; }
.stButton>button[kind="secondary"] { border: 1px solid #D9534F; }
/* Ensure caption text is readable */
.stCaption { color: #606770; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("A Smart, Evidence-Based Nutrition Tracker That Actually Gets You...")

# ------ Sidebar for User Input with Form ------
with st.sidebar.form(key='user_input_form'):
    st.header("Let’s Get Personal 📊")
    
    # SUGGESTION 11: UNITS TOGGLE
    st.session_state.units = st.radio("Units", ["Metric (kg, cm)", "Imperial (lbs, in)"], horizontal=True, key='units_radio')

    all_inputs = {}
    standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
    advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

    for field_name, field_config in standard_fields.items():
        # Rename height_cm/weight_kg to generic names for the toggle
        generic_name = 'height' if 'height' in field_name else 'weight' if 'weight' in field_name else field_name
        value = create_unified_input(generic_name, field_config)
        all_inputs[generic_name] = value

    with st.expander("Advanced Settings ⚙️"):
        for field_name, field_config in advanced_fields.items():
            value = create_unified_input(field_name, field_config)
            if 'convert' in field_config:
                value = field_config['convert'](value)
            all_inputs[field_name] = value
    
    # SUGGESTION 6: VALIDATE ON SUBMIT
    submitted = st.form_submit_button("Calculate My Targets")
    if submitted:
        required_fields = {
            field: config['label'] for field, config in CONFIG['form_fields'].items() if config.get('required')
        }
        missing_fields = [label for field, label in required_fields.items() if all_inputs.get(field) is None]
        
        if not missing_fields:
            st.session_state.form_submitted = True
            st.session_state.update(all_inputs) # Save inputs to state
            # SUGGESTION 12: MOTIVATIONAL NOTIFICATION
            st.toast("🚀 Targets calculated! Let's get started.", icon="🎉")
        else:
            st.session_state.form_submitted = False
            for field in missing_fields:
                st.error(f"Please fill in the '{field}' field.")

# Process final values only after successful submission
if st.session_state.form_submitted:
    # Build inputs from session state for calculations
    user_inputs_from_state = {key.replace('user_', ''): val for key, val in st.session_state.items() if key.startswith('user_')}
    final_values = get_final_values(user_inputs_from_state)
    targets = calculate_personalized_targets(**final_values)
else:
    # Use defaults if form not submitted
    final_values = get_final_values({key: val for key, val in DEFAULTS.items() if key in ['age', 'sex', 'activity_level', 'goal', 'protein_per_kg', 'fat_percentage']})
    final_values['weight_kg'] = DEFAULTS['weight_kg']
    final_values['height_cm'] = DEFAULTS['height_cm']
    targets = calculate_personalized_targets(**final_values)

# ------ SUGGESTION 15: DYNAMIC SIDEBAR SUMMARY ------
def render_sidebar_summary(totals, targets):
    st.sidebar.divider()
    st.sidebar.subheader("Live Summary")
    if sum(totals.values()) > 0 and st.session_state.form_submitted:
        for nutrient, config in CONFIG['nutrient_configs'].items():
            actual = totals[nutrient]
            target_val = targets[config['target_key']]
            progress = min(actual / target_val, 1.0) if target_val > 0 else 0
            st.sidebar.progress(progress, text=f"{config['label']}: {actual:.0f}/{target_val:.0f}")
    else:
        st.sidebar.caption("Log food to see progress.")

# This calculation needs to happen regardless of form submission for the summary to update live
live_totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
render_sidebar_summary(live_totals, targets)

# ---------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# ---------------------------------------------------------------------------

if not st.session_state.form_submitted:
    st.info("👈 Pop your details into the sidebar and hit 'Calculate' to get your personalized targets.")
    st.header("Sample Daily Targets for Reference")
else:
    goal_labels = {'weight_loss': 'Weight Loss', 'weight_maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Custom Nutrition Roadmap for {goal_label} 🎯")

# Display hydration needs using the correct weight unit
display_weight = final_values.get('weight') or DEFAULTS['weight_kg']
hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 4,
        'metrics': [
            ("BMR", f"{targets['bmr']} kcal", None, "Basal Metabolic Rate: Calories your body burns at complete rest."),
            ("TDEE", f"{targets['tdee']} kcal", None, "Total Daily Energy Expenditure: Your BMR plus calories burned from activity."),
            ("Caloric Adj.", f"{targets['caloric_adjustment']:+} kcal", None, "The daily calorie surplus or deficit to meet your goal."),
            ("Est. Weekly Change", f"{targets['estimated_weekly_change']:+.2f} kg", None, "Your estimated weight change per week based on the caloric adjustment.")
        ]
    },
    {
        'title': 'Your Daily Nutrition Targets', 'columns': 4,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            ("Protein", f"{targets['protein_g']} g", f"{targets['protein_percent']:.0f}%"),
            ("Carbohydrates", f"{targets['carb_g']} g", f"{targets['carb_percent']:.0f}%"),
            ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}%"),
        ]
    }
]

for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# ---------------------------------------------------------------------------
# Cell 10: Enhanced Evidence-Based Tips and Context
# ---------------------------------------------------------------------------

# SUGGESTION 3: COLLAPSE LONG SECTIONS
with st.expander("Expand for Your Evidence-Based Game Plan & The Science 📚", expanded=False):
    tab1, tab2, tab3 = st.tabs(["🏆 The Big Three", "🧠 Mindset & Strategy", "🔬 The Science"])
    with tab1:
        st.subheader("💧 Master Your Hydration Game")
        st.markdown("Daily Goal: Aim for about 35 ml per kilogram of body weight.")
    with tab2:
        st.subheader("Mindset Is Everything 🧠")
        st.markdown("The 80/20 principle is your best friend. Aim for consistency, not perfection.")
    with tab3:
        st.subheader("Understanding Your Metabolism")
        st.markdown("Your BMR is your baseline, TDEE is your total burn. We use these to create your targets.")

# ---------------------------------------------------------------------------
# Cell 11: (No Cell 11 in original, this section is for new features)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# ---------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")

# SUGGESTION 5: ADD FOOD SEARCH
search_term = st.text_input("🔍 Search for a food item", placeholder="e.g., Tofu, Almonds...")

# SUGGESTION 4: SAVE/LOAD FUNCTIONALITY
with st.container(border=True):
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        uploaded_file = st.file_uploader("Load Progress From File", type=['json'], key='progress_uploader')
        if uploaded_file is not None:
            progress_data = json.load(uploaded_file)
            st.session_state.food_selections = progress_data.get('food_selections', {})
            user_data = progress_data.get('user_inputs', {})
            for key, value in user_data.items():
                st.session_state[f'user_{key}'] = value
            st.toast("Progress loaded successfully! Please click 'Calculate' to update targets.", icon="✅")
            st.rerun()
    with c2:
        if st.button("🔄 Reset Selections", use_container_width=True):
            st.session_state.food_selections = {}
            st.rerun()
    with c3:
        # Prepare data for download
        user_inputs_for_save = {key.replace('user_', ''): val for key, val in st.session_state.items() if key.startswith('user_')}
        progress_to_save = {
            'user_inputs': user_inputs_for_save,
            'food_selections': st.session_state.food_selections
        }
        st.download_button(
            label="💾 Save Progress",
            data=json.dumps(progress_to_save, indent=4),
            file_name="nutrition_progress.json",
            mime="application/json",
            use_container_width=True,
            help="Saves your inputs and food selections to a file."
        )

available_categories = [cat for cat, items in sorted(foods.items()) if items]
tabs = st.tabs(available_categories)
for i, category in enumerate(available_categories):
    with tabs[i]:
        items = foods[category]
        if search_term:
            filtered_items = [item for item in items if search_term.lower() in item['name'].lower()]
        else:
            filtered_items = items
        
        sorted_items = sorted(filtered_items, key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
        if sorted_items:
            render_food_grid(sorted_items, category, columns=2)
        else:
            st.caption("No food items match your search.")

# ---------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking
# ---------------------------------------------------------------------------

st.header("Today’s Scorecard 📊")
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Nutrition Snapshot")
        summary_metrics = [("Calories", f"{totals['calories']:.0f} kcal"), ("Protein", f"{totals['protein']:.0f} g"), ("Carbs", f"{totals['carbs']:.0f} g"), ("Fat", f"{totals['fat']:.0f} g")]
        display_metrics_grid(summary_metrics, 2)

    with col2:
        st.subheader("Macronutrient Split")
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
        if sum(macro_values) > 0:
            fig = go.Figure(go.Pie(labels=['Protein', 'Carbs', 'Fat'], values=macro_values, hole=.4, marker_colors=['#D9534F', '#feca57', '#48dbfb'], textinfo='label+percent', insidetextorientation='radial'))
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=250)
            st.plotly_chart(fig, use_container_width=True)

    with st.expander("Your Food Choices Today"):
        for item in selected_foods:
            food, servings = item['food'], item['servings']
            st.write(f"**{food['name']}** - {servings} serving(s)")
    
    # SUGGESTION 13: EXPORT DAILY SUMMARY
    summary_data = {
        "Metric": ["Target", "Consumed", "Remaining"],
        "Calories (kcal)": [targets['total_calories'], totals['calories'], targets['total_calories'] - totals['calories']],
        "Protein (g)": [targets['protein_g'], totals['protein'], targets['protein_g'] - totals['protein']],
        "Carbs (g)": [targets['carb_g'], totals['carbs'], targets['carb_g'] - totals['carbs']],
        "Fat (g)": [targets['fat_g'], totals['fat'], targets['fat_g'] - totals['fat']],
    }
    summary_df = pd.DataFrame(summary_data)
    st.download_button(
        "📄 Export Summary to CSV",
        summary_df.to_csv(index=False),
        "daily_nutrition_summary.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.info("Log some food items from the categories above to see your progress!")

# ---------------------------------------------------------------------------
# Cell 14: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()

# SUGGESTION 14: GATHER USER FEEDBACK
with st.expander("Give Feedback"):
    with st.form("feedback_form", clear_on_submit=True):
        feedback_text = st.text_area("How can we improve this app?")
        submitted = st.form_submit_button("Submit")
        if submitted:
            # In a real app, you would email this or save it to a database
            st.success("Thank you for your feedback!")

st.markdown("### The Fine Print ⚠️")
st.caption("This tool is for informational purposes only. Consult a healthcare provider before making significant dietary changes.")

# ---------------------------------------------------------------------------
# Cell 15: Session State Management and Performance
# ---------------------------------------------------------------------------
# Clean up logic is already present, which is good practice.
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {k: v for k, v in st.session_state.food_selections.items() if v > 0}
