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
   - Click "Calculate" to validate your inputs and generate your plan.
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
                   'min': 0.0, 'max': 250.0, 'step': 1.0,
                   'placeholder': 'Enter your height', 'required': True},
        'weight': {'type': 'number', 'label': 'Weight',
                   'min': 0.0, 'max': 200.0, 'step': 0.5,
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
                           'help': 'Define your daily protein target in grams per kilogram of body weight',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number',
                           'label': 'Fat Intake (% of calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Set the share of your daily calories that should come from healthy fats',
                           'convert': lambda x: x / 100 if x else None,
                           'advanced': True, 'required': False}
    },
    'metric_tooltips': {
        "Basal Metabolic Rate (BMR)": "The number of calories your body needs to function at rest. Think of it as your metabolic baseline.",
        "Total Daily Energy Expenditure (TDEE)": "Your BMR plus the calories you burn through daily activities and exercise. This is your total daily calorie burn.",
        "Daily Caloric Adjustment": "The surplus or deficit of calories applied to your TDEE to help you achieve your goal (e.g., -500 kcal for weight loss).",
        "Estimated Weekly Weight Change": "A projection of your weekly weight change based on your caloric adjustment. Note: 7,700 kcal is roughly equivalent to 1 kg of body fat.",
        "Water": "Your estimated daily fluid needs. This can vary based on activity, climate, and individual factors."
    },
    'emoji_tooltips': {
        '🥇': "Gold Medal: A nutritional all-star! High in its target nutrient and very calorie-efficient.",
        '🔥': "High Calorie: One of the more calorie-dense options in its group.",
        '💪': "High Protein: A true protein powerhouse.",
        '🍚': "High Carb: A carbohydrate champion.",
        '🥑': "High Fat: A healthy fat hero."
    }
}


# ---------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables if they don't exist."""
    session_vars = {
        'food_selections': {},
        'calculation_done': False,
        'show_notification': False,
        'search_query': "",
        'unit_system': "Metric (kg/cm)",
        'user_age': None, 'user_height': None, 'user_weight': None,
        'user_sex': DEFAULTS['sex'],
        'user_activity_level': DEFAULTS['activity_level'],
        'user_goal': DEFAULTS['goal'],
        'user_protein_per_kg': None, 'user_fat_percentage': None
    }
    for var, default_value in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default_value


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration."""
    session_key = f'user_{field_name}'
    input_key = f'input_{field_name}'  # Use a unique key for the widget itself

    # Special handling for unit-dependent fields
    if field_name == 'height':
        label = "Height (cm)" if st.session_state.unit_system == "Metric (kg/cm)" else "Height (in)"
        min_val, max_val = (140, 220) if st.session_state.unit_system == "Metric (kg/cm)" else (55, 87)
    elif field_name == 'weight':
        label = "Weight (kg)" if st.session_state.unit_system == "Metric (kg/cm)" else "Weight (lbs)"
        min_val, max_val = (40.0, 150.0) if st.session_state.unit_system == "Metric (kg/cm)" else (88.0, 330.0)
    else:
        label = field_config['label']
        min_val = field_config.get('min')
        max_val = field_config.get('max')

    if field_config['type'] == 'number':
        value = container.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=st.session_state[session_key],
            step=field_config['step'],
            placeholder=field_config.get('placeholder'),
            help=field_config.get('help'),
            key=input_key
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
            try:
                index = next(i for i, (_, val) in enumerate(options) if val == current_value)
            except StopIteration:
                index = 0
            selection = container.selectbox(
                label, options, index=index, format_func=lambda x: x[0], key=input_key
            )
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(label, options, index=index, key=input_key)

    st.session_state[session_key] = value
    return value


def get_final_values():
    """Processes all user inputs from session state and applies defaults."""
    user_inputs = {key.replace('user_', ''): value for key, value in st.session_state.items() if key.startswith('user_')}
    final_values = {}

    # Convert imperial units to metric for calculation
    if st.session_state.unit_system == "Imperial (lbs/in)":
        if user_inputs.get('weight') is not None:
            final_values['weight_kg'] = user_inputs['weight'] * 0.453592
        else:
            final_values['weight_kg'] = DEFAULTS['weight_kg']

        if user_inputs.get('height') is not None:
            final_values['height_cm'] = user_inputs['height'] * 2.54
        else:
            final_values['height_cm'] = DEFAULTS['height_cm']
    else:
        final_values['weight_kg'] = user_inputs.get('weight') if user_inputs.get('weight') is not None else DEFAULTS['weight_kg']
        final_values['height_cm'] = user_inputs.get('height') if user_inputs.get('height') is not None else DEFAULTS['height_cm']

    # Process other fields
    for field, value in user_inputs.items():
        if field not in ['weight', 'height']:
            final_values[field] = value if value is not None else DEFAULTS.get(field)

    goal = final_values.get('goal', DEFAULTS['goal'])
    if goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[goal]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']

    return final_values


def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculates daily fluid needs based on body weight and activity."""
    base_needs = weight_kg * 35  # Baseline is 35 milliliters per kilogram
    activity_bonus = {'sedentary': 0, 'lightly_active': 300, 'moderately_active': 500, 'very_active': 700, 'extremely_active': 1000}
    climate_multiplier = {'cold': 0.9, 'temperate': 1.0, 'hot': 1.2, 'very_hot': 1.4}
    total_ml = (base_needs + activity_bonus.get(activity_level, 500)) * climate_multiplier.get(climate, 1.0)
    return round(total_ml)


def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics with tooltips in a configurable layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            label, value, *rest = metric_info
            delta = rest[0] if rest else None
            tooltip = CONFIG['metric_tooltips'].get(label)
            
            # Use markdown for label with tooltip if available
            if tooltip:
                label_html = f'{label} <span title="{tooltip}">&#9432;</span>'
                st.markdown(f"**{label_html}**", unsafe_allow_html=True)
                st.metric(label="", value=value, delta=delta, label_visibility="collapsed")
            else:
                 st.metric(label, value, delta)

def get_progress_bar_html(percent, text):
    """Generates HTML for a color-coded progress bar."""
    if percent >= 80:
        color = "#2ecc71"  # Green
    elif percent >= 50:
        color = "#f1c40f"  # Yellow
    else:
        color = "#e74c3c"  # Red

    return f"""
    <div style="margin-bottom: 0.5rem;">
        <div style="font-size: 0.9rem; margin-bottom: 0.2rem;">{text}</div>
        <div style="background-color: #e0e0e0; border-radius: 5px; height: 20px; width: 100%;">
            <div style="background-color: {color}; width: {percent}%; height: 100%; border-radius: 5px; text-align: center; color: white; font-weight: bold;">
            </div>
        </div>
    </div>
    """


def create_progress_tracking(totals, targets):
    """Creates color-coded progress bars for nutritional targets."""
    st.subheader("Your Daily Dashboard 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = min(actual / target * 100, 100) if target > 0 else 0
        text = f"{config['label']}: {actual:.0f} / {target:.0f} {config['unit']} ({percent:.0f}%)"
        st.markdown(get_progress_bar_html(percent, text), unsafe_allow_html=True)


def calculate_daily_totals(food_selections, foods):
    """Calculates the total daily nutrition from all selected foods."""
    totals = {nutrient: 0 for nutrient in CONFIG['nutrient_configs'].keys()}
    selected_foods = []
    
    all_foods_flat = {food['name']: food for category_items in foods.values() for food in category_items}

    for food_name, servings in food_selections.items():
        if servings > 0 and food_name in all_foods_flat:
            food = all_foods_flat[food_name]
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


def calculate_estimated_weekly_change(daily_caloric_adjustment, unit_system="Metric (kg/cm)"):
    """Calculates the estimated weekly weight change from a caloric adjustment."""
    # 7,700 kcal is approx. 1 kg of body fat. 1 kg = 2.20462 lbs.
    kg_change = (daily_caloric_adjustment * 7) / 7700
    if unit_system == "Imperial (lbs/in)":
        return kg_change * 2.20462
    return kg_change


def calculate_personalized_targets(age, height_cm, weight_kg, sex='male',
                                   activity_level='moderately_active',
                                   goal='weight_gain', protein_per_kg=None,
                                   fat_percentage=None, unit_system="Metric (kg/cm)"):
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
    carb_g = carb_calories / 4 if carb_calories > 0 else 0

    estimated_weekly_change = calculate_estimated_weekly_change(caloric_adjustment, unit_system)

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 2),
        'goal': goal
    }

    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
    else: # Handle edge case of zero calories
        targets['protein_percent'] = 0
        targets['carb_percent'] = 0
        targets['fat_percent'] = 0

    return targets


# ---------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_process_foods(file_path):
    """Loads and processes the food database, including assigning emojis. Cached for performance."""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in df['category'].unique()}
    for _, row in df.iterrows():
        foods[row['category']].append({
            'name': f"{row['name']} ({row['serving_unit']})",
            'calories': row['calories'], 'protein': row['protein'],
            'carbs': row['carbs'], 'fat': row['fat']
        })
    return assign_food_emojis(foods)


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

    for items in foods.values():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = any(food_name in top_list for top_list in top_foods['calories'].values())

            if is_high_calorie and is_top_nutrient: food['emoji'] = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: food['emoji'] = emoji_mapping['high_calorie']
            elif food_name in top_foods['protein']: food['emoji'] = emoji_mapping['protein']
            elif food_name in top_foods['carbs']: food['emoji'] = emoji_mapping['carbs']
            elif food_name in top_foods['fat']: food['emoji'] = emoji_mapping['fat']
            else: food['emoji'] = ''
    return foods


def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        emoji = food.get('emoji', '')
        tooltip = CONFIG['emoji_tooltips'].get(emoji, "")
        st.markdown(f'<h5><span title="{tooltip}">{emoji}</span> {food["name"]}</h5>', unsafe_allow_html=True)

        key_base = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)
        
        col1, col2 = st.columns([2, 1.2])
        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key_base}_{k}", type=button_type, help=f"Set to {k} servings", use_container_width=True):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()
        with col2:
            custom_serving = st.number_input(
                "Custom Servings", min_value=0.0, max_value=20.0, # Capped at 20 servings
                value=float(current_serving), step=0.1, key=f"{key_base}_custom", label_visibility="collapsed"
            )
            if custom_serving != current_serving:
                if custom_serving > 0: st.session_state.food_selections[food['name']] = custom_serving
                elif food['name'] in st.session_state.food_selections: del st.session_state.food_selections[food['name']]
                st.rerun()

        caption_text = f"Per Serving: {food['calories']:.0f} kcal | {food['protein']:.1f}g P | {food['carbs']:.1f}g C | {food['fat']:.1f}g F"
        st.caption(caption_text)


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
foods = load_and_process_foods('nutrition_results.csv')

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.5rem; }
[data-testid="stMetricLabel"] { font-size: 1rem; }
[data-testid="stCaption"] { color: #555; }
.stButton>button[kind="primary"] { background-color: #ff6b6b; color: black; border: 1px solid #ff6b6b; }
.stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; color: #ff6b6b }
h5 { margin-bottom: 0.1rem; }
span[title] {
    text-decoration: none;
    border-bottom: 1px dotted #777;
    cursor: help;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")

# ------ Sidebar for User Input ------
with st.sidebar:
    st.header("Let’s Get Personal 📊")
    
    st.session_state.unit_system = st.radio(
        "Unit System",
        ["Metric (kg/cm)", "Imperial (lbs/in)"],
        key="unit_system_radio",
        horizontal=True,
    )
    
    with st.form(key="user_input_form"):
        standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
        advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

        for field_name, field_config in standard_fields.items():
            create_unified_input(field_name, field_config)
        
        with st.expander("Advanced Settings ⚙️"):
            for field_name, field_config in advanced_fields.items():
                value = create_unified_input(field_name, field_config)
                if 'convert' in field_config:
                    st.session_state[f'user_{field_name}'] = field_config['convert'](value)

        submitted = st.form_submit_button("Calculate My Plan", use_container_width=True, type="primary")

    if submitted:
        required_fields_map = {'age': 'Age', 'height': 'Height', 'weight': 'Weight'}
        missing_fields = [label for field, label in required_fields_map.items() if st.session_state[f'user_{field}'] is None]
        
        if missing_fields:
            st.error(f"Hold on! Please fill in these required fields: {', '.join(missing_fields)}")
        else:
            st.session_state.calculation_done = True
            st.session_state.show_notification = True
    
    # ------ Dynamic Sidebar Summary ------
    if st.session_state.calculation_done and st.session_state.food_selections:
        st.divider()
        st.subheader("Today's Progress")
        final_values = get_final_values()
        targets = calculate_personalized_targets(**final_values, unit_system=st.session_state.unit_system)
        totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
        
        for nutrient, config in CONFIG['nutrient_configs'].items():
            actual = totals[nutrient]
            target = targets[config['target_key']]
            st.markdown(f"**{config['label']}:** {actual:.0f} / {target:.0f} {config['unit']}")
        st.divider()

    with st.sidebar.expander("Your Activity Level Decoded"):
        st.markdown("""
        * **🧑‍💻 Sedentary**: You're basically married to your desk chair
        * **🏃 Lightly Active**: You squeeze in walks or workouts one to three times a week
        * **🚴 Moderately Active**: You're sweating it out three to five days a week
        * **🏋️ Very Active**: You might actually be part treadmill
        * **🤸 Extremely Active**: You live in the gym and sweat is your second skin
        
        *💡 If you’re torn between two levels, pick the lower one. It’s better to underestimate your burn than to overeat and stall.*
        """)


# ------ Initial Welcome Message ------
if not st.session_state.calculation_done:
    st.info("👈 Pop your details into the sidebar and click **'Calculate My Plan'** to get started!")
    st.markdown("""
    Welcome aboard! This isn’t just another calorie counter—it’s your personalized guide, built on rock-solid science to help you smash your goals. Whether you’re aiming to shed a few pounds, hold steady, or bulk up, we’ve crunched the numbers so you can focus on enjoying your food.
    
    Let’s get rolling—your journey to feeling awesome starts now! 🚀
    """)


# ---------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# ---------------------------------------------------------------------------

if st.session_state.calculation_done:
    final_values = get_final_values()
    targets = calculate_personalized_targets(**final_values, unit_system=st.session_state.unit_system)

    # Motivational Notification
    if st.session_state.show_notification:
        change_val = targets['estimated_weekly_change']
        unit = "lbs" if st.session_state.unit_system == "Imperial (lbs/in)" else "kg"
        st.toast(f"Plan calculated! You're on track for a weekly change of ~{change_val:+.2f} {unit}. 🔥", icon='🎉')
        st.session_state.show_notification = False

    goal_labels = {'weight_loss': 'Weight Loss', 'weight_maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Custom Nutrition Roadmap for {goal_label} 🎯")

    hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
    unit_label = "lbs" if st.session_state.unit_system == "Imperial (lbs/in)" else "kg"
    
    metrics_config = [
        {'title': 'Metabolic Information', 'columns': 4,
         'metrics': [
             ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal"),
             ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal"),
             ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
             ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} {unit_label}")
         ]},
        {'title': 'Your Daily Nutrition Targets', 'columns': 5,
         'metrics': [
             ("Total Calories", f"{targets['total_calories']} kcal"),
             ("Protein", f"{targets['protein_g']} g", f"{targets['protein_percent']:.0f}% of calories"),
             ("Carbohydrates", f"{targets['carb_g']} g", f"{targets['carb_percent']:.0f}% of calories"),
             ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}% of calories"),
             ("Water", f"{hydration_ml} ml", f"~{hydration_ml/250:.1f} cups")
         ]}
    ]
    for config in metrics_config:
        st.subheader(config['title'])
        display_metrics_grid(config['metrics'], config['columns'])
        st.divider()


    # ---------------------------------------------------------------------------
    # Cell 10: Enhanced Evidence-Based Tips (Collapsible)
    # ---------------------------------------------------------------------------
    
    with st.expander("Your Evidence-Based Game Plan 📚", expanded=False):
        # The content from the original Cell 10 tabs is placed here.
        # This makes the main view cleaner by default.
        tab1, tab2, tab3, tab4 = st.tabs([
            "The Big Three to Win At Nutrition 🏆", "Level Up Your Progress Tracking 📊",
            "Mindset Is Everything 🧠", "The Science Behind the Magic 🔬"
        ])
        with tab1:
            st.subheader("💧 Master Your Hydration Game")
            st.markdown("Daily Goal: Shot for about 35 ml per kilogram of your body weight daily. Training Bonus: Tack on an extra 500-750 ml per hour of sweat time. Fat Loss Hack: Chugging 500 ml of water before meals can boost fullness by by 13%.")
            st.subheader("😴 Sleep Like Your Goals Depend on It")
            st.markdown("The Shocking Truth: Getting less than 7 hours of sleep can torpedo your fat loss by more than half. Daily Goal: Shoot for 7-9 hours and try to keep a consistent schedule. Set the Scene: Keep your cave dark, cool (18-20°C), and screen-free for at least an hour before lights out.")
        with tab2:
            st.subheader("Go Beyond the Scale 📸")
            st.markdown("The Bigger Picture: Snap a few pics every month. Use the same pose, lighting, and time of day. The mirror doesn't lie. Size Up Your Wins: Measure your waist, hips, arms, and thighs monthly. The Quiet Victories: Pay attention to how you feel. Your energy levels, sleep quality, gym performance, and hunger patterns tell a story numbers can’t.")
        with tab3:
            st.subheader("Mindset Is Everything 🧠")
            st.markdown("The 80/20 principle is your best defense against the perfectionist trap. Start Small, Win Big: Weeks 1–2: Focus on hitting your calorie targets. Weeks 3–4: Start layering in protein tracking. Week 5+: Fine-tune your carb and fat intake.")
        with tab4:
            st.subheader("Understanding Your Metabolism")
            st.markdown("Your Basal Metabolic Rate (BMR) is the energy your body needs just to keep the lights on. Your Total Daily Energy Expenditure (TDEE) builds on that baseline by factoring in how active you are. Not all calories are created equal. Prioritize Protein, then Fiber-Rich Carbs, then Healthy Fats.")


    # ---------------------------------------------------------------------------
    # Cell 12: Food Selection Interface
    # ---------------------------------------------------------------------------

    st.header("Track Your Daily Intake 🥗")
    st.markdown("Select your foods below to see how your intake stacks up against your daily targets.")

    # ------ Search, Save, and Load Controls ------
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    with c1:
        st.session_state.search_query = st.text_input("🔎 Search for a food...", value=st.session_state.search_query, key="food_search")
    with c2:
        if st.button("🔄 Reset Selections", use_container_width=True, key='reset_all_selections'):
            st.session_state.food_selections = {}
            st.rerun()
    with c3:
        # Save progress to JSON
        json_data = json.dumps(st.session_state.food_selections, indent=2)
        st.download_button(
            label="💾 Save Progress",
            data=json_data,
            file_name="my_nutrition_progress.json",
            mime="application/json",
            use_container_width=True,
            key='save_progress_button'
        )
    with c4:
        # Load progress from JSON
        uploaded_file = st.file_uploader("📂 Load Progress", type="json", label_visibility="collapsed", key="load_progress_uploader")
        if uploaded_file is not None:
            try:
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                loaded_selections = json.load(stringio)
                if isinstance(loaded_selections, dict):
                    st.session_state.food_selections = loaded_selections
                    st.toast("Progress loaded successfully!", icon="✅")
                    st.rerun()
                else:
                    st.error("Invalid file format. Please upload a valid JSON file.")
            except Exception as e:
                st.error(f"Error loading file: {e}")

    # ------ Food Selection with Tabs ------
    available_categories = [cat for cat, items in sorted(foods.items()) if items]
    tabs = st.tabs(available_categories)

    for i, category in enumerate(available_categories):
        with tabs[i]:
            items_in_category = foods[category]
            
            # Filter based on search query
            if st.session_state.search_query:
                filtered_items = [
                    item for item in items_in_category
                    if st.session_state.search_query.lower() in item['name'].lower()
                ]
            else:
                filtered_items = items_in_category

            if not filtered_items:
                st.caption("No food found with that name in this category.")
            else:
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories'])
                )
                render_food_grid(sorted_items, category, columns=2)


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
            st.subheader("Macronutrient Split")
            macro_values = [totals['protein'], totals['carbs'], totals['fat']]
            if sum(macro_values) > 0:
                fig = go.Figure(go.Pie(
                    labels=['Protein', 'Carbs', 'Fat'], values=macro_values, hole=.4,
                    marker_colors=['#ff6b6b', '#feca57', '#48dbfb'], textinfo='label+percent',
                    insidetextorientation='radial'
                ))
                fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=250)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption("Select foods to see the macronutrient split.")

        with col2:
            st.subheader("Logged Foods")
            summary_df = pd.DataFrame([{
                'Food': item['food']['name'],
                'Servings': item['servings'],
                'Calories (kcal)': item['food']['calories'] * item['servings'],
                'Protein (g)': item['food']['protein'] * item['servings'],
            } for item in selected_foods])
            st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            # Export to CSV
            csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Summary (CSV)",
                csv,
                "daily_summary.csv",
                "text/csv",
                key='download-csv',
                use_container_width=True
            )
            
    else:
        st.info("Start selecting foods from the tabs above to see your progress here!")


# ---------------------------------------------------------------------------
# Cell 14: Footer, Disclaimers, and Feedback
# ---------------------------------------------------------------------------

st.divider()
with st.expander("The Science We Stand On & The Fine Print ⚠️"):
    st.markdown("""
    ### The Science We Stand On 📚
    This tracker isn't built on guesswork—it's grounded in peer-reviewed research. We rely on the **Mifflin-St Jeor equation** to calculate your Basal Metabolic Rate (BMR), a method endorsed by the Academy of Nutrition and Dietetics. Our protein recommendations are based on guidelines from the International Society of Sports Nutrition.

    ### The Fine Print ⚠️
    This tool is a guide, not a medical diagnosis. Your individual needs may vary due to genetics, health conditions, or other factors. Always consult a qualified healthcare provider before making significant dietary changes. Listen to your body and adjust as needed.
    """)

# ------ User Feedback Form ------
st.subheader("How Can We Improve? 💬")
with st.form("feedback_form", clear_on_submit=True):
    feedback_text = st.text_area("Share your thoughts, suggestions, or issues here.", key="feedback_text_area")
    submitted_feedback = st.form_submit_button("Submit Feedback")
    if submitted_feedback:
        if feedback_text:
            # In a real app, this would send the feedback to a server/database.
            st.success("Thank you for your feedback! We appreciate you helping us improve. 👍")
        else:
            st.warning("Please enter some feedback before submitting.")
