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

[Previous docstring content remains the same...]
"""

# ---------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# ---------------------------------------------------------------------------

import math
import json
import csv
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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
    'fat_percentage': 0.25,
    'use_metric': True
}

# ------ Unit Conversion Constants ------
CONVERSION_FACTORS = {
    'kg_to_lbs': 2.20462,
    'cm_to_inches': 0.393701,
    'lbs_to_kg': 0.453592,
    'inches_to_cm': 2.54
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
    'emoji_tooltips': {
        '🥇': 'Gold Medal: Nutritional all-star! High in target nutrient and calorie-efficient',
        '🔥': 'High Calorie: One of the more calorie-dense options in its group',
        '💪': 'High Protein: A true protein powerhouse',
        '🍚': 'High Carb: A carbohydrate champion',
        '🥑': 'High Fat: A healthy fat hero'
    },
    'metric_tooltips': {
        'BMR': 'Basal Metabolic Rate: Calories your body burns at rest to maintain basic functions',
        'TDEE': 'Total Daily Energy Expenditure: BMR plus calories burned through activity',
        'Caloric Adjustment': 'Daily calorie difference from maintenance to reach your goal',
        'Weekly Change': 'Estimated weight change per week based on caloric deficit/surplus'
    },
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
        'height_cm': {'type': 'number', 'label': 'Height',
                      'min': 140, 'max': 220, 'step': 1,
                      'placeholder': 'Enter your height', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight',
                      'min': 40.0, 'max': 150.0, 'step': 0.5,
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
    }
}

# ---------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables."""
    session_vars = (
        ['food_selections', 'form_submitted', 'form_errors', 'use_metric', 'search_term'] +
        [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    )

    for var in session_vars:
        if var not in st.session_state:
            if var == 'food_selections':
                st.session_state[var] = {}
            elif var == 'form_submitted':
                st.session_state[var] = False
            elif var == 'form_errors':
                st.session_state[var] = []
            elif var == 'use_metric':
                st.session_state[var] = True
            elif var == 'search_term':
                st.session_state[var] = ""
            else:
                st.session_state[var] = None

def convert_units(value, from_unit, to_unit):
    """Converts between metric and imperial units."""
    conversion_key = f"{from_unit}_to_{to_unit}"
    if conversion_key in CONVERSION_FACTORS:
        return value * CONVERSION_FACTORS[conversion_key]
    return value

def get_unit_labels(use_metric=True):
    """Returns appropriate unit labels based on user preference."""
    if use_metric:
        return {'weight': 'kg', 'height': 'cm'}
    else:
        return {'weight': 'lbs', 'height': 'inches'}

def validate_user_inputs(inputs, use_metric=True):
    """Validates user inputs and returns error messages if any."""
    errors = []
    
    # Convert limits based on unit system
    if use_metric:
        weight_min, weight_max = 40.0, 150.0
        height_min, height_max = 140, 220
    else:
        weight_min, weight_max = 88.0, 330.0  # ~40-150 kg in lbs
        height_min, height_max = 55, 87  # ~140-220 cm in inches
    
    if not inputs.get('age'):
        errors.append("Please enter your age")
    elif inputs['age'] < 16 or inputs['age'] > 80:
        errors.append("Age must be between 16 and 80 years")
    
    if not inputs.get('weight_kg'):
        errors.append("Please enter your weight")
    elif inputs['weight_kg'] < weight_min or inputs['weight_kg'] > weight_max:
        unit = get_unit_labels(use_metric)['weight']
        errors.append(f"Weight must be between {weight_min} and {weight_max} {unit}")
        
    if not inputs.get('height_cm'):
        errors.append("Please enter your height")
    elif inputs['height_cm'] < height_min or inputs['height_cm'] > height_max:
        unit = get_unit_labels(use_metric)['height']
        errors.append(f"Height must be between {height_min} and {height_max} {unit}")
    
    if not inputs.get('sex'):
        errors.append("Please select your biological sex")
    
    if not inputs.get('activity_level'):
        errors.append("Please select your activity level")
        
    if not inputs.get('goal'):
        errors.append("Please select your goal")
    
    return errors

def create_unified_input(field_name, field_config, container=st.sidebar, use_metric=True):
    """Creates an input widget based on a unified configuration."""
    session_key = f'user_{field_name}'
    
    # Adjust labels and values for unit system
    if field_name in ['weight_kg', 'height_cm']:
        units = get_unit_labels(use_metric)
        if field_name == 'weight_kg':
            field_config = field_config.copy()
            field_config['label'] = f"Weight ({units['weight']})"
            if not use_metric:
                field_config['min'] = 88.0  # ~40 kg
                field_config['max'] = 330.0  # ~150 kg
                field_config['step'] = 1.0
        elif field_name == 'height_cm':
            field_config = field_config.copy()
            field_config['label'] = f"Height ({units['height']})"
            if not use_metric:
                field_config['min'] = 55  # ~140 cm
                field_config['max'] = 87  # ~220 cm
                field_config['step'] = 1

    if field_config['type'] == 'number':
        if field_config.get('advanced'):
            default_val = DEFAULTS.get(field_name, 0)
            display_val = (
                int(default_val * 100)
                if field_name == 'fat_percentage'
                else default_val
            )
            placeholder = f"Default: {display_val}"
        else:
            placeholder = field_config.get('placeholder')

        # Add max_value constraint for servings
        max_value = field_config.get('max', 10.0)
        if field_name in ['weight_kg', 'height_cm']:
            max_value = field_config['max']

        value = container.number_input(
            field_config['label'],
            min_value=field_config['min'],
            max_value=max_value,
            value=st.session_state[session_key],
            step=field_config['step'],
            placeholder=placeholder,
            help=field_config.get('help'),
            key=f"input_{field_name}"
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
            # Find the index of the current value, default to 0 if not found
            index = next(
                (i for i, (_, val) in enumerate(options) if val == current_value),
                next((i for i, (_, val) in enumerate(options) if val == DEFAULTS[field_name]), 0)
            )
            selection = container.selectbox(
                field_config['label'],
                options,
                index=index,
                format_func=lambda x: x[0],
                key=f"select_{field_name}"
            )
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(
                field_config['label'],
                options,
                index=index,
                key=f"select_{field_name}_basic"
            )

    st.session_state[session_key] = value
    return value

def get_final_values(user_inputs, use_metric=True):
    """Processes all user inputs and applies default values where needed."""
    final_values = {}

    for field, value in user_inputs.items():
        final_values[field] = value if value is not None else DEFAULTS[field]

    # Convert imperial to metric if needed
    if not use_metric:
        if final_values.get('weight_kg'):
            final_values['weight_kg'] = convert_units(final_values['weight_kg'], 'lbs', 'kg')
        if final_values.get('height_cm'):
            final_values['height_cm'] = convert_units(final_values['height_cm'], 'inches', 'cm')

    # Apply goal-specific defaults for advanced settings if they are not set
    goal = final_values['goal']
    if goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[goal]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']

    return final_values

def save_progress_to_json(food_selections, user_inputs, targets):
    """Creates a JSON export of current progress."""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'user_inputs': user_inputs,
        'targets': targets,
        'food_selections': food_selections,
        'app_version': 'v2.0'
    }
    return json.dumps(export_data, indent=2)

def load_progress_from_json(json_data):
    """Loads progress from JSON data."""
    try:
        data = json.loads(json_data)
        return data.get('food_selections', {}), data.get('user_inputs', {}), data.get('targets', {})
    except json.JSONDecodeError:
        return {}, {}, {}

def create_csv_export(totals, targets, selected_foods):
    """Creates a CSV export of daily summary."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers and data
    writer.writerow(['Date', datetime.now().strftime('%Y-%m-%d')])
    writer.writerow([])
    writer.writerow(['Targets vs Actual'])
    writer.writerow(['Nutrient', 'Target', 'Actual', 'Percentage'])
    
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']]
        actual = totals[nutrient]
        percentage = (actual / target * 100) if target > 0 else 0
        writer.writerow([config['label'], f"{target:.1f} {config['unit']}", 
                        f"{actual:.1f} {config['unit']}", f"{percentage:.1f}%"])
    
    writer.writerow([])
    writer.writerow(['Foods Consumed'])
    writer.writerow(['Food', 'Servings', 'Calories', 'Protein', 'Carbs', 'Fat'])
    
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        writer.writerow([
            food['name'], servings,
            f"{food['calories'] * servings:.1f}",
            f"{food['protein'] * servings:.1f}",
            f"{food['carbs'] * servings:.1f}",
            f"{food['fat'] * servings:.1f}"
        ])
    
    return output.getvalue()

def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculates daily fluid needs based on body weight and activity."""
    base_needs = weight_kg * 35  # Baseline is 35 milliliters per kilogram

    activity_bonus = {
        'sedentary': 0,
        'lightly_active': 300,
        'moderately_active': 500,
        'very_active': 700,
        'extremely_active': 1000
    }

    climate_multiplier = {
        'cold': 0.9,
        'temperate': 1.0,
        'hot': 1.2,
        'very_hot': 1.4
    }

    total_ml = (
        (base_needs + activity_bonus.get(activity_level, 500)) *
        climate_multiplier.get(climate, 1.0)
    )
    return round(total_ml)

def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics in a configurable column layout."""
    columns = st.columns(num_columns)

    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                label, value = metric_info
                st.metric(label, value)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                st.metric(label, value, delta)
            elif len(metric_info) == 4:
                label, value, delta, help_text = metric_info
                st.metric(label, value, delta, help=help_text)

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
        return (
            f"Adding just {suggestion_servings} serving of {best_food['name']} will give you a solid {best_food[nutrient]:.0f} grams of {nutrient}."
        )
    return None

def create_enhanced_progress_tracking(totals, targets, foods):
    """Creates enhanced progress bars with color coding and recommendations."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")

    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation and building',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and overall health'
    }

    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = min(actual / target * 100, 100) if target > 0 else 0

        # Color coding based on progress
        if percent >= 80:
            progress_color = "🟢"  # Green - excellent
            bar_color = "#28a745"
        elif percent >= 50:
            progress_color = "🟡"  # Yellow - good
            bar_color = "#ffc107"
        else:
            progress_color = "🔴"  # Red - needs attention
            bar_color = "#dc3545"

        col1, col2 = st.columns([4, 1])
        with col1:
            st.progress(
                percent / 100,
                text=(
                    f"{config['label']}: {actual:.0f}/{target:.0f} {config['unit']} "
                    f"({percent:.0f}%)"
                )
            )
        with col2:
            st.markdown(f"### {progress_color}")

        if actual < target:
            deficit = target - actual
            purpose = purpose_map.get(nutrient, 'for optimal nutrition')
            base_rec = (
                f"You've got {deficit:.0f} more {config['unit']} of "
                f"{config['label'].lower()} to go {purpose}."
            )

            if nutrient in ['protein', 'carbs', 'fat']:
                food_suggestion = find_best_food_for_nutrient(
                    nutrient, deficit, foods
                )
                if food_suggestion:
                    base_rec += f" Looking for a suggestion? {food_suggestion}"

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

def generate_personalized_recommendations(totals, targets, final_values):
    """Generates personalized tips based on current intake and goals."""
    recommendations = []
    goal = final_values['goal']
    hydration_ml = calculate_hydration_needs(
        final_values['weight_kg'], final_values['activity_level']
    )
    recommendations.append(
        f"💧 Your Estimated Daily Hydration Goal: {hydration_ml} ml. That's roughly {hydration_ml/250:.1f} cups of water throughout the day."
    )

    if goal == 'weight_loss':
        recommendations.extend([
            "The Shocking Truth: Getting less than 7 hours of sleep can torpedo your fat loss by a more than half.",
            "Daily Goal: Shoot for 7-9 hours and try to keep a consistent schedule.",
            "Set the Scene: Keep your cave dark, cool (18-20°C), and screen-free for at least an hour before lights out.",
            "Morning Ritual: Weigh yourself first thing after using the bathroom, before eating or drinking, in minimal clothing",
            "Look for Trends, Not Blips: Watch your weekly average instead of getting hung up on daily fluctuations. Your weight can swing 2-3 pounds daily.",
            "Hold the Line: Don't tweak your plan too soon. Wait for two or more weeks of stalled progress before making changes.",
            "Leaf Your Hunger Behind: Load your plate with low-calorie, high-volume foods like leafy greens, cucumbers, and berries. They're light on calories but big on satisfaction."
        ])
    elif goal == 'weight_gain':
        recommendations.extend([
            "Drink Your Calories: Liquid calories from smoothies, milk, and protein shakes go down way easier than another full meal",
            "Fat is Fuel: Load up healthy fats like nuts, oils, and avocados.",
            "Push Your Limits: Give your body a reason to grow! Make sure you're consistently challenging yourself in the gym.",
            "Turn Up the Heat: If you've been stuck for over two weeks, bump up your intake by 100-150 calories to get the ball rolling again."
        ])
    else:  # This handles the 'weight_maintenance' goal
        recommendations.extend([
            "⚖️ **Flexible Tracking:** Monitor your intake five days per week "
            "instead of seven for a more sustainable and flexible approach "
            "to maintenance.",
            "📅 **Regular Check-ins:** Weigh yourself weekly and take body "
            "measurements monthly to catch any significant changes early.",
            "🎯 **The 80/20 Balance:** Aim for 80 percent of your diet to "
            "consist of nutrient-dense foods, with 20 percent flexibility "
            "for social situations."
        ])

    protein_per_meal = targets['protein_g'] / 4
    recommendations.append(
        "Spread the Love: Instead of cramming your protein into one or two giant meals, aim for 20-40 grams with each of your 3-4 daily meals. This works out to roughly 0.4-0.5 grams per kilogram of body weight per meal."
    )
    recommendations.append(
        "Frame Your Fitness: Get some carbs and 20–40g protein before and within two hours of wrapping up your workout."
    )
    recommendations.append(
        "The Night Shift: Try 20-30g of casein protein before bed for keeping your muscles fed while you snooze"
    )

    return recommendations

def create_sidebar_summary(totals, targets, use_metric=True):
    """Creates a dynamic summary in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Today's Progress")
    
    # Quick progress indicators
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = min(actual / target * 100, 100) if target > 0 else 0
        
        if percent >= 80:
            icon = "✅"
        elif percent >= 50:
            icon = "⚠️"
        else:
            icon = "❌"
        
        st.sidebar.metric(
            f"{icon} {config['label']}", 
            f"{actual:.0f}/{target:.0f} {config['unit']}",
            f"{percent:.0f}%"
        )

# ---------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# ---------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculates the Basal Metabolic Rate using the Mifflin-St Jeor equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """Calculates Total Daily Energy Expenditure based on activity level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Calculates the estimated weekly weight change from a caloric adjustment."""
    # This is based on the approximation that one kilogram of body fat
    # contains approximately 7,700 kilocalories.
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

    protein_per_kg_final = (
        protein_per_kg if protein_per_kg is not None
        else goal_config['protein_per_kg']
    )
    fat_percentage_final = (
        fat_percentage if fat_percentage is not None
        else goal_config['fat_percentage']
    )

    protein_g = protein_per_kg_final * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage_final
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4
    estimated_weekly_change = calculate_estimated_weekly_change(
        caloric_adjustment
    )

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

    # Handle edge case: prevent division by zero
    if targets['total_calories'] > 0:
        targets['protein_percent'] = (
            (targets['protein_calories'] / targets['total_calories']) * 100
        )
        targets['carb_percent'] = (
            (targets['carb_calories'] / targets['total_calories']) * 100
        )
        targets['fat_percent'] = (
            (targets['fat_calories'] / targets['total_calories']) * 100
        )
    else:
        targets['protein_percent'] = 0
        targets['carb_percent'] = 0
        targets['fat_percent'] = 0

    return targets

# ---------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# ---------------------------------------------------------------------------

@st.cache_data
def load_and_process_food_database(file_path):
    """Loads and processes the vegetarian food database with cached emoji assignment."""
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
    
    # Assign emojis as part of cached function
    foods = assign_food_emojis(foods)
    return foods

def assign_food_emojis(foods):
    """Assigns emojis to foods based on a unified ranking system."""
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'calories': {}}

    for category, items in foods.items():
        if not items:
            continue

        sorted_by_calories = sorted(
            items, key=lambda x: x['calories'], reverse=True
        )
        top_foods['calories'][category] = [
            food['name'] for food in sorted_by_calories[:3]
        ]

        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(
                items, key=lambda x: x[map_info['sort_by']], reverse=True
            )
            top_foods[map_info['key']] = [
                food['name'] for food in sorted_by_nutrient[:3]
            ]

    all_top_nutrient_foods = {
        food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]
    }

    emoji_mapping = {
        'high_cal_nutrient': '🥇', 'high_calorie': '🔥',
        'protein': '💪', 'carbs': '🍚', 'fat': '🥑'
    }

    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])

            if is_high_calorie and is_top_nutrient:
                food['emoji'] = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie:
                food['emoji'] = emoji_mapping['high_calorie']
            elif food_name in top_foods['protein']:
                food['emoji'] = emoji_mapping['protein']
            elif food_name in top_foods['carbs']:
                food['emoji'] = emoji_mapping['carbs']
            elif food_name in top_foods['fat']:
                food['emoji'] = emoji_mapping['fat']
            else:
                food['emoji'] = ''
    return foods

def filter_foods_by_search(foods, search_term):
    """Filters foods based on search term."""
    if not search_term:
        return foods
    
    filtered_foods = {}
    search_lower = search_term.lower()
    
    for category, items in foods.items():
        filtered_items = [
            food for food in items 
            if search_lower in food['name'].lower()
        ]
        if filtered_items:
            filtered_foods[category] = filtered_items
    
    return filtered_foods

def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        # Add emoji tooltip
        emoji = food.get('emoji', '')
        emoji_tooltip = CONFIG['emoji_tooltips'].get(emoji, '')
        
        if emoji and emoji_tooltip:
            st.subheader(f"{emoji} {food['name']}")
            st.caption(emoji_tooltip)
        else:
            st.subheader(f"{emoji} {food['name']}")
        
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)

        col1, col2 = st.columns([2, 1.2])

        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(
                        f"{k}",
                        key=f"{key}_{k}",
                        type=button_type,
                        help=f"Set to {k} servings",
                        use_container_width=True
                    ):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()

        with col2:
            custom_serving = st.number_input(
                "Custom",
                min_value=0.0, 
                max_value=10.0,  # Cap max servings to prevent absurd values
                value=float(current_serving), 
                step=0.1,
                key=f"{key}_custom",
                label_visibility="collapsed"
            )

        if custom_serving != current_serving:
            if custom_serving > 0:
                st.session_state.food_selections[food['name']] = custom_serving
            elif food['name'] in st.session_state.food_selections:
                del st.session_state.food_selections[food['name']]
            st.rerun()

        caption_text = (
            f"Per Serving: {food['calories']} kcal | {food['protein']}g protein | "
            f"{food['carbs']}g carbs | {food['fat']}g fat"
        )
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

# ------ Initialize Session State ------
initialize_session_state()

# ------ Load Food Database with Cached Emoji Assignment ------
foods = load_and_process_food_database('nutrition_results.csv')

# ------ Apply Enhanced CSS for Better Accessibility and Design ------
st.markdown("""
<style>
/* Hide input instructions */
[data-testid="InputInstructions"] { display: none; }

/* Enhanced button styling */
.stButton>button[kind="primary"] { 
    background-color: #ff6b6b; 
    color: white; 
    border: 1px solid #ff6b6b;
    font-weight: 600;
}
.stButton>button[kind="secondary"] { 
    border: 1px solid #ff6b6b; 
    color: #ff6b6b;
    font-weight: 500;
}

/* Improved sidebar styling */
.sidebar .sidebar-content { 
    background-color: #f8f9fa; 
    padding: 1rem;
}

/* Better contrast for captions */
.caption {
    color: #495057 !important;
    font-size: 0.875rem;
}

/* Enhanced metric containers */
[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e9ecef;
    border-radius: 0.5rem;
    padding: 1rem;
    box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
}

/* Improved progress bar visibility */
.stProgress > div > div > div {
    background-color: #28a745;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .stColumns > div {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    
    .stButton > button {
        width: 100% !important;
        margin-bottom: 0.5rem;
    }
    
    [data-testid="metric-container"] {
        margin-bottom: 1rem;
    }
}

/* Better spacing */
.element-container {
    margin-bottom: 1rem;
}

/* Enhanced container borders */
[data-testid="stContainer"] {
    border-radius: 0.5rem;
    border-color: #dee2e6;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Application Title and Enhanced Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
A Smart, Evidence-Based Nutrition Tracker That Actually Gets You

Welcome aboard!

Hey there! Welcome to your new nutrition buddy. This isn't just another calorie counter—it's your personalized guide, built on rock-solid science to help you smash your goals. Whether you're aiming to shed a few pounds, hold steady, or bulk up, we've crunched the numbers so you can focus on enjoying your food.

Let's get rolling—your journey to feeling awesome starts now! 🚀
""")

# ------ Sidebar for User Input with Unit Toggle ------
st.sidebar.header("Let's Get Personal 📊")

# Unit system toggle
use_metric = st.sidebar.toggle(
    "Use Metric Units (kg/cm)", 
    value=st.session_state.use_metric,
    help="Toggle between metric (kg/cm) and imperial (lbs/inches) units",
    key="unit_toggle"
)
st.session_state.use_metric = use_metric

# Form submission tracking
if not st.session_state.form_submitted:
    st.sidebar.info("👇 Fill out your details below and click 'Calculate My Targets' to get started!")

all_inputs = {}
standard_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')
}
advanced_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')
}

# Standard input fields
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar, use_metric=use_metric)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Advanced settings expander
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(
        field_name, field_config, container=advanced_expander, use_metric=use_metric
    )
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Calculate button with validation
if st.sidebar.button("🎯 Calculate My Targets", type="primary", use_container_width=True, key="calculate_btn"):
    validation_errors = validate_user_inputs(all_inputs, use_metric)
    if validation_errors:
        st.session_state.form_errors = validation_errors
        st.session_state.form_submitted = False
    else:
        st.session_state.form_errors = []
        st.session_state.form_submitted = True
        st.rerun()

# Display validation errors
if st.session_state.form_errors:
    st.sidebar.error("Please fix the following issues:")
    for error in st.session_state.form_errors:
        st.sidebar.error(f"• {error}")

# Save/Load Progress Section
st.sidebar.markdown("---")
st.sidebar.subheader("💾 Save & Load Progress")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Save", help="Save your current progress", key="save_btn", use_container_width=True):
        if st.session_state.form_submitted:
            final_values = get_final_values(all_inputs, use_metric)
            targets = calculate_personalized_targets(**final_values)
            json_data = save_progress_to_json(st.session_state.food_selections, all_inputs, targets)
            st.download_button(
                "📥 Download Progress",
                data=json_data,
                file_name=f"nutrition_progress_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                key="download_progress"
            )
        else:
            st.sidebar.warning("Please calculate your targets first!")

with col2:
    uploaded_file = st.file_uploader(
        "📤 Load", 
        type=['json'], 
        help="Load previously saved progress",
        key="load_progress",
        label_visibility="collapsed"
    )
    if uploaded_file:
        json_data = uploaded_file.read().decode()
        food_selections, user_inputs, saved_targets = load_progress_from_json(json_data)
        if food_selections:
            st.session_state.food_selections = food_selections
            st.sidebar.success("Progress loaded successfully!")
            st.rerun()

# ------ Activity Level Guide in Sidebar ------
with st.sidebar.container(border=True):
    st.markdown("##### Your Activity Level Decoded")
    st.markdown("---")
    st.markdown("""
* **🧑‍💻 Sedentary**: You're basically married to your desk chair
* **🏃 Lightly Active**: You squeeze in walks or workouts one to three times a week
* **🚴 Moderately Active**: You're sweating it out three to five days a week
* **🏋️ Very Active**: You might actually be part treadmill
* **🤸 Extremely Active**: You live in the gym and sweat is your second skin

*💡 If you're torn between two levels, pick the lower one. It's better to underestimate your burn than to overeat and stall.*
    """)

# ------ Process Final Values ------
final_values = get_final_values(all_inputs, use_metric)

# ------ Check for Form Submission ------
user_has_calculated_targets = st.session_state.form_submitted and not st.session_state.form_errors

# ------ Calculate Personalized Targets ------
if user_has_calculated_targets:
    targets = calculate_personalized_targets(**final_values)
    
    # Show motivational message
    goal_messages = {
        'weight_loss': f"🎉 Awesome! You're set up for sustainable weight loss of about {abs(targets['estimated_weekly_change']):.1f} kg per week!",
        'weight_gain': f"💪 Perfect! You're on track to gain approximately {targets['estimated_weekly_change']:.1f} kg per week with this plan!",
        'weight_maintenance': "⚖️ Excellent! Your plan is designed to maintain your current weight while optimizing your nutrition!"
    }
    st.success(goal_messages.get(targets['goal'], "Great job setting up your nutrition plan!"))
else:
    # Use default values for display
    targets = calculate_personalized_targets(**final_values)

# ---------------------------------------------------------------------------
# Cell 9: Enhanced Target Display System with Tooltips
# ---------------------------------------------------------------------------

if not user_has_calculated_targets:
    st.info(
        "👈 Enter your details in the sidebar and click 'Calculate My Targets' to get your personalized nutrition plan."
    )
    st.header("Sample Daily Targets for Reference")
    st.caption(
        "These are example targets. Please enter your information in the "
        "sidebar for personalized calculations."
    )
else:
    goal_labels = {
        'weight_loss': 'Weight Loss',
        'weight_maintenance': 'Weight Maintenance',
        'weight_gain': 'Weight Gain'
    }
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Custom Daily Nutrition Roadmap for {goal_label} 🎯")

st.info(
    "🎯 **The 80/20 Rule**: Try to hit your targets about 80% of the time. This gives you wiggle room for birthday cake, date nights, and those inevitable moments when life throws you a curveball. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap."
)

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Enhanced Metrics Display with Tooltips ------
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal", None, CONFIG['metric_tooltips']['BMR']),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal", None, CONFIG['metric_tooltips']['TDEE']),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal", None, CONFIG['metric_tooltips']['Caloric Adjustment']),
            ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kg", None, CONFIG['metric_tooltips']['Weekly Change']),
            ("", "")
        ]
    },
    {
        'title': 'Your Daily Nutrition Targets', 'columns': 5,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            ("Protein", f"{targets['protein_g']} g",
             f"{targets['protein_percent']:.0f}% of your calories"),
            ("Carbohydrates", f"{targets['carb_g']} g",
             f"{targets['carb_percent']:.0f}% of your calories"),
            ("Fat", f"{targets['fat_g']} g",
             f"{targets['fat_percent']:.0f}% of your calories"),
            ("Water", f"{hydration_ml} ml",
             f"~{hydration_ml/250:.1f} cups")
        ]
    }
]

# ------ Display All Metric Sections ------
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# ---------------------------------------------------------------------------
# Cell 10: Collapsible Evidence-Based Tips and Context
# ---------------------------------------------------------------------------

st.header("Your Evidence-Based Game Plan 📚")

with st.expander("🏆 The Big Three to Win At Nutrition", expanded=False):
    st.subheader("💧 Master Your Hydration Game")
    st.markdown("""
    Daily Goal: Shot for about 35 ml per kilogram of your body weight daily. 
    Training Bonus: Tack on an extra 500-750 ml per hour of sweat time
    Fat Loss Hack: Chugging 500 ml of water before meals can boost fullness by by 13%. Your stomach will thank you, and so will your waistline.
    """)

    st.subheader("😴 Sleep Like Your Goals Depend on It")
    st.markdown("""
    The Shocking Truth: Getting less than 7 hours of sleep can torpedo your fat loss by a more than half.
    Daily Goal: Shoot for 7-9 hours and try to keep a consistent schedule.
    Set the Scene: Keep your cave dark, cool (18-20°C), and screen-free for at least an hour before lights out.
    """)

    st.subheader("📅 Follow Your Wins")
    st.markdown("""
    Morning Ritual: Weigh yourself first thing after using the bathroom, before eating or drinking, in minimal clothing
    Look for Trends, Not Blips: Watch your weekly average instead of getting hung up on daily fluctuations. Your weight can swing 2-3 pounds daily. 
    Hold the Line: Don't tweak your plan too soon. Wait for two or more weeks of stalled progress before making changes.
    """)

with st.expander("📊 Level Up Your Progress Tracking", expanded=False):
    st.subheader("Go Beyond the Scale 📸")
    st.markdown("""
    The Bigger Picture: Snap a few pics every month. Use the same pose, lighting, and time of day. The mirror doesn't lie.
    Size Up Your Wins: Measure your waist, hips, arms, and thighs monthly
    The Quiet Victories: Pay attention to how you feel. Your energy levels, sleep quality, gym performance, and hunger patterns tell a story numbers can't.
    """)

with st.expander("🧠 Mindset Is Everything", expanded=False):
    st.markdown("""
    The 80/20 principle is your best defense against the perfectionist trap. It's about ditching that mindset that makes you throw in the towel after one "bad" meal. Instead of trying to master everything at once, build your habits gradually and you'll be far more likely to stick with them for the long haul.

    Start Small, Win Big:

    Weeks 1–2: Your only job is to focus on hitting your calorie targets. Don't worry about anything else!
    Weeks 3–4: Once calories feel like second nature, start layering in protein tracking
    Week 5 and Beyond: With calories and protein in the bag, you can now fine-tune your carb and fat intake

    When Progress Stalls 🔄

    Hit a Weight Loss Plateau?

    Guess Less, Stress Less: Before you do anything else, double-check how accurately you're logging your food. Little things can add up!
    Activity Audit: Take a fresh look at your activity level. Has it shifted?
    Walk it Off: Try adding 10-15 minutes of walking to your daily routine before cutting calories further. It's a simple way to boost progress without tightening the belt just yet.
    Step Back to Leap Forwarde: Consider a "diet break" every 6-8 weeks. Eating at your maintenance calories for a week or two can give your metabolism and your mind a well-deserved reset.
    Leaf Your Hunger Behind: Load your plate with low-calorie, high-volume foods like leafy greens, cucumbers, and berries. They're light on calories but big on satisfaction.

    Struggling to Gain Weight?

    Drink Your Calories: Liquid calories from smoothies, milk, and protein shakes go down way easier than another full meal 
    Fat is Fuel: Load up healthy fats like nuts, oils, and avocados. 
    Push Your Limits: Give your body a reason to grow! Make sure you're consistently challenging yourself in the gym.
    Turn Up the Heat: If you've been stuck for over two weeks, bump up your intake by 100-150 calories to get the ball rolling again.

    Pace Your Protein

    Spread the Love: Instead of cramming your protein into one or two giant meals, aim for 20-40 grams with each of your 3-4 daily meals. This works out to roughly 0.4-0.5 grams per kilogram of body weight per meal.
    Frame Your Fitness: Get some carbs and 20–40g protein before and within two hours of wrapping up your workout.
    The Night Shift: Try 20-30g of casein protein before bed for keeping your muscles fed while you snooze
    """)

with st.expander("🔬 The Science Behind the Magic", expanded=False):
    st.subheader("Understanding Your Metabolism")
    st.markdown("""
    Your Basal Metabolic Rate (BMR) is the energy your body needs just to keep the lights on. Your Your Total Daily Energy Expenditure (TDEE) builds on that baseline by factoring in how active you are throughout the day.

    The Smart Eater's Cheat Sheet

    Not all calories are created equal. Some foods fill you up, while others leave you rummaging through the pantry an hour later. Here's the pecking order:

    Protein: Protein is the undisputed king of fullness! It digests slowly, steadies blood sugar, and even burns a few extra calories in the process. Eggs, Greek yogurt, chicken, tofu, and lentils are all your hunger-busting best friends.

    Fiber-Rich Carbohydrates: Veggies, fruits, and whole grains are the unsung heroes of fullness. They fill you up, slow things down, and bulk up meals without blowing your calorie budget.

    Healthy Fats: Think of nuts, olive oil, and avocados as the smooth operators delivering steady, long-lasting energy that keeps you powered throughout the day.

    Processed Stuff: These foods promise the world but leave you hanging. They're fine for a cameo appearance, but you can't build a winning strategy around them.

    As a great rule of thumb, aim for 14 grams of fibre for every 1,000 calories you consume, which usually lands between 25 and 38 grams daily. Ramp up gradually to avoid digestive drama.

    Your Nutritional Supporting Cast

    Going plant-based? There are a few tiny but mighty micronutrients to keep an eye on. They may not get top billing, but they're essential for keeping the show running smoothly.

    The Watch List:

    B₁₂: B₁₂ keeps your cells and nerves firing like a well-oiled machine. It's almost exclusively found in animal products, so if you're running a plant-powered show, you'll need reinforcements. A trusty supplement is often the easiest way to keep your levels topped up and your brain buzzing.
    Iron: Iron is the taxi service that shuttles oxygen all over your body. When it's running low, you'll feel like a sloth on a Monday morning. Load up on leafy greens, lentils, and fortified grains, and team them with a hit of vitamin C—think bell peppers or citrus—to supercharge absorption.
    Calcium: This multitasker helps build bones, power muscles, and keeps your heart thumping to a steady beat. While dairy is the classic go-to, you can also get your fix from kale, almonds, tofu, and fortified plant milks.
    Zinc: Think of zinc as your immune system's personal security detail. You'll find it hanging out in nuts, seeds, and whole grains. Keep your zinc levels up, and you'll be dodging colds like a ninja.
    Iodine: Your thyroid is the command center for your metabolism, and iodine is its right-hand mineral. A pinch of iodized salt is usually all it takes.
    Omega-3s (EPA/DHA): These healthy fats are premium fuel for your brain, heart, and emotional well-being. If fish isn't on your plate, fortified foods or supplements can help you stay sharp and serene.

    The good news? Fortified foods and targeted supplements have your back. Plant milks, cereals, and nutritional yeast are often spiked with B₁₂, calcium, or iodine. Supplements are a safety net, but don't overdo it. It's always best to chat with a doctor or dietitian to build a plan that's right for you.
    """)

# ---------------------------------------------------------------------------
# Cell 11: Enhanced Personalized Recommendations System
# ---------------------------------------------------------------------------

if user_has_calculated_targets:
    st.header("Your Personalized Action Steps 🎯")
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
    recommendations = generate_personalized_recommendations(
        totals, targets, final_values
    )
    for rec in recommendations:
        st.info(rec)

# ---------------------------------------------------------------------------
# Cell 12: Enhanced Food Selection Interface with Search
# ---------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")
st.markdown(
    "Pick how many servings of each food you're having to see how your choices stack up against your daily targets."
)

# Food search functionality
st.subheader("🔍 Quick Food Search")
search_term = st.text_input(
    "Search for foods...", 
    placeholder="e.g., 'oats', 'banana', 'tofu'",
    value=st.session_state.search_term,
    key="food_search"
)
st.session_state.search_term = search_term

# Show search results if there's a search term
if search_term:
    filtered_foods = filter_foods_by_search(foods, search_term)
    if filtered_foods:
        st.write(f"Found foods matching '{search_term}':")
        for category, items in filtered_foods.items():
            st.subheader(f"{category}")
            render_food_grid(items[:4], category, columns=2)  # Show top 4 results per category
    else:
        st.info("No foods found. Try a different search term.")
    st.markdown("---")

with st.expander("💡 Need a hand with food choices? Check out the emoji guide below!"):
    st.markdown("""
    * **🥇 Gold Medal**: A nutritional all-star! High in its target nutrient and very calorie-efficient.
    * **🔥 High Calorie**: One of the more calorie-dense options in its group.
    * **💪 High Protein**: A true protein powerhouse.
    * **🍚 High Carb**: A carbohydrate champion.
    * **🥑 High Fat**: A healthy fat hero.
    """)

# Quick meal templates
st.subheader("⚡ Quick Meal Templates")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🥣 High Protein Breakfast", use_container_width=True, key="template_breakfast"):
        # Pre-populate with high protein breakfast foods (adjust based on your database)
        template_foods = {
            "Greek Yogurt, Plain (1 cup)": 1.0,
            "Oats, Rolled (1/2 cup dry)": 1.0,
            "Almonds (1 oz)": 1.0
        }
        for food_name, servings in template_foods.items():
            if any(food_name in [item['name'] for item in category_items] 
                   for category_items in foods.values()):
                st.session_state.food_selections[food_name] = servings
        st.success("High protein breakfast template loaded!")
        st.rerun()

with col2:
    if st.button("🥙 Balanced Lunch", use_container_width=True, key="template_lunch"):
        template_foods = {
            "Quinoa, Cooked (1 cup)": 1.0,
            "Black Beans, Cooked (1/2 cup)": 1.0,
            "Avocado (1/2 medium)": 1.0,
            "Mixed Greens (2 cups)": 1.0
        }
        for food_name, servings in template_foods.items():
            if any(food_name in [item['name'] for item in category_items] 
                   for category_items in foods.values()):
                st.session_state.food_selections[food_name] = servings
        st.success("Balanced lunch template loaded!")
        st.rerun()

with col3:
    if st.button("🍽️ Power Dinner", use_container_width=True, key="template_dinner"):
        template_foods = {
            "Tofu, Firm (4 oz)": 1.0,
            "Brown Rice, Cooked (1 cup)": 1.0,
            "Broccoli, Steamed (1 cup)": 1.0,
            "Olive Oil (1 tbsp)": 1.0
        }
        for food_name, servings in template_foods.items():
            if any(food_name in [item['name'] for item in category_items] 
                   for category_items in foods.values()):
                st.session_state.food_selections[food_name] = servings
        st.success("Power dinner template loaded!")
        st.rerun()

# Clear selections button
if st.button("🗑️ Clear All Selections", type="secondary", key="clear_all"):
    st.session_state.food_selections = {}
    st.success("All food selections cleared!")
    st.rerun()

# Food category tabs
tab_names = list(foods.keys())
tabs = st.tabs(tab_names)

for tab, (category, items) in zip(tabs, foods.items()):
    with tab:
        if items:
            # Sort items by emoji priority, then alphabetically
            sorted_items = sorted(
                items,
                key=lambda x: (
                    CONFIG['emoji_order'].get(x.get('emoji', ''), 4),
                    x['name']
                )
            )
            render_food_grid(sorted_items, category, columns=2)
        else:
            st.info(f"No foods available in {category}")

# ---------------------------------------------------------------------------
# Cell 13: Enhanced Progress Tracking and Recommendations
# ---------------------------------------------------------------------------

st.header("Your Daily Progress Dashboard 📊")

# Calculate current totals
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

# Create sidebar summary
create_sidebar_summary(totals, targets, st.session_state.use_metric)

# Enhanced progress tracking with recommendations
recommendations = create_enhanced_progress_tracking(totals, targets, foods)

# Show personalized recommendations if any deficits exist
if recommendations:
    st.subheader("🎯 Smart Recommendations")
    for rec in recommendations:
        st.info(rec)

# ---------------------------------------------------------------------------
# Cell 14: Enhanced Visualization Dashboard
# ---------------------------------------------------------------------------

st.subheader("Visual Progress Overview 📈")

# Create comparison charts
col1, col2 = st.columns(2)

with col1:
    # Macronutrient comparison chart
    fig_macros = go.Figure()
    
    macros = ['Protein', 'Carbohydrates', 'Fat']
    actual_values = [totals['protein'], totals['carbs'], totals['fat']]
    target_values = [targets['protein_g'], targets['carb_g'], targets['fat_g']]
    
    fig_macros.add_trace(go.Bar(
        name='Actual',
        x=macros,
        y=actual_values,
        marker_color='#ff6b6b'
    ))
    
    fig_macros.add_trace(go.Bar(
        name='Target',
        x=macros,
        y=target_values,
        marker_color='#4ecdc4'
    ))
    
    fig_macros.update_layout(
        title='Macronutrients: Actual vs Target (g)',
        barmode='group',
        yaxis_title='Grams',
        height=400
    )
    
    st.plotly_chart(fig_macros, use_container_width=True)

with col2:
    # Calorie breakdown pie chart
    if totals['calories'] > 0:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Protein', 'Carbohydrates', 'Fat'],
            values=[
                totals['protein'] * 4,  # Protein calories
                totals['carbs'] * 4,    # Carb calories
                totals['fat'] * 9       # Fat calories
            ],
            hole=0.4,
            marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1']
        )])
        
        fig_pie.update_layout(
            title=f'Calorie Distribution<br>Total: {totals["calories"]:.0f} kcal',
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Add some foods to see your calorie distribution!")

# ---------------------------------------------------------------------------
# Cell 15: Export and Summary Features
# ---------------------------------------------------------------------------

st.header("Export Your Progress 📋")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Download CSV Summary", use_container_width=True, key="export_csv"):
        if selected_foods:
            csv_data = create_csv_export(totals, targets, selected_foods)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"nutrition_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_csv"
            )
        else:
            st.warning("Please select some foods first!")

with col2:
    if st.button("📱 Share Progress", use_container_width=True, key="share_progress"):
        if selected_foods:
            # Create a shareable summary
            share_text = f"""
🍽️ My Nutrition Progress - {datetime.now().strftime('%Y-%m-%d')}

🎯 Daily Targets:
• Calories: {targets['total_calories']} kcal
• Protein: {targets['protein_g']}g
• Carbs: {targets['carb_g']}g  
• Fat: {targets['fat_g']}g

📊 Today's Intake:
• Calories: {totals['calories']:.0f} kcal ({totals['calories']/targets['total_calories']*100:.0f}%)
• Protein: {totals['protein']:.0f}g ({totals['protein']/targets['protein_g']*100:.0f}%)
• Carbs: {totals['carbs']:.0f}g ({totals['carbs']/targets['carb_g']*100:.0f}%)
• Fat: {totals['fat']:.0f}g ({totals['fat']/targets['fat_g']*100:.0f}%)

Created with Personal Nutrition Coach 🍽️
            """
            st.text_area("Copy this summary to share:", share_text, height=200)
        else:
            st.warning("Please select some foods first!")

with col3:
    if st.button("🎯 Reset Day", use_container_width=True, key="reset_day"):
        st.session_state.food_selections = {}
        st.success("Day reset! Start fresh with your food selections.")
        st.rerun()

# ---------------------------------------------------------------------------
# Cell 16: Food Selection Summary
# ---------------------------------------------------------------------------

if selected_foods:
    st.subheader("Today's Food Log 📝")
    
    # Create a summary table
    summary_data = []
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        summary_data.append({
            'Food': food['name'],
            'Servings': f"{servings:.1f}",
            'Calories': f"{food['calories'] * servings:.0f}",
            'Protein (g)': f"{food['protein'] * servings:.1f}",
            'Carbs (g)': f"{food['carbs'] * servings:.1f}",
            'Fat (g)': f"{food['fat'] * servings:.1f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)
    
    # Add totals row
    st.markdown("**Daily Totals:**")
    totals_col1, totals_col2, totals_col3, totals_col4 = st.columns(4)
    with totals_col1:
        st.metric("Total Calories", f"{totals['calories']:.0f} kcal")
    with totals_col2:
        st.metric("Total Protein", f"{totals['protein']:.1f} g")
    with totals_col3:
        st.metric("Total Carbs", f"{totals['carbs']:.1f} g")
    with totals_col4:
        st.metric("Total Fat", f"{totals['fat']:.1f} g")

# ---------------------------------------------------------------------------
# Cell 17: User Feedback and Support Section
# ---------------------------------------------------------------------------

st.header("Help Us Improve! 💬")

with st.expander("📝 Share Your Feedback", expanded=False):
    st.markdown("""
    Your thoughts help us make this app better for everyone! Whether it's a bug report, 
    feature request, or just general feedback, we'd love to hear from you.
    """)
    
    feedback_type = st.selectbox(
        "What type of feedback do you have?",
        ["General Feedback", "Bug Report", "Feature Request", "Question"],
        key="feedback_type"
    )
    
    feedback_text = st.text_area(
        "Tell us more:",
        placeholder="Share your thoughts, suggestions, or questions here...",
        height=100,
        key="feedback_text"
    )
    
    if st.button("📤 Submit Feedback", key="submit_feedback"):
        if feedback_text.strip():
            # In a real app, this would send to a database or email
            st.success("Thank you for your feedback! We appreciate you taking the time to help us improve.")
            # Here you would typically save the feedback to a database or send an email
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'type': feedback_type,
                'feedback': feedback_text,
                'user_goal': targets.get('goal', 'unknown')
            }
            # st.write("Debug - Feedback data:", feedback_data)  # Remove in production
        else:
            st.warning("Please enter some feedback before submitting.")

# ---------------------------------------------------------------------------
# Cell 18: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.markdown("---")

# Quick tips section
st.subheader("💡 Quick Tips for Success")
tips_col1, tips_col2 = st.columns(2)

with tips_col1:
    st.markdown("""
    **🎯 Consistency Beats Perfection**
    - Aim for 80% adherence rather than 100%
    - One "off" meal doesn't ruin your progress
    - Focus on weekly averages, not daily perfection
    
    **📱 Make It Easier**
    - Pre-log your meals when possible
    - Batch cook proteins and grains
    - Keep healthy snacks visible and accessible
    """)

with tips_col2:
    st.markdown("""
    **⚖️ Listen to Your Body**
    - Hunger and fullness cues matter
    - Energy levels are important indicators
    - Adjust portions based on how you feel
    
    **🔄 Stay Flexible**
    - Swap similar foods when needed
    - Adjust timing to fit your schedule
    - Remember: this is a lifestyle, not a diet
    """)

# App information
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p><strong>Personal Nutrition Coach v2.0</strong></p>
    <p>Built with ❤️ using evidence-based nutrition science</p>
    <p>Remember: This tool provides general guidance. For personalized medical advice, consult with a healthcare professional.</p>
</div>
""", unsafe_allow_html=True)

# Debug information (remove in production)
if st.checkbox("🔧 Show Debug Info", key="debug_mode"):
    st.subheader("Debug Information")
    st.write("Session State Food Selections:", st.session_state.food_selections)
    st.write("Form Submitted:", st.session_state.form_submitted)
    st.write("Form Errors:", st.session_state.form_errors)
    st.write("Use Metric:", st.session_state.use_metric)
    st.write("Final Values:", final_values)
    st.write("Targets:", targets)
    st.write("Totals:", totals)

# ---------------------------------------------------------------------------
# End of Application
# ---------------------------------------------------------------------------

