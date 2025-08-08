#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
#
#       A Personalized, Evidence-Based Nutrition Tracker
#                   for Goal-Specific Meal Planning
#
# ---------------------------------------------------------------------------
"""
"""
This script implements an interactive, evidence-based nutrition tracker using
Streamlit. It is designed to help users achieve personalized nutrition goals,
such as weight loss, maintenance, or gain, with a focus on vegetarian food
sources.

Core Functionality and Scientific Basis
- Basal Metabolic Rate (BMR) Calculation: The application uses the Mifflin-St
  Jeor equation, which is recognized by organizations like the Academy of
  Nutrition and Dietetics for its accuracy.
  - For Males: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
  - For Females: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

- Total Daily Energy Expenditure (TDEE): The BMR is multiplied by a
  scientifically validated activity factor to estimate the total number of
  calories burned in a day, including physical activity.

- Goal-Specific Caloric Adjustments
  - Weight Loss: A 20 percent caloric deficit from TDEE.
  - Weight Maintenance: Caloric intake is set equal to TDEE.
  - Weight Gain: A 10 percent caloric surplus over TDEE.

- Macronutrient Strategy: The script follows a protein-first approach.
  1. Protein intake is determined based on grams per kilogram of body weight.
  2. Fat intake is set as a percentage of total daily calories.
  3. Carbohydrate intake is calculated from the remaining caloric budget.

Implementation Details
- The user interface is built with Streamlit, providing interactive widgets
  for user input and data visualization.
- The food database is managed using the Pandas library.
- Progress visualizations are created with Streamlit's native components and
  Plotly for generating detailed charts.

Usage Documentation
1. Prerequisites: Ensure the required Python libraries are installed. You can
   install them using pip:
   pip install streamlit pandas plotly reportlab

2. Running the Application: Save this script as a Python file, for example,
   `nutrition_app.py`, and run it from your terminal using the following
   command:
   streamlit run nutrition_app.py

3. Interacting with the Application
   - Use the sidebar to enter your personal details, such as age, height,
     weight, sex, activity level, and primary nutrition goal.
   - Your personalized daily targets for calories and macronutrients will be
     calculated and displayed automatically.
   - Navigate through the food tabs to select the number of servings for
     each food item you consume.
   - The daily summary section will update in real time to show your
     progress toward your targets.
"""
# -----------------------------------------------------------------------------
# # Cell 1: Import Required Libraries and Modules
# -----------------------------------------------------------------------------
import math
import json
import io
from datetime import datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


# -----------------------------------------------------------------------------
# # Cell 2: Page Configuration and Initial Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Your Personal Nutrition Coach 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -----------------------------------------------------------------------------
# # Cell 3: Unified Configuration Constants
# -----------------------------------------------------------------------------
# ------ Default Parameter Values ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 57.5,
    'sex': "Male",
    'activity_level': "lightly_active",
    'goal': "weight_gain",
    'protein_per_kg': 2.0,
    'fat_percentage': 0.25,
    'units': 'metric'
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

# ------ Goal-Specific Targets Based on Evidence-Based Guidelines ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # A 20 percent deficit from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,  # Zero percent deficit from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # A 10 percent surplus over TDEE
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
}

# ------ Unified Configuration for All Application Components ------
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
        'age': {'type': 'number', 'label': 'Age (in Years)',
                'min': 16, 'max': 80, 'step': 1,
                'placeholder': 'Enter your age', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (in Centimeters)',
                      'min': 140, 'max': 220, 'step': 1,
                      'placeholder': 'Enter your height', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (in Kilograms)',
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
                           'help': 'Define your daily protein target in grams'
                                   ' per kilogram of body weight',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number',
                           'label': 'Fat Intake (% of Calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Set the share of your daily calories that'
                                   ' should come from healthy fats',
                           'convert': lambda x: x / 100 if x else None,
                           'advanced': True, 'required': False}
    }
}

# ------ Emoji Tooltips ------
EMOJI_TOOLTIPS = {
    '🥇': 'Gold Medal: This food is a nutritional all-star that is high'
         ' in its target nutrient and very calorie efficient.',
    '🔥': 'High Calorie: This food is one of the more calorie-dense'
         ' options in its group.',
    '💪': 'High Protein: This food is a true protein powerhouse.',
    '🍚': 'High Carbohydrate: This food is a carbohydrate champion.',
    '🥑': 'High Fat: This food is a healthy fat hero.'
}

# ------ Metric Tooltips ------
METRIC_TOOLTIPS = {
    'BMR': 'Basal Metabolic Rate is the energy your body needs to'
           ' keep vital functions running.',
    'TDEE': 'Total Daily Energy Expenditure is your BMR plus the'
            ' calories burned through activity.',
    'Caloric Adjustment': 'This is how many calories above or below TDEE'
                          ' are needed to reach your goal.',
    'Protein': 'Protein is essential for muscle building, repair,'
               ' and satiety.',
    'Carbohydrates': 'Carbohydrates are your body\'s preferred energy source'
                     ' for brain and muscle function.',
    'Fat': 'Fat is important for hormone production, nutrient absorption,'
           ' and cell health.'
}

# ------ Centralized Tip and Recommendation Content ------
TIPS_CONTENT = {
    'hydration': [
        "**Daily Goal**: Aim for approximately 35 milliliters of water per"
        " kilogram of your body weight each day.",
        "**Training Bonus**: Add an extra 500 to 750 milliliters of water"
        " per hour of exercise.",
        "**Mealtime Strategy**: Consuming 500 milliliters of water before"
        " meals has been shown to increase satiety by approximately 13"
        " percent."
    ],
    'sleep': [
        "**Scientific Fact**: Obtaining less than seven hours of sleep can"
        " reduce fat loss by more than 50 percent.",
        "**Daily Goal**: Aim for seven to nine hours of sleep per night and"
        " maintain a consistent schedule.",
        "**Optimal Environment**: Keep your bedroom dark, cool between 18 and"
        " 20 degrees Celsius, and free of screens for at least one hour"
        " before sleep."
    ],
    'tracking_wins': [
        "**Morning Routine**: Weigh yourself first thing in the morning after"
        " using the bathroom and before eating or drinking.",
        "**Focus on Trends**: Monitor your weekly average weight instead of"
        " focusing on daily fluctuations, which can vary by one to two"
        " kilograms.",
        "**Maintain Consistency**: Do not adjust your plan prematurely. Wait"
        " for at least two weeks of stalled progress before making changes."
    ],
    'beyond_the_scale': [
        "**Visual Progress**: Take progress photos each month in the same"
        " pose, lighting, and at the same time of day.",
        "**Track Measurements**: Measure your waist, hips, arms, and thighs"
        " on a monthly basis to track changes in body composition.",
        "**Subjective Markers**: Pay attention to your energy levels, sleep"
        " quality, gym performance, and hunger patterns as indicators of"
        " progress."
    ],
    'protein_pacing': [
        "**Distribute Intake**: Aim for 20 to 40 grams of protein with each"
        " of your three to four daily meals, which is about 0.4 to 0.5"
        " grams per kilogram of body weight per meal.",
        "**Frame Your Workouts**: Consume a meal containing both"
        " carbohydrates and 20 to 40 grams of protein before and within two"
        " hours after your workout.",
        "**Nighttime Nutrition**: Consider consuming 20 to 30 grams of casein"
        " protein before bed to support muscle protein synthesis overnight."
    ],
    'weight_loss_plateau': [
        "**Ensure Accuracy**: Before making changes, double check the"
        " accuracy of your food logging, as small discrepancies can"
        " accumulate.",
        "**Activity Audit**: Re-evaluate your activity level to determine if"
        " it has changed.",
        "**Increase Activity**: Add 10 to 15 minutes of walking to your daily"
        " routine before reducing calories further.",
        "**Implement a Diet Break**: Consider taking a diet break every six"
        " to eight weeks by eating at your maintenance calorie level for one"
        " to two weeks.",
        "**Increase Food Volume**: Prioritize low calorie, high volume foods"
        " such as leafy greens, cucumbers, and berries to increase satiety."
    ],
    'weight_gain_stalls': [
        "**Consume Liquid Calories**: Smoothies, milk, and protein shakes are"
        " an effective way to increase calorie intake without excessive"
        " fullness.",
        "**Incorporate Healthy Fats**: Increase your consumption of calorie"
        " dense, healthy fats from sources like nuts, oils, and avocados.",
        "**Progressive Overload**: Ensure your training program provides a"
        " consistent and challenging stimulus for muscle growth.",
        "**Increase Caloric Intake**: If progress has stalled for over two"
        " weeks, increase your daily intake by 100 to 150 calories."
    ]
}


# -----------------------------------------------------------------------------
# # Cell 4: Unit Conversion Functions
# -----------------------------------------------------------------------------
def kg_to_lbs(kg):
    """Converts a weight in kilograms to pounds."""
    return kg * 2.20462 if kg else 0


def lbs_to_kg(lbs):
    """Converts a weight in pounds to kilograms."""
    return lbs / 2.20462 if lbs else 0


def cm_to_inches(cm):
    """Converts a height in centimeters to inches."""
    return cm / 2.54 if cm else 0


def inches_to_cm(inches):
    """Converts a height in inches to centimeters."""
    return inches * 2.54 if inches else 0


def format_weight(weight_kg, units):
    """Formats a weight value based on the selected unit system."""
    if units == 'imperial':
        return f"{kg_to_lbs(weight_kg):.1f} lbs"
    return f"{weight_kg:.1f} kg"


def format_height(height_cm, units):
    """Formats a height value based on the selected unit system."""
    if units == 'imperial':
        total_inches = cm_to_inches(height_cm)
        feet = int(total_inches // 12)
        inches = total_inches % 12
        return f"{feet}'{inches:.0f}\""
    return f"{height_cm:.0f} cm"


# -----------------------------------------------------------------------------
# # Cell 5: Unified Helper Functions
# -----------------------------------------------------------------------------
def initialize_session_state():
    """Initializes all required session state variables if they do not exist."""
    session_vars = (
        ['food_selections', 'form_submitted', 'show_motivational_message',
         'food_search'] +
        [f'user_{field}' for field in CONFIG['form_fields'].keys()] +
        ['user_units']
    )

    for var in session_vars:
        if var not in st.session_state:
            if var == 'food_selections':
                st.session_state[var] = {}
            elif var == 'user_units':
                st.session_state[var] = 'metric'
            elif var in ['form_submitted', 'show_motivational_message']:
                st.session_state[var] = False
            elif var == 'food_search':
                st.session_state[var] = ""
            else:
                st.session_state[var] = None


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration dictionary."""
    session_key = f'user_{field_name}'
    widget_key = f'input_{field_name}'

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

        # Handle unit conversion for display purposes
        min_val = field_config['min']
        max_val = field_config['max']
        step_val = field_config['step']
        current_value = st.session_state[session_key]

        if (field_name == 'weight_kg' and
                st.session_state.get('user_units') == 'imperial'):
            label = 'Weight (in Pounds)'
            min_val, max_val = kg_to_lbs(min_val), kg_to_lbs(max_val)
            step_val = 1.0
            if current_value:
                current_value = kg_to_lbs(current_value)
        elif (field_name == 'height_cm' and
                st.session_state.get('user_units') == 'imperial'):
            label = 'Height (in Inches)'
            min_val, max_val = cm_to_inches(min_val), cm_to_inches(max_val)
            step_val = 1.0
            if current_value:
                current_value = cm_to_inches(current_value)
        else:
            label = field_config['label']

        value = container.number_input(
            label,
            min_value=min_val,
            max_value=max_val,
            value=current_value,
            step=step_val,
            placeholder=placeholder,
            help=field_config.get('help'),
            key=widget_key
        )

        # Convert the value back to metric for internal storage
        if (field_name == 'weight_kg' and
                st.session_state.get('user_units') == 'imperial' and value):
            value = lbs_to_kg(value)
        elif (field_name == 'height_cm' and
                st.session_state.get('user_units') == 'imperial' and value):
            value = inches_to_cm(value)

    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
            # Find the index of the current value, or default to 0
            index = next(
                (i for i, (_, val) in enumerate(options)
                 if val == current_value),
                next((i for i, (_, val) in enumerate(options)
                      if val == DEFAULTS[field_name]), 0)
            )
            selection = container.selectbox(
                field_config['label'],
                options,
                index=index,
                format_func=lambda x: x[0],
                key=widget_key
            )
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(
                field_config['label'],
                options,
                index=index,
                key=widget_key
            )

    st.session_state[session_key] = value
    return value


def validate_user_inputs(user_inputs):
    """Validates required user inputs and returns a list of error messages."""
    errors = []
    required_fields = [
        field for field, config in CONFIG['form_fields'].items()
        if config.get('required')
    ]

    for field in required_fields:
        if user_inputs.get(field) is None:
            field_label = CONFIG['form_fields'][field]['label']
            errors.append(f"Please enter your {field_label.lower()}")

    return errors


def get_final_values(user_inputs):
    """Processes user inputs and applies default values where necessary."""
    final_values = {}

    for field, value in user_inputs.items():
        final_values[field] = value if value is not None else DEFAULTS[field]

    # Apply goal specific defaults for advanced settings if they are not set
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
                help_text = METRIC_TOOLTIPS.get(label.split('(')[0].strip())
                st.metric(label, value, help=help_text)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                help_text = METRIC_TOOLTIPS.get(label.split('(')[0].strip())
                st.metric(label, value, delta, help=help_text)


def get_progress_color(percent):
    """Returns a color indicator for a progress bar based on percentage."""
    if percent >= 80:
        return "🟢"  # Green
    elif percent >= 50:
        return "🟡"  # Yellow
    else:
        return "🔴"  # Red


def render_progress_bars(totals, targets):
    """Renders a set of progress bars for all tracked nutrients."""
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals.get(nutrient, 0)
        target = targets.get(config['target_key'], 1)
        target = target if target > 0 else 1  # Avoid division by zero

        percent = min((actual / target) * 100, 100)
        color_indicator = get_progress_color(percent)

        st.progress(
            percent / 100,
            text=(
                f"{color_indicator} {config['label']}: {percent:.0f}% of your"
                f" daily target ({target:.0f} {config['unit']})"
            )
        )


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")

    # Call the dedicated function to render progress bars
    render_progress_bars(totals, targets)

    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation and building',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and overall health'
    }

    deficits = {}

    # Collect deficits for each nutrient
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        if actual < target:
            deficit = target - actual
            deficits[nutrient] = {
                'amount': deficit,
                'unit': config['unit'],
                'label': config['label'].lower(),
                'purpose': purpose_map.get(nutrient, 'for optimal nutrition')
            }

    # Create combined recommendations with multiple suggestions
    if deficits:
        all_foods = [item for sublist in foods.values() for item in sublist]
        food_suggestions = []

        for food in all_foods:
            coverage_score = 0
            nutrients_helped = []

            for nutrient, deficit_info in deficits.items():
                if nutrient != 'calories' and food[nutrient] > 0:
                    help_percentage = min(
                        food[nutrient] / deficit_info['amount'], 1.0
                    )
                    if help_percentage > 0.1:
                        coverage_score += help_percentage
                        nutrients_helped.append(nutrient)

            if coverage_score > 0 and len(nutrients_helped) > 1:
                food_suggestions.append({
                    'food': food,
                    'nutrients_helped': nutrients_helped,
                    'score': coverage_score
                })

        food_suggestions.sort(key=lambda x: x['score'], reverse=True)
        top_suggestions = food_suggestions[:3]

        deficit_summary = []
        for nutrient, deficit_info in deficits.items():
            deficit_summary.append(
                f"{deficit_info['amount']:.0f}g more {deficit_info['label']} "
                f"{deficit_info['purpose']}"
            )

        if len(deficit_summary) > 1:
            summary_text = ("You still need " +
                            ", ".join(deficit_summary[:-1]) +
                            f", and {deficit_summary[-1]}.")
        else:
            summary_text = f"You still need {deficit_summary[0]}."

        recommendations.append(summary_text)

        if top_suggestions:
            for i, suggestion in enumerate(top_suggestions):
                food = suggestion['food']
                nutrients_helped = suggestion['nutrients_helped']
                nutrient_benefits = [
                    f"{food[n]:.0f}g {n}" for n in nutrients_helped
                ]

                if len(nutrient_benefits) > 1:
                    benefits_text = (", ".join(nutrient_benefits[:-1]) +
                                     f", and {nutrient_benefits[-1]}")
                else:
                    benefits_text = nutrient_benefits[0]

                if i == 0:
                    recommendations.append(
                        f"🎯 A strategic choice is one serving of "
                        f"{food['name']}, which would provide "
                        f"{benefits_text}."
                    )
                else:
                    recommendations.append(
                        f"💡 An alternative option is {food['name']}, which "
                        f"provides {benefits_text}."
                    )
        else:
            biggest_deficit = max(
                deficits.items(), key=lambda x: x[1]['amount']
            )
            nutrient, deficit_info = biggest_deficit

            best_single_food = max(
                all_foods,
                key=lambda x: x.get(nutrient, 0),
                default=None
            )

            if best_single_food and best_single_food.get(nutrient, 0) > 0:
                recommendations.append(
                    f"💡 Consider adding {best_single_food['name']}, as it is"
                    f" rich in {deficit_info['label']}, containing "
                    f"{best_single_food[nutrient]:.0f}g per serving."
                )

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


def save_progress_to_json(food_selections, user_inputs):
    """Saves the current progress to a JSON formatted string."""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'food_selections': food_selections,
        'user_inputs': user_inputs
    }
    return json.dumps(progress_data, indent=2)


def load_progress_from_json(json_data):
    """Loads progress from a JSON data string."""
    try:
        data = json.loads(json_data)
        return data.get('food_selections', {}), data.get('user_inputs', {})
    except json.JSONDecodeError:
        return {}, {}


def prepare_summary_data(totals, targets, selected_foods):
    """Prepares a standardized summary data structure for exports."""
    summary_data = {
        'nutrition_summary': [],
        'consumed_foods': []
    }

    # Prepare the nutrition summary section
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = (actual / target * 100) if target > 0 else 0
        summary_data['nutrition_summary'].append({
            'label': config['label'],
            'actual': actual,
            'target': target,
            'unit': config['unit'],
            'percent': percent
        })

    # Prepare the consumed foods list
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        summary_data['consumed_foods'].append({
            'name': food['name'],
            'servings': servings,
            'calories': food['calories'] * servings,
            'protein': food['protein'] * servings,
            'carbs': food['carbs'] * servings,
            'fat': food['fat'] * servings
        })

    return summary_data


def create_pdf_summary(totals, targets, selected_foods, user_info):
    """Creates a PDF summary of the daily nutrition data."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Add the document title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Daily Nutrition Summary")

    # Add the date
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80,
                 f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    # Add user information
    y_pos = height - 120
    p.drawString(50, y_pos, f"Age: {user_info.get('age', 'N/A')}")
    p.drawString(200, y_pos,
                 f"Weight: {user_info.get('weight_kg', 'N/A'):.1f} kg")
    goal_display = user_info.get('goal', 'N/A').replace('_', ' ').title()
    p.drawString(350, y_pos, f"Goal: {goal_display}")

    # Add the nutrition summary section
    y_pos -= 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y_pos, "Nutrition Summary")

    y_pos -= 30
    p.setFont("Helvetica", 12)
    for item in summary_data['nutrition_summary']:
        p.drawString(
            50, y_pos,
            f"{item['label']}: {item['actual']:.0f}/{item['target']:.0f} "
            f"{item['unit']} ({item['percent']:.0f}%)"
        )
        y_pos -= 20

    # Add the selected foods list
    if summary_data['consumed_foods']:
        y_pos -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Foods Consumed")

        y_pos -= 30
        p.setFont("Helvetica", 10)
        # Limit to 20 items to prevent page overflow
        for item in summary_data['consumed_foods'][:20]:
            p.drawString(50, y_pos,
                         f"• {item['name']}: {item['servings']} serving(s)")
            y_pos -= 15
            if y_pos < 50:  # Prevent text from going off the page
                break

    p.save()
    buffer.seek(0)
    return buffer


def create_csv_summary(totals, targets, selected_foods):
    """Creates a CSV summary of the daily nutrition data."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    data = []

    # Add nutrition summary from the prepared data
    for item in summary_data['nutrition_summary']:
        data.append({
            'Category': 'Nutrition Summary',
            'Item': item['label'],
            'Actual': f"{item['actual']:.0f} {item['unit']}",
            'Target': f"{item['target']:.0f} {item['unit']}",
            'Percentage': f"{item['percent']:.0f}%"
        })

    # Add selected foods from the prepared data
    for item in summary_data['consumed_foods']:
        data.append({
            'Category': 'Foods Consumed',
            'Item': item['name'],
            'Servings': item['servings'],
            'Calories': f"{item['calories']:.0f} kcal",
            'Protein': f"{item['protein']:.1f} g",
            'Carbohydrates': f"{item['carbs']:.1f} g",
            'Fat': f"{item['fat']:.1f} g"
        })

    df = pd.DataFrame(data)
    df.rename(columns={'Carbs': 'Carbohydrates'}, inplace=True)
    return df.to_csv(index=False).encode('utf-8')


# -----------------------------------------------------------------------------
# # Cell 6: Nutritional Calculation Functions
# -----------------------------------------------------------------------------
def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculates the Basal Metabolic Rate using the Mifflin-St Jeor equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculates the Total Daily Energy Expenditure based on activity level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Calculates the estimated weekly weight change from a caloric adjustment.

    This calculation is based on the approximation that one kilogram of body
    fat contains approximately 7,700 kilocalories.
    """
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(age, height_cm, weight_kg, sex='male',
                                   activity_level='moderately_active',
                                   goal='weight_gain', protein_per_kg=None,
                                   fat_percentage=None):
    """Calculates personalized daily nutritional targets based on user inputs."""
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
        'protein_g': round(protein_g),
        'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 3),
        'goal': goal
    }

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


# -----------------------------------------------------------------------------
# # Cell 7: Food Database Processing Functions
# -----------------------------------------------------------------------------
@st.cache_data
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


@st.cache_data
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
            is_high_calorie = food_name in top_foods['calories'].get(
                category, []
            )

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
    """Filters the food dictionary based on a search term."""
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
        # Add a tooltip for the emoji, if one exists
        emoji_with_tooltip = food.get('emoji', '')
        if emoji_with_tooltip and emoji_with_tooltip in EMOJI_TOOLTIPS:
            st.markdown(f"**{emoji_with_tooltip}** {food['name']}")
            st.caption(EMOJI_TOOLTIPS[emoji_with_tooltip])
        else:
            st.subheader(f"{emoji_with_tooltip} {food['name']}")

        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'],
                                                               0.0)

        col1, col2 = st.columns([2, 1.2])

        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) \
                        else "secondary"
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
                min_value=0.0, max_value=20.0,
                value=float(current_serving), step=0.1,
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
            f"Per Serving: {food['calories']} kcal | {food['protein']}g"
            f" protein | {food['carbs']}g carbs | {food['fat']}g fat"
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


# -----------------------------------------------------------------------------
# # Cell 8: Initialize Application
# -----------------------------------------------------------------------------
# ------ Initialize Session State ------
initialize_session_state()

# ------ Load Food Database and Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# ------ Apply Custom CSS for Enhanced Styling ------
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
.stButton>button[kind="primary"] {
    background-color: #ff6b6b;
    color: white;
    border: 1px solid #ff6b6b;
}
.stButton>button[kind="secondary"] {
    border: 1px solid #ff6b6b;
    color: #333;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.stMetric > div > div > div > div {
    color: #262730;
}
.stProgress .st-bo {
    background-color: #e0e0e0;
}
.stProgress .st-bp {
    background-color: #ff6b6b;
}
/* Improved contrast for captions */
.stCaption {
    color: #555555 !important;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# # Cell 9: Application Title and Unified Input Interface
# -----------------------------------------------------------------------------
st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
### A Smart, Evidence-Based Nutrition Tracker That Gets You

Welcome to your personalized nutrition guide. This tool is built on established
scientific principles to help you achieve your goals, whether that involves
weight loss, maintenance, or muscle gain. We handle the calculations so you can
focus on enjoying your food.

Your journey to improved health and performance starts now. 🚀
""")

# ------ Sidebar for User Input ------
st.sidebar.header("Your Profile and Goals 📊")

# Units toggle using a switch widget for a cleaner interface
units = st.sidebar.toggle(
    "Toggle for Imperial Units",
    value=(st.session_state.get('user_units', 'metric') == 'imperial'),
    key='units_toggle',
    help="Toggle on for Imperial units (lbs, inches) or off for Metric"
         " units (kg, cm)."
)
st.session_state.user_units = 'imperial' if units else 'metric'

all_inputs = {}
standard_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')
}
advanced_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')
}

for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config,
                                 container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(
        field_name, field_config, container=advanced_expander
    )
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Calculate button with input validation
if st.sidebar.button("🧮 Calculate My Targets", type="primary",
                     key="calculate_button"):
    validation_errors = validate_user_inputs(all_inputs)
    if validation_errors:
        for error in validation_errors:
            st.sidebar.error(error)
    else:
        st.session_state.form_submitted = True
        st.session_state.show_motivational_message = True
        st.rerun()

# Save and Load Progress Section
st.sidebar.divider()
st.sidebar.subheader("💾 Save or Load Your Progress")

# Save progress button
if st.sidebar.button("Save Current Progress", key="save_progress",
                     type="primary"):
    progress_json = save_progress_to_json(st.session_state.food_selections,
                                          all_inputs)
    st.sidebar.download_button(
        "📥 Download Progress File",
        data=progress_json,
        file_name=f"nutrition_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="download_progress"
    )

# Load progress file uploader
uploaded_file = st.sidebar.file_uploader(
    "Load a Progress File", type="json", key="upload_progress"
)
if uploaded_file is not None:
    content = uploaded_file.read().decode()
    food_selections, user_inputs = load_progress_from_json(content)

    # Update the session state with loaded data
    st.session_state.food_selections.update(food_selections)
    for key, value in user_inputs.items():
        if f'user_{key}' in st.session_state:
            st.session_state[f'user_{key}'] = value

    st.sidebar.success("Your progress was loaded successfully! ✅")
    st.rerun()

# ------ Activity Level Guide in the Sidebar ------
st.sidebar.divider()
with st.sidebar.container(border=True):
    st.markdown("##### A Guide to Activity Levels")
    st.markdown("""
* **🧑‍💻 Sedentary**: You have a desk job with little to no exercise.
* **🏃 Lightly Active**: You engage in light walks or workouts one to three
  times per week.
* **🚴 Moderately Active**: You exercise at a moderate intensity three to five
  days per week.
* **🏋️ Very Active**: You participate in heavy exercise six to seven days
  per week.
* **🤸 Extremely Active**: You have a physically demanding job or train two
  times per day.

*💡 If you are unsure which level to choose, it is best to select the lower
one to avoid overestimating your caloric expenditure.*
    """)

# ------ Dynamic Sidebar Summary ------
if st.session_state.form_submitted:
    final_values = get_final_values(all_inputs)
    targets = calculate_personalized_targets(**final_values)
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Quick Summary")

    progress_calories = min(
        totals['calories'] / targets['total_calories'] * 100, 100
    ) if targets['total_calories'] > 0 else 0
    progress_protein = min(
        totals['protein'] / targets['protein_g'] * 100, 100
    ) if targets['protein_g'] > 0 else 0

    st.sidebar.metric(
        "Calories Progress",
        f"{progress_calories:.0f}%",
        f"{totals['calories']:.0f} / {targets['total_calories']:.0f} kcal"
    )
    st.sidebar.metric(
        "Protein Progress",
        f"{progress_protein:.0f}%",
        f"{totals['protein']:.0f} / {targets['protein_g']:.0f} g"
    )

# ------ Process Final Values ------
final_values = get_final_values(all_inputs)

# ------ Check for User Input ------
user_has_entered_info = st.session_state.form_submitted

# ------ Calculate Personalized Targets ------
targets = calculate_personalized_targets(**final_values)

# Show a motivational message after the form is submitted
if st.session_state.show_motivational_message and user_has_entered_info:
    goal_messages = {
        'weight_loss': f"🎉 Your plan is set for success. You are on track to"
                       f" lose approximately"
                       f" {abs(targets['estimated_weekly_change']):.2f} kg"
                       f" per week. Consistency will be key to your success.",
        'weight_maintenance': f"🎯 Your maintenance plan is now active. You"
                              f" are set to maintain your current weight of"
                              f" {format_weight(final_values['weight_kg'], st.session_state.get('user_units', 'metric'))}"
                              f" while optimizing your nutrition.",
        'weight_gain': f"💪 Your muscle building journey starts now. You are"
                       f" targeting a gain of about"
                       f" {targets['estimated_weekly_change']:.2f} kg per"
                       f" week. Remember to fuel your body and train hard."
    }

    message = goal_messages.get(
        targets['goal'], "🚀 Your plan is ready. Let us begin."
    )
    st.success(message)

    # Reset the flag so the message does not show on every rerun
    if st.button("✨ Acknowledged", key="dismiss_message"):
        st.session_state.show_motivational_message = False
        st.rerun()


# -----------------------------------------------------------------------------
# # Cell 10: Unified Target Display System
# -----------------------------------------------------------------------------
if not user_has_entered_info:
    st.info(
        "👈 Please enter your details in the sidebar and click 'Calculate My"
        " Targets' to generate your personalized daily targets."
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
    "🎯 **The 80/20 Rule**: Aim to meet your nutritional targets approximately"
    " 80 percent of the time. This approach allows for flexibility and can"
    " improve long term adherence by accommodating social events and personal"
    " preferences."
)

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Unified Metrics Display Configuration ------
units_display = st.session_state.get('user_units', 'metric')
weight_display = format_weight(final_values['weight_kg'], units_display)

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Weight", weight_display),
            ("BMR", f"{targets['bmr']} kcal"),
            ("TDEE", f"{targets['tdee']} kcal"),
            ("Daily Caloric Adjustment",
             f"{targets['caloric_adjustment']:+} kcal"),
            ("Estimated Weekly Weight Change",
             f"{targets['estimated_weekly_change']:+.2f} kg")
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


# -----------------------------------------------------------------------------
# # Cell 11: Enhanced Evidence-Based Tips and Context
# -----------------------------------------------------------------------------
with st.expander("📚 Your Evidence-Based Game Plan", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs([
        "Core Nutritional Principles 🏆", "Advanced Progress Tracking 📊",
        "The Mindset for Success 🧠", "The Science Behind the Calculations 🔬"
    ])

    with tab1:
        st.subheader("💧 Master Your Hydration")
        for tip in TIPS_CONTENT['hydration']:
            st.markdown(f"* {tip}")

        st.subheader("😴 Prioritize Your Sleep")
        for tip in TIPS_CONTENT['sleep']:
            st.markdown(f"* {tip}")

        st.subheader("📅 Track Your Progress Consistently")
        for tip in TIPS_CONTENT['tracking_wins']:
            st.markdown(f"* {tip}")

    with tab2:
        st.subheader("Go Beyond the Scale 📸")
        for tip in TIPS_CONTENT['beyond_the_scale']:
            st.markdown(f"* {tip}")

    with tab3:
        st.subheader("The Mindset for Success 🧠")
        st.markdown("""
        The 80/20 principle is an effective strategy for avoiding a perfectionist
        mindset. Rather than abandoning your plan after one suboptimal meal,
        this approach promotes long term consistency.

        **A Gradual Approach to Habit Formation:**

        * **Weeks 1 to 2**: Focus exclusively on meeting your calorie targets.
        * **Weeks 3 to 4**: After mastering calorie tracking, begin to monitor
          your protein intake.
        * **Week 5 and Beyond**: Once calories and protein are consistently
          tracked, you can fine tune your carbohydrate and fat intake.

        ---
        **When Progress Stalls** 🔄
        """)

        st.markdown("##### Have You Hit a Weight Loss Plateau?")
        for tip in TIPS_CONTENT['weight_loss_plateau']:
            st.markdown(f"* {tip}")

        st.markdown("##### Are You Struggling to Gain Weight?")
        for tip in TIPS_content['weight_gain_stalls']:
            st.markdown(f"* {tip}")

        st.markdown("--- \n ##### Pace Your Protein Intake")
        for tip in TIPS_CONTENT['protein_pacing']:
            st.markdown(f"* {tip}")

    with tab4:
        st.subheader("Understanding Your Metabolism")
        st.markdown("""
        Your Basal Metabolic Rate (BMR) represents the energy your body
        requires for basic physiological functions. Your Total Daily Energy
        Expenditure (TDEE) expands on this by including the energy burned
        through physical activity.

        **A Guide to Satiety**

        Different foods affect hunger and fullness differently.

        * **Protein**: Protein is the most satiating macronutrient. It digests
          slowly, helps stabilize blood sugar, and has a higher thermic effect
          of food.

        * **Fiber-Rich Carbohydrates**: Vegetables, fruits, and whole grains
          are high in fiber and water, which increases food volume and promotes
          satiety.

        * **Healthy Fats**: Nuts, olive oil, and avocados provide sustained
          energy release, contributing to long term fullness.

        * **Processed Foods**: Highly processed foods are often less satiating
          and should be consumed in moderation.

        A general guideline is to consume 14 grams of fiber for every 1,000
        calories, which typically amounts to 25 to 38 grams per day.

        **Key Micronutrients for Vegetarians**

        A vegetarian diet requires attention to certain micronutrients that are
        less abundant in plant based foods.

        **The Watch List:**

        * **Vitamin B₁₂**: This vitamin is crucial for nerve function and cell
          metabolism. It is almost exclusively found in animal products, so a
          supplement is often recommended.
        * **Iron**: Iron is essential for oxygen transport. Plant based sources
          like leafy greens and lentils should be consumed with a source of
          vitamin C to enhance absorption.
        * **Calcium**: This mineral is vital for bone health and muscle
          function. It can be found in kale, almonds, tofu, and fortified
          plant milks.
        * **Zinc**: Zinc is important for immune function. It is available in
          nuts, seeds, and whole grains.
        * **Iodine**: Iodine is necessary for thyroid function. A small amount
          of iodized salt can typically meet daily requirements.
        * **Omega-3s (EPA/DHA)**: These fats are important for brain and heart
          health. Fortified foods or algae based supplements are good sources.

        Fortified foods such as plant milks and cereals can help meet these
        micronutrient needs. It is advisable to consult a doctor or dietitian
        to create a suitable plan.
        """)


# -----------------------------------------------------------------------------
# # Cell 12: [REMOVED]
# -----------------------------------------------------------------------------
# This section was removed to eliminate redundancy with the content provided
# in Cell 11 under "Your Evidence-Based Game Plan".


# -----------------------------------------------------------------------------
# # Cell 13: Food Selection Interface
# -----------------------------------------------------------------------------
st.header("Track Your Daily Intake 🥗")

# Food search functionality
search_col, reset_col = st.columns([3, 1])
with search_col:
    search_term = st.text_input(
        "🔍 Search for Foods",
        value=st.session_state.food_search,
        placeholder="Type a food name to filter the results...",
        key="food_search_input"
    )
    st.session_state.food_search = search_term

with reset_col:
    st.write("")  # Spacer for vertical alignment
    st.write("")  # Spacer for vertical alignment
    if st.button("🔄 Clear Search", key="clear_search"):
        st.session_state.food_search = ""
        st.rerun()

st.markdown(
    "Select the number of servings for each food you consume to see how your"
    " choices compare to your daily targets."
)

with st.expander("💡 View the Emoji Guide for Food Choices"):
    st.markdown("""
    * **🥇 Gold Medal**: A nutritional all-star that is high in its target
      nutrient and very calorie efficient.
    * **🔥 High Calorie**: One of the more calorie-dense options in its group.
    * **💪 High Protein**: A true protein powerhouse.
    * **🍚 High Carbohydrate**: A carbohydrate champion.
    * **🥑 High Fat**: A healthy fat hero.
    """)

if st.button("🔄 Reset All Food Selections", type="secondary",
             key="reset_foods"):
    st.session_state.food_selections = {}
    st.rerun()

# Filter foods based on the search term
filtered_foods = filter_foods_by_search(foods, search_term)

if not filtered_foods and search_term:
    st.warning(f"No foods were found matching '{search_term}'. Please try a"
               " different search term or clear the search. 🙁")
elif filtered_foods:
    # ------ Food Selection with Tabs ------
    available_categories = [
        cat for cat, items in sorted(filtered_foods.items()) if items
    ]
    tabs = st.tabs(available_categories)

    for i, category in enumerate(available_categories):
        items = filtered_foods[category]
        sorted_items_in_category = sorted(
            items,
            key=lambda x: (
                CONFIG['emoji_order'].get(x.get('emoji', ''), 4),
                -x['calories']
            )
        )
        with tabs[i]:
            render_food_grid(sorted_items_in_category, category, columns=2)


# -----------------------------------------------------------------------------
# # Cell 14: Daily Summary and Progress Tracking
# -----------------------------------------------------------------------------
st.header("Today's Scorecard 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)

    # Export functionality
    st.subheader("📥 Export Your Summary")
    col1, col2 = st.columns(2)

    with col1:
        pdf_buffer = create_pdf_summary(
            totals, targets, selected_foods, final_values
        )
        st.download_button(
            "📄 Download PDF Report",
            data=pdf_buffer,
            file_name=f"Nutrition_Summary_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            key="download_pdf_button"
        )

    with col2:
        csv_data = create_csv_summary(totals, targets, selected_foods)
        st.download_button(
            "📊 Download CSV Data",
            data=csv_data,
            file_name=f"Nutrition_Data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_csv_button"
        )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Today's Nutrition Snapshot")
        summary_metrics = [
            ("Calories Consumed", f"{totals['calories']:.0f} kcal"),
            ("Protein Intake", f"{totals['protein']:.0f} g"),
            ("Carbohydrates", f"{totals['carbs']:.0f} g"),
            ("Fat Intake", f"{totals['fat']:.0f} g")
        ]
        display_metrics_grid(summary_metrics, 2)

    with col2:
        st.subheader("Your Macronutrient Split (in Grams)")
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
        if sum(macro_values) > 0:
            fig = go.Figure(go.Pie(
                labels=['Protein', 'Carbohydrates', 'Fat'],
                values=macro_values,
                hole=.4,
                marker_colors=['#ff6b6b', '#feca57', '#48dbfb'],
                textinfo='label+percent',
                insidetextorientation='radial'
            ))
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Please select foods to see the macronutrient split.")

    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)

    with st.expander("Review Your Food Choices for Today"):
        st.subheader("What You Have Logged")
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings

            st.write(f"**{food['name']}** - {servings} serving(s)")
            st.write(
                f"→ {total_cals:.0f} kcal | {total_protein:.1f}g protein | "
                f"{total_carbs:.1f}g carbs | {total_fat:.1f}g fat"
            )
else:
    st.info(
        "You have not selected any foods yet. Please add items from the"
        " categories above to begin tracking your intake. 🥗"
    )
    st.subheader("Progress Snapshot")
    render_progress_bars(totals, targets)


# -----------------------------------------------------------------------------
# # Cell 15: User Feedback Section
# -----------------------------------------------------------------------------
st.divider()
st.header("💬 Help Us Improve!")
st.markdown(
    "Your feedback helps us make this application better. Please share your "
    "thoughts below."
)

with st.form("feedback_form", clear_on_submit=True):
    feedback_type = st.selectbox(
        "What Type of Feedback Would You Like to Share?",
        ["General Feedback", "Bug Report", "Feature Request",
         "Success Story"],
        key="feedback_type"
    )

    feedback_text = st.text_area(
        "How Can We Improve?",
        placeholder="Tell us about your experience, suggest new features, or "
                    "report any issues you have encountered...",
        height=100,
        key="feedback_text"
    )

    if st.form_submit_button("📤 Submit Feedback", type="primary"):
        if feedback_text.strip():
            # In a real application, this would save to a database
            st.success(f"Thank you for your {feedback_type.lower()}! Your "
                       "input helps us make the application better for "
                       "everyone. 🙏")
        else:
            st.error("Please enter some feedback before submitting. 📝")


# -----------------------------------------------------------------------------
# # Cell 16: Footer and Additional Resources
# -----------------------------------------------------------------------------
st.divider()
st.markdown("""
### The Scientific Foundation 📚

This tracker is grounded in peer reviewed research and evidence based
guidelines. We use the Mifflin-St Jeor equation to calculate your Basal
Metabolic Rate (BMR), a method that is endorsed by the Academy of Nutrition
and Dietetics. To estimate your Total Daily Energy Expenditure (TDEE), we use
activity multipliers derived from exercise physiology research. Our protein
recommendations are based on guidelines from the International Society of
Sports Nutrition.

All caloric adjustments are based on sustainable rates that research has shown
to produce lasting results.

### Disclaimer ⚠️

This tool should be considered a starting point, as individual results may vary
due to factors such as genetics, health conditions, and medications. It is
always advisable to consult a qualified healthcare provider before making
significant dietary changes. Please monitor your energy levels, performance,
and overall well being, and adjust your plan as needed.
""")

st.success(
    "Thank you for using the nutrition tracker. Remember that consistency is "
    "the key to achieving your goals. 🥳"
)


# -----------------------------------------------------------------------------
# # Cell 17: Session State Management and Performance
# -----------------------------------------------------------------------------
# ------ Clean Up Session State to Prevent Memory Issues ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# ------ Performance Optimization ------
# Ensure all widgets have unique keys to reduce unnecessary reruns.
# This practice has been handled throughout the code with explicit key
# parameters.

# ------ Session State Cleanup for Unused Variables ------
# Remove any temporary variables that might accumulate in the session state.
temp_keys = [key for key in st.session_state.keys() if key.startswith('temp_')]
for key in temp_keys:
    del st.session_state[key]
