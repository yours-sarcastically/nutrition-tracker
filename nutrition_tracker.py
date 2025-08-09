#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# An Interactive, Evidence-Based Nutrition Tracker for 
# Personalized Meal Planning
# ---------------------------------------------------------------------------

"""
This script implements an interactive, evidence-based nutrition tracker using
Streamlit. It is designed to help users achieve personalized nutrition goals,
such as weight loss, maintenance, or muscle gain, with a focus on vegetarian
food sources.

Core Functionality and Scientific Basis
----------------------------------------
- Basal Metabolic Rate (BMR) Calculation: The application uses the Mifflin-St
  Jeor equation, which is widely recognized by organizations like the Academy
  of Nutrition and Dietetics for its accuracy in estimating resting metabolic
  rate.
  - For Males: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
  - For Females: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

- Total Daily Energy Expenditure (TDEE): The calculated BMR is multiplied by a
  scientifically validated activity factor to estimate the total number of
  calories burned in a day, including all physical activity.

- Goal-Specific Caloric Adjustments:
  - Weight Loss: A conservative 20 percent caloric deficit from TDEE.
  - Weight Maintenance: Caloric intake is set equal to TDEE.
  - Weight Gain: A controlled 10 percent caloric surplus over TDEE.

- Macronutrient Strategy: The script follows a protein-first approach, which
  is consistent with modern nutrition science for optimizing body composition.
  1. Protein intake is determined based on grams per kilogram of body weight.
  2. Fat intake is set as a percentage of the total daily calories.
  3. Carbohydrate intake is calculated from the remaining caloric budget.

Implementation Details
----------------------
- The user interface is built with Streamlit, providing interactive widgets
  for user input and data visualization.
- The food database is managed using the Pandas library for efficient data
  handling.
- Progress visualizations are created with Streamlit's native components and
  Plotly for generating detailed charts and graphs.

Usage Documentation
-------------------
1. Prerequisites: Ensure you have the required Python libraries installed.
   You can install them using pip:
   pip install streamlit pandas plotly reportlab

2. Running the Application: Save this script as a Python file (for example,
   `nutrition_app.py`) and run it from your terminal using the following
   command:
   streamlit run nutrition_app.py

3. Interacting with the Application:
   - Use the sidebar to enter your personal details, such as age, height,
     weight, sex, activity level, and primary nutrition goal.
   - Your personalized daily targets for calories and macronutrients will be
     calculated and displayed automatically after you click the calculate
     button.
   - Navigate through the food tabs to select the number of servings for
     each food item you consume throughout the day.
   - The daily summary section will update in real time to show your
     progress toward your targets.
"""

# # --------------------------------------------------------------------------- # #
# # Cell 1: Import Required Libraries and Modules                                 # #
# # --------------------------------------------------------------------------- # #
import io
import json
import math
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# # --------------------------------------------------------------------------- # #
# # Cell 2: Page Configuration and Initial Setup                                  # #
# # --------------------------------------------------------------------------- # #

# ------ Configure the Streamlit Page ------
st.set_page_config(
    page_title="Your Personal Nutrition Coach 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# # --------------------------------------------------------------------------- # #
# # Cell 3: Unified Configuration Constants                                       # #
# # --------------------------------------------------------------------------- # #

# ------ Default Parameter Values Based on Published Research ------
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
    'sedentary': "🧑‍💻 **Sedentary**: You are mostly tied to your desk chair.",
    'lightly_active': (
        "🏃 **Lightly Active**: You engage in light walks or workouts one to "
        "three times a week."
    ),
    'moderately_active': (
        "🚴 **Moderately Active**: You exercise or have an active job three "
        "to five days a week."
    ),
    'very_active': "🏋️ **Very Active**: You are highly active most days.",
    'extremely_active': "🤸 **Extremely Active**: You live in the gym."
}

# ------ Goal-Specific Targets Based on an Evidence-Based Guide ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # A 20% deficit from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,    # A 0% adjustment from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # A 10% surplus over TDEE
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
        'calories': {
            'unit': 'kcal', 'label': 'Calories',
            'target_key': 'total_calories'
        },
        'protein': {
            'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'
        },
        'carbs': {
            'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'
        },
        'fat': {
            'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'
        }
    },
    'form_fields': {
        'age': {
            'type': 'number', 'label': 'Age (in years)', 'min': 16,
            'max': 80, 'step': 1, 'placeholder': 'Enter your age',
            'required': True
        },
        'height_cm': {
            'type': 'number', 'label': 'Height (in centimeters)',
            'min': 140, 'max': 220, 'step': 1,
            'placeholder': 'Enter your height', 'required': True
        },
        'weight_kg': {
            'type': 'number', 'label': 'Weight (in kilograms)', 'min': 40.0,
            'max': 150.0, 'step': 0.5, 'placeholder': 'Enter your weight',
            'required': True
        },
        'sex': {
            'type': 'selectbox', 'label': 'Biological Sex',
            'options': ["Male", "Female"], 'required': True
        },
        'activity_level': {
            'type': 'selectbox', 'label': 'Activity Level', 'options': [
                ("Sedentary", "sedentary"),
                ("Lightly Active", "lightly_active"),
                ("Moderately Active", "moderately_active"),
                ("Very Active", "very_active"),
                ("Extremely Active", "extremely_active")
            ], 'required': True
        },
        'goal': {
            'type': 'selectbox', 'label': 'Your Goal', 'options': [
                ("Weight Loss", "weight_loss"),
                ("Weight Maintenance", "weight_maintenance"),
                ("Weight Gain", "weight_gain")
            ], 'required': True
        },
        'protein_per_kg': {
            'type': 'number', 'label': 'Protein Goal (g/kg)',
            'min': 1.2, 'max': 3.0, 'step': 0.1,
            'help': (
                'Define your daily protein target in grams per kilogram of '
                'body weight'
            ),
            'advanced': True, 'required': False
        },
        'fat_percentage': {
            'type': 'number', 'label': 'Fat Intake (% of calories)',
            'min': 15, 'max': 40, 'step': 1,
            'help': (
                'Set the percentage of your daily calories that should come '
                'from healthy fats'
            ),
            'convert': lambda x: x / 100 if x else None,
            'advanced': True, 'required': False
        }
    }
}

# ------ Emoji Tooltips ------
EMOJI_TOOLTIPS = {
    '🥇': (
        'Gold Medal: A nutritional all-star that is high in its target '
        'nutrient and very calorie-efficient'
    ),
    '🔥': 'High Calorie: One of the more calorie-dense options in its group',
    '💪': 'High Protein: A true protein powerhouse',
    '🍚': 'High Carbohydrate: A carbohydrate champion',
    '🥑': 'High Fat: A healthy fat hero'
}

# ------ Metric Tooltips ------
METRIC_TOOLTIPS = {
    'BMR': (
        'Basal Metabolic Rate is the energy your body needs to keep vital '
        'functions running at rest'
    ),
    'TDEE': (
        'Total Daily Energy Expenditure is your BMR plus the calories '
        'burned through daily activity'
    ),
    'Caloric Adjustment': (
        'This is the number of calories above or below your TDEE needed to '
        'reach your goal'
    ),
    'Protein': 'Essential for building muscle, repair, and satiety',
    'Carbohydrates': (
        'The body\'s preferred energy source for brain and muscle function'
    ),
    'Fat': 'Important for hormone production, nutrient absorption, and health'
}

# # --------------------------------------------------------------------------- # #
# # Cell 4: Unit Conversion Functions                                             # #
# # --------------------------------------------------------------------------- # #

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
    """Formats a weight value based on the selected unit preference."""
    if units == 'imperial':
        return f"{kg_to_lbs(weight_kg):.1f} lbs"
    return f"{weight_kg:.1f} kg"


def format_height(height_cm, units):
    """Formats a height value based on the selected unit preference."""
    if units == 'imperial':
        total_inches = cm_to_inches(height_cm)
        feet = int(total_inches // 12)
        inches = total_inches % 12
        return f"{feet}'{inches:.0f}\""
    return f"{height_cm:.0f} cm"


# # --------------------------------------------------------------------------- # #
# # Cell 5: Unified Helper Functions                                              # #
# # --------------------------------------------------------------------------- # #

def initialize_session_state():
    """Initializes all required session state variables if not present."""
    session_vars = (
        ['food_selections', 'form_submitted', 'show_motivational_message',
         'food_search', 'form_errors', 'food_input_modes'] +
        [f'user_{field}' for field in CONFIG['form_fields'].keys()] +
        ['user_units']
    )

    for var in session_vars:
        if var not in st.session_state:
            if var == 'food_selections':
                st.session_state[var] = {}
            elif var == 'food_input_modes':
                st.session_state[var] = {}
            elif var == 'user_units':
                st.session_state[var] = 'metric'
            elif var in ['form_submitted', 'show_motivational_message']:
                st.session_state[var] = False
            elif var == 'food_search':
                st.session_state[var] = ""
            elif var == 'form_errors':
                st.session_state[var] = []
            else:
                st.session_state[var] = None


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates a user input widget based on a unified configuration."""
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

        # Handle unit conversion for display
        min_val = field_config['min']
        max_val = field_config['max']
        step_val = field_config['step']
        current_value = st.session_state[session_key]

        user_units = st.session_state.get('user_units')
        if field_name == 'weight_kg' and user_units == 'imperial':
            label = 'Weight (in pounds)'
            min_val, max_val = kg_to_lbs(min_val), kg_to_lbs(max_val)
            step_val = 1.0
            if current_value:
                current_value = kg_to_lbs(current_value)
        elif field_name == 'height_cm' and user_units == 'imperial':
            label = 'Height (in inches)'
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

        # Convert back to metric for storage
        if field_name == 'weight_kg' and user_units == 'imperial' and value:
            value = lbs_to_kg(value)
        elif field_name == 'height_cm' and user_units == 'imperial' and value:
            value = inches_to_cm(value)

    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        options = field_config['options']
        if field_name in ['activity_level', 'goal']:
            # Find index of the current value, or default if not found
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
    """Validates required user inputs and returns a list of errors."""
    errors = []
    required_fields = [
        field for field, config in CONFIG['form_fields'].items()
        if config.get('required')
    ]

    for field in required_fields:
        if user_inputs.get(field) is None:
            field_label = CONFIG['form_fields'][field]['label']
            errors.append(f"Please enter your {field_label.lower()}.")

    return errors


def get_final_values(user_inputs):
    """Processes all user inputs and applies default values where needed."""
    final_values = {}

    for field, value in user_inputs.items():
        final_values[field] = value if value is not None else DEFAULTS[field]

    # Apply goal-specific defaults for advanced settings if they are not set
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
    base_needs = weight_kg * 35  # Baseline is 35 ml per kg

    activity_bonus = {
        'sedentary': 0, 'lightly_active': 300, 'moderately_active': 500,
        'very_active': 700, 'extremely_active': 1000
    }
    climate_multiplier = {
        'cold': 0.9, 'temperate': 1.0, 'hot': 1.2, 'very_hot': 1.4
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
            label = metric_info[0]
            value = metric_info[1]
            help_text = METRIC_TOOLTIPS.get(label.split('(')[0].strip())

            if len(metric_info) == 2:
                st.metric(label, value, help=help_text)
            elif len(metric_info) == 3:
                delta = metric_info[2]
                st.metric(label, value, delta, help=help_text)


def get_progress_color(percent):
    """Returns a color for the progress bar based on a percentage."""
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
                f"{color_indicator} {config['label']}: {percent:.0f}% of your "
                f"daily target ({target:.0f} {config['unit']})"
            )
        )


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")

    # Call the dedicated function to render progress bars
    render_progress_bars(totals, targets)

    purpose_map = {
        'calories': 'to reach your energy target',
        'protein': 'for muscle preservation and growth',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and overall health'
    }
    deficits = {}

    # Collect deficits
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        if actual < target:
            deficits[nutrient] = {
                'amount': target - actual,
                'unit': config['unit'],
                'label': config['label'].lower(),
                'purpose': purpose_map.get(nutrient, 'for optimal nutrition')
            }

    # Create combined recommendations if there are deficits
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
                    'food': food, 'nutrients_helped': nutrients_helped,
                    'score': coverage_score
                })

        food_suggestions.sort(key=lambda x: x['score'], reverse=True)
        top_suggestions = food_suggestions[:3]
        deficit_summary = [
            f"{info['amount']:.0f}g more {info['label']} {info['purpose']}"
            for _, info in deficits.items()
        ]

        # Format deficit summary text
        if len(deficit_summary) > 1:
            summary_text = (
                "You still need " + ", ".join(deficit_summary[:-1]) +
                f", and {deficit_summary[-1]}."
            )
        else:
            summary_text = f"You still need {deficit_summary[0]}."
        recommendations.append(summary_text)

        if top_suggestions:
            # Format smart pick recommendation
            first_sug = top_suggestions[0]
            benefits = [f"{first_sug['food'][n]:.0f}g {n}"
                        for n in first_sug['nutrients_helped']]
            if len(benefits) > 1:
                benefits_text = ", ".join(benefits[:-1]) + f", and {benefits[-1]}"
            else:
                benefits_text = benefits[0]
            recommendations.append(
                f"🎯 **Smart Pick**: One serving of {first_sug['food']['name']} "
                f"would give you {benefits_text}, which knocks out "
                "multiple targets at once!"
            )

            # Format alternative options
            if len(top_suggestions) > 1:
                alt_foods = []
                for suggestion in top_suggestions[1:]:
                    food, helped = suggestion['food'], suggestion['nutrients_helped']
                    benefits = [f"{food[n]:.0f}g {n}" for n in helped]
                    if len(benefits) > 1:
                        b_text = ", ".join(benefits[:-1]) + f", and {benefits[-1]}"
                    else:
                        b_text = benefits[0]
                    alt_foods.append(f"{food['name']} (provides {b_text})")

                if len(alt_foods) == 1:
                    alt_text = alt_foods[0]
                elif len(alt_foods) == 2:
                    alt_text = f"{alt_foods[0]} or {alt_foods[1]}"
                else:
                    alt_text = ", ".join(alt_foods[:-1]) + f", or {alt_foods[-1]}"
                recommendations.append(
                    "💡 **Alternative Options**: "
                    f"{alt_text} are also great ways to meet multiple goals."
                )
        else:
            # Suggest best food for biggest deficit if no multi-purpose foods
            biggest_deficit = max(deficits.items(), key=lambda x: x[1]['amount'])
            nutrient, info = biggest_deficit
            best_food = max(
                all_foods, key=lambda x: x.get(nutrient, 0), default=None
            )
            if best_food and best_food.get(nutrient, 0) > 0:
                recommendations.append(
                    f"💡 **Alternative Option**: Try adding "
                    f"{best_food['name']}, since it is packed with "
                    f"{best_food[nutrient]:.0f}g of {info['label']}."
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
    """Loads progress from JSON data."""
    try:
        data = json.loads(json_data)
        return data.get('food_selections', {}), data.get('user_inputs', {})
    except json.JSONDecodeError:
        return {}, {}


def prepare_summary_data(totals, targets, selected_foods):
    """Prepares a standardized summary data structure for exports."""
    summary_data = {'nutrition_summary': [], 'consumed_foods': []}

    # Prepare nutrition summary
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = (actual / target * 100) if target > 0 else 0
        summary_data['nutrition_summary'].append({
            'label': config['label'], 'actual': actual, 'target': target,
            'unit': config['unit'], 'percent': percent
        })

    # Prepare consumed foods list
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        summary_data['consumed_foods'].append({
            'name': food['name'], 'servings': servings,
            'calories': food['calories'] * servings,
            'protein': food['protein'] * servings,
            'carbs': food['carbs'] * servings, 'fat': food['fat'] * servings
        })

    return summary_data


def create_pdf_summary(totals, targets, selected_foods, user_info):
    """Creates a PDF summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title and date
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Daily Nutrition Summary")
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"Date: {datetime.now():%Y-%m-%d}")

    # User info
    y_pos = height - 120
    p.drawString(50, y_pos, f"Age: {user_info.get('age', 'N/A')}")
    p.drawString(200, y_pos, f"Weight: {user_info.get('weight_kg', 'N/A')} kg")
    p.drawString(350, y_pos, f"Goal: {user_info.get('goal', 'N/A')}")

    # Nutrition summary table
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

    # Selected foods list
    if summary_data['consumed_foods']:
        y_pos -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Foods Consumed")
        y_pos -= 30
        p.setFont("Helvetica", 10)
        for item in summary_data['consumed_foods'][:20]:  # Limit to one page
            p.drawString(50, y_pos, f"• {item['name']}: {item['servings']} servings")
            y_pos -= 15
            if y_pos < 50:  # Prevent going off the page
                break

    p.save()
    buffer.seek(0)
    return buffer


def create_csv_summary(totals, targets, selected_foods):
    """Creates a CSV summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    data = []

    # Add nutrition summary from prepared data
    for item in summary_data['nutrition_summary']:
        data.append({
            'Category': 'Nutrition Summary', 'Item': item['label'],
            'Actual': f"{item['actual']:.0f} {item['unit']}",
            'Target': f"{item['target']:.0f} {item['unit']}",
            'Percentage': f"{item['percent']:.0f}%"
        })

    # Add selected foods from prepared data
    for item in summary_data['consumed_foods']:
        data.append({
            'Category': 'Foods Consumed', 'Item': item['name'],
            'Servings': item['servings'],
            'Calories': f"{item['calories']:.0f} kcal",
            'Protein': f"{item['protein']:.1f} g",
            'Carbohydrates': f"{item['carbs']:.1f} g",
            'Fat': f"{item['fat']:.1f} g"
        })

    df = pd.DataFrame(data)
    return df.to_csv(index=False)


# # --------------------------------------------------------------------------- # #
# # Cell 6: Nutritional Calculation Functions                                     # #
# # --------------------------------------------------------------------------- # #

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculates Basal Metabolic Rate using the Mifflin-St Jeor equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculates Total Daily Energy Expenditure from BMR and activity."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Estimates weekly weight change from a daily caloric adjustment."""
    # Based on the approximation that one kg of fat contains 7,700 kcal
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


# # --------------------------------------------------------------------------- # #
# # Cell 7: Food Database Processing Functions                                    # #
# # --------------------------------------------------------------------------- # #

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
                'carbs': row['carbs'], 'fat': row['fat'],
                'serving_size_g': row.get('serving_size_g', 100)  # Add serving size
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


def get_food_input_mode(food_name):
    """Gets the current input mode for a food item."""
    return st.session_state.food_input_modes.get(food_name, 'servings')


def set_food_input_mode(food_name, mode):
    """Sets the input mode for a food item."""
    st.session_state.food_input_modes[food_name] = mode


def render_food_item(food, category):
    """Renders a single food item with its interaction controls and mode toggle."""
    with st.container(border=True):
        # Header with emoji and toggle
        col_title, col_toggle = st.columns([3, 1])
        
        with col_title:
            emoji_with_tooltip = food.get('emoji', '')
            st.subheader(f"{emoji_with_tooltip} {food['name']}")
            if emoji_with_tooltip and emoji_with_tooltip in EMOJI_TOOLTIPS:
                st.caption(EMOJI_TOOLTIPS[emoji_with_tooltip])
        
        with col_toggle:
            # Mode toggle switch
            food_name = food['name']
            current_mode = get_food_input_mode(food_name)
            
            # Create toggle with icons
            mode_toggle = st.toggle(
                "⚖️ 100g",
                value=(current_mode == 'grams'),
                key=f"mode_toggle_{category}_{food_name}",
                help="Toggle between servings (🥄) and grams (⚖️) input mode"
            )
            
            new_mode = 'grams' if mode_toggle else 'servings'
            if new_mode != current_mode:
                set_food_input_mode(food_name, new_mode)
                st.rerun()

        key_prefix = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)
        
        # Display nutritional info based on mode
        serving_size_g = food.get('serving_size_g', 100)
        
        if current_mode == 'servings':
            # Servings mode
            st.caption(
                f"Per Serving ({serving_size_g}g): {food['calories']} kcal | "
                f"{food['protein']}g protein | {food['carbs']}g carbs | {food['fat']}g fat"
            )
            
            col1, col2 = st.columns([2, 1.2])
            
            with col1:
                button_cols = st.columns(5)
                for k in range(1, 6):
                    with button_cols[k - 1]:
                        btn_type = "primary" if current_serving == float(k) else "secondary"
                        if st.button(
                            f"{k}", key=f"{key_prefix}_serving_{k}", type=btn_type,
                            help=f"Set to {k} servings", use_container_width=True
                        ):
                            st.session_state.food_selections[food['name']] = float(k)
                            st.rerun()
            
            with col2:
                custom_serving = st.number_input(
                    "Custom", min_value=0.0, max_value=20.0,
                    value=float(current_serving), step=0.5,
                    key=f"{key_prefix}_custom_serving", label_visibility="collapsed"
                )
        
        else:
            # Grams mode
            # Calculate per 100g values
            cal_per_100g = (food['calories'] / serving_size_g) * 100
            protein_per_100g = (food['protein'] / serving_size_g) * 100
            carbs_per_100g = (food['carbs'] / serving_size_g) * 100
            fat_per_100g = (food['fat'] / serving_size_g) * 100
            
            st.caption(
                f"Per 100g: {cal_per_100g:.0f} kcal | {protein_per_100g:.1f}g protein | "
                f"{carbs_per_100g:.1f}g carbs | {fat_per_100g:.1f}g fat"
            )
            
            # Show conversion hint
            equivalent_servings = current_serving
            grams_equivalent = equivalent_servings * serving_size_g
            if grams_equivalent > 0:
                st.caption(f"💡 {grams_equivalent:.0f}g = {equivalent_servings:.1f} servings")
            
            col1, col2 = st.columns([2, 1.2])
            
            with col1:
                button_cols = st.columns(5)
                gram_amounts = [25, 50, 100, 150, 200]
                
                for i, grams in enumerate(gram_amounts):
                    with button_cols[i]:
                        # Convert grams to servings for comparison and storage
                        servings_equivalent = grams / serving_size_g
                        btn_type = "primary" if abs(current_serving - servings_equivalent) < 0.01 else "secondary"
                        
                        if st.button(
                            f"{grams}g", key=f"{key_prefix}_gram_{grams}", type=btn_type,
                            help=f"Set to {grams}g ({servings_equivalent:.1f} servings)", 
                            use_container_width=True
                        ):
                            st.session_state.food_selections[food['name']] = servings_equivalent
                            st.rerun()
            
            with col2:
                # Custom gram input
                current_grams = current_serving * serving_size_g
                custom_grams = st.number_input(
                    "Custom (g)", min_value=0.0, max_value=500.0,
                    value=float(current_grams), step=5.0,
                    key=f"{key_prefix}_custom_grams", label_visibility="collapsed"
                )
                custom_serving = custom_grams / serving_size_g

        # Handle input changes
        if current_mode == 'servings':
            if custom_serving != current_serving:
                if custom_serving > 0:
                    st.session_state.food_selections[food['name']] = custom_serving
                elif food['name'] in st.session_state.food_selections:
                    del st.session_state.food_selections[food['name']]
                st.rerun()
        else:
            if abs(custom_serving - current_serving) > 0.01:
                if custom_serving > 0:
                    st.session_state.food_selections[food['name']] = custom_serving
                elif food['name'] in st.session_state.food_selections:
                    del st.session_state.food_selections[food['name']]
                st.rerun()


def render_food_grid(items, category, columns=2):
    """Renders a grid of food items for a given category."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)


# # --------------------------------------------------------------------------- # #
# # Cell 8: Initialize Application                                                # #
# # --------------------------------------------------------------------------- # #

# ------ Initialize Session State ------
initialize_session_state()

# ------ Load Food Database and Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# ------ Apply Custom CSS for Enhanced Styling ------
st.markdown("""
<style>
    html { font-size: 100%; }
    [data-testid="InputInstructions"] { display: none; }
    .stButton>button[kind="primary"] {
        background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b;
    }
    .stButton>button[kind="secondary"] {
        border: 1px solid #ff6b6b; color: #333;
    }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    .stMetric > div > div > div > div { color: #262730; }
    .stProgress .st-bo { background-color: #e0e0e0; }
    .stProgress .st-bp { background-color: #ff6b6b; }
    .stCaption { color: #555555 !important; }
</style>
""", unsafe_allow_html=True)


# # --------------------------------------------------------------------------- # #
# # Cell 9: Application Title and Unified Input Interface                         # #
# # --------------------------------------------------------------------------- # #

# ------ Main Application Title and Introduction ------
st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
### A Smart, Evidence-Based Nutrition Tracker That Gets You

Welcome aboard! This is not just another calorie counter. It is your
personalized guide, built on rock-solid science to help you achieve your
goals. Whether you are aiming to shed a few pounds, maintain your current
physique, or build muscle, we have crunched the numbers so you can focus on
enjoying your food. Let us get started! 🚀
""")

# ------ Sidebar for User Input ------
st.sidebar.header("Let's Get Personal 📊")

# ------ Unit Selection Toggle ------
use_imperial = st.sidebar.toggle(
    "Use Imperial Units",
    value=(st.session_state.get('user_units', 'metric') == 'imperial'),
    key='units_toggle',
    help="Toggle on for Imperial (lbs, inches) or off for Metric (kg, cm)."
)
st.session_state.user_units = 'imperial' if use_imperial else 'metric'

# ------ Unified Input Form ------
all_inputs = {}
standard_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')
}
advanced_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')
}

for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
with advanced_expander:
    for field_name, field_config in advanced_fields.items():
        value = create_unified_input(field_name, field_config, advanced_expander)
        if 'convert' in field_config:
            value = field_config['convert'](value)
        all_inputs[field_name] = value

# ------ Form Submission and Validation ------
if st.sidebar.button("Calculate My Targets 🧮", type="primary", key="calculate_button"):
    validation_errors = validate_user_inputs(all_inputs)
    st.session_state.form_errors = validation_errors
    if not validation_errors:
        st.session_state.form_submitted = True
        st.session_state.show_motivational_message = True
    else:
        st.session_state.form_submitted = False
    st.rerun()

# ------ Display Validation Errors ------
if st.session_state.get('form_errors'):
    for error in st.session_state.form_errors:
        st.sidebar.error(f"• {error}")

st.sidebar.divider()

# ------ Save and Load Progress ------
st.sidebar.subheader("Save Your Progress 💾")
progress_json = save_progress_to_json(
    st.session_state.food_selections, all_inputs
)
st.sidebar.download_button(
    "Download 📥", 
    data=progress_json,
    file_name=f"nutrition_progress_{datetime.now():%Y%m%d_%H%M%S}.json",
    mime="application/json", 
    key="download_progress",
    type="primary"
)

# ------ Load Progress 📂 ------
st.sidebar.subheader("Load Your Progress 📂")
uploaded_file = st.sidebar.file_uploader(
    "Upload a progress file", type="json", key="upload_progress"
)
if uploaded_file is not None:
    if 'last_uploaded_file' not in st.session_state or st.session_state.last_uploaded_file != uploaded_file.name:
        content = uploaded_file.read().decode()
        food_selections, user_inputs = load_progress_from_json(content)
        st.session_state.food_selections.update(food_selections)
        for key, value in user_inputs.items():
            if f'user_{key}' in st.session_state:
                st.session_state[f'user_{key}'] = value
        st.session_state.last_uploaded_file = uploaded_file.name
        st.sidebar.success("Progress loaded successfully! 📂")
        st.rerun()

st.sidebar.divider()

# ------ Activity Level Guide in Sidebar ------
with st.sidebar.container(border=True):
    st.markdown("#### Your Activity Level, Decoded")
    for key in ACTIVITY_MULTIPLIERS:
        description = ACTIVITY_DESCRIPTIONS.get(key, "")
        st.markdown(f"* {description}")
    st.markdown(
        "💡 *If you are torn between two levels, pick the lower one. It is "
        "better to underestimate your calorie burn than to overestimate it.*"
    )

# ------ Dynamic Sidebar Summary ------
if st.session_state.form_submitted:
    final_values = get_final_values(all_inputs)
    targets = calculate_personalized_targets(**final_values)
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)

    st.sidebar.divider()
    st.sidebar.markdown("### Quick Summary 📊")
    progress_calories = (
        min(totals['calories'] / targets['total_calories'] * 100, 100)
        if targets['total_calories'] > 0 else 0
    )
    progress_protein = (
        min(totals['protein'] / targets['protein_g'] * 100, 100)
        if targets['protein_g'] > 0 else 0
    )

    st.sidebar.metric(
        "Calories Progress", f"{progress_calories:.0f}%",
        f"{totals['calories']:.0f} / {targets['total_calories']:.0f} kcal"
    )
    st.sidebar.metric(
        "Protein Progress", f"{progress_protein:.0f}%",
        f"{totals['protein']:.0f} / {targets['protein_g']:.0f} g"
    )

# ------ Process Final Values and Calculate Targets ------
final_values = get_final_values(all_inputs)
user_has_entered_info = st.session_state.form_submitted
targets = calculate_personalized_targets(**final_values)

# ------ Display Motivational Message on First Calculation ------
if st.session_state.show_motivational_message and user_has_entered_info:
    goal_messages = {
        'weight_loss': (
            f"🎉 Awesome! You are set up for success! With this plan, you are "
            f"on track to lose approximately "
            f"{abs(targets['estimated_weekly_change']):.2f} kg per week. "
            "Stay consistent, and you will achieve your goal! 💪"
        ),
        'weight_maintenance': (
            f"🎯 Perfect! Your maintenance plan is locked and loaded. You "
            f"are all set to maintain your current weight of "
            f"{format_weight(final_values['weight_kg'], st.session_state.get('user_units', 'metric'))} "
            "while optimizing your nutrition. ⚖️"
        ),
        'weight_gain': (
            f"💪 Let us grow! Your muscle-building journey starts now. You "
            f"are targeting a healthy gain of about "
            f"{targets['estimated_weekly_change']:.2f} kg per week. Fuel up "
            "and lift heavy! 🚀"
        )
    }
    message = goal_messages.get(
        targets['goal'],
        "🚀 You are all set! Let us achieve those nutrition goals!"
    )
    st.success(message)

    if st.button("Got It! ✨", key="dismiss_message", type="primary"):
        st.session_state.show_motivational_message = False
        st.rerun()


# # --------------------------------------------------------------------------- # #
# # Cell 10: Unified Target Display System                                        # #
# # --------------------------------------------------------------------------- # #

if not user_has_entered_info:
    st.info(
        "👈 Please enter your details in the sidebar and click 'Calculate My "
        "Targets' to get your personalized daily recommendations."
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
    "🎯 **The 80/20 Rule**: Aim to meet your targets about 80 percent of "
    "the time. This gives you flexibility for social events and helps you "
    "build a sustainable, long-term habit."
)

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Unified Metrics Display Configuration ------
units_display = st.session_state.get('user_units', 'metric')
weight_display = format_weight(final_values['weight_kg'], units_display)

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5, 'metrics': [
            ("Weight", weight_display), ("BMR", f"{targets['bmr']} kcal"),
            ("TDEE", f"{targets['tdee']} kcal"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
            ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kg")
        ]
    },
    {
        'title': 'Your Daily Nutrition Targets', 'columns': 5, 'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            ("Protein", f"{targets['protein_g']} g",
             f"{targets['protein_percent']:.0f}% of your calories"),
            ("Carbohydrates", f"{targets['carb_g']} g",
             f"{targets['carb_percent']:.0f}% of your calories"),
            ("Fat", f"{targets['fat_g']} g",
             f"{targets['fat_percent']:.0f}% of your calories"),
            ("Water", f"{hydration_ml} ml", f"~{hydration_ml/250:.1f} cups")
        ]
    }
]

# ------ Display All Metric Sections ------
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()


# # --------------------------------------------------------------------------- # #
# # Cell 11: Enhanced Evidence-Based Tips and Context                             # #
# # --------------------------------------------------------------------------- # #

with st.expander("📚 Your Evidence-Based Game Plan", expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "The Big Three to Win at Nutrition 🏆",
        "Level Up Your Progress Tracking 📊",
        "Mindset Is Everything 🧠",
        "Own Your Energy 🧗",
        "The Science Behind the Magic 🔬"
    ])

    with tab1:
        st.header("Master Your Hydration Game 💧")
        st.markdown("""
        * **Daily Goal**: Shoot for approximately 35 ml per kilogram of your
          body weight daily.
        * **Training Bonus**: Add an extra 500 to 750 ml per hour of exercise.
        * **Fat Loss Hack**: Drinking 500 ml of water before meals can
          increase feelings of fullness by 13 percent.
        """)
        st.divider()
        st.header("Sleep Like Your Goals Depend on It 😴")
        st.markdown("""
        * **The Shocking Truth**: Getting less than seven hours of sleep can
          reduce fat loss by more than half.
        * **Daily Goal**: Aim for seven to nine hours and try to maintain a
          consistent schedule.
        * **Set the Scene**: Keep your room dark, cool (18-20°C), and avoid
          screens for at least an hour before bedtime.
        """)
        st.divider()
        st.header("Follow Your Wins 📅")
        st.markdown("""
        * **Morning Ritual**: Weigh yourself first thing in the morning after
          using the bathroom and before eating or drinking.
        * **Look for Trends, Not Blips**: Monitor your weekly average weight
          instead of focusing on daily fluctuations.
        * **Hold the Line**: Do not adjust your plan too quickly. Wait for at
          least two weeks of stalled progress before making changes.
        """)

    with tab2:
        st.header("Go Beyond the Scale 📸")
        st.markdown("""
        * **The Bigger Picture**: Take progress photos every month. Use the
          same pose, lighting, and time of day for consistency.
        * **Size Up Your Wins**: Measure your waist, hips, arms, and thighs
          monthly to track changes in body composition.
        * **The Quiet Victories**: Pay attention to non-scale victories like
          increased energy levels, better sleep quality, and improved gym
          performance.
        """)

    with tab3:
        st.header("Mindset Is Everything 🧠")
        st.markdown(
            "The 80/20 principle is your best defense against the "
            "perfectionist trap. It is about ditching the all-or-nothing "
            "mindset. Build your habits gradually, and you will be far more "
            "likely to stick with them for the long haul."
        )
        st.subheader("Start Small, and Win Big")
        st.markdown("""
        * **Weeks 1–2**: Your only job is to focus on hitting your calorie targets.
        * **Weeks 3–4**: Once calories feel natural, start layering in protein.
        * **Week 5 and Beyond**: With calories and protein managed, you can
          now fine-tune your carbohydrate and fat intake.
        """)
        st.divider()
        st.subheader("When Progress Stalls 🔄")
        st.markdown("#### Did You Hit a Weight Loss Plateau?")
        st.markdown("""
        * **Guess Less, Stress Less**: Before you do anything else, double-check
          how accurately you are logging your food.
        * **Activity Audit**: Take a fresh look at your activity level.
        * **Walk It Off**: Try adding 10 to 15 minutes of walking to your daily
          routine before reducing calories further.
        * **Step Back to Leap Forward**: Consider a diet break every six to
          eight weeks. Eating at maintenance calories for one or two weeks
          can provide a metabolic and psychological reset.
        * **Increase Food Volume**: Load your plate with low-calorie,
          high-volume foods like leafy greens and berries.
        """)
        st.markdown("#### Are You Struggling to Gain Weight?")
        st.markdown("""
        * **Drink Your Calories**: Liquid calories from smoothies, milk, and
          protein shakes are easier to consume than another full meal.
        * **Fat Is Your Fuel**: Increase healthy fats like nuts, oils, and avocados.
        * **Push Your Limits**: Ensure you are consistently challenging
          yourself in the gym to stimulate muscle growth.
        * **Turn Up the Heat**: If you have been stuck for over two weeks,
          increase your intake by 100 to 150 calories.
        """)
        st.divider()
        st.subheader("Pace Your Protein 💪")
        st.markdown("""
        * **Spread the Love**: Aim for 20 to 40 grams of protein with each of
          your three to four daily meals.
        * **Frame Your Fitness**: Consume carbohydrates and 20 to 40 grams
          of protein before and within two hours of your workout.
        * **The Night Shift**: Consider 20 to 30 grams of casein protein
          before bed to support muscle recovery overnight.
        """)

    with tab4:
        st.header("Own Your Energy 🧗")
        st.subheader("Build Your Foundation with Resistance Training 💪")
        st.markdown("""
        This is your non-negotiable, no matter your goal. Lifting weights
        tells your body to build or hold onto precious muscle, which is the
        engine of your metabolism.

        * **For Fat Loss**: More muscle means you burn more calories at rest.
        * **For Bulking Up**: Exercise signals where to direct the protein.
        * **The Game Plan**: Start with two to three sessions of 20 to 40
          minutes per week.
        * **Find What You Love**: Fitness should be enjoyable. Dance, bike, or
          hike. Pick an activity that makes you happy.
        """)
        st.divider()
        st.subheader("Use NEAT as Your Fitness Piggy Bank 🏃")
        st.markdown("""
        NEAT stands for Non-Exercise Activity Thermogenesis. It is the
        calories you burn just by living your life.

        * Adding just **10 to 20 minutes of walking** to your day can be the
          difference between a plateau and progress.
        """)

    with tab5:
        st.header("Understanding Your Metabolism 🔬")
        st.markdown(
            "Your Basal Metabolic Rate (BMR) is the energy your body needs "
            "just to keep the lights on. Your Total Daily Energy Expenditure "
            "(TDEE) builds on that baseline by factoring in how active you "
            "are throughout the day."
        )
        st.divider()
        st.subheader("The Smart Eater's Cheat Sheet 🍽️")
        st.markdown("""
        * **Protein**: Protein is the king of satiety. It digests slowly and
          steadies blood sugar.
        * **Fiber-Rich Carbohydrates**: Vegetables, fruits, and whole grains
          fill you up without adding excessive calories.
        * **Healthy Fats**: Nuts, olive oil, and avocados provide steady,
          long-lasting energy.
        * **Processed Foods**: These are fine occasionally but should not be
          the foundation of your diet.

        Aim for 14 grams of fiber for every 1,000 calories you consume.
        """)
        st.divider()
        st.subheader("Your Nutritional Supporting Cast 🌱")
        st.markdown("""
        A plant-based diet requires attention to a few key micronutrients.

        **The Watch List:**

        * **B₁₂**: Essential for cell and nerve function, B₁₂ is almost
          exclusively found in animal products. A supplement is often necessary.
        * **Iron**: Iron transports oxygen throughout your body. Pair
          plant-based sources with vitamin C to enhance absorption.
        * **Calcium**: Crucial for bones and muscles, calcium can be found in
          kale, almonds, tofu, and fortified plant milks.
        * **Zinc**: Important for your immune system, zinc is found in nuts,
          seeds, and whole grains.
        * **Iodine**: Your thyroid needs iodine to regulate your metabolism.
          A pinch of iodized salt is usually sufficient.
        * **Omega-3s (EPA/DHA)**: These fats support brain and heart health.
          Consider supplements if you do not consume fish.

        Fortified foods and targeted supplements can help fill any gaps.
        It is always best to consult with a healthcare provider.
        """)


# # --------------------------------------------------------------------------- # #
# # Cell 12: Food Selection Interface                                             # #
# # --------------------------------------------------------------------------- # #

st.header("Track Your Daily Intake 🥗")

# ------ Food Search and Reset Functionality ------
search_col, reset_col = st.columns([3, 1])
with search_col:
    search_term = st.text_input(
        "Search for foods", placeholder="🔍 Type a food name to filter results...",
        key="food_search_input", label_visibility="collapsed"
    )
    st.session_state.food_search = search_term

with reset_col:
    if st.button("Clear Search 🔄", key="clear_search", type="primary"):
        st.session_state.food_search = ""
        st.rerun()

st.markdown(
    "Select the number of servings for each food to see how your choices "
    "compare to your daily targets. Use the toggle switch (🥄 Servings | ⚖️ 100g) "
    "to switch between serving-based and gram-based input modes."
)

# ------ Emoji Guide Expander ------
with st.expander("💡 Need help with food choices? Check the emoji guide below!"):
    for emoji, tooltip in EMOJI_TOOLTIPS.items():
        label = tooltip.split(':')[0]
        description = ':'.join(tooltip.split(':')[1:]).strip()
        st.markdown(f"* **{emoji} {label}**: {description}")

if st.button(
    "Start Fresh: Reset All Food Selections 🔄", type="primary", key="reset_foods"
):
    st.session_state.food_selections = {}
    st.session_state.food_input_modes = {}
    st.rerun()

# ------ Filter and Display Food Items ------
filtered_foods = filter_foods_by_search(foods, search_term)

if not filtered_foods and search_term:
    st.warning(
        f"No foods were found matching '{search_term}'. Please try a "
        "different search term or clear the search. 🙁"
    )
elif filtered_foods:
    available_categories = [
        cat for cat, items in sorted(filtered_foods.items()) if items
    ]
    tabs = st.tabs(available_categories)

    for i, category in enumerate(available_categories):
        items = filtered_foods[category]
        sorted_items = sorted(
            items,
            key=lambda x: (
                CONFIG['emoji_order'].get(x.get('emoji', ''), 4),
                -x['calories']
            )
        )
        with tabs[i]:
            render_food_grid(sorted_items, category, columns=2)


# # --------------------------------------------------------------------------- # #
# # Cell 13: Daily Summary and Progress Tracking                                  # #
# # --------------------------------------------------------------------------- # #

st.header("Today's Scorecard 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)

    # ------ Export and Share Functionality ------
    st.subheader("Export Your Summary 📥")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Download PDF Report 📄", key="export_pdf", use_container_width=True):
            pdf_buffer = create_pdf_summary(
                totals, targets, selected_foods, final_values
            )
            st.download_button(
                "Download PDF 📥", data=pdf_buffer,
                file_name=f"nutrition_summary_{datetime.now():%Y%m%d}.pdf",
                mime="application/pdf", key="download_pdf_button"
            )
    with col2:
        if st.button("Download CSV Data 📊", key="export_csv", use_container_width=True):
            csv_data = create_csv_summary(totals, targets, selected_foods)
            st.download_button(
                "Download CSV 📥", data=csv_data,
                file_name=f"nutrition_data_{datetime.now():%Y%m%d}.csv",
                mime="text/csv", key="download_csv_button"
            )
    with col3:
        if st.button("Share Progress 📱", key="share_progress", use_container_width=True):
            share_text = f"""
My Nutrition Progress - {datetime.now():%Y-%m-%d} 🍽️

📊 Today's Intake:
- Calories: {totals['calories']:.0f} / {targets['total_calories']:.0f} kcal
- Protein: {totals['protein']:.0f} / {targets['protein_g']:.0f} g
- Carbohydrates: {totals['carbs']:.0f} / {targets['carb_g']:.0f} g
- Fat: {totals['fat']:.0f} / {targets['fat_g']:.0f} g

Created with the Personal Nutrition Coach.
            """
            st.info("Please copy the summary below to share! 📋")
            st.text_area(
                "Shareable Summary:", share_text, height=200,
                label_visibility="collapsed"
            )

    st.divider()
    col1, col2 = st.columns([1, 1])

    # ------ Enhanced Visualization Dashboard ------
    with col1:
        st.subheader("Your Daily Fuel Mix")
        fig_macros = go.Figure()
        
        macros = ['Protein (g)', 'Carbohydrates (g)', 'Fat (g)', 'Calories (kcal)']
        actual_values = [totals['protein'], totals['carbs'], totals['fat'], totals['calories']]
        target_values = [targets['protein_g'], targets['carb_g'], targets['fat_g'], targets['total_calories']]

        fig_macros.add_trace(go.Bar(
            name='Actual', x=macros, y=actual_values, marker_color='#ff6b6b'
        ))
        fig_macros.add_trace(go.Bar(
            name='Target', x=macros, y=target_values, marker_color='#4ecdc4'
        ))
        fig_macros.update_layout(
            title_text='Nutrition Report Card', barmode='group',
            yaxis_title='Amount', height=400, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_macros, use_container_width=True)

    with col2:
        st.subheader("Your Macronutrient Split")
        if totals['calories'] > 0:
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Protein', 'Carbohydrates', 'Fat'],
                values=[
                    totals['protein'] * 4, totals['carbs'] * 4,
                    totals['fat'] * 9
                ],
                hole=0.4, marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1']
            )])
            fig_pie.update_layout(
                title=f'Total: {totals["calories"]:.0f} kcal', height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("Please select some foods to see the macronutrient split.")

    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)

    # ------ Food Selection Summary Table ------
    with st.expander("Your Food Choices Today", expanded=True):
        st.subheader("What You Have Logged")
        prepared_data = prepare_summary_data(totals, targets, selected_foods)
        consumed_foods_list = prepared_data['consumed_foods']
        if consumed_foods_list:
            display_data = [{
                'Food': item['name'], 'Servings': f"{item['servings']:.1f}",
                'Calories (kcal)': f"{item['calories']:.0f}",
                'Protein (g)': f"{item['protein']:.1f}",
                'Carbohydrates (g)': f"{item['carbs']:.1f}",
                'Fat (g)': f"{item['fat']:.1f}"
            } for item in consumed_foods_list]
            df_summary = pd.DataFrame(display_data)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        else:
            st.caption("No foods have been logged yet.")

else:
    st.info(
        "You have not selected any foods yet. Please add some items from the "
        "categories above to start tracking your intake! 🍎"
    )
    st.subheader("Progress Snapshot")
    render_progress_bars(totals, targets)


# # --------------------------------------------------------------------------- # #
# # Cell 14: User Feedback Section                                                # #
# # --------------------------------------------------------------------------- # #

st.divider()
st.header("Help Us Improve! 💬")
st.markdown(
    "Your feedback helps us make this app even better. Please share your "
    "thoughts below."
)

with st.form("feedback_form", clear_on_submit=True):
    feedback_type = st.selectbox(
        "What type of feedback would you like to share?",
        ["General Feedback", "Bug Report", "Feature Request", "Success Story"],
        key="feedback_type"
    )
    feedback_text = st.text_area(
        "How can we improve?",
        placeholder=(
            "Tell us about your experience, suggest new features, or "
            "report any issues you encountered..."
        ),
        height=100,
        key="feedback_text"
    )
    if st.form_submit_button("Submit Feedback 📤", type="primary"):
        if feedback_text.strip():
            # In a real app, this would save to a database or send an email
            st.success(
                f"Thank you for your {feedback_type.lower()}! Your input "
                "helps us improve the app for everyone. 🙏"
            )
        else:
            st.error("Please enter some feedback before submitting. 🙁")


# # --------------------------------------------------------------------------- # #
# # Cell 15: Footer and Additional Resources                                      # #
# # --------------------------------------------------------------------------- # #

st.divider()
st.markdown("""
### The Science We Stand On 📚

This tracker is not built on guesswork. It is grounded in peer-reviewed
research and evidence-based guidelines. We rely on the **Mifflin-St Jeor
equation** to calculate your Basal Metabolic Rate (BMR), a method that is
widely regarded as the gold standard and endorsed by the Academy of Nutrition
and Dietetics. To estimate your Total Daily Energy Expenditure (TDEE), we use
well-established activity multipliers from exercise physiology research. For
protein recommendations, our targets are based on official guidelines from the
International Society of Sports Nutrition. All calorie adjustments adhere to
conservative, sustainable rates that research has consistently shown lead to
lasting, meaningful results.

### The Fine Print ⚠️

Think of this tool as your launchpad, but remember that everyone is different.
Your mileage may vary due to genetics, health conditions, medications, and
other factors that a calculator cannot account for. It is always wise to
consult a qualified healthcare provider before making significant dietary
shifts. Above all, tune into your body by keeping tabs on your energy levels
and performance, and adjust things as needed. We are here to help, but you
know yourself best!
""")

st.success(
    "You made it to the finish line! Thanks for joining us on this nutrition "
    "adventure. Keep showing up for yourself. You have got this! 🥳"
)


# # --------------------------------------------------------------------------- # #
# # Cell 16: Session State Management and Performance                             # #
# # --------------------------------------------------------------------------- # #

# ------ Clean Up Session State to Prevent Memory Issues ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# ------ Session State Cleanup for Unused Variables ------
# Remove any temporary variables that might accumulate over time
temp_keys = [key for key in st.session_state.keys() if key.startswith('temp_')]
for key in temp_keys:
    del st.session_state[key]
