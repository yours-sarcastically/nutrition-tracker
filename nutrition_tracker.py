#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker Using Streamlit And Mifflin St Jeor Equations
# -----------------------------------------------------------------------------

"""
This script provides an interactive nutrition tracking interface implemented
with Streamlit. Its purpose is to calculate individualized daily calorie and
macronutrient targets using the Mifflin St Jeor basal metabolic rate (BMR)
equations and standard activity multipliers to estimate total daily energy
expenditure (TDEE). It supports goal-specific caloric adjustments for weight
loss, maintenance, or gain and assists users in logging vegetarian food
choices toward those targets.

Algorithmic Overview
1. Basal Metabolic Rate (Mifflin St Jeor):
   Male: BMR = 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
   Female: BMR = 10 * weight_kg + 6.25 * height_cm - 5 * age - 161
2. Total Daily Energy Expenditure:
   TDEE = BMR * activity_multiplier
   Activity multipliers follow widely adopted reference values ranging from
   1.2 (sedentary) to 1.9 (extremely active)
3. Goal Adjustment:
   Caloric targets are adjusted by fixed percentages relative to TDEE for
   weight change or maintenance scenarios
4. Macronutrient Distribution:
   a. Protein grams = protein_per_kg * weight_kg
   b. Fat calories = fat_percentage * total_calories
      Fat grams = fat_calories / 9
   c. Carbohydrate calories = total_calories - (protein_calories + fat_calories)
      Carbohydrate grams = carbohydrate_calories / 4
5. Estimated Weekly Weight Change:
   Uses 7,700 kilocalories ≈ 1 kilogram body mass change approximation:
   weekly_change_kg = (daily_caloric_adjustment * 7) / 7700

Features
- User input via sidebar with validation
- Metric or imperial unit handling for height and weight input (internal
  storage remains metric)
- Automatic goal-specific defaults for protein and fat if advanced values
  are not supplied
- Vegetarian food database ingestion (CSV) with serving-based nutrient
  accumulation
- Dynamic progress bars for calories, protein, carbohydrates, and fat
- Recommendation logic that identifies foods aiding multiple nutrient
  deficits
- Export functions for daily summaries (PDF and CSV)
- Session state persistence for user inputs and food selections
- JSON-based save and load of progress
- Hydration estimation based on weight and activity
- Evidence-based contextual guidance displayed through expandable sections

Data Flow Summary
1. User enters demographic and goal data in sidebar
2. Script validates inputs and calculates BMR, TDEE, and macro targets
3. User selects foods; each food contributes per-serving energy and macro data
4. Aggregated totals are compared to targets and displayed with progress
5. Optional recommendations suggest foods covering multiple remaining gaps
6. User may export results or save/load progress

Usage
Command line execution:
    streamlit run nutrition_app.py

Required dependencies:
    streamlit
    pandas
    plotly
    reportlab

CSV Food File Format (nutrition_results.csv expected columns):
    name, category, serving_unit, calories, protein, carbs, fat

Interface Workflow
1. Open the application
2. Enter personal data and click Calculate My Targets
3. Browse foods, set servings, and monitor real-time progress
4. Use export buttons to download a PDF or CSV
5. Save or load a session via JSON if desired

Command-Line Invocation
    streamlit run nutrition_app.py

No additional command-line arguments are implemented. All interaction occurs
through the graphical interface.

Limitations
- The weight change estimate is an approximation and does not account for
  adaptive metabolic responses
- Activity multipliers are generalized and may not reflect individual
  variability
- Food database accuracy depends on source data integrity

This module is structured into notebook-style cells with explicit section
dividers to support Jupyter-like organization inside a single script file.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries And Modules
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
# Cell 2: Page Configuration And Initial Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Your Personal Nutrition Coach 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cell 3: Unified Configuration Constants
# -----------------------------------------------------------------------------

# ------ Default Parameter Values Based On Research ------
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

# ------ Activity Level Multipliers For TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Activity Level Descriptions ------
ACTIVITY_DESCRIPTIONS = {
    'sedentary': "🧑‍💻 Sedentary: Desk dominated routine",
    'lightly_active': "🏃 Lightly Active: Activity one to three days per week",
    'moderately_active': "🚴 Moderately Active: Activity three to five days per week",
    'very_active': "🏋️ Very Active: Intense training six to seven days per week",
    'extremely_active': "🤸 Extremely Active: Physical job or multiple daily sessions"
}

# ------ Goal Specific Targets ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
}

# ------ Unified Configuration For App Components ------
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
            'unit': 'g', 'label': 'Protein',
            'target_key': 'protein_g'
        },
        'carbs': {
            'unit': 'g', 'label': 'Carbohydrates',
            'target_key': 'carb_g'
        },
        'fat': {
            'unit': 'g', 'label': 'Fat',
            'target_key': 'fat_g'
        }
    },
    'form_fields': {
        'age': {
            'type': 'number', 'label': 'Age (in Years)',
            'min': 16, 'max': 80, 'step': 1,
            'placeholder': 'Enter your age', 'required': True
        },
        'height_cm': {
            'type': 'number', 'label': 'Height (in Centimeters)',
            'min': 140, 'max': 220, 'step': 1,
            'placeholder': 'Enter your height', 'required': True
        },
        'weight_kg': {
            'type': 'number', 'label': 'Weight (in Kilograms)',
            'min': 40.0, 'max': 150.0, 'step': 0.5,
            'placeholder': 'Enter your weight', 'required': True
        },
        'sex': {
            'type': 'selectbox', 'label': 'Biological Sex',
            'options': ["Male", "Female"], 'required': True
        },
        'activity_level': {
            'type': 'selectbox', 'label': 'Activity Level',
            'options': [
                ("Sedentary", "sedentary"),
                ("Lightly Active", "lightly_active"),
                ("Moderately Active", "moderately_active"),
                ("Very Active", "very_active"),
                ("Extremely Active", "extremely_active")
            ],
            'required': True
        },
        'goal': {
            'type': 'selectbox', 'label': 'Your Goal',
            'options': [
                ("Weight Loss", "weight_loss"),
                ("Weight Maintenance", "weight_maintenance"),
                ("Weight Gain", "weight_gain")
            ],
            'required': True
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
            'type': 'number', 'label': 'Fat Intake (% of Calories)',
            'min': 15, 'max': 40, 'step': 1,
            'help': (
                'Set the share of your daily calories that should come from '
                'fats'
            ),
            'convert': lambda x: x / 100 if x else None,
            'advanced': True, 'required': False
        }
    }
}

# ------ Emoji Tooltips ------
EMOJI_TOOLTIPS = {
    '🥇': (
        'Gold Medal: High in its target nutrient and relatively calorie '
        'efficient'
    ),
    '🔥': 'High Calorie: Higher energy density within its group',
    '💪': 'High Protein: Higher protein content relative to peers',
    '🍚': 'High Carb: Higher carbohydrate content relative to peers',
    '🥑': 'High Fat: Higher fat content relative to peers'
}

# ------ Metric Tooltips ------
METRIC_TOOLTIPS = {
    'BMR': (
        'Basal Metabolic Rate is the energy needed for essential functions'
    ),
    'TDEE': (
        'Total Daily Energy Expenditure equals BMR plus activity energy'
    ),
    'Caloric Adjustment': (
        'Difference between TDEE and assigned goal calories'
    ),
    'Protein': 'Protein supports muscle maintenance and repair',
    'Carbohydrates': 'Carbohydrates supply energy for brain and muscle use',
    'Fat': 'Fat supports hormone production and nutrient absorption'
}

# ------ Centralized Tip Content ------
TIPS_CONTENT = {
    'hydration': [
        "Daily Goal: About 35 milliliters per kilogram of body weight",
        "Training Bonus: Add 500 to 750 milliliters per training hour",
        (
            "Pre Meal Water: 500 milliliters before meals may increase "
            "fullness"
        )
    ]
}

# -----------------------------------------------------------------------------
# Cell 4: Unit Conversion Functions
# -----------------------------------------------------------------------------

def kg_to_lbs(kg):
    """Convert kilograms to pounds."""
    return kg * 2.20462 if kg else 0


def lbs_to_kg(lbs):
    """Convert pounds to kilograms."""
    return lbs / 2.20462 if lbs else 0


def cm_to_inches(cm):
    """Convert centimeters to inches."""
    return cm / 2.54 if cm else 0


def inches_to_cm(inches):
    """Convert inches to centimeters."""
    return inches * 2.54 if inches else 0


def format_weight(weight_kg, units):
    """Format weight based on unit preference."""
    if units == 'imperial':
        return f"{kg_to_lbs(weight_kg):.1f} lbs"
    return f"{weight_kg:.1f} kg"


def format_height(height_cm, units):
    """Format height based on unit preference."""
    if units == 'imperial':
        total_inches = cm_to_inches(height_cm)
        feet = int(total_inches // 12)
        inches = total_inches % 12
        return f"{feet}'{inches:.0f}\""
    return f"{height_cm:.0f} cm"


# -----------------------------------------------------------------------------
# Cell 5: Unified Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize required session state variables."""
    session_vars = (
        ['food_selections', 'form_submitted', 'show_motivational_message',
         'food_search', 'form_errors'] +
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
            elif var == 'form_errors':
                st.session_state[var] = []
            else:
                st.session_state[var] = None


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Create an input widget from configuration."""
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
            index = next(
                (i for i, (_, val) in enumerate(options)
                 if val == current_value),
                next(
                    (i for i, (_, val) in enumerate(options)
                     if val == DEFAULTS[field_name]),
                    0
                )
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
            index = (
                options.index(current_value)
                if current_value in options else 0
            )
            value = container.selectbox(
                field_config['label'],
                options,
                index=index,
                key=widget_key
            )
    st.session_state[session_key] = value
    return value


def validate_user_inputs(user_inputs):
    """Validate required user inputs and return messages."""
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
    """Process user inputs and apply defaults where needed."""
    final_values = {}
    for field, value in user_inputs.items():
        final_values[field] = value if value is not None else DEFAULTS[field]
    goal = final_values['goal']
    if goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[goal]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']
    return final_values


def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculate daily fluid needs in milliliters."""
    base_needs = weight_kg * 35
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
    """Display metrics in a grid layout."""
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
    """Return color indicator emoji based on percentage."""
    if percent >= 80:
        return "🟢"
    if percent >= 50:
        return "🟡"
    return "🔴"


def render_progress_bars(totals, targets):
    """Render progress bars for nutrients."""
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals.get(nutrient, 0)
        target = targets.get(config['target_key'], 1)
        target = target if target > 0 else 1
        percent = min((actual / target) * 100, 100)
        color_indicator = get_progress_color(percent)
        st.progress(
            percent / 100,
            text=(
                f"{color_indicator} {config['label']}: "
                f"{percent:.0f}% of daily target "
                f"({target:.0f} {config['unit']})"
            )
        )


def create_progress_tracking(totals, targets, foods):
    """Generate progress bars and recommendations."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")
    render_progress_bars(totals, targets)
    purpose_map = {
        'calories': 'to reach the target',
        'protein': 'for muscle support',
        'carbs': 'for energy',
        'fat': 'for hormone support'
    }
    deficits = {}
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        if actual < target:
            deficit = target - actual
            deficits[nutrient] = {
                'amount': deficit,
                'unit': config['unit'],
                'label': config['label'].lower(),
                'purpose': purpose_map.get(
                    nutrient, 'for nutrition support'
                )
            }
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
                f"{deficit_info['amount']:.0f}g more "
                f"{deficit_info['label']} {deficit_info['purpose']}"
            )
        if len(deficit_summary) > 1:
            summary_text = (
                "You still need: " +
                ", ".join(deficit_summary[:-1]) +
                f", and {deficit_summary[-1]}"
            )
        else:
            summary_text = f"You still need: {deficit_summary[0]}"
        recommendations.append(summary_text)
        if top_suggestions:
            first_suggestion = top_suggestions[0]
            food = first_suggestion['food']
            nutrients_helped = first_suggestion['nutrients_helped']
            nutrient_benefits = [
                f"{food[n]:.0f}g {n}" for n in nutrients_helped
            ]
            if len(nutrient_benefits) > 1:
                benefits_text = (
                    ", ".join(nutrient_benefits[:-1]) +
                    f", and {nutrient_benefits[-1]}"
                )
            else:
                benefits_text = nutrient_benefits[0]
            recommendations.append(
                f"Suggested Food: One serving of {food['name']} adds "
                f"{benefits_text}"
            )
            if len(top_suggestions) > 1:
                alternative_foods = []
                for suggestion in top_suggestions[1:]:
                    food = suggestion['food']
                    nutrients_helped = suggestion['nutrients_helped']
                    nutrient_benefits = [
                        f"{food[n]:.0f}g {n}" for n in nutrients_helped
                    ]
                    if len(nutrient_benefits) > 1:
                        benefits_text = (
                            ", ".join(nutrient_benefits[:-1]) +
                            f", and {nutrient_benefits[-1]}"
                        )
                    else:
                        benefits_text = nutrient_benefits[0]
                    alternative_foods.append(
                        f"{food['name']} (adds {benefits_text})"
                    )
                if len(alternative_foods) == 1:
                    alternatives_text = alternative_foods[0]
                elif len(alternative_foods) == 2:
                    alternatives_text = (
                        f"{alternative_foods[0]} or {alternative_foods[1]}"
                    )
                else:
                    alternatives_text = (
                        ", ".join(alternative_foods[:-1]) +
                        f", or {alternative_foods[-1]}"
                    )
                recommendations.append(
                    f"Alternative Options: {alternatives_text}"
                )
        else:
            biggest_deficit = max(
                deficits.items(), key=lambda x: x[1]['amount']
            )
            nutrient, deficit_info = biggest_deficit
            all_foods = [item for sublist in foods.values()
                         for item in sublist]
            best_single_food = max(
                all_foods,
                key=lambda x: x.get(nutrient, 0),
                default=None
            )
            if best_single_food and best_single_food.get(nutrient, 0) > 0:
                recommendations.append(
                    "Consider Adding: "
                    f"{best_single_food['name']} contributes "
                    f"{best_single_food[nutrient]:.0f}g of "
                    f"{deficit_info['label']}"
                )
    return recommendations


def calculate_daily_totals(food_selections, foods):
    """Calculate total nutrient intake from selected foods."""
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
    """Serialize current progress to JSON."""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'food_selections': food_selections,
        'user_inputs': user_inputs
    }
    return json.dumps(progress_data, indent=2)


def load_progress_from_json(json_data):
    """Deserialize progress from JSON content."""
    try:
        data = json.loads(json_data)
        return data.get('food_selections', {}), data.get('user_inputs', {})
    except json.JSONDecodeError:
        return {}, {}


def prepare_summary_data(totals, targets, selected_foods):
    """Prepare standardized summary structure for export."""
    summary_data = {
        'nutrition_summary': [],
        'consumed_foods': []
    }
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
    """Create a PDF summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    buffer = io.BytesIO()
    pdf_canvas = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf_canvas.setFont("Helvetica-Bold", 16)
    pdf_canvas.drawString(50, height - 50, "Daily Nutrition Summary")
    pdf_canvas.setFont("Helvetica", 12)
    pdf_canvas.drawString(
        50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d')}"
    )
    y_pos = height - 120
    pdf_canvas.drawString(
        50, y_pos, f"Age: {user_info.get('age', 'N/A')}"
    )
    pdf_canvas.drawString(
        200, y_pos, f"Weight: {user_info.get('weight_kg', 'N/A')} kg"
    )
    pdf_canvas.drawString(
        350, y_pos, f"Goal: {user_info.get('goal', 'N/A')}"
    )
    y_pos -= 40
    pdf_canvas.setFont("Helvetica-Bold", 14)
    pdf_canvas.drawString(50, y_pos, "Nutrition Summary")
    y_pos -= 30
    pdf_canvas.setFont("Helvetica", 12)
    for item in summary_data['nutrition_summary']:
        pdf_canvas.drawString(
            50, y_pos,
            f"{item['label']}: {item['actual']:.0f}/"
            f"{item['target']:.0f} {item['unit']} "
            f"({item['percent']:.0f}%)"
        )
        y_pos -= 20
    if summary_data['consumed_foods']:
        y_pos -= 20
        pdf_canvas.setFont("Helvetica-Bold", 14)
        pdf_canvas.drawString(50, y_pos, "Foods Consumed")
        y_pos -= 30
        pdf_canvas.setFont("Helvetica", 10)
        for item in summary_data['consumed_foods'][:20]:
            pdf_canvas.drawString(
                50, y_pos,
                f"• {item['name']}: {item['servings']} serving(s)"
            )
            y_pos -= 15
            if y_pos < 50:
                break
    pdf_canvas.save()
    buffer.seek(0)
    return buffer


def create_csv_summary(totals, targets, selected_foods):
    """Create a CSV summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    data = []
    for item in summary_data['nutrition_summary']:
        data.append({
            'Category': 'Nutrition Summary',
            'Item': item['label'],
            'Actual': f"{item['actual']:.0f} {item['unit']}",
            'Target': f"{item['target']:.0f} {item['unit']}",
            'Percentage': f"{item['percent']:.0f}%"
        })
    for item in summary_data['consumed_foods']:
        data.append({
            'Category': 'Foods Consumed',
            'Item': item['name'],
            'Servings': item['servings'],
            'Calories': f"{item['calories']:.0f} kcal",
            'Protein': f"{item['protein']:.1f} g",
            'Carbs': f"{item['carbs']:.1f} g",
            'Fat': f"{item['fat']:.1f} g"
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


# -----------------------------------------------------------------------------
# Cell 6: Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculate BMR using the Mifflin St Jeor equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculate TDEE from BMR and activity level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Estimate weekly weight change from daily caloric adjustment."""
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(
    age, height_cm, weight_kg, sex='male',
    activity_level='moderately_active', goal='weight_gain',
    protein_per_kg=None, fat_percentage=None
):
    """Calculate daily personalized nutrient targets."""
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
        'bmr': round(bmr),
        'tdee': round(tdee),
        'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g),
        'protein_calories': round(protein_calories),
        'fat_g': round(fat_g),
        'fat_calories': round(fat_calories),
        'carb_g': round(carb_g),
        'carb_calories': round(carb_calories),
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
# Cell 7: Food Database Processing Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """Load vegetarian food database from CSV."""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in df['category'].unique()}
    for _, row in df.iterrows():
        category = row['category']
        if category in foods:
            foods[category].append({
                'name': f"{row['name']} ({row['serving_unit']})",
                'calories': row['calories'],
                'protein': row['protein'],
                'carbs': row['carbs'],
                'fat': row['fat']
            })
    return foods


@st.cache_data
def assign_food_emojis(foods):
    """Assign emojis to foods using ranking rules."""
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
        'high_cal_nutrient': '🥇',
        'high_calorie': '🔥',
        'protein': '💪',
        'carbs': '🍚',
        'fat': '🥑'
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
    """Filter foods based on search term."""
    if not search_term:
        return foods
    filtered_foods = {}
    search_lower = search_term.lower()
    for category, items in foods.items():
        filtered_items = [
            food for food in items if search_lower in food['name'].lower()
        ]
        if filtered_items:
            filtered_foods[category] = filtered_items
    return filtered_foods


def render_food_item(food, category):
    """Render a single food item with controls."""
    with st.container(border=True):
        emoji_with_tooltip = food.get('emoji', '')
        if emoji_with_tooltip and emoji_with_tooltip in EMOJI_TOOLTIPS:
            st.subheader(f"{emoji_with_tooltip} {food['name']}")
            st.caption(EMOJI_TOOLTIPS[emoji_with_tooltip])
        else:
            st.subheader(f"{emoji_with_tooltip} {food['name']}")
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(
            food['name'], 0.0
        )
        col1, col2 = st.columns([2, 1.2])
        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = (
                        "primary"
                        if current_serving == float(k)
                        else "secondary"
                    )
                    if st.button(
                        f"{k}",
                        key=f"{key}_{k}",
                        type=button_type,
                        help=f"Set {k} servings",
                        use_container_width=True
                    ):
                        st.session_state.food_selections[
                            food['name']
                        ] = float(k)
                        st.rerun()
        with col2:
            custom_serving = st.number_input(
                "Custom",
                min_value=0.0,
                max_value=20.0,
                value=float(current_serving),
                step=0.5,
                key=f"{key}_custom",
                label_visibility="collapsed"
            )
        if custom_serving != current_serving:
            if custom_serving > 0:
                st.session_state.food_selections[
                    food['name']
                ] = custom_serving
            elif food['name'] in st.session_state.food_selections:
                del st.session_state.food_selections[food['name']]
            st.rerun()
        caption_text = (
            f"Per Serving: {food['calories']} kcal | "
            f"{food['protein']}g protein | "
            f"{food['carbs']}g carbs | {food['fat']}g fat"
        )
        st.caption(caption_text)


def render_food_grid(items, category, columns=2):
    """Render a grid of food items for a category."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)


# -----------------------------------------------------------------------------
# Cell 8: Initialize Application
# -----------------------------------------------------------------------------

# ------ Initialize Session State ------
initialize_session_state()

# ------ Load Food Database And Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# ------ Apply Custom CSS Styling ------
st.markdown(
    """
<style>
html { font-size: 100%; }
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
.stMetric > div > div > div > div { color: #262730; }
.stProgress .st-bo { background-color: #e0e0e0; }
.stProgress .st-bp { background-color: #ff6b6b; }
.stCaption { color: #555555 !important; }
</style>
""",
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Cell 9: Application Title And Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown(
    "Provide your information to calculate daily nutrition targets"
)

# ------ Sidebar For User Input ------
st.sidebar.header("Enter Personal Information 📊")
units = st.sidebar.toggle(
    "Use Imperial Units",
    value=(st.session_state.get('user_units', 'metric') == 'imperial'),
    key='units_toggle',
    help="Toggle for Imperial (pounds and inches) or Metric (kilograms and centimeters)"
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
    value = create_unified_input(field_name, field_config, container=st.sidebar)
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
if st.sidebar.button(
    "🧮 Calculate My Targets", type="primary", key="calculate_button"
):
    validation_errors = validate_user_inputs(all_inputs)
    st.session_state.form_errors = validation_errors
    if not validation_errors:
        st.session_state.form_submitted = True
        st.session_state.show_motivational_message = True
    else:
        st.session_state.form_submitted = False
    st.rerun()
if st.session_state.get('form_errors'):
    for error in st.session_state.form_errors:
        st.sidebar.error(f"• {error}")

# ------ Save Or Load Progress ------
st.sidebar.divider()
st.sidebar.subheader("Save Your Progress 💾")
if st.sidebar.button("Save", key="save_progress", type="primary"):
    progress_json = save_progress_to_json(
        st.session_state.food_selections, all_inputs
    )
    st.sidebar.download_button(
        "Download",
        data=progress_json,
        file_name=(
            f"nutrition_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
        mime="application/json",
        key="download_progress"
    )
st.sidebar.subheader("Load Progress 📂")
uploaded_file = st.sidebar.file_uploader(
    "Load", type="json", key="upload_progress"
)
if uploaded_file is not None:
    content = uploaded_file.read().decode()
    food_selections, user_inputs = load_progress_from_json(content)
    st.session_state.food_selections.update(food_selections)
    for key, value in user_inputs.items():
        if f'user_{key}' in st.session_state:
            st.session_state[f'user_{key}'] = value
    st.sidebar.success("Progress loaded 📂")
    st.rerun()

# ------ Activity Level Guide ------
with st.sidebar.container(border=True):
    st.markdown("#### Activity Level Reference")
    for key in ACTIVITY_MULTIPLIERS:
        description = ACTIVITY_DESCRIPTIONS.get(key, "")
        st.markdown(f"* {description}")
    st.markdown(
        "If uncertain between two levels select the lower level"
    )

# ------ Dynamic Sidebar Summary ------
if st.session_state.form_submitted:
    final_values = get_final_values(all_inputs)
    targets = calculate_personalized_targets(**final_values)
    totals, _ = calculate_daily_totals(
        st.session_state.food_selections, foods
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Summary 📊")
    progress_calories = (
        min(
            totals['calories'] / targets['total_calories'] * 100, 100
        )
        if targets['total_calories'] > 0 else 0
    )
    progress_protein = (
        min(
            totals['protein'] / targets['protein_g'] * 100, 100
        )
        if targets['protein_g'] > 0 else 0
    )
    st.sidebar.metric(
        "Calories Progress",
        f"{progress_calories:.0f}%",
        f"{totals['calories']:.0f}/"
        f"{targets['total_calories']:.0f} kcal"
    )
    st.sidebar.metric(
        "Protein Progress",
        f"{progress_protein:.0f}%",
        f"{totals['protein']:.0f}/{targets['protein_g']:.0f} g"
    )

# ------ Process Final Values ------
final_values = get_final_values(all_inputs)

# ------ Check Submission ------
user_has_entered_info = st.session_state.form_submitted

# ------ Calculate Targets ------
targets = calculate_personalized_targets(**final_values)

# ------ Motivational Message ------
if st.session_state.show_motivational_message and user_has_entered_info:
    goal_messages = {
        'weight_loss': (
            f"Target indicates an estimated loss of "
            f"{abs(targets['estimated_weekly_change']):.2f} kg per week 🎯"
        ),
        'weight_maintenance': (
            f"Maintenance plan prepared for weight "
            f"{format_weight(final_values['weight_kg'], st.session_state.get('user_units', 'metric'))} ⚖️"
        ),
        'weight_gain': (
            f"Target indicates an estimated gain of "
            f"{targets['estimated_weekly_change']:.2f} kg per week 📈"
        )
    }
    message = goal_messages.get(
        targets['goal'],
        "Targets calculated"
    )
    st.success(message)
    if st.button("Dismiss", key="dismiss_message"):
        st.session_state.show_motivational_message = False
        st.rerun()

# -----------------------------------------------------------------------------
# Cell 10: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info(
        "Enter data in the sidebar and select Calculate My Targets"
    )
    st.header("Sample Daily Targets For Reference")
    st.caption(
        "These are example values. Enter personal data for specific targets"
    )
else:
    goal_labels = {
        'weight_loss': 'Weight Loss',
        'weight_maintenance': 'Weight Maintenance',
        'weight_gain': 'Weight Gain'
    }
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(
        f"Custom Daily Nutrition Targets For {goal_label} 🎯"
    )
st.info(
    "Aim to meet targets most days to support consistency 🎯"
)
hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)
units_display = st.session_state.get('user_units', 'metric')
weight_display = format_weight(final_values['weight_kg'], units_display)
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Weight", weight_display),
            ("BMR", f"{targets['bmr']} kcal"),
            ("TDEE", f"{targets['tdee']} kcal"),
            (
                "Daily Caloric Adjustment",
                f"{targets['caloric_adjustment']:+} kcal"
            ),
            (
                "Estimated Weekly Weight Change",
                f"{targets['estimated_weekly_change']:+.2f} kg"
            )
        ]
    },
    {
        'title': 'Daily Nutrition Targets', 'columns': 5,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            ("Protein", f"{targets['protein_g']} g",
             f"{targets['protein_percent']:.0f}% of calories"),
            ("Carbohydrates", f"{targets['carb_g']} g",
             f"{targets['carb_percent']:.0f}% of calories"),
            ("Fat", f"{targets['fat_g']} g",
             f"{targets['fat_percent']:.0f}% of calories"),
            ("Water", f"{hydration_ml} ml",
             f"~{hydration_ml/250:.1f} cups")
        ]
    }
]
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# -----------------------------------------------------------------------------
# Cell 11: Evidence Based Tips And Context
# -----------------------------------------------------------------------------

with st.expander("Guidance And Reference Information 📚", expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Hydration Sleep And Tracking",
        "Body Composition Monitoring",
        "Mindset And Habits",
        "Energy And Activity",
        "Metabolism And Nutrients"
    ])
    with tab1:
        st.header("Hydration Guidelines")
        st.markdown(
            "* Daily Goal: About 35 milliliters per kilogram of body weight\n"
            "* Training Bonus: Add 500 to 750 milliliters per training hour\n"
            "* Pre Meal Water: 500 milliliters before meals may increase satiety"
        )
        st.divider()
        st.header("Sleep Guidelines")
        st.markdown(
            "* Target 7 to 9 hours per night\n"
            "* Maintain consistent sleep and wake times\n"
            "* Keep room dark and limit screens pre sleep"
        )
        st.divider()
        st.header("Scale Data Practices")
        st.markdown(
            "* Weigh at consistent time daily\n"
            "* Track rolling weekly averages\n"
            "* Assess trends before making changes"
        )
    with tab2:
        st.header("Progress Monitoring")
        st.markdown(
            "* Use periodic photos under consistent conditions\n"
            "* Track circumference measures monthly\n"
            "* Note energy and performance patterns"
        )
    with tab3:
        st.header("Habit Sequencing")
        st.markdown(
            "* Establish calorie consistency first\n"
            "* Add protein tracking after initial adherence\n"
            "* Introduce carbohydrate and fat refinement later"
        )
        st.subheader("Adjustment Guidelines")
        st.markdown(
            "* Verify logging accuracy during plateaus\n"
            "* Reassess activity classification if routine shifts\n"
            "* Consider modest activity increases before further calorie cuts"
        )
        st.subheader("Protein Distribution")
        st.markdown(
            "* Even spacing of protein servings supports synthesis\n"
            "* Include protein pre and post resistance training"
        )
    with tab4:
        st.header("Resistance Training Role")
        st.markdown(
            "* Supports muscle retention during deficit\n"
            "* Raises adaptive capacity during surplus\n"
            "* Plan two to three sessions as baseline"
        )
        st.divider()
        st.header("Non Exercise Activity")
        st.markdown(
            "* Incremental walking can increase total daily expenditure\n"
            "* Low intensity additions reduce recovery burden"
        )
    with tab5:
        st.header("Metabolic Components")
        st.markdown(
            "* BMR represents baseline physiological maintenance\n"
            "* TDEE combines BMR plus activity energy"
        )
        st.divider()
        st.header("Macronutrient Characteristics")
        st.markdown(
            "* Protein supports tissue repair and satiety\n"
            "* Complex carbohydrates support training performance\n"
            "* Dietary fats support endocrine function"
        )
        st.divider()
        st.header("Key Micronutrients In Plant Focused Patterns")
        st.markdown(
            "* Monitor B12 iron calcium zinc iodine and omega 3 sources"
        )

# -----------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Track Daily Intake 🥗")
search_col, reset_col = st.columns([3, 1])
with search_col:
    search_term = st.text_input(
        "Search Foods",
        placeholder="Type food name to filter results",
        key="food_search_input",
        label_visibility="collapsed"
    )
    st.session_state.food_search = search_term
with reset_col:
    if st.button("Clear Search", key="clear_search", type="primary"):
        st.session_state.food_search = ""
        st.rerun()
st.markdown(
    "Select servings to accumulate daily nutrient totals"
)
with st.expander("Emoji Guide 💡"):
    for emoji, tooltip in EMOJI_TOOLTIPS.items():
        label = tooltip.split(':')[0]
        description = ':'.join(tooltip.split(':')[1:]).strip()
        st.markdown(f"* **{emoji} {label}**: {description}")
if st.button(
    "Reset All Food Selections",
    type="primary",
    key="reset_foods"
):
    st.session_state.food_selections = {}
    st.rerun()
filtered_foods = filter_foods_by_search(foods, search_term)
if not filtered_foods and search_term:
    st.warning(
        f"No foods found matching '{search_term}'"
    )
elif filtered_foods:
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
# Cell 13: Daily Summary And Progress Tracking
# -----------------------------------------------------------------------------

st.header("Daily Summary 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)
if selected_foods:
    recommendations = create_progress_tracking(
        totals, targets, foods
    )
    st.subheader("Export Summary 📥")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "Prepare PDF Report",
            key="export_pdf",
            use_container_width=True
        ):
            pdf_buffer = create_pdf_summary(
                totals, targets, selected_foods, final_values
            )
            st.download_button(
                "Download PDF",
                data=pdf_buffer,
                file_name=(
                    f"nutrition_summary_{datetime.now().strftime('%Y%m%d')}.pdf"
                ),
                mime="application/pdf",
                key="download_pdf_button"
            )
    with col2:
        if st.button(
            "Prepare CSV Data",
            key="export_csv",
            use_container_width=True
        ):
            csv_data = create_csv_summary(
                totals, targets, selected_foods
            )
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=(
                    f"nutrition_data_{datetime.now().strftime('%Y%m%d')}.csv"
                ),
                mime="text/csv",
                key="download_csv_button"
            )
    with col3:
        if st.button(
            "Create Share Text",
            key="share_progress",
            use_container_width=True
        ):
            share_text = (
                f"Nutrition Progress - {datetime.now().strftime('%Y-%m-%d')}\n\n"
                f"Intake:\n"
                f"- Calories: {totals['calories']:.0f} / "
                f"{targets['total_calories']:.0f} kcal\n"
                f"- Protein: {totals['protein']:.0f} / "
                f"{targets['protein_g']:.0f} g\n"
                f"- Carbs: {totals['carbs']:.0f} / "
                f"{targets['carb_g']:.0f} g\n"
                f"- Fat: {totals['fat']:.0f} / {targets['fat_g']:.0f} g\n"
            )
            st.info("Copy the summary below")
            st.text_area(
                "Shareable Summary",
                share_text,
                height=200,
                label_visibility="collapsed"
            )
    st.divider()
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Macronutrient Comparison")
        fig_macros = go.Figure()
        macros = ['Protein', 'Carbohydrates', 'Fat']
        actual_values = [
            totals['protein'], totals['carbs'], totals['fat']
        ]
        target_values = [
            targets['protein_g'], targets['carb_g'], targets['fat_g']
        ]
        fig_macros.add_trace(
            go.Bar(
                name='Actual', x=macros, y=actual_values,
                marker_color='#ff6b6b'
            )
        )
        fig_macros.add_trace(
            go.Bar(
                name='Target', x=macros, y=target_values,
                marker_color='#4ecdc4'
            )
        )
        fig_macros.update_layout(
            title_text='Macronutrient Comparison',
            barmode='group',
            yaxis_title='Grams',
            height=400,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=40, b=10)
        )
        st.plotly_chart(fig_macros, use_container_width=True)
    with col2:
        st.subheader("Macronutrient Energy Split")
        if totals['calories'] > 0:
            fig_pie = go.Figure(
                data=[go.Pie(
                    labels=['Protein', 'Carbohydrates', 'Fat'],
                    values=[
                        totals['protein'] * 4,
                        totals['carbs'] * 4,
                        totals['fat'] * 9
                    ],
                    hole=0.4,
                    marker_colors=['#ff6b6b', '#4ecdc4', '#45b7d1']
                )]
            )
            fig_pie.update_layout(
                title=f"Total Energy: {totals['calories']:.0f} kcal",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("Select foods to view macronutrient energy split")
    if recommendations:
        st.subheader("Recommendations")
        for rec in recommendations:
            st.info(rec)
    with st.expander("Logged Food Details", expanded=True):
        st.subheader("Foods Logged")
        prepared_data = prepare_summary_data(
            totals, targets, selected_foods
        )
        consumed_foods_list = prepared_data['consumed_foods']
        if consumed_foods_list:
            display_data = [
                {
                    'Food': item['name'],
                    'Servings': f"{item['servings']:.1f}",
                    'Calories (kcal)': f"{item['calories']:.0f}",
                    'Protein (g)': f"{item['protein']:.1f}",
                    'Carbs (g)': f"{item['carbs']:.1f}",
                    'Fat (g)': f"{item['fat']:.1f}"
                } for item in consumed_foods_list
            ]
            df_summary = pd.DataFrame(display_data)
            st.dataframe(
                df_summary, use_container_width=True, hide_index=True
            )
        else:
            st.caption("No foods logged")
else:
    st.info(
        "Add food selections to begin tracking intake"
    )
    st.subheader("Progress Snapshot")
    render_progress_bars(totals, targets)

# -----------------------------------------------------------------------------
# Cell 14: User Feedback Section
# -----------------------------------------------------------------------------

st.divider()
st.header("Feedback Submission 💬")
st.markdown(
    "Provide feedback or feature requests to support improvements"
)
with st.form("feedback_form", clear_on_submit=True):
    feedback_type = st.selectbox(
        "Select Feedback Type",
        ["General Feedback", "Bug Report", "Feature Request", "Success Story"],
        key="feedback_type"
    )
    feedback_text = st.text_area(
        "Enter Feedback",
        placeholder=(
            "Describe your experience or suggest a feature or report an issue"
        ),
        height=100,
        key="feedback_text"
    )
    if st.form_submit_button(
        "Submit Feedback", type="primary"
    ):
        if feedback_text.strip():
            st.success(
                f"Thank you for your {feedback_type.lower()}"
            )
        else:
            st.error("Enter feedback before submitting")

# -----------------------------------------------------------------------------
# Cell 15: Footer And References
# -----------------------------------------------------------------------------

st.divider()
st.markdown(
    "### Scientific References 📚\n\n"
    "This tracker applies the Mifflin St Jeor equation for basal metabolic "
    "rate estimation and standard activity multipliers for TDEE estimation. "
    "Protein ranges align with position stand recommendations from sports "
    "nutrition organizations.\n\n"
    "### Advisory ⚠️\n\n"
    "Values are estimates and do not replace personalized assessment by a "
    "qualified professional. Monitor individual responses and adjust as "
    "needed."
)
st.success(
    "Session complete. Continue consistent tracking for trend analysis ✅"
)

# -----------------------------------------------------------------------------
# Cell 16: Session State Management And Performance
# -----------------------------------------------------------------------------

# ------ Trim Food Selections If Excess Size ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# ------ Widget Key Management ------
# All widgets use explicit keys to reduce unintended reruns

# ------ Remove Temporary Session Keys ------
temp_keys = [
    key for key in st.session_state.keys() if key.startswith('temp_')
]
for key in temp_keys:
    del st.session_state[key]

# -----------------------------------------------------------------------------
# End Of Script
# -----------------------------------------------------------------------------
