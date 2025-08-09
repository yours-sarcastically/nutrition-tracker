#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------------
# Personalized Evidence-Based Streamlit Nutrition Tracker Using The Mifflin St Jeor
# Equation And Goal-Specific Macronutrient Periodization
# -------------------------------------------------------------------------------

"""
This script implements an interactive, evidence-based nutrition tracking
application using Streamlit. Its primary purpose is to generate personalized
daily calorie and macronutrient targets based on the scientifically validated
Mifflin St Jeor equation and to help users monitor progress through a
structured vegetarian food logging interface.

Core Scientific Methods And Equations
1. Basal Metabolic Rate (BMR)
   The script uses the Mifflin St Jeor equation, widely supported in the
   scientific and clinical nutrition community for its accuracy in estimating
   resting energy needs.
   Male:
       BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
   Female:
       BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
2. Total Daily Energy Expenditure (TDEE)
   TDEE = BMR × Activity Factor
   Activity factors are selected from a standardized map of lifestyle
   categories ranging from sedentary (1.2) to extremely active (1.9).
3. Goal-Specific Caloric Adjustment
   Weight Loss: 20 percent caloric deficit (TDEE × -0.20)
   Weight Maintenance: Neutral adjustment (TDEE × 0.00)
   Weight Gain: 10 percent surplus (TDEE × +0.10)
4. Macronutrient Allocation Strategy
   a. Protein: Assigned using a target grams per kilogram of body weight
      based on goal (range 1.6–2.0 g/kg by default, adjustable in advanced
      settings)
   b. Fat: Allocated as a configurable percentage of total calories
   c. Carbohydrates: Derived from remaining available calories
   Conversions:
       Protein calories = protein_g × 4
       Fat calories = fat_g × 9
       Carbohydrate calories = carb_g × 4
5. Estimated Weekly Weight Change
   Approximated using:
       Weekly Change (kg) = (Daily Caloric Adjustment × 7) / 7700
   (Based on the approximation that one kilogram of body fat contains about
   7,700 kilocalories)

Application Features
1. Unified Configuration System
   A centralized CONFIG dictionary defines all input fields, nutrient
   display properties, and food categorization logic for maintainability.
2. Dynamic Goal Integration
   Goal-specific default protein and fat allocations automatically populate
   unless manually overridden in advanced settings.
3. Food Selection Interface
   Users log foods from a vegetarian database, adjust serving counts via
   quick buttons or custom inputs, and instantly view cumulative nutrition.
4. Real-Time Feedback
   Progress bars, recommendation generation, and adaptive suggestions help
   users close nutrient gaps efficiently.
5. Export And Persistence
   - JSON save and load functionality for resuming sessions
   - PDF summary export with key metrics and selected foods
   - CSV export containing both nutrient summary and logged items
6. Evidence-Based Guidance
   Expandable educational sections provide structured, research-informed
   nutrition, hydration, sleep, mindset, and micronutrient guidance.
7. Unit Flexibility
   Metric or Imperial display toggling while storing values internally in
   metric units for standardization.
8. Hydration Estimation
   A simple weight- and activity-based model provides daily water guidance.

Usage Instructions
1. Installation
   Ensure Python 3.9 or later is installed. Then install dependencies:
       pip install streamlit pandas plotly reportlab
2. Running The Application
   Save this script as nutrition_app.py then launch with:
       streamlit run nutrition_app.py
3. User Workflow
   a. Open the sidebar
   b. Enter age, height, weight, sex, activity level, and goal
   c. Optionally expand Advanced Settings to adjust protein and fat targets
   d. Click Calculate My Targets
   e. Use the food tabs to log intake by servings
   f. Review real-time dashboard, recommendations, charts, and summaries
   g. Export results (PDF or CSV) or save progress as JSON
4. Command Line Context
   Although primarily launched through Streamlit, the script can be invoked
   from a shell using:
       streamlit run nutrition_app.py
   No additional command line parameters are required. All interaction is
   performed through the graphical interface in the browser.

Design And Implementation Notes
- State Management: Streamlit session_state stores user inputs, selected
  foods, search filters, and control flags
- Performance: Caching decorators (st.cache_data) applied to static data
  loaders and emoji assignment logic
- Accessibility: Text outputs use complete sentences without contractions.
  Each major UI section and operation is thoroughly documented with
  explanatory comments
- Style Compliance: The script adheres to PEP 8 standards for spacing,
  indentation, naming, and line length (long logical strings are wrapped
  where practical without altering functionality)

Limitations And Assumptions
- The weight change estimation is a simplified energetic model and does
  not account for adaptive thermogenesis or lean mass changes
- Food database must include required columns: category, name, serving_unit,
  calories, protein, carbs, fat
- This tool does not substitute for medical advice. Users with clinical
  conditions should consult a licensed professional

Outputs And Visual Elements
- Progress Bars: Display percent of targets with color-coded emoji markers
- Recommendation Engine: Suggests foods that simultaneously close multiple
  nutrient gaps
- Charts: Plotly bar comparison and macronutrient split pie chart
- Reports: PDF concise summary, CSV tabular detail, JSON session snapshot

Emoji Usage Policy
Each interactive or display output is limited to at most one emoji to
enhance readability while avoiding visual overload.

Enjoy refining your nutrition strategy with structured, science-aligned
guidance tailored to your daily goals.
"""

# -------------------------------------------------------------------------------
# Cell 1: Import Required Libraries And Modules
# -------------------------------------------------------------------------------

import math
import json
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

# -------------------------------------------------------------------------------
# Cell 2: Page Configuration And Initial Setup
# -------------------------------------------------------------------------------

st.set_page_config(
    page_title="Your Personal Nutrition Coach 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------------------
# Cell 3: Unified Configuration Constants
# -------------------------------------------------------------------------------

# ------ Default Parameter Values Based On Published Research ------
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
    'sedentary': "🧑‍💻 **Sedentary**: You are basically married to your desk chair",
    'lightly_active': (
        "🏃 **Lightly Active**: You squeeze in walks or workouts one to three "
        "times a week"
    ),
    'moderately_active': (
        "🚴 **Moderately Active**: You are training three to five days a week"
    ),
    'very_active': "🏋️ **Very Active**: You perform hard exercise most days",
    'extremely_active': (
        "🤸 **Extremely Active**: You train intensely and have a highly active job"
    )
}

# ------ Goal-Specific Targets Based On An Evidence-Based Guide ------
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

# ------ Unified Configuration For All App Components ------
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
            'type': 'number', 'label': 'Age (in years)',
            'min': 16, 'max': 80, 'step': 1,
            'placeholder': 'Enter your age', 'required': True
        },
        'height_cm': {
            'type': 'number', 'label': 'Height (in centimeters)',
            'min': 140, 'max': 220, 'step': 1,
            'placeholder': 'Enter your height', 'required': True
        },
        'weight_kg': {
            'type': 'number', 'label': 'Weight (in kilograms)',
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
            ], 'required': True
        },
        'goal': {
            'type': 'selectbox', 'label': 'Your Goal',
            'options': [
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
                'Set the share of your daily calories that should come from '
                'healthy fats'
            ),
            'convert': lambda x: x / 100 if x else None,
            'advanced': True, 'required': False
        }
    }
}

# ------ Emoji Tooltips ------
EMOJI_TOOLTIPS = {
    '🥇': (
        'Gold Medal: Nutritional all-star. High in its target nutrient and '
        'very calorie efficient'
    ),
    '🔥': (
        'High Calorie: One of the more calorie dense options in its group'
    ),
    '💪': 'High Protein: A true protein powerhouse',
    '🍚': 'High Carb: A carbohydrate champion',
    '🥑': 'High Fat: A healthy fat hero'
}

# ------ Metric Tooltips ------
METRIC_TOOLTIPS = {
    'BMR': (
        'Basal Metabolic Rate - the energy your body needs for vital '
        'functions at rest'
    ),
    'TDEE': (
        'Total Daily Energy Expenditure - your BMR plus calories burned '
        'through activity'
    ),
    'Caloric Adjustment': (
        'How many calories above or below TDEE are used to align with your goal'
    ),
    'Protein': 'Essential for muscle building, repair, and satiety',
    'Carbohydrates': (
        'Primary energy source for brain performance and muscular work'
    ),
    'Fat': (
        'Important for hormone production, nutrient absorption, and cellular '
        'health'
    )
}

# ------ Centralized Tip And Recommendation Content ------
TIPS_CONTENT = {
    'hydration': [
        "**Daily Goal**: Shoot for about 35 ml per kilogram of your body weight daily",
        "**Training Bonus**: Add an extra 500 to 750 ml per hour of purposeful exercise",
        (
            "**Fullness Aid**: Drinking 500 ml of water before meals may boost "
            "satiety by approximately 13 percent"
        )
    ],
    'sleep': [
        (
            "**Sleep Quality Matters**: Getting fewer than seven hours of sleep "
            "can reduce fat loss efficiency significantly"
        ),
        "**Daily Goal**: Aim for seven to nine hours on a consistent schedule",
        (
            "**Environment Setup**: Keep your room dark, cool (18–20°C), and "
            "avoid screens for one hour before bed"
        )
    ],
    'tracking_wins': [
        (
            "**Morning Routine**: Weigh yourself after using the bathroom and "
            "before eating while wearing minimal clothing"
        ),
        (
            "**Weekly Trends Over Daily Swings**: Focus on the rolling weekly "
            "average rather than day to day spikes"
        ),
        (
            "**Adjustment Patience**: Wait for two or more weeks of stalled "
            "progress before modifying targets"
        )
    ],
    'beyond_the_scale': [
        (
            "**Visual Progress**: Take consistent photos monthly in comparable "
            "lighting and posture"
        ),
        "**Tape Measurements**: Track waist, hips, arms, and thighs monthly",
        (
            "**Internal Feedback**: Monitor energy, sleep, performance, mood, "
            "and hunger for a broad perspective"
        )
    ],
    'protein_pacing': [
        (
            "**Distribute Intake**: Aim for 20 to 40 grams of protein in each "
            "of three to four meals"
        ),
        (
            "**Training Window**: Include carbohydrates and 20 to 40 grams of "
            "protein pre workout and within two hours post workout"
        ),
        (
            "**Night Support**: Consider 20 to 30 grams of slow digesting casein "
            "protein before sleep"
        )
    ],
    'weight_loss_plateau': [
        (
            "**Logging Accuracy**: Reassess tracking precision before making "
            "caloric cuts"
        ),
        "**Activity Review**: Confirm your daily movement has not decreased",
        (
            "**Low Impact Addition**: Add 10 to 15 minutes of daily walking "
            "before reducing intake further"
        ),
        (
            "**Diet Break**: A maintenance phase of one to two weeks every "
            "six to eight weeks can aid adherence"
        ),
        (
            "**Volume Foods**: Favor high fiber vegetables, berries, and lean "
            "protein for satiety"
        )
    ],
    'weight_gain_stalls': [
        (
            "**Caloric Density**: Use smoothies, milk, and liquid calories to "
            "ease digestive load"
        ),
        "**Healthy Fats**: Include nuts, seeds, oils, and avocado",
        (
            "**Progressive Overload**: Provide a muscular growth signal via "
            "consistent resistance training"
        ),
        (
            "**Incremental Increase**: Raise daily intake by 100 to 150 "
            "calories after a two week stall"
        )
    ]
}

# -------------------------------------------------------------------------------
# Cell 4: Unit Conversion Functions
# -------------------------------------------------------------------------------


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
    """Format a weight value for display in the user's preferred units."""
    if units == 'imperial':
        return f"{kg_to_lbs(weight_kg):.1f} lbs"
    return f"{weight_kg:.1f} kg"


def format_height(height_cm, units):
    """Format a height value for display in the user's preferred units."""
    if units == 'imperial':
        total_inches = cm_to_inches(height_cm)
        feet = int(total_inches // 12)
        inches = total_inches % 12
        return f"{feet}'{inches:.0f}\""
    return f"{height_cm:.0f} cm"


# -------------------------------------------------------------------------------
# Cell 5: Unified Helper Functions
# -------------------------------------------------------------------------------


def initialize_session_state():
    """Initialize all required session state variables for first run."""
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
    """Create a single input widget from the unified configuration."""
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
            label = 'Weight (in pounds)'
            min_val, max_val = kg_to_lbs(min_val), kg_to_lbs(max_val)
            step_val = 1.0
            if current_value:
                current_value = kg_to_lbs(current_value)
        elif (field_name == 'height_cm' and
              st.session_state.get('user_units') == 'imperial'):
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
    """Validate required user inputs and return a list of error messages."""
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
    """Process all user inputs and apply goal defaults to advanced fields."""
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
    """Calculate daily fluid needs based on body weight and activity."""
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
    """Display a grid of metrics using a specified column layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                label, value = metric_info
                help_text = METRIC_TOOLTIPS.get(
                    label.split('(')[0].strip()
                )
                st.metric(label, value, help=help_text)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                help_text = METRIC_TOOLTIPS.get(
                    label.split('(')[0].strip()
                )
                st.metric(label, value, delta, help=help_text)


def get_progress_color(percent):
    """Return an emoji color indicator based on completion percent."""
    if percent >= 80:
        return "🟢"
    if percent >= 50:
        return "🟡"
    return "🔴"


def render_progress_bars(totals, targets):
    """Render progress bars for each primary nutrient and calories."""
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
                f"{percent:.0f}% of your daily target "
                f"({target:.0f} {config['unit']})"
            )
        )


def create_progress_tracking(totals, targets, foods):
    """Generate recommendations based on remaining nutrient gaps."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")
    render_progress_bars(totals, targets)
    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation and building',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and overall health'
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
                'purpose': purpose_map.get(nutrient, 'for optimal nutrition')
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
                        food[nutrient] / deficit_info['amount'],
                        1.0
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
                f", and {deficit_summary[-1]}."
            )
        else:
            summary_text = f"You still need: {deficit_summary[0]}."
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
                f"Smart pick: One serving of {food['name']} provides "
                f"{benefits_text} which helps multiple targets"
            )
            if len(top_suggestions) > 1:
                alternative_foods = []
                for suggestion in top_suggestions[1:]:
                    alt_food = suggestion['food']
                    nutrients_helped = suggestion['nutrients_helped']
                    alt_benefits = [
                        f"{alt_food[n]:.0f}g {n}" for n in nutrients_helped
                    ]
                    if len(alt_benefits) > 1:
                        alt_text = (
                            ", ".join(alt_benefits[:-1]) +
                            f", and {alt_benefits[-1]}"
                        )
                    else:
                        alt_text = alt_benefits[0]
                    alternative_foods.append(
                        f"{alt_food['name']} (provides {alt_text})"
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
                    f"Alternative options: {alternatives_text}"
                )
        else:
            biggest_deficit = max(
                deficits.items(), key=lambda x: x[1]['amount']
            )
            nutrient, deficit_info = biggest_deficit
            best_single_food = max(
                all_foods, key=lambda x: x.get(nutrient, 0), default=None
            )
            if best_single_food and best_single_food.get(nutrient, 0) > 0:
                recommendations.append(
                    f"Alternative option: Add {best_single_food['name']} "
                    f"which supplies {best_single_food[nutrient]:.0f}g of "
                    f"{deficit_info['label']}"
                )
    return recommendations


def calculate_daily_totals(food_selections, foods):
    """Calculate aggregate nutrient totals from all chosen foods."""
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
    """Serialize current session progress to a JSON string."""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'food_selections': food_selections,
        'user_inputs': user_inputs
    }
    return json.dumps(progress_data, indent=2)


def load_progress_from_json(json_data):
    """Deserialize JSON progress data into selection and input structures."""
    try:
        data = json.loads(json_data)
        return data.get('food_selections', {}), data.get('user_inputs', {})
    except json.JSONDecodeError:
        return {}, {}


def prepare_summary_data(totals, targets, selected_foods):
    """Prepare structured summary data for exports."""
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
    """Create a PDF summary file object for download."""
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
                50, y_pos, f"• {item['name']}: {item['servings']} serving(s)"
            )
            y_pos -= 15
            if y_pos < 50:
                break
    pdf_canvas.save()
    buffer.seek(0)
    return buffer


def create_csv_summary(totals, targets, selected_foods):
    """Generate a CSV representation of the daily nutrition summary."""
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


# -------------------------------------------------------------------------------
# Cell 6: Nutritional Calculation Functions
# -------------------------------------------------------------------------------


def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculate Basal Metabolic Rate using the Mifflin St Jeor equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure from BMR and activity level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Estimate weekly weight change from daily caloric adjustment."""
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(
    age,
    height_cm,
    weight_kg,
    sex='male',
    activity_level='moderately_active',
    goal='weight_gain',
    protein_per_kg=None,
    fat_percentage=None
):
    """Compute personalized daily metabolic and macronutrient targets."""
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


# -------------------------------------------------------------------------------
# Cell 7: Food Database Processing Functions
# -------------------------------------------------------------------------------


@st.cache_data
def load_food_database(file_path):
    """Load a vegetarian food database from a CSV file path."""
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
    """Assign emojis to foods based on relative nutrient ranking."""
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
            is_high_calorie = (
                food_name in top_foods['calories'].get(category, [])
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
    """Filter foods across all categories using a case insensitive term."""
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
    """Render a single food card with serving selection controls."""
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
                        if current_serving == float(k) else "secondary"
                    )
                    if st.button(
                        f"{k}",
                        key=f"{key}_{k}",
                        type=button_type,
                        help=f"Set to {k} servings",
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
            f"Per Serving: {food['calories']} kcal | {food['protein']}g "
            f"protein | {food['carbs']}g carbs | {food['fat']}g fat"
        )
        st.caption(caption_text)


def render_food_grid(items, category, columns=2):
    """Render a grid of food items for a given category."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)


# -------------------------------------------------------------------------------
# Cell 8: Initialize Application
# -------------------------------------------------------------------------------

# ------ Initialize Session State ------
initialize_session_state()

# ------ Load Food Database And Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# ------ Apply Custom CSS For Enhanced Styling ------
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
.stMetric > div > div > div > div {
    color: #262730;
}
.stProgress .st-bo { background-color: #e0e0e0; }
.stProgress .st-bp { background-color: #ff6b6b; }
.stCaption { color: #555555 !important; }
</style>
""",
    unsafe_allow_html=True
)

# -------------------------------------------------------------------------------
# Cell 9: Application Title And Unified Input Interface
# -------------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown(
    """
A Smart Evidence Based Nutrition Tracker That Actually Supports Your Progress

Welcome aboard.

This experience is your structured nutrition companion. It is built on
validated metabolic equations and macronutrient allocation principles to
assist you in reliably aligning daily habits with long term body
composition goals. Enter your information to generate personalized targets
and then log foods to visualize your daily trajectory toward success.
"""
)

# ------ Sidebar For User Input ------
st.sidebar.header("User Profile And Goal Setup 📊")
units = st.sidebar.toggle(
    "Use Imperial Units",
    value=(st.session_state.get('user_units', 'metric') == 'imperial'),
    key='units_toggle',
    help="Toggle on for Imperial (lbs and inches) or off for Metric (kg and cm)"
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
    value = create_unified_input(
        field_name, field_config, container=st.sidebar
    )
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

# ------ Save And Load Progress ------
st.sidebar.divider()
st.sidebar.subheader("Save Current Progress 💾")
if st.sidebar.button("Save", key="save_progress", type="primary"):
    progress_json = save_progress_to_json(
        st.session_state.food_selections, all_inputs
    )
    st.sidebar.download_button(
        "Download",
        data=progress_json,
        file_name=(
            f"nutrition_progress_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ),
        mime="application/json",
        key="download_progress"
    )
st.sidebar.subheader("Load Previous Progress 📂")
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
    st.sidebar.success("Progress loaded successfully")
    st.rerun()

# ------ Activity Level Guide ------
with st.sidebar.container(border=True):
    st.markdown("#### Activity Level Reference")
    for key in ACTIVITY_MULTIPLIERS:
        description = ACTIVITY_DESCRIPTIONS.get(key, "")
        st.markdown(f"* {description}")
    st.markdown(
        "Tip: If unsure between levels choose the lower level for "
        "greater dietary accuracy"
    )

# ------ Dynamic Sidebar Summary ------
if st.session_state.form_submitted:
    final_values = get_final_values(all_inputs)
    targets = calculate_personalized_targets(**final_values)
    totals, _ = calculate_daily_totals(
        st.session_state.food_selections, foods
    )
    st.sidebar.divider()
    st.sidebar.markdown("### Quick Summary 📊")
    progress_calories = (
        min(
            totals['calories'] / targets['total_calories'] * 100, 100
        ) if targets['total_calories'] > 0 else 0
    )
    progress_protein = (
        min(
            totals['protein'] / targets['protein_g'] * 100, 100
        ) if targets['protein_g'] > 0 else 0
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

final_values = get_final_values(all_inputs)
user_has_entered_info = st.session_state.form_submitted
targets = calculate_personalized_targets(**final_values)
if st.session_state.show_motivational_message and user_has_entered_info:
    goal_messages = {
        'weight_loss': (
            f"Your plan targets an estimated weekly loss of "
            f"{abs(targets['estimated_weekly_change']):.2f} kg"
        ),
        'weight_maintenance': (
            "Your plan supports stable maintenance while optimizing recovery"
        ),
        'weight_gain': (
            f"Your plan supports a steady gain of "
            f"{targets['estimated_weekly_change']:.2f} kg per week"
        )
    }
    message = goal_messages.get(
        targets['goal'],
        "Targets calculated successfully"
    )
    st.success(message)
    if st.button("Dismiss Message", key="dismiss_message"):
        st.session_state.show_motivational_message = False
        st.rerun()

# -------------------------------------------------------------------------------
# Cell 10: Unified Target Display System
# -------------------------------------------------------------------------------

if not user_has_entered_info:
    st.info(
        "Enter your details in the sidebar and select Calculate My Targets "
        "to generate personalized values"
    )
    st.header("Sample Daily Targets For Reference")
    st.caption(
        "These are only example targets. Provide your information for "
        "personalized estimates"
    )
else:
    goal_labels = {
        'weight_loss': 'Weight Loss',
        'weight_maintenance': 'Weight Maintenance',
        'weight_gain': 'Weight Gain'
    }
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(
        f"Your Custom Daily Nutrition Roadmap For {goal_label} 🎯"
    )

st.info(
    "Guideline: Aim to meet your targets about 80 percent of the time. "
    "Consistent adherence with structured flexibility supports long term "
    "success"
)
hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)
units_display = st.session_state.get('user_units', 'metric')
weight_display = format_weight(final_values['weight_kg'], units_display)

metrics_config = [
    {
        'title': 'Metabolic Information',
        'columns': 5,
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
        'title': 'Your Daily Nutrition Targets',
        'columns': 5,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            (
                "Protein",
                f"{targets['protein_g']} g",
                f"{targets['protein_percent']:.0f}% of calories"
            ),
            (
                "Carbohydrates",
                f"{targets['carb_g']} g",
                f"{targets['carb_percent']:.0f}% of calories"
            ),
            (
                "Fat",
                f"{targets['fat_g']} g",
                f"{targets['fat_percent']:.0f}% of calories"
            ),
            (
                "Water",
                f"{hydration_ml} ml",
                f"Approximately {hydration_ml/250:.1f} cups"
            )
        ]
    }
]
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# -------------------------------------------------------------------------------
# Cell 11: Enhanced Evidence Based Tips And Context
# -------------------------------------------------------------------------------

with st.expander("Evidence Based Guidance 📚", expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Core Daily Practices",
        "Progress Tracking",
        "Mindset Strategy",
        "Activity And Energy",
        "Physiology And Support"
    ])
    with tab1:
        st.header("Hydration Principles")
        st.markdown(
            "* Daily Goal: Shoot for about 35 ml per kilogram of body weight\n"
            "* Training Bonus: Add 500 to 750 ml per hour of intense exercise\n"
            "* Pre Meal Water: A 500 ml serving before meals may help satiety"
        )
        st.divider()
        st.header("Sleep Optimization")
        st.markdown(
            "* Sleep Duration: Aim for seven to nine hours nightly\n"
            "* Consistency: Align bed and wake times when possible\n"
            "* Environment: Keep the room cool and dark with reduced blue light"
        )
        st.divider()
        st.header("Scale Data Strategy")
        st.markdown(
            "* Daily Weigh In: Same time each morning after the restroom\n"
            "* Weekly Average: Use weekly means to reduce noise\n"
            "* Adjustment Timing: Wait two weeks before plan changes"
        )
    with tab2:
        st.header("Holistic Progress Assessment")
        st.markdown(
            "* Photos: Monthly consistent angle and lighting\n"
            "* Measurements: Track waist, hips, arms, thighs monthly\n"
            "* Internal Cues: Monitor energy, mood, sleep, recovery"
        )
    with tab3:
        st.header("Mindset Framework")
        st.markdown(
            "Sustainable progress is developed through consistent execution "
            "of foundational routines with iterative refinement rather than "
            "perfectionism"
        )
        st.subheader("Habit Layering Sequence")
        st.markdown(
            "* Weeks 1 To 2: Focus on total calorie consistency\n"
            "* Weeks 3 To 4: Add protein tracking\n"
            "* Week 5 Plus: Refine carbohydrate and fat distribution"
        )
        st.divider()
        st.subheader("Stall Management")
        st.markdown(
            "* Reassess tracking accuracy\n"
            "* Add modest walking volume\n"
            "* Employ diet breaks strategically\n"
            "* Increase intake slightly if weight gain stalls"
        )
    with tab4:
        st.header("Activity Structure")
        st.subheader("Resistance Training")
        st.markdown(
            "Two to three sessions per week of multi joint patterns form a "
            "base for muscle retention and growth"
        )
        st.subheader("Non Exercise Movement")
        st.markdown(
            "Incremental walking increases daily expenditure with minimal "
            "recovery demand"
        )
    with tab5:
        st.header("Metabolic Overview")
        st.markdown(
            "BMR reflects resting energy. TDEE accounts for movement and "
            "thermic effects"
        )
        st.divider()
        st.subheader("Macronutrient Hierarchy")
        st.markdown(
            "* Protein: Highest satiety and structural role\n"
            "* Fiber Rich Carbohydrates: Volume and micronutrients\n"
            "* Dietary Fats: Hormonal and structural support"
        )
        st.divider()
        st.subheader("Plant Focused Micronutrients")
        st.markdown(
            "Monitor B12, iron, calcium, zinc, iodine, and omega 3s via "
            "fortified foods or supplementation as appropriate"
        )

# -------------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# -------------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")
search_col, reset_col = st.columns([3, 1])
with search_col:
    search_term = st.text_input(
        "Search For Foods",
        placeholder="Type a food name to filter results",
        key="food_search_input",
        label_visibility="collapsed"
    )
    st.session_state.food_search = search_term
with reset_col:
    if st.button("Clear Search", key="clear_search", type="primary"):
        st.session_state.food_search = ""
        st.rerun()
st.markdown(
    "Select the number of servings for each item to build your daily intake "
    "and evaluate progress versus targets"
)
with st.expander(
    "Emoji Guide For Food Highlights", expanded=False
):
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
        f"No foods found matching '{search_term}'. Adjust your search"
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

# -------------------------------------------------------------------------------
# Cell 13: Daily Summary And Progress Tracking
# -------------------------------------------------------------------------------

st.header("Today’s Scorecard 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)
if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)
    st.subheader("Export Your Summary 📥")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button(
            "Generate PDF Report",
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
            "Generate CSV Data",
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
            "Create Share Summary",
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
                f"- Fat: {totals['fat']:.0f} / "
                f"{targets['fat_g']:.0f} g\n\n"
                f"Created With Personal Nutrition Coach"
            )
            st.info("Copy the summary below to share")
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
        st.subheader("Macronutrient Split")
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
                    marker_colors=[
                        '#ff6b6b', '#4ecdc4', '#45b7d1'
                    ]
                )]
            )
            fig_pie.update_layout(
                title=f"Total: {totals['calories']:.0f} kcal",
                height=400
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption(
                "Select foods to display the macronutrient split"
            )
    if recommendations:
        st.subheader("Personalized Recommendations")
        for rec in recommendations:
            st.info(rec)
    with st.expander("Food Choices Today", expanded=True):
        st.subheader("Logged Items")
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
                df_summary,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.caption("No foods logged yet")
else:
    st.info(
        "No foods selected yet. Add items from the categories above to "
        "start tracking"
    )
    st.subheader("Progress Snapshot")
    render_progress_bars(totals, targets)

# -------------------------------------------------------------------------------
# Cell 14: User Feedback Section
# -------------------------------------------------------------------------------

st.divider()
st.header("User Feedback 💬")
st.markdown(
    "Your input helps refine functionality and enhance clarity. Share your "
    "experience or suggestions below"
)
with st.form("feedback_form", clear_on_submit=True):
    feedback_type = st.selectbox(
        "Feedback Type",
        [
            "General Feedback",
            "Bug Report",
            "Feature Request",
            "Success Story"
        ],
        key="feedback_type"
    )
    feedback_text = st.text_area(
        "Your Message",
        placeholder=(
            "Describe your experience, request a feature, or report an issue"
        ),
        height=100,
        key="feedback_text"
    )
    if st.form_submit_button(
        "Submit Feedback",
        type="primary"
    ):
        if feedback_text.strip():
            st.success(
                f"Thank you for your {feedback_type.lower()} submission"
            )
        else:
            st.error("Please enter feedback before submitting")

# -------------------------------------------------------------------------------
# Cell 15: Footer And Additional Resources
# -------------------------------------------------------------------------------

st.divider()
st.markdown(
    """
### Scientific Foundation 📚

This tracker uses the Mifflin St Jeor equation for BMR estimation and
activity multipliers from exercise physiology research for TDEE. Protein
targets draw from consensus position stands of sports nutrition bodies and
weight management literature. Caloric adjustments are intentionally modest
to support sustainability and reduce lean mass loss risk.

### Disclaimer ⚠️

Individual variation exists in metabolic response, nutrient partitioning,
and adaptive thermogenesis. Consult qualified healthcare professionals
before major dietary or training changes, especially if managing medical
conditions. Use subjective feedback along with quantitative data to guide
iterative adjustments.
"""
)
st.success(
    "End Of Session. Continue consistent execution and refine iteratively"
)

# -------------------------------------------------------------------------------
# Cell 16: Session State Management And Performance
# -------------------------------------------------------------------------------

# ------ Clean Up Session State To Prevent Memory Issues ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items()
        if v > 0
    }

# ------ Performance Optimization ------
# Widget keys are explicitly defined throughout for rerun stability

# ------ Session State Cleanup For Temporary Variables ------
temp_keys = [
    key for key in st.session_state.keys() if key.startswith('temp_')
]
for key in temp_keys:
    del st.session_state[key]
