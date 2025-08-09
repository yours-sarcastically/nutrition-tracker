#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker for Goal-Specific Meal Planning
# -----------------------------------------------------------------------------

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

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
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
# Cell 2: Page Configuration and Initial Setup
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
    'sedentary': "🧑‍💻 **Sedentary**: You are basically married to your desk chair.",
    'lightly_active': "🏃 **Lightly Active**: You squeeze in walks or workouts one to three times a week.",
    'moderately_active': "🚴 **Moderately Active**: You are sweating it out three to five days a week.",
    'very_active': "🏋️ **Very Active**: You might actually be part treadmill.",
    'extremely_active': "🤸 **Extremely Active**: You live in the gym and sweat is your second skin."
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
                'placeholder': 'Enter your age', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (in centimeters)',
                      'min': 140, 'max': 220, 'step': 1,
                      'placeholder': 'Enter your height', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (in kilograms)',
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

# ------ Emoji Tooltips ------
EMOJI_TOOLTIPS = {
    '🥇': 'Gold Medal: Nutritional all-star! High in its target nutrient and very calorie-efficient.',
    '🔥': 'High Calorie: One of the more calorie-dense options in its group.',
    '💪': 'High Protein: A true protein powerhouse.',
    '🍚': 'High Carb: A carbohydrate champion.',
    '🥑': 'High Fat: A healthy fat hero.'
}

# ------ Metric Tooltips ------
METRIC_TOOLTIPS = {
    'BMR': 'Basal Metabolic Rate - the energy your body needs just to keep vital functions running',
    'TDEE': 'Total Daily Energy Expenditure - your BMR plus calories burned through activity',
    'Caloric Adjustment': 'How many calories above or below TDEE to reach your goal',
    'Protein': 'Essential for muscle building, repair, and satiety',
    'Carbohydrates': 'Your body\'s preferred energy source for brain and muscle function',
    'Fat': 'Important for hormone production, nutrient absorption, and cell health'
}

# ------ Centralized Tip and Recommendation Content ------
TIPS_CONTENT = {
    'hydration': [
        "**Daily Goal**: Shot for about 35 ml per kilogram of your body weight daily.",
        "**Training Bonus**: Tack on an extra 500-750 ml per hour of sweat time.",
        "**Fat Loss Hack**: Chugging 500 ml of water before meals can boost fullness by by 13 percent. Your stomach will thank you, and so will your waistline."
    ],
    'sleep': [
        "**The Shocking Truth**: Getting less than 7 hours of sleep can torpedo your fat loss by a more than half.",
        "**Daily Goal**: Shoot for 7-9 hours and try to keep a consistent schedule.",
        "**Set the Scene**: Keep your cave dark, cool (18-20°C), and screen-free for at least an hour before lights out."
    ],
    'tracking_wins': [
        "**Morning Ritual**: Weigh yourself first thing after using the bathroom, before eating or drinking, in minimal clothing.",
        "**Look for Trends, Not Blips**: Watch your weekly average instead of getting hung up on daily fluctuations. Your weight can swing 2-3 pounds daily.",
        "**Hold the Line**: Do not tweak your plan too soon. Wait for two or more weeks of stalled progress before making changes."
    ],
    'beyond_the_scale': [
        "**The Bigger Picture**: Snap a few pics every month. Use the same pose, lighting, and time of day. The mirror does not lie.",
        "**Size Up Your Wins**: Measure your waist, hips, arms, and thighs monthly.",
        "**The Quiet Victories**: Pay attention to how you feel. Your energy levels, sleep quality, gym performance, and hunger patterns tell a story numbers cannot."
    ],
    'protein_pacing': [
        "**Spread the Love**: Instead of cramming your protein into one or two giant meals, aim for 20-40 grams with each of your 3-4 daily meals. This works out to roughly 0.4-0.5 grams per kilogram of body weight per meal.",
        "**Frame Your Fitness**: Get some carbs and 20–40g protein before and within two hours of wrapping up your workout.",
        "**The Night Shift**: Try 20-30g of casein protein before bed for keeping your muscles fed while you snooze."
    ],
    'weight_loss_plateau': [
        "**Guess Less, Stress Less**: Before you do anything else, double-check how accurately you are logging your food. Little things can add up!",
        "**Activity Audit**: Take a fresh look at your activity level. Has it shifted?",
        "**Walk it Off**: Try adding 10-15 minutes of walking to your daily routine before cutting calories further. It is a simple way to boost progress without tightening the belt just yet.",
        "**Step Back to Leap Forward**: Consider a diet break every 6-8 weeks. Eating at your maintenance calories for a week or two can give your metabolism and your mind a well-deserved reset.",
        "**Leaf Your Hunger Behind**: Load your plate with low-calorie, high-volume foods like leafy greens, cucumbers, and berries. They are light on calories but big on satisfaction."
    ],
    'weight_gain_stalls': [
        "**Drink Your Calories**: Liquid calories from smoothies, milk, and protein shakes go down way easier than another full meal.",
        "**Fat is Fuel**: Load up healthy fats like nuts, oils, and avocados.",
        "**Push Your Limits**: Give your body a reason to grow! Make sure you are consistently challenging yourself in the gym.",
        "**Turn Up the Heat**: If you have been stuck for over two weeks, bump up your intake by 100-150 calories to get the ball rolling again."
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
    """Initializes all required session state variables."""
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
    """Creates an input widget based on a unified configuration."""
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
        min_val, max_val, step_val = field_config['min'], field_config['max'], field_config['step']
        current_value = st.session_state[session_key]

        if field_name == 'weight_kg' and st.session_state.get('user_units') == 'imperial':
            label = 'Weight (in pounds)'
            min_val, max_val = kg_to_lbs(min_val), kg_to_lbs(max_val)
            step_val = 1.0
            if current_value:
                current_value = kg_to_lbs(current_value)
        elif field_name == 'height_cm' and st.session_state.get('user_units') == 'imperial':
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
        if field_name == 'weight_kg' and st.session_state.get('user_units') == 'imperial' and value:
            value = lbs_to_kg(value)
        elif field_name == 'height_cm' and st.session_state.get('user_units') == 'imperial' and value:
            value = inches_to_cm(value)

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
    """Validate required user inputs and return error messages."""
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
    """Get color for progress bar based on percentage."""
    if percent >= 80:
        return "🟢"  # Green
    elif percent >= 50:
        return "🟡"  # Yellow
    else:
        return "🔴"  # Red

def render_progress_bars(totals, targets):
    """Renders a set of progress bars for all nutrients."""
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals.get(nutrient, 0)
        target = targets.get(config['target_key'], 1)
        target = target if target > 0 else 1  # Avoid division by zero

        percent = min((actual / target) * 100, 100)
        color_indicator = get_progress_color(percent)

        st.progress(
            percent / 100,
            text=(
                f"{color_indicator} {config['label']}: {percent:.0f}% of your daily target "
                f"({target:.0f} {config['unit']})"
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

    # Collect deficits
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
                    help_percentage = min(food[nutrient] / deficit_info['amount'], 1.0)
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
            summary_text = "You still need: " + ", ".join(deficit_summary[:-1]) + f", and {deficit_summary[-1]}."
        else:
            summary_text = f"You still need: {deficit_summary[0]}."

        recommendations.append(summary_text)

        if top_suggestions:
            # First suggestion as smart pick
            first_suggestion = top_suggestions[0]
            food = first_suggestion['food']
            nutrients_helped = first_suggestion['nutrients_helped']

            nutrient_benefits = [f"{food[n]:.0f}g {n}" for n in nutrients_helped]

            if len(nutrient_benefits) > 1:
                benefits_text = ", ".join(nutrient_benefits[:-1]) + f", and {nutrient_benefits[-1]}"
            else:
                benefits_text = nutrient_benefits[0]

            recommendations.append(
                f"🎯 Smart pick: One serving of {food['name']} would give you {benefits_text}, "
                f"knocking out multiple targets at once!"
            )

            # Consolidate alternative options into one statement
            if len(top_suggestions) > 1:
                alternative_foods = []
                for suggestion in top_suggestions[1:]:
                    food = suggestion['food']
                    nutrients_helped = suggestion['nutrients_helped']
                    nutrient_benefits = [f"{food[n]:.0f}g {n}" for n in nutrients_helped]

                    if len(nutrient_benefits) > 1:
                        benefits_text = ", ".join(nutrient_benefits[:-1]) + f", and {nutrient_benefits[-1]}"
                    else:
                        benefits_text = nutrient_benefits[0]

                    alternative_foods.append(f"{food['name']} (provides {benefits_text})")

                # Create single consolidated alternative statement
                if len(alternative_foods) == 1:
                    alternatives_text = alternative_foods[0]
                elif len(alternative_foods) == 2:
                    alternatives_text = f"{alternative_foods[0]} or {alternative_foods[1]}"
                else:
                    alternatives_text = ", ".join(alternative_foods[:-1]) + f", or {alternative_foods[-1]}"

                recommendations.append(
                    f"💡 Alternative options: {alternatives_text} are also great ways to hit multiple goals!"
                )
        else:
            biggest_deficit = max(deficits.items(), key=lambda x: x[1]['amount'])
            nutrient, deficit_info = biggest_deficit

            best_single_food = max(
                all_foods,
                key=lambda x: x.get(nutrient, 0),
                default=None
            )

            if best_single_food and best_single_food.get(nutrient, 0) > 0:
                recommendations.append(
                    f"💡 Alternative option: Try adding {best_single_food['name']} - it is packed with "
                    f"{best_single_food[nutrient]:.0f}g of {deficit_info['label']}."
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
    """Save current progress to JSON."""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'food_selections': food_selections,
        'user_inputs': user_inputs
    }
    return json.dumps(progress_data, indent=2)

def load_progress_from_json(json_data):
    """Load progress from JSON data."""
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

    # Prepare nutrition summary
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

    # Prepare consumed foods list
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
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Title
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Daily Nutrition Summary")

    # Date
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d')}")

    # User info
    y_pos = height - 120
    p.drawString(50, y_pos, f"Age: {user_info.get('age', 'N/A')}")
    p.drawString(200, y_pos, f"Weight: {user_info.get('weight_kg', 'N/A')} kg")
    p.drawString(350, y_pos, f"Goal: {user_info.get('goal', 'N/A')}")

    # Nutrition summary
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

    # Selected foods
    if summary_data['consumed_foods']:
        y_pos -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Foods Consumed")

        y_pos -= 30
        p.setFont("Helvetica", 10)
        for item in summary_data['consumed_foods'][:20]:  # Limit to prevent overflow
            p.drawString(50, y_pos, f"• {item['name']}: {item['servings']} serving(s)")
            y_pos -= 15
            if y_pos < 50:  # Prevent going off page
                break

    p.save()
    buffer.seek(0)
    return buffer

def create_csv_summary(totals, targets, selected_foods):
    """Create a CSV summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    data = []

    # Add nutrition summary from prepared data
    for item in summary_data['nutrition_summary']:
        data.append({
            'Category': 'Nutrition Summary',
            'Item': item['label'],
            'Actual': f"{item['actual']:.0f} {item['unit']}",
            'Target': f"{item['target']:.0f} {item['unit']}",
            'Percentage': f"{item['percent']:.0f}%"
        })

    # Add selected foods from prepared data
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
    # contains approximately 7,700 kilocalories
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
    """Filter foods based on search term."""
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
        emoji_with_tooltip = food.get('emoji', '')
        if emoji_with_tooltip and emoji_with_tooltip in EMOJI_TOOLTIPS:
            # Use st.subheader for consistency
            st.subheader(f"{emoji_with_tooltip} {food['name']}")
            st.caption(EMOJI_TOOLTIPS[emoji_with_tooltip])
        else:
            st.subheader(f"{emoji_with_tooltip} {food['name']}")

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
                min_value=0.0, max_value=20.0,  # Capped max servings
                value=float(current_serving), step=0.5,
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

# -----------------------------------------------------------------------------
# Cell 8: Initialize Application
# -----------------------------------------------------------------------------

# ------ Initialize Session State ------
initialize_session_state()

# ------ Load Food Database and Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# ------ Apply Custom CSS for Enhanced Styling ------
st.markdown("""
<style>
html {
    font-size: 110%; /* Makes all text 10% larger */
}

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
# Cell 9: Application Title and Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
A Smart, Evidence-Based Nutrition Tracker That Actually Gets You

Welcome aboard!

Hey there! Welcome to your new nutrition buddy. This is not just another calorie counter—it is your personalized guide, built on rock-solid science to help you smash your goals. Whether you are aiming to shed a few pounds, hold steady, or bulk up, we have crunched the numbers so you can focus on enjoying your food.

Let us get rolling—your journey to feeling awesome starts now! 🚀
""")

# ------ Sidebar for User Input ------
st.sidebar.header("Let's Get Personal 📊")

# Units toggle - now using switch instead of radio buttons
units = st.sidebar.toggle(
    "Use Imperial Units",
    value=(st.session_state.get('user_units', 'metric') == 'imperial'),
    key='units_toggle',
    help="Toggle on for Imperial (lbs, inches) or off for Metric (kg, cm)"
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

# Calculate button with enhanced validation
if st.sidebar.button("🧮 Calculate My Targets", type="primary", key="calculate_button"):
    validation_errors = validate_user_inputs(all_inputs)
    st.session_state.form_errors = validation_errors
    if not validation_errors:
        st.session_state.form_submitted = True
        st.session_state.show_motivational_message = True
    else:
        st.session_state.form_submitted = False
    st.rerun()

# Display validation errors from session state
if st.session_state.get('form_errors'):
    for error in st.session_state.form_errors:
        st.sidebar.error(f"• {error}")

# Save/Load Progress - Save button first, then Load JSON section below
st.sidebar.divider()
st.sidebar.subheader("💾 Save Your Progress")

if st.sidebar.button("Save", key="save_progress", type="primary"):
    progress_json = save_progress_to_json(st.session_state.food_selections, all_inputs)
    st.sidebar.download_button(
        "📥 Download",
        data=progress_json,
        file_name=f"nutrition_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        key="download_progress"
    )

# Load JSON section below the save button
st.sidebar.subheader("📂 Load Progress")
uploaded_file = st.sidebar.file_uploader("Load", type="json", key="upload_progress")
if uploaded_file is not None:
    content = uploaded_file.read().decode()
    food_selections, user_inputs = load_progress_from_json(content)

    # Update session state
    st.session_state.food_selections.update(food_selections)
    for key, value in user_inputs.items():
        if f'user_{key}' in st.session_state:
            st.session_state[f'user_{key}'] = value

    st.sidebar.success("Progress loaded successfully! 📂")
    st.rerun()

# ------ Activity Level Guide in Sidebar ------
with st.sidebar.container(border=True):
    st.markdown("#### Your Activity Level Decoded")

    # Dynamically generate the list from the updated dictionary
    for key in ACTIVITY_MULTIPLIERS:  # Iterate in a consistent order
        description = ACTIVITY_DESCRIPTIONS.get(key, "")
        st.markdown(f"* {description}")

    st.markdown("""
    💡 *If you are torn between two levels, pick the lower one. It is better to underestimate your burn than to overeat and stall.*
    """)

# ------ Dynamic Sidebar Summary ------
if st.session_state.form_submitted:
    final_values = get_final_values(all_inputs)
    targets = calculate_personalized_targets(**final_values)
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)

    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Quick Summary")

    progress_calories = min(totals['calories'] / targets['total_calories'] * 100, 100) if targets['total_calories'] > 0 else 0
    progress_protein = min(totals['protein'] / targets['protein_g'] * 100, 100) if targets['protein_g'] > 0 else 0

    st.sidebar.metric(
        "Calories Progress",
        f"{progress_calories:.0f}%",
        f"{totals['calories']:.0f}/{targets['total_calories']:.0f} kcal"
    )
    st.sidebar.metric(
        "Protein Progress",
        f"{progress_protein:.0f}%",
        f"{totals['protein']:.0f}/{targets['protein_g']:.0f} g"
    )

# ------ Process Final Values ------
final_values = get_final_values(all_inputs)

# ------ Check for User Input ------
user_has_entered_info = st.session_state.form_submitted

# ------ Calculate Personalized Targets ------
targets = calculate_personalized_targets(**final_values)

# Show motivational message
if st.session_state.show_motivational_message and user_has_entered_info:
    goal_messages = {
        'weight_loss': f"🎉 Awesome! You are set up for success! With your plan, you are on track to lose approximately {abs(targets['estimated_weekly_change']):.2f} kg per week. Stay consistent and you have got this! 💪",
        'weight_maintenance': f"🎯 Perfect! Your maintenance plan is locked and loaded! You are all set to maintain your current weight of {format_weight(final_values['weight_kg'], st.session_state.get('user_units', 'metric'))} while optimizing your nutrition. ⚖️",
        'weight_gain': f"💪 Let us grow! Your muscle-building journey starts now! You are targeting a healthy gain of about {targets['estimated_weekly_change']:.2f} kg per week. Fuel up and lift heavy! 🚀"
    }

    message = goal_messages.get(targets['goal'], "🚀 You are all set! Let us crush those nutrition goals!")
    st.success(message)

    # Reset the flag so message does not show on every rerun
    if st.button("✨ Got it!", key="dismiss_message"):
        st.session_state.show_motivational_message = False
        st.rerun()

# -----------------------------------------------------------------------------
# Cell 10: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info(
        "👈 Enter your details in the sidebar and click 'Calculate My Targets' to get your personalized daily targets."
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
    "🎯 **The 80/20 Rule**: Try to hit your targets about 80 percent of the time. This gives you wiggle room for birthday cake, date nights, and those inevitable moments when life throws you a curveball. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap."
)

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Unified Metrics Display Configuration ------
units_display = st.session_state.get('user_units', 'metric')
weight_display = format_weight(final_values['weight_kg'], units_display)

# Updated metrics config - weight added to Metabolic Information, 5 columns maintained
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Weight", weight_display),
            ("BMR", f"{targets['bmr']} kcal"),
            ("TDEE", f"{targets['tdee']} kcal"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
            ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kg")
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
# Cell 11: Enhanced Evidence-Based Tips and Context
# -----------------------------------------------------------------------------

with st.expander("📚 Your Evidence-Based Game Plan", expanded=False):
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "The Big Three to Win At Nutrition 🏆", "Level Up Your Progress Tracking 📊",
        "Mindset Is Everything 🧠", "🧗 Own Your Energy", "The Science Behind the Magic 🔬"
    ])

    with tab1:
        # Heading
        st.subheader("💧 Master Your Hydration Game")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['hydration']:
            st.markdown(f"• {tip}")
        
        # Divider between sections
        st.divider()

        # Heading
        st.subheader("😴 Sleep Like Your Goals Depend on It")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['sleep']:
            st.markdown(f"• {tip}")

        # Divider between sections
        st.divider()

        # Heading
        st.subheader("📅 Follow Your Wins")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['tracking_wins']:
            st.markdown(f"• {tip}")

    with tab2:
        # Heading
        st.subheader("📸 Go Beyond the Scale")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['beyond_the_scale']:
            st.markdown(f"• {tip}")

    with tab3:
        # Heading
        st.subheader("🧠 Mindset Is Everything")
        
        # Main content paragraph
        st.markdown("""
        The 80/20 principle is your best defense against the perfectionist trap. It is about ditching that mindset that makes you throw in the towel after one "bad" meal. Instead of trying to master everything at once, build your habits gradually and you will be far more likely to stick with them for the long haul.
        
        **Start Small, Win Big:**
        • **Weeks 1–2**: Your only job is to focus on hitting your calorie targets. Do not worry about anything else!
        • **Weeks 3–4**: Once calories feel like second nature, start layering in protein tracking.
        • **Week 5 and Beyond**: With calories and protein in the bag, you can now fine-tune your carb and fat intake.
        """)
        
        # Divider between sections
        st.divider()

        # Subheading
        st.subheader("🔄 When Progress Stalls")
        
        # Sub-subheading
        st.markdown("#### Hit a Weight Loss Plateau?")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['weight_loss_plateau']:
            st.markdown(f"• {tip}")
        
        # Sub-subheading
        st.markdown("#### Struggling to Gain Weight?")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['weight_gain_stalls']:
            st.markdown(f"• {tip}")

        # Divider between sections
        st.divider()

        # Subheading
        st.subheader("💪 Pace Your Protein")
        
        # Content with consistent bullet formatting
        for tip in TIPS_CONTENT['protein_pacing']:
            st.markdown(f"• {tip}")

    with tab4:
        # Heading
        st.subheader("🧗 Own Your Energy")
        
        # Sub-subheading
        st.markdown("#### Build Your Foundation with Resistance Training")
        
        # Main content paragraph
        st.markdown("""
        This is your non-negotiable, no matter your goal. Lifting weights (or using your bodyweight!) tells your body to build or hold onto precious muscle, which is the engine of your metabolism.
        
        • **🎯 For Fat Loss**: More muscle means you burn more calories even while you are chilling on the couch. It is the secret to keeping the weight off for good.
        • **💪 For Bulking Up**: Exercise tells your body where to send all that protein you are eating. Without it, you are just a shaky tent in a windstorm.
        • **💡 The Game Plan**: A good starting point is **2-3 sessions of 20-40 minutes** a week. Push, pull, squat—make those muscles sing!
        • **❤️ Find What You Love**: If it feels like torture, switch it up. Fitness should spark joy, not dread! Dance, bike, chase your dog. Pick whatever makes you grin!
        """)

        # Divider between sections
        st.divider()
        
        # Sub-subheading
        st.markdown("#### NEAT: Your Sneaky Fitness Piggy Bank")
        
        # Main content paragraph
        st.markdown("""
        NEAT stands for Non-Exercise Activity Thermogenesis. It is a fancy term for all the calories you burn just by living your life.
        
        • Adding just **10-20 minutes of walking** to your day can be the difference between a plateau and progress. It is low-stress and the results add up big time.
        
        **Pro Tip**: The best workout is the one you will actually do. Make it fun, and you are golden!
        """)

    with tab5:
        # Heading
        st.subheader("🔬 Understanding Your Metabolism")
        
        # Main content paragraph
        st.markdown("""
        Your Basal Metabolic Rate (BMR) is the energy your body needs just to keep the lights on. Your Total Daily Energy Expenditure (TDEE) builds on that baseline by factoring in how active you are throughout the day.
        """)

        # Divider between sections
        st.divider()

        # Sub-subheading
        st.markdown("#### The Smart Eater's Cheat Sheet")
        
        # Main content paragraph
        st.markdown("""
        Not all calories are created equal. Some foods fill you up, while others leave you rummaging through the pantry an hour later. Here is the pecking order:
        
        • **Protein**: Protein is the undisputed king of fullness! It digests slowly, steadies blood sugar, and even burns a few extra calories in the process. Eggs, Greek yogurt, chicken, tofu, and lentils are all your hunger-busting best friends.
        • **Fiber-Rich Carbohydrates**: Veggies, fruits, and whole grains are the unsung heroes of fullness. They fill you up, slow things down, and bulk up meals without blowing your calorie budget.
        • **Healthy Fats**: Think of nuts, olive oil, and avocados as the smooth operators delivering steady, long-lasting energy that keeps you powered throughout the day.
        • **Processed Stuff**: These foods promise the world but leave you hanging. They are fine for a cameo appearance, but you cannot build a winning strategy around them.
        
        As a great rule of thumb, aim for 14 grams of fiber for every 1,000 calories you consume, which usually lands between 25 and 38 grams daily. Ramp up gradually to avoid digestive drama.
        """)

        # Divider between sections
        st.divider()

        # Sub-subheading
        st.markdown("#### Your Nutritional Supporting Cast")
        
        # Main content paragraph
        st.markdown("""
        Going plant-based? There are a few tiny but mighty micronutrients to keep an eye on. They may not get top billing, but they are essential for keeping the show running smoothly.
        
        **The Watch List:**
        
        • **B₁₂**: B₁₂ keeps your cells and nerves firing like a well-oiled machine. It is almost exclusively found in animal products, so if you are running a plant-powered show, you will need reinforcements. A trusty supplement is often the easiest way to keep your levels topped up and your brain buzzing.
        • **Iron**: Iron is the taxi service that shuttles oxygen all over your body. When it is running low, you will feel like a sloth on a Monday morning. Load up on leafy greens, lentils, and fortified grains, and team them with a hit of vitamin C—think bell peppers or citrus—to supercharge absorption.
        • **Calcium**: This multitasker helps build bones, power muscles, and keeps your heart thumping to a steady beat. While dairy is the classic go-to, you can also get your fix from kale, almonds, tofu, and fortified plant milks.
        • **Zinc**: Think of zinc as your immune system's personal security detail. You will find it hanging out in nuts, seeds, and whole grains. Keep your zinc levels up, and you will be dodging colds like a ninja.
        • **Iodine**: Your thyroid is the command center for your metabolism, and iodine is its right-hand mineral. A pinch of iodized salt is usually all it takes.
        • **Omega-3s (EPA/DHA)**: These healthy fats are premium fuel for your brain, heart, and emotional well-being. If fish is not on your plate, fortified foods or supplements can help you stay sharp and serene.
        
        The good news? Fortified foods and targeted supplements have your back. Plant milks, cereals, and nutritional yeast are often spiked with B₁₂, calcium, or iodine. Supplements are a safety net, but do not overdo it. It is always best to chat with a doctor or dietitian to build a plan that is right for you.
        """)

# -----------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")

# Food search functionality
search_col, reset_col = st.columns([3, 1])

with search_col:
    search_term = st.text_input(
        "Search for foods",  # The label is still needed for Streamlit's backend
        placeholder="🔍 Type food name to filter results...", # Visual text is here
        key="food_search_input",
        label_visibility="collapsed"  # This hides the label from the UI
    )
    st.session_state.food_search = search_term

with reset_col:
    if st.button("🔄 Clear Search", key="clear_search", type="primary"):
        st.session_state.food_search = ""
        st.rerun()

st.markdown(
    "Pick how many servings of each food you are having to see how your choices stack up against your daily targets."
)

# Emoji guide
with st.expander("💡 Need a hand with food choices? Check out the emoji guide below!"):
    # Dynamically generate the guide from the EMOJI_TOOLTIPS dictionary
    for emoji, tooltip in EMOJI_TOOLTIPS.items():
        # The tooltip often includes a label like "Gold Medal:", so we extract it
        label = tooltip.split(':')[0]
        description = ':'.join(tooltip.split(':')[1:]).strip()
        st.markdown(f"* **{emoji} {label}**: {description}")

if st.button("🔄 Start Fresh: Reset All Food Selections", type="primary", key="reset_foods"):
    st.session_state.food_selections = {}
    st.rerun()

# Filter foods based on search
filtered_foods = filter_foods_by_search(foods, search_term)

if not filtered_foods and search_term:
    st.warning(f"No foods found matching '{search_term}'. Try a different search term or clear the search.")
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
                CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']
            )
        )
        with tabs[i]:
            render_food_grid(sorted_items_in_category, category, columns=2)

# -----------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking
# -----------------------------------------------------------------------------

st.header("Today's Scorecard 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)

    # Export functionality with Share button
    st.subheader("📥 Export Your Summary")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Download PDF Report", key="export_pdf", use_container_width=True):
            pdf_buffer = create_pdf_summary(totals, targets, selected_foods, final_values)
            st.download_button(
                "📥 Download PDF",
                data=pdf_buffer,
                file_name=f"nutrition_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="download_pdf_button"
            )

    with col2:
        if st.button("📊 Download CSV Data", key="export_csv", use_container_width=True):
            csv_data = create_csv_summary(totals, targets, selected_foods)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"nutrition_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_csv_button"
            )

    with col3:
        if st.button("📱 Share Progress", key="share_progress", use_container_width=True):
            share_text = f"""
🍽️ My Nutrition Progress - {datetime.now().strftime('%Y-%m-%d')}

📊 Today's Intake:
- Calories: {totals['calories']:.0f} / {targets['total_calories']:.0f} kcal
- Protein: {totals['protein']:.0f} / {targets['protein_g']:.0f} g
- Carbs: {totals['carbs']:.0f} / {targets['carb_g']:.0f} g
- Fat: {totals['fat']:.0f} / {targets['fat_g']:.0f} g

Created with Personal Nutrition Coach 🍽️
            """
            st.info("Copy the summary below to share! 📋")
            st.text_area("Shareable Summary:", share_text, height=200, label_visibility="collapsed")

    st.divider()
    col1, col2 = st.columns([1, 1])

    # Enhanced Visualization Dashboard
    with col1:
        st.subheader("Today's Fuel Mix")
        fig_macros = go.Figure()

        macros = ['Protein', 'Carbohydrates', 'Fat']
        actual_values = [totals['protein'], totals['carbs'], totals['fat']]
        target_values = [targets['protein_g'], targets['carb_g'], targets['fat_g']]

        fig_macros.add_trace(go.Bar(
            name='Actual', x=macros, y=actual_values, marker_color='#ff6b6b'
        ))

        fig_macros.add_trace(go.Bar(
            name='Target', x=macros, y=target_values, marker_color='#4ecdc4'
        ))

        fig_macros.update_layout(
            title_text='Macronutrient Comparison', barmode='group',
            yaxis_title='Grams', height=400, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig_macros, use_container_width=True)

    with col2:
        st.subheader("Your Macronutrient Split")
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
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
                title=f'Total: {totals["calories"]:.0f} kcal',
                height=400
            )

            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.caption("Please select foods to see the macronutrient split.")

    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)

    # Food Selection Summary Table
    with st.expander("Your Food Choices Today", expanded=True):
        st.subheader("What You've Logged")

        # Reuse the existing helper function to get calculated food data
        prepared_data = prepare_summary_data(totals, targets, selected_foods)
        consumed_foods_list = prepared_data['consumed_foods']

        # Reformat the data for display in the DataFrame
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
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
        else:
            st.caption("No foods logged yet.")

else:
    st.info(
        "Have not picked any foods yet? No worries! Go ahead and add some items from the categories above to start tracking your intake! 🍎"
    )
    st.subheader("Progress Snapshot")
    render_progress_bars(totals, targets)

# -----------------------------------------------------------------------------
# Cell 14: User Feedback Section
# -----------------------------------------------------------------------------

st.divider()
st.header("💬 Help Us Improve!")
st.markdown("Your feedback helps us make this app even better. Share your thoughts below:")

with st.form("feedback_form", clear_on_submit=True):
    feedback_type = st.selectbox(
        "What type of feedback would you like to share?",
        ["General Feedback", "Bug Report", "Feature Request", "Success Story"],
        key="feedback_type"
    )

    feedback_text = st.text_area(
        "How can we improve?",
        placeholder="Tell us about your experience, suggest new features, or report any issues you encountered...",
        height=100,
        key="feedback_text"
    )

    if st.form_submit_button("📤 Submit Feedback", type="primary"):
        if feedback_text.strip():
            # In a real app, this would save to a database
            st.success(f"Thank you for your {feedback_type.lower()}! Your input helps us make the app better for everyone. 🙏")
        else:
            st.error("Please enter some feedback before submitting.")

# -----------------------------------------------------------------------------
# Cell 15: Footer and Additional Resources
# -----------------------------------------------------------------------------

st.divider()
st.markdown("""
### The Science We Stand On 📚

This tracker is not built on guesswork—it is grounded in peer-reviewed research and evidence-based guidelines. We rely on the Mifflin-St Jeor equation to calculate your Basal Metabolic Rate (BMR). This method is widely regarded as the gold standard and is strongly endorsed by the Academy of Nutrition and Dietetics. To estimate your Total Daily Energy Expenditure (TDEE), we use well-established activity multipliers derived directly from exercise physiology research. For protein recommendations, our targets are based on official guidelines from the International Society of Sports Nutrition.

When it comes to any calorie adjustments, we stick to conservative, sustainable rates that research has consistently shown lead to lasting, meaningful results. We are all about setting you up for success, one step at a time!

### The Fine Print ⚠️

Think of this tool as your launchpad, but remember—everyone is different. Your mileage may vary due to factors like genetics, health conditions, medications, and other factors that a calculator simply cannot account for. It is always wise to consult a qualified healthcare provider before making any big dietary shifts. Above all, tune into your body—keep tabs on your energy levels, performance, and tweak things as needed. We are here to help, but you know yourself best!
""")

st.success(
    "You made it to the finish line! Thanks for sticking with us on this nutrition adventure. Remember, the sun does not rush to rise, but it always shows up. Keep shining—you have got this! 🥳"
)

# -----------------------------------------------------------------------------
# Cell 16: Session State Management and Performance
# -----------------------------------------------------------------------------

# ------ Clean Up Session State to Prevent Memory Issues ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# ------ Performance Optimization ------
# Ensure all widgets have unique keys to reduce reruns
# This is handled throughout the code with explicit key parameters

# ------ Session state cleanup for unused variables ------
# Remove any temporary variables that might accumulate
temp_keys = [key for key in st.session_state.keys() if key.startswith('temp_')]
for key in temp_keys:
    del st.session_state[key]
