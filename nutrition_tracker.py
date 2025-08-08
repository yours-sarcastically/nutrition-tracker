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
import json
import io
from datetime import datetime
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

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


# ---------------------------------------------------------------------------
# Cell 4: Unit Conversion Functions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Cell 5: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables."""
    session_vars = (
        ['food_selections', 'form_submitted', 'show_motivational_message', 'food_search'] +
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


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")

    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation and building',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and overall health'
    }

    deficits = {}
    
    # First pass: create progress bars and collect deficits
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
        percent = min(actual / target * 100, 100)
        color_indicator = get_progress_color(percent)

        st.progress(
            percent / 100,
            text=(
                f"{color_indicator} {config['label']}: {percent:.0f}% of your daily target "
                f"({target:.0f} {config['unit']})"
            )
        )

        if actual < target:
            deficit = target - actual
            deficits[nutrient] = {
                'amount': deficit,
                'unit': config['unit'],
                'label': config['label'].lower(),
                'purpose': purpose_map.get(nutrient, 'for optimal nutrition')
            }

    # Second pass: create combined recommendations with multiple suggestions
    if deficits:
        # Find the best multi-nutrient food suggestions
        all_foods = [item for sublist in foods.values() for item in sublist]
        food_suggestions = []
        
        for food in all_foods:
            coverage_score = 0
            nutrients_helped = []
            
            for nutrient, deficit_info in deficits.items():
                if nutrient != 'calories' and food[nutrient] > 0:
                    # Calculate how much this food helps with this deficit
                    help_percentage = min(food[nutrient] / deficit_info['amount'], 1.0)
                    if help_percentage > 0.1:  # Only consider if it helps by at least 10%
                        coverage_score += help_percentage
                        nutrients_helped.append(nutrient)
            
            if coverage_score > 0 and len(nutrients_helped) > 1:
                food_suggestions.append({
                    'food': food,
                    'nutrients_helped': nutrients_helped,
                    'score': coverage_score
                })
        
        # Sort by coverage score and take top suggestions
        food_suggestions.sort(key=lambda x: x['score'], reverse=True)
        top_suggestions = food_suggestions[:3]  # Get top 3 suggestions

        # Create summary message
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
        
        # Add multiple smart food suggestions
        if top_suggestions:
            for i, suggestion in enumerate(top_suggestions):
                food = suggestion['food']
                nutrients_helped = suggestion['nutrients_helped']
                
                nutrient_benefits = []
                for nutrient in nutrients_helped:
                    amount = food[nutrient]
                    nutrient_benefits.append(f"{amount:.0f}g {nutrient}")
                
                if len(nutrient_benefits) > 1:
                    benefits_text = ", ".join(nutrient_benefits[:-1]) + f", and {nutrient_benefits[-1]}"
                else:
                    benefits_text = nutrient_benefits[0]
                
                if i == 0:
                    recommendations.append(
                        f"🎯 Smart pick: One serving of {food['name']} would give you {benefits_text}, "
                        f"knocking out multiple targets at once!"
                    )
                else:
                    recommendations.append(
                        f"💡 Alternative option: {food['name']} provides {benefits_text}, "
                        f"another great way to hit multiple goals!"
                    )
        else:
            # Fallback to individual suggestions for the biggest deficit
            biggest_deficit = max(deficits.items(), key=lambda x: x[1]['amount'])
            nutrient, deficit_info = biggest_deficit
            
            best_single_food = max(
                all_foods, 
                key=lambda x: x[nutrient] if x[nutrient] > 0 else 0,
                default=None
            )
            
            if best_single_food and best_single_food[nutrient] > 0:
                recommendations.append(
                    f"💡 Try adding {best_single_food['name']} - it's packed with "
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


def generate_personalized_recommendations(totals, targets, final_values):
    """Generates personalized tips based on current intake and goals."""
    recommendations = []
    goal = final_values['goal']

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


def create_pdf_summary(totals, targets, selected_foods, user_info):
    """Create a PDF summary of the daily nutrition."""
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
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = (actual / target * 100) if target > 0 else 0
        
        p.drawString(50, y_pos, f"{config['label']}: {actual:.0f}/{target:.0f} {config['unit']} ({percent:.0f}%)")
        y_pos -= 20
    
    # Selected foods
    if selected_foods:
        y_pos -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Foods Consumed")
        
        y_pos -= 30
        p.setFont("Helvetica", 10)
        for item in selected_foods[:20]:  # Limit to prevent overflow
            food = item['food']
            servings = item['servings']
            p.drawString(50, y_pos, f"• {food['name']}: {servings} serving(s)")
            y_pos -= 15
            if y_pos < 50:  # Prevent going off page
                break
    
    p.save()
    buffer.seek(0)
    return buffer


def create_csv_summary(totals, targets, selected_foods):
    """Create a CSV summary of the daily nutrition."""
    data = []
    
    # Add nutrition summary
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = (actual / target * 100) if target > 0 else 0
        
        data.append({
            'Category': 'Nutrition Summary',
            'Item': config['label'],
            'Actual': f"{actual:.0f} {config['unit']}",
            'Target': f"{target:.0f} {config['unit']}",
            'Percentage': f"{percent:.0f}%"
        })
    
    # Add selected foods
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        total_cals = food['calories'] * servings
        
        data.append({
            'Category': 'Foods Consumed',
            'Item': food['name'],
            'Servings': servings,
            'Calories': f"{total_cals:.0f} kcal",
            'Protein': f"{food['protein'] * servings:.1f} g",
            'Carbs': f"{food['carbs'] * servings:.1f} g",
            'Fat': f"{food['fat'] * servings:.1f} g"
        })
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


# ---------------------------------------------------------------------------
# Cell 6: Nutritional Calculation Functions
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
# Cell 7: Food Database Processing Functions
# ---------------------------------------------------------------------------

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
            st.markdown(f"**{emoji_with_tooltip}** {food['name']}")
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
# Cell 8: Initialize Application
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Cell 9: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
A Smart, Evidence-Based Nutrition Tracker That Actually Gets You

Welcome aboard!

Hey there! Welcome to your new nutrition buddy. This isn't just another calorie counter—it's your personalized guide, built on rock-solid science to help you smash your goals. Whether you're aiming to shed a few pounds, hold steady, or bulk up, we've crunched the numbers so you can focus on enjoying your food.

Let's get rolling—your journey to feeling awesome starts now! 🚀
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

# Calculate button with validation
if st.sidebar.button("🧮 Calculate My Targets", type="primary", key="calculate_button"):
    validation_errors = validate_user_inputs(all_inputs)
    if validation_errors:
        for error in validation_errors:
            st.sidebar.error(error)
    else:
        st.session_state.form_submitted = True
        st.session_state.show_motivational_message = True
        st.rerun()

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
    
    st.sidebar.success("Progress loaded successfully!")
    st.rerun()

# ------ Activity Level Guide in Sidebar ------
st.sidebar.divider()
with st.sidebar.container(border=True):
    st.markdown("##### Your Activity Level Decoded")
    st.markdown("""
* **🧑‍💻 Sedentary**: You're basically married to your desk chair
* **🏃 Lightly Active**: You squeeze in walks or workouts one to three times a week
* **🚴 Moderately Active**: You're sweating it out three to five days a week
* **🏋️ Very Active**: You might actually be part treadmill
* **🤸 Extremely Active**: You live in the gym and sweat is your second skin

💡 If you're torn between two levels, pick the lower one. It's better to underestimate your burn than to overeat and stall.*
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
        'weight_loss': f"🎉 Awesome! You're set up for success! With your plan, you're on track to lose approximately {abs(targets['estimated_weekly_change']):.2f} kg per week. Stay consistent and you've got this!",
        'weight_maintenance': f"🎯 Perfect! Your maintenance plan is locked and loaded! You're all set to maintain your current weight of {format_weight(final_values['weight_kg'], st.session_state.get('user_units', 'metric'))} while optimizing your nutrition.",
        'weight_gain': f"💪 Let's grow! Your muscle-building journey starts now! You're targeting a healthy gain of about {targets['estimated_weekly_change']:.2f} kg per week. Fuel up and lift heavy!"
    }
    
    message = goal_messages.get(targets['goal'], "🚀 You're all set! Let's crush those nutrition goals!")
    st.success(message)
    
    # Reset the flag so message doesn't show on every rerun
    if st.button("✨ Got it!", key="dismiss_message"):
        st.session_state.show_motivational_message = False
        st.rerun()

# ---------------------------------------------------------------------------
# Cell 10: Unified Target Display System
# ---------------------------------------------------------------------------

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
    "🎯 **The 80/20 Rule**: Try to hit your targets about 80% of the time. This gives you wiggle room for birthday cake, date nights, and those inevitable moments when life throws you a curveball. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap."
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

# ---------------------------------------------------------------------------
# Cell 11: Enhanced Evidence-Based Tips and Context (Collapsed by default)
# ---------------------------------------------------------------------------

with st.expander("📚 Your Evidence-Based Game Plan", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs([
        "The Big Three to Win At Nutrition 🏆", "Level Up Your Progress Tracking 📊",
        "Mindset Is Everything 🧠", "The Science Behind the Magic 🔬"
    ])

    with tab1:
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

    with tab2:
        st.subheader("Go Beyond the Scale 📸")
        st.markdown("""
        The Bigger Picture: Snap a few pics every month. Use the same pose, lighting, and time of day. The mirror doesn't lie.
        Size Up Your Wins: Measure your waist, hips, arms, and thighs monthly
        The Quiet Victories: Pay attention to how you feel. Your energy levels, sleep quality, gym performance, and hunger patterns tell a story numbers can't.
        """)

    with tab3:
        st.subheader("Mindset Is Everything 🧠")
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

    with tab4:
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
# Cell 12: Personalized Recommendations System
# ---------------------------------------------------------------------------

if user_has_entered_info:
    st.header("Your Personalized Action Steps 🎯")
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
    recommendations = generate_personalized_recommendations(
        totals, targets, final_values
    )
    for rec in recommendations:
        st.info(rec)

# ---------------------------------------------------------------------------
# Cell 13: Food Selection Interface
# ---------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")

# Food search functionality
search_col, reset_col = st.columns([3, 1])
with search_col:
    search_term = st.text_input(
        "🔍 Search for foods",
        value=st.session_state.food_search,
        placeholder="Type food name to filter results...",
        key="food_search_input"
    )
    st.session_state.food_search = search_term

with reset_col:
    st.write("")  # Spacer
    st.write("")  # Spacer
    if st.button("🔄 Clear Search", key="clear_search"):
        st.session_state.food_search = ""
        st.rerun()

st.markdown(
    "Pick how many servings of each food you're having to see how your choices stack up against your daily targets."
)

with st.expander("💡 Need a hand with food choices? Check out the emoji guide below!"):
    st.markdown("""
    * **🥇 Gold Medal**: A nutritional all-star! High in its target nutrient and very calorie-efficient.
    * **🔥 High Calorie**: One of the more calorie-dense options in its group.
    * **💪 High Protein**: A true protein powerhouse.
    * **🍚 High Carb**: A carbohydrate champion.
    * **🥑 High Fat**: A healthy fat hero.
    """)

if st.button("🔄 Start Fresh: Reset All Food Selections", type="secondary", key="reset_foods"):
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

# ---------------------------------------------------------------------------
# Cell 14: Daily Summary and Progress Tracking
# ---------------------------------------------------------------------------

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
        if st.button("📄 Download PDF Report", key="export_pdf"):
            pdf_buffer = create_pdf_summary(totals, targets, selected_foods, final_values)
            st.download_button(
                "📥 Download PDF",
                data=pdf_buffer,
                file_name=f"nutrition_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                key="download_pdf_button"
            )
    
    with col2:
        if st.button("📊 Download CSV Data", key="export_csv"):
            csv_data = create_csv_summary(totals, targets, selected_foods)
            st.download_button(
                "📥 Download CSV",
                data=csv_data,
                file_name=f"nutrition_data_{datetime.now().strftime('%Y%m%d')}.csv",
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
        st.subheader("Your Macronutrient Split (in grams)")
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
        if sum(macro_values) > 0:
            fig = go.Figure(go.Pie(
                labels=['Protein', 'Carbs', 'Fat'],
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

    with st.expander("Your Food Choices Today"):
        st.subheader("What You've Logged")
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
        "Haven't picked any foods yet? No worries! Go ahead and add some items from the categories above to start tracking your intake!"
    )
    st.subheader("Progress Snapshot")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
        st.progress(
            0.0,
            text=(
                f"🔴 {config['label']}: 0% of daily target ({target:.0f} "
                f"{config['unit']})"
            )
        )

# ---------------------------------------------------------------------------
# Cell 15: User Feedback Section
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Cell 16: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()
st.markdown("""
### The Science We Stand On 📚

This tracker isn't built on guesswork—it's grounded in peer-reviewed research and evidence-based guidelines. We rely on the Mifflin-St Jeor equation to calculate your Basal Metabolic Rate (BMR). This method is widely regarded as the gold standard and is strongly endorsed by the Academy of Nutrition and Dietetics. To estimate your Total Daily Energy Expenditure (TDEE), we use well-established activity multipliers derived directly from exercise physiology research. For protein recommendations, our targets are based on official guidelines from the International Society of Sports Nutrition.
 
When it comes to any calorie adjustments, we stick to conservative, sustainable rates that research has consistently shown lead to lasting, meaningful results.  We're all about setting you up for success, one step at a time!

### The Fine Print ⚠️

Think of this tool as your launchpad, but remember—everyone's different. Your mileage may vary due to factors like genetics, health conditions, medications, and other factors that a calculator simply can't account for. It's always wise to consult a qualified healthcare provider before making any big dietary shifts. Above all, tune into your body—keep tabs on your energy levels, performance,and tweak things as needed. We're here to help, but you know yourself best!
""")

st.success(
    "You made it to the finish line! Thanks for sticking with us on this nutrition adventure. Remember, the sun doesn't rush to rise, but it always shows up. Keep shining—you've got this! 🥳"
)

# ---------------------------------------------------------------------------
# Cell 17: Session State Management and Performance
# ---------------------------------------------------------------------------

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
