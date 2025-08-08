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
import base64
from io import StringIO
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

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
                'placeholder': 'Enter your age in years', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (in centimeters)',
                      'min': 140, 'max': 220, 'step': 1,
                      'placeholder': 'Enter your height in cm', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (in kilograms)',
                      'min': 40.0, 'max': 150.0, 'step': 0.5,
                      'placeholder': 'Enter your weight in kg', 'required': True},
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

# ------ Unit Conversion Functions ------
def kg_to_lbs(kg):
    return kg * 2.20462

def lbs_to_kg(lbs):
    return lbs / 2.20462

def cm_to_inches(cm):
    return cm / 2.54

def inches_to_cm(inches):
    return inches * 2.54

def ml_to_cups(ml):
    return ml / 236.588

# ---------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables."""
    session_vars = (
        ['food_selections', 'form_validated', 'validation_errors', 'food_search', 'user_feedback'] +
        [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    )

    for var in session_vars:
        if var not in st.session_state:
            if var == 'food_selections':
                st.session_state[var] = {}
            elif var == 'form_validated':
                st.session_state[var] = False
            elif var == 'validation_errors':
                st.session_state[var] = []
            elif var == 'food_search':
                st.session_state[var] = ""
            elif var == 'user_feedback':
                st.session_state[var] = ""
            else:
                st.session_state[var] = None


def validate_form_inputs(user_inputs):
    """Validates required form inputs and returns errors."""
    errors = []
    required_fields = [
        field for field, config in CONFIG['form_fields'].items()
        if config.get('required')
    ]
    
    for field in required_fields:
        if user_inputs.get(field) is None:
            field_label = CONFIG['form_fields'][field]['label']
            errors.append(f"Please enter your {field_label.lower()}!")
    
    return errors


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration."""
    session_key = f'user_{field_name}'
    
    # Handle units conversion for display
    units = st.session_state.get('user_units', 'metric')
    
    if field_config['type'] == 'number':
        # Adjust labels and values for imperial units
        if units == 'imperial':
            if field_name == 'height_cm':
                field_config = field_config.copy()
                field_config['label'] = 'Height (in inches)'
                field_config['min'] = 55  # ~140cm
                field_config['max'] = 87  # ~220cm
                field_config['placeholder'] = 'Enter your height in inches'
                # Convert stored metric value to imperial for display
                current_value = st.session_state[session_key]
                if current_value:
                    current_value = cm_to_inches(current_value)
            elif field_name == 'weight_kg':
                field_config = field_config.copy()
                field_config['label'] = 'Weight (in pounds)'
                field_config['min'] = 88  # ~40kg
                field_config['max'] = 330  # ~150kg
                field_config['step'] = 1.0
                field_config['placeholder'] = 'Enter your weight in lbs'
                # Convert stored metric value to imperial for display
                current_value = st.session_state[session_key]
                if current_value:
                    current_value = kg_to_lbs(current_value)
            else:
                current_value = st.session_state[session_key]
        else:
            current_value = st.session_state[session_key]
            
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

        value = container.number_input(
            field_config['label'],
            min_value=field_config['min'],
            max_value=field_config['max'],
            value=current_value,
            step=field_config['step'],
            placeholder=placeholder,
            help=field_config.get('help'),
            key=f"input_{field_name}"
        )
        
        # Convert imperial back to metric for storage
        if units == 'imperial':
            if field_name == 'height_cm' and value:
                value = inches_to_cm(value)
            elif field_name == 'weight_kg' and value:
                value = lbs_to_kg(value)
                
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
                key=f"input_{field_name}"
            )
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(
                field_config['label'],
                options,
                index=index,
                key=f"input_{field_name}"
            )

    st.session_state[session_key] = value
    return value


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
                st.metric(label, value)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                st.metric(label, value, delta)


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


def get_progress_color(percent):
    """Returns color based on progress percentage."""
    if percent >= 80:
        return "#28a745"  # Green
    elif percent >= 50:
        return "#ffc107"  # Yellow
    else:
        return "#dc3545"  # Red


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

    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
        percent = min(actual / target * 100, 100)
        color = get_progress_color(percent)

        # Create colored progress bar
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: bold;">{config['label']}</span>
                <span>{percent:.0f}% of target ({targets[config['target_key']]:.0f} {config['unit']})</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 10px; height: 20px;">
                <div style="background-color: {color}; height: 100%; width: {percent}%; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if actual < targets[config['target_key']]:
            deficit = targets[config['target_key']] - actual
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
    
    units = st.session_state.get('user_units', 'metric')
    if units == 'imperial':
        hydration_display = f"{ml_to_cups(hydration_ml):.1f} cups"
    else:
        hydration_display = f"{hydration_ml} ml (~{hydration_ml/250:.1f} cups)"
    
    recommendations.append(
        f"💧 Your Estimated Daily Hydration Goal: {hydration_display} of water throughout the day."
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


def save_progress_to_json():
    """Saves current progress to JSON format."""
    progress_data = {
        'food_selections': st.session_state.food_selections,
        'user_inputs': {
            f'user_{field}': st.session_state.get(f'user_{field}')
            for field in CONFIG['form_fields'].keys()
        },
        'timestamp': pd.Timestamp.now().isoformat()
    }
    return json.dumps(progress_data, indent=2)


def load_progress_from_json(json_data):
    """Loads progress from JSON data."""
    try:
        data = json.loads(json_data)
        st.session_state.food_selections = data.get('food_selections', {})
        user_inputs = data.get('user_inputs', {})
        for key, value in user_inputs.items():
            if key in st.session_state:
                st.session_state[key] = value
        return True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False


def create_pdf_summary(totals, targets, selected_foods, final_values):
    """Creates a PDF summary of the daily nutrition data."""
    buffer = StringIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Daily Nutrition Summary", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # User info
    units = st.session_state.get('user_units', 'metric')
    if units == 'imperial':
        weight_display = f"{kg_to_lbs(final_values['weight_kg']):.1f} lbs"
        height_display = f"{cm_to_inches(final_values['height_cm']):.1f} inches"
    else:
        weight_display = f"{final_values['weight_kg']} kg"
        height_display = f"{final_values['height_cm']} cm"
    
    user_info = f"""
    <b>User Information:</b><br/>
    Age: {final_values['age']} years<br/>
    Weight: {weight_display}<br/>
    Height: {height_display}<br/>
    Goal: {final_values['goal'].replace('_', ' ').title()}<br/>
    """
    story.append(Paragraph(user_info, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Targets vs Actual
    targets_text = f"""
    <b>Daily Targets vs Actual:</b><br/>
    Calories: {totals['calories']:.0f} / {targets['total_calories']} kcal<br/>
    Protein: {totals['protein']:.0f} / {targets['protein_g']} g<br/>
    Carbs: {totals['carbs']:.0f} / {targets['carb_g']} g<br/>
    Fat: {totals['fat']:.0f} / {targets['fat_g']} g<br/>
    """
    story.append(Paragraph(targets_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Selected foods
    if selected_foods:
        story.append(Paragraph("<b>Foods Consumed:</b>", styles['Normal']))
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            food_text = f"{food['name']} - {servings} serving(s)"
            story.append(Paragraph(food_text, styles['Normal']))
    
    doc.build(story)
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data


def create_csv_summary(totals, targets, selected_foods):
    """Creates a CSV summary of the daily nutrition data."""
    data = {
        'Metric': ['Calories', 'Protein (g)', 'Carbs (g)', 'Fat (g)'],
        'Target': [targets['total_calories'], targets['protein_g'], 
                  targets['carb_g'], targets['fat_g']],
        'Actual': [totals['calories'], totals['protein'], 
                  totals['carbs'], totals['fat']],
        'Percentage': [
            (totals['calories'] / targets['total_calories'] * 100) if targets['total_calories'] > 0 else 0,
            (totals['protein'] / targets['protein_g'] * 100) if targets['protein_g'] > 0 else 0,
            (totals['carbs'] / targets['carb_g'] * 100) if targets['carb_g'] > 0 else 0,
            (totals['fat'] / targets['fat_g'] * 100) if targets['fat_g'] > 0 else 0
        ]
    }
    
    df = pd.DataFrame(data)
    return df.to_csv(index=False)


def update_sidebar_summary(totals, targets):
    """Updates the sidebar with a dynamic summary."""
    with st.sidebar:
        st.markdown("---")
        st.subheader("📊 Today's Progress")
        
        for nutrient, config in CONFIG['nutrient_configs'].items():
            actual = totals[nutrient]
            target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
            percent = min(actual / target * 100, 100)
            
            # Create a simple progress indicator
            progress_bars = "█" * int(percent // 10) + "░" * (10 - int(percent // 10))
            st.markdown(f"**{config['label']}**: {percent:.0f}%")
            st.markdown(f"`{progress_bars}`")


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

    # Handle division by zero for macro percentages
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
        # Add tooltips to emojis
        emoji_tooltips = {
            '🥇': 'Gold Medal: Nutritional all-star! High in target nutrient and calorie-efficient',
            '🔥': 'High Calorie: One of the more calorie-dense options',
            '💪': 'High Protein: A true protein powerhouse',
            '🍚': 'High Carb: A carbohydrate champion',
            '🥑': 'High Fat: A healthy fat hero'
        }
        
        emoji = food.get('emoji', '')
        emoji_tooltip = emoji_tooltips.get(emoji, '')
        
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
            # Cap max servings to prevent absurd values
            custom_serving = st.number_input(
                "Custom",
                min_value=0.0, max_value=20.0,  # Capped at 20 servings
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

        # Add tooltips to metrics
        caption_text = (
            f"Per Serving: {food['calories']} kcal | {food['protein']}g protein | "
            f"{food['carbs']}g carbs | {food['fat']}g fat"
        )
        st.caption(caption_text)
        
        # Add metric explanations on hover (using help text)
        with st.expander("ℹ️ Nutrition Info", expanded=False):
            st.markdown(f"""
            **TDEE**: Total Daily Energy Expenditure - the total calories you burn in a day  
            **Macros**: Protein (muscle building), Carbs (energy), Fat (hormones & health)  
            **Serving**: One standard portion as defined in our database
            """)


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

# ------ Load Food Database and Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# ------ Apply Custom CSS for Enhanced Styling and Better Contrast ------
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
.stButton>button[kind="primary"] { 
    background-color: #ff6b6b; 
    color: white; 
    border: 1px solid #ff6b6b;
    font-weight: bold;
}
.stButton>button[kind="secondary"] { 
    border: 1px solid #ff6b6b; 
    color: #333;
    background-color: white;
}
.sidebar .sidebar-content { background-color: #f0f2f6; }

/* Improve contrast for better readability */
.stMarkdown p, .stMarkdown li {
    color: #333333;
}
.stCaption {
    color: #666666 !important;
    font-size: 0.875rem;
}
.metric-label {
    color: #333333 !important;
    font-weight: 600;
}
.metric-value {
    color: #1f77b4 !important;
    font-weight: bold;
}

/* Progress bar styling */
.progress-container {
    background-color: #e9ecef;
    border-radius: 10px;
    height: 20px;
    margin: 0.5rem 0;
}
.progress-bar {
    height: 100%;
    border-radius: 10px;
    transition: width 0.3s ease;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
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

# Add units toggle
units = st.sidebar.radio(
    "📏 Measurement Units",
    options=['metric', 'imperial'],
    format_func=lambda x: 'Metric (kg/cm)' if x == 'metric' else 'Imperial (lbs/inches)',
    key="user_units",
    help="Choose your preferred measurement system"
)

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

# ------ Form Validation ------
validation_errors = validate_form_inputs(all_inputs)
st.session_state.validation_errors = validation_errors

# Calculate button with validation
if st.sidebar.button("🎯 Calculate My Targets", type="primary", key="calculate_targets"):
    if validation_errors:
        st.session_state.form_validated = False
        for error in validation_errors:
            st.sidebar.error(error)
    else:
        st.session_state.form_validated = True
        st.sidebar.success("✅ Targets calculated successfully!")

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

# ------ Save/Load Progress Section ------
with st.sidebar.expander("💾 Save/Load Progress"):
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Save Progress", key="save_progress"):
            progress_json = save_progress_to_json()
            st.download_button(
                label="📥 Download",
                data=progress_json,
                file_name=f"nutrition_progress_{pd.Timestamp.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                key="download_progress"
            )
    
    with col2:
        uploaded_file = st.file_uploader(
            "📤 Load Progress",
            type=['json'],
            key="upload_progress",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            json_data = uploaded_file.read().decode('utf-8')
            if load_progress_from_json(json_data):
                st.success("✅ Progress loaded!")
                st.rerun()

# ------ Process Final Values ------
final_values = get_final_values(all_inputs)

if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_ml = calculate_hydration_needs(
        final_values['weight_kg'], final_values['activity_level']
    )
    
    if units == 'imperial':
        hydration_display = f"💧 Daily Hydration: ~{ml_to_cups(hydration_ml):.1f} cups"
    else:
        hydration_display = f"💧 Daily Hydration: {hydration_ml} ml (~{hydration_ml/250:.1f} cups)"
    
    st.sidebar.info(hydration_display)

# ------ Check for User Input and Form Validation ------
required_fields = [
    field for field, config in CONFIG['form_fields'].items()
    if config.get('required')
]
user_has_entered_info = all(
    (all_inputs.get(field) is not None)
    for field in required_fields
)

form_is_valid = st.session_state.get('form_validated', False) and not validation_errors

# ------ Calculate Personalized Targets ------
targets = calculate_personalized_targets(**final_values)

# Calculate totals for sidebar summary
totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
if user_has_entered_info and form_is_valid:
    update_sidebar_summary(totals, targets)

# ---------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# ---------------------------------------------------------------------------

if not form_is_valid:
    if validation_errors:
        st.warning("⚠️ Please complete all required fields and click 'Calculate My Targets' to see your personalized nutrition plan.")
    else:
        st.info("👈 Pop your details into the sidebar and click 'Calculate My Targets' to get your personalized daily targets.")
    
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
    
    # Add motivational notification
    weekly_change = targets['estimated_weekly_change']
    if weekly_change > 0:
        change_text = f"gain {weekly_change:.2f} kg"
    elif weekly_change < 0:
        change_text = f"lose {abs(weekly_change):.2f} kg"
    else:
        change_text = "maintain your current weight"
    
    st.success(f"🎉 Great job setting up! You're on track to {change_text} per week following this plan.")

st.info(
    "🎯 **The 80/20 Rule**: Try to hit your targets about 80% of the time. This gives you wiggle room for birthday cake, date nights, and those inevitable moments when life throws you a curveball. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap."
)

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Unified Metrics Display Configuration ------
if units == 'imperial':
    weight_display = f"{kg_to_lbs(final_values['weight_kg']):.1f} lbs"
    height_display = f"{cm_to_inches(final_values['height_cm']):.1f} inches"
    hydration_display = f"{ml_to_cups(hydration_ml):.1f} cups"
else:
    weight_display = f"{final_values['weight_kg']} kg"
    height_display = f"{final_values['height_cm']} cm"
    hydration_display = f"{hydration_ml} ml (~{hydration_ml/250:.1f} cups)"

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
            ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kg"),
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
            ("Water", hydration_display)
        ]
    }
]

# ------ Display All Metric Sections ------
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# ---------------------------------------------------------------------------
# Cell 10: Enhanced Evidence-Based Tips and Context (Collapsed by Default)
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
# Cell 11: Personalized Recommendations System
# ---------------------------------------------------------------------------

if form_is_valid:
    st.header("Your Personalized Action Steps 🎯")
    recommendations = generate_personalized_recommendations(
        totals, targets, final_values
    )
    for rec in recommendations:
        st.info(rec)

# ---------------------------------------------------------------------------
# Cell 12: Food Selection Interface with Search
# ---------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")
st.markdown(
    "Pick how many servings of each food you're having to see how your choices stack up against your daily targets."
)

# Add search functionality
search_term = st.text_input(
    "🔍 Search for foods",
    value=st.session_state.food_search,
    placeholder="Type to search for specific foods...",
    key="food_search_input",
    help="Search across all food categories to quickly find what you're looking for"
)

if search_term != st.session_state.food_search:
    st.session_state.food_search = search_term

# Filter foods based on search
filtered_foods = filter_foods_by_search(foods, st.session_state.food_search)

with st.expander("💡 Need a hand with food choices? Check out the emoji guide below!"):
    st.markdown("""
    * **🥇 Gold Medal**: A nutritional all-star! High in its target nutrient and very calorie-efficient.
    * **🔥 High Calorie**: One of the more calorie-dense options in its group.
    * **💪 High Protein**: A true protein powerhouse.
    * **🍚 High Carb**: A carbohydrate champion.
    * **🥑 High Fat**: A healthy fat hero.
    """)

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("🔄 Start Fresh: Reset All Food Selections", type="secondary", key="reset_foods"):
        st.session_state.food_selections = {}
        st.rerun()

with col2:
    # Show search results count
    if st.session_state.food_search:
        total_results = sum(len(items) for items in filtered_foods.values())
        st.caption(f"Found {total_results} foods matching '{st.session_state.food_search}'")

# ------ Food Selection with Tabs ------
if filtered_foods:
    available_categories = [
        cat for cat, items in sorted(filtered_foods.items()) if items
    ]
    
    if available_categories:
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
    else:
        st.info("No foods found matching your search. Try a different term or clear the search box.")
else:
    if st.session_state.food_search:
        st.info("No foods found matching your search. Try a different term or clear the search box.")
    else:
        st.info("Loading food database...")

# ---------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking with Export Options
# ---------------------------------------------------------------------------

st.header("Today's Scorecard 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)

# Export options
if selected_foods and form_is_valid:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export PDF Summary", key="export_pdf"):
            try:
                pdf_data = create_pdf_summary(totals, targets, selected_foods, final_values)
                st.download_button(
                    label="📥 Download PDF",
                    data=pdf_data,
                    file_name=f"nutrition_summary_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                    mime="application/pdf",
                    key="download_pdf"
                )
            except Exception as e:
                st.error("PDF export temporarily unavailable. Please try CSV export instead.")
    
    with col2:
        if st.button("📊 Export CSV Data", key="export_csv"):
            csv_data = create_csv_summary(totals, targets, selected_foods)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"nutrition_data_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                key="download_csv"
            )

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)
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
        color = get_progress_color(0)
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: bold;">{config['label']}</span>
                <span>0% of daily target ({targets[config['target_key']]:.0f} {config['unit']})</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 10px; height: 20px;">
                <div style="background-color: {color}; height: 100%; width: 0%; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 14: User Feedback Section
# ---------------------------------------------------------------------------

st.divider()
st.header("Help Us Improve! 💬")

with st.expander("📝 Share Your Feedback", expanded=False):
    st.markdown("We'd love to hear from you! Your feedback helps us make this tool better for everyone.")
    
    feedback_text = st.text_area(
        "How can we improve this nutrition tracker?",
        placeholder="Tell us what you like, what could be better, or suggest new features...",
        key="feedback_input",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("📤 Submit Feedback", key="submit_feedback"):
            if feedback_text.strip():
                # In a real app, you'd send this to a database or email
                st.success("Thank you for your feedback! We really appreciate it. 🙏")
                st.session_state.user_feedback = feedback_text
            else:
                st.warning("Please enter some feedback before submitting.")

# ---------------------------------------------------------------------------
# Cell 15: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()
st.markdown("""
### The Science We Stand On 📚

This tracker isn't built on guesswork—it's grounded in peer-reviewed research and evidence-based guidelines. We rely on the Mifflin-St Jeor equation to calculate your Basal Metabolic Rate (BMR). This method is widely regarded as the gold standard and is strongly endorsed by the Academy of Nutrition and Dietetics. To estimate your Total Daily Energy Expenditure (TDEE), we use well-established activity multipliers derived directly from exercise physiology research. For protein recommendations, our targets are based on official guidelines from the International Society of Sports Nutrition.
 
When it comes to any calorie adjustments, we stick to conservative, sustainable rates that research has consistently shown lead to lasting, meaningful results.  We're all about setting you up for success, one step at a time!

### The Fine Print ⚠️

Think of this tool as your launchpad, but remember—everyone's different. Your mileage may vary due to factors like genetics, health conditions, medications, and other factors that a calculator simply can't account for. It's always wise to consult a qualified healthcare provider before making any big dietary shifts. Above all, tune into your body—keep tabs on your energy levels, performance, and tweak things as needed. We're here to help, but you know yourself best!
""")

st.success(
    "You made it to the finish line! Thanks for sticking with us on this nutrition adventure. Remember, the sun doesn't rush to rise, but it always shows up. Keep shining—you've got this! 🥳"
)

# ---------------------------------------------------------------------------
# Cell 16: Session State Management and Performance Optimization
# ---------------------------------------------------------------------------

# ------ Clean Up Session State to Prevent Memory Issues ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# ------ Optimize Performance by Reducing Unnecessary Reruns ------
# Ensure all widgets have unique keys to prevent conflicts
# This is handled throughout the code with explicit key parameters

# ------ Handle Edge Cases ------
# Division by zero protection is implemented in calculate_personalized_targets()
# Max servings are capped in render_food_item() at 20 servings
# Form validation prevents incomplete submissions
# Search functionality gracefully handles empty results
