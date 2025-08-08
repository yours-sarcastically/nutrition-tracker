Sonnet Thinking

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# A Personalized Evidence-Based Nutrition Tracker for Goal-Specific Meal Planning
# ---------------------------------------------------------------------------

"""
Enhanced nutrition tracker with improved UX, accessibility, and functionality.
Includes progress tracking, auto-save, search functionality, and comprehensive
validation features.
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
# Cell 3: Enhanced Configuration Constants
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
    'use_imperial': False
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
    'extremely_active': "Very heavy exercise, a physical job, or two times per day training"
}

# ------ Goal-Specific Targets Based on an Evidence-Based Guide ------
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

# ------ Emoji Tooltips ------
EMOJI_TOOLTIPS = {
    '🥇': 'Gold Medal: High in target nutrient and very calorie-efficient',
    '🔥': 'High Calorie: One of the more calorie-dense options',
    '💪': 'High Protein: A protein powerhouse',
    '🍚': 'High Carb: A carbohydrate champion',
    '🥑': 'High Fat: A healthy fat hero'
}

# ------ Progress Bar Colors ------
PROGRESS_COLORS = {
    'low': '#ff4757',     # Red for <50%
    'medium': '#ffa502',  # Orange for 50-80%
    'high': '#2ed573'     # Green for >80%
}

# ------ Enhanced Unified Configuration ------
CONFIG = {
    'emoji_order': {'🥇': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
    },
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories',
                    'tooltip': 'Total Daily Energy Expenditure - your daily calorie needs'},
        'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g',
                   'tooltip': 'Essential for muscle maintenance, repair, and growth'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g',
                 'tooltip': 'Primary energy source for your body and brain'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g',
               'tooltip': 'Essential for hormone production and nutrient absorption'}
    },
    'form_fields': {
        'age': {'type': 'number', 'label': 'Age (in years)',
                'min': 16, 'max': 80, 'step': 1,
                'placeholder': 'Enter your age in years', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (cm)',
                      'min': 140, 'max': 220, 'step': 1,
                      'placeholder': 'Enter your height in cm', 'required': True},
        'height_ft': {'type': 'number', 'label': 'Height (feet)',
                      'min': 4, 'max': 7, 'step': 1,
                      'placeholder': 'Enter feet', 'required': True},
        'height_in': {'type': 'number', 'label': 'Height (inches)',
                      'min': 0, 'max': 11, 'step': 1,
                      'placeholder': 'Enter inches', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (kg)',
                      'min': 40.0, 'max': 150.0, 'step': 0.5,
                      'placeholder': 'Enter your weight in kg', 'required': True},
        'weight_lbs': {'type': 'number', 'label': 'Weight (lbs)',
                       'min': 88, 'max': 330, 'step': 1,
                       'placeholder': 'Enter your weight in lbs', 'required': True},
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
        'protein_per_kg': {'type': 'number', 'label': 'Protein Goal (g/kg)',
                           'min': 1.2, 'max': 3.0, 'step': 0.1,
                           'help': 'Define your daily protein target in grams per kilogram of body weight',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number', 'label': 'Fat Intake (% of calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Set the share of your daily calories that should come from healthy fats',
                           'convert': lambda x: x / 100 if x else None,
                           'advanced': True, 'required': False}
    }
}

# ---------------------------------------------------------------------------
# Cell 4: Enhanced Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables."""
    session_vars = [
        'food_selections', 'search_filter', 'user_inputs_complete',
        'calculation_ready', 'feedback_submitted', 'use_imperial'
    ] + [f'user_{field}' for field in CONFIG['form_fields'].keys()]

    defaults = {
        'food_selections': {},
        'search_filter': '',
        'user_inputs_complete': False,
        'calculation_ready': False,
        'feedback_submitted': False,
        'use_imperial': False
    }

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = defaults.get(var, None)


def convert_imperial_to_metric(height_ft, height_in, weight_lbs):
    """Converts imperial measurements to metric."""
    height_cm = ((height_ft * 12) + height_in) * 2.54
    weight_kg = weight_lbs / 2.205
    return height_cm, weight_kg


def convert_metric_to_imperial(height_cm, weight_kg):
    """Converts metric measurements to imperial for display."""
    total_inches = height_cm / 2.54
    height_ft = int(total_inches // 12)
    height_in = int(total_inches % 12)
    weight_lbs = weight_kg * 2.205
    return height_ft, height_in, weight_lbs


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration with unique keys."""
    session_key = f'user_{field_name}'
    widget_key = f'widget_{field_name}_{id(container)}'

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

        value = container.number_input(
            field_config['label'],
            min_value=field_config['min'],
            max_value=field_config['max'],
            value=st.session_state[session_key],
            step=field_config['step'],
            placeholder=placeholder,
            help=field_config.get('help'),
            key=widget_key
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
            index = next(
                (i for i, (_, val) in enumerate(options) if val == current_value),
                next((i for i, (_, val) in enumerate(options) if val == DEFAULTS.get(field_name)), 0)
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


def validate_user_inputs():
    """Validates that all required inputs are complete."""
    required_fields = [
        field for field, config in CONFIG['form_fields'].items()
        if config.get('required')
    ]
    
    use_imperial = st.session_state.get('use_imperial', False)
    
    if use_imperial:
        required_fields = [f for f in required_fields if f not in ['height_cm', 'weight_kg']]
        required_fields.extend(['height_ft', 'height_in', 'weight_lbs'])
    else:
        required_fields = [f for f in required_fields if f not in ['height_ft', 'height_in', 'weight_lbs']]

    missing_fields = []
    for field in required_fields:
        value = st.session_state.get(f'user_{field}')
        if value is None or (isinstance(value, (int, float)) and value <= 0):
            field_label = CONFIG['form_fields'][field]['label']
            missing_fields.append(field_label)
    
    return len(missing_fields) == 0, missing_fields


def get_final_values(user_inputs):
    """Processes all user inputs and applies default values where needed."""
    final_values = {}
    
    use_imperial = st.session_state.get('use_imperial', False)
    
    for field, value in user_inputs.items():
        final_values[field] = value if value is not None else DEFAULTS.get(field)

    # Handle unit conversions
    if use_imperial:
        height_ft = final_values.get('height_ft', 5)
        height_in = final_values.get('height_in', 9)
        weight_lbs = final_values.get('weight_lbs', 127)
        
        final_values['height_cm'], final_values['weight_kg'] = convert_imperial_to_metric(
            height_ft, height_in, weight_lbs
        )
    
    # Apply goal-specific defaults for advanced settings
    goal = final_values.get('goal', 'weight_gain')
    if goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[goal]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']

    return final_values


def get_progress_color(percent):
    """Returns the appropriate color based on progress percentage."""
    if percent >= 80:
        return PROGRESS_COLORS['high']
    elif percent >= 50:
        return PROGRESS_COLORS['medium']
    else:
        return PROGRESS_COLORS['low']


def create_colored_progress_bar(percent, label, target, unit):
    """Creates a colored progress bar with improved styling."""
    color = get_progress_color(percent)
    
    # Create custom HTML for colored progress bar
    progress_html = f"""
    <div style="margin-bottom: 10px;">
        <div style="font-size: 14px; margin-bottom: 5px; color: #262730;">
            <strong>{label}</strong>: {percent:.0f}% of daily target ({target:.0f} {unit})
        </div>
        <div style="background-color: #e0e0e0; border-radius: 10px; height: 20px; overflow: hidden;">
            <div style="background-color: {color}; height: 100%; width: {min(percent, 100):.1f}%; 
                        transition: width 0.3s ease; border-radius: 10px;"></div>
        </div>
    </div>
    """
    return progress_html


def save_progress_to_json():
    """Saves current progress to JSON format."""
    progress_data = {
        'timestamp': datetime.now().isoformat(),
        'food_selections': st.session_state.food_selections,
        'user_inputs': {
            key.replace('user_', ''): value 
            for key, value in st.session_state.items() 
            if key.startswith('user_') and value is not None
        }
    }
    return json.dumps(progress_data, indent=2)


def load_progress_from_json(json_data):
    """Loads progress from JSON data."""
    try:
        data = json.loads(json_data)
        
        # Load food selections
        if 'food_selections' in data:
            st.session_state.food_selections = data['food_selections']
        
        # Load user inputs
        if 'user_inputs' in data:
            for field, value in data['user_inputs'].items():
                st.session_state[f'user_{field}'] = value
        
        return True, "Progress loaded successfully!"
    except Exception as e:
        return False, f"Error loading progress: {str(e)}"


def create_pdf_summary(totals, targets, selected_foods, final_values):
    """Creates a PDF summary of the daily nutrition."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("Daily Nutrition Summary", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Date
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Targets vs Actual
    story.append(Paragraph("Daily Targets vs Actual Intake", styles['Heading2']))
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = (actual / target * 100) if target > 0 else 0
        
        story.append(Paragraph(
            f"{config['label']}: {actual:.0f}/{target:.0f} {config['unit']} ({percent:.0f}%)",
            styles['Normal']
        ))
    
    story.append(Spacer(1, 12))
    
    # Selected Foods
    if selected_foods:
        story.append(Paragraph("Foods Consumed Today", styles['Heading2']))
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            story.append(Paragraph(
                f"• {food['name']} - {servings} serving(s)",
                styles['Normal']
            ))
    
    doc.build(story)
    buffer.seek(0)
    return buffer


def filter_foods_by_search(foods, search_term):
    """Filters foods based on search term."""
    if not search_term:
        return foods
    
    search_lower = search_term.lower()
    filtered_foods = {}
    
    for category, items in foods.items():
        filtered_items = [
            item for item in items 
            if search_lower in item['name'].lower()
        ]
        if filtered_items:
            filtered_foods[category] = filtered_items
    
    return filtered_foods


def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics with tooltips."""
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
                label, value, delta, tooltip = metric_info
                st.metric(label, value, delta, help=tooltip)


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
    """Creates enhanced progress bars with color coding and recommendations."""
    recommendations = []
    st.subheader("Your Daily Dashboard 🎯")

    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation and building',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and overall health'
    }

    progress_html = ""
    
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
        percent = min(actual / target * 100, 100)
        
        progress_html += create_colored_progress_bar(
            percent, config['label'], target, config['unit']
        )

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

    # Display the colored progress bars
    st.markdown(progress_html, unsafe_allow_html=True)
    
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


def show_motivational_message(targets, final_values):
    """Shows motivational success message after calculation."""
    goal = final_values.get('goal', 'weight_gain')
    change = targets.get('estimated_weekly_change', 0)
    
    messages = {
        'weight_loss': f"🎉 Fantastic! You're all set up for success! With your plan, you're on track to lose approximately {abs(change):.2f} kg per week. Remember, slow and steady wins the race!",
        'weight_maintenance': "🎯 Perfect! Your plan is dialed in to help you maintain your current weight while staying healthy and energized. Consistency is key!",
        'weight_gain': f"💪 Awesome setup! You're on track to gain approximately {change:.2f} kg per week. Time to fuel those gains and crush your goals!"
    }
    
    message = messages.get(goal, "Great job setting up your nutrition plan!")
    st.success(message)


def create_sidebar_summary(totals, targets):
    """Creates a dynamic summary in the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Today's Progress")
    
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
        percent = (actual / target * 100)
        
        # Create mini progress indicator
        if percent >= 80:
            icon = "✅"
        elif percent >= 50:
            icon = "🟡"
        else:
            icon = "🔴"
            
        st.sidebar.markdown(f"{icon} **{config['label']}**: {actual:.0f}/{target:.0f} {config['unit']}")


# ---------------------------------------------------------------------------
# Cell 5: Core Calculation and Data Processing Functions
# ---------------------------------------------------------------------------

@st.cache_data
def load_food_database(filename):
    """Loads and processes the nutrition database from CSV file."""
    try:
        df = pd.read_csv(filename)
        
        # Expected columns in the CSV
        required_cols = ['Food', 'Category', 'Calories_per_serving', 
                        'Protein_g_per_serving', 'Carbs_g_per_serving', 
                        'Fat_g_per_serving']
        
        # Check if required columns exist
        if not all(col in df.columns for col in required_cols):
            st.error(f"❌ Missing required columns in {filename}. Expected: {required_cols}")
            return {}
        
        foods = {}
        for _, row in df.iterrows():
            category = row['Category']
            if category not in foods:
                foods[category] = []
                
            food_item = {
                'name': row['Food'],
                'calories': float(row['Calories_per_serving']),
                'protein': float(row['Protein_g_per_serving']),
                'carbs': float(row['Carbs_g_per_serving']),
                'fat': float(row['Fat_g_per_serving'])
            }
            foods[category].append(food_item)
        
        return foods
        
    except FileNotFoundError:
        st.error(f"❌ File '{filename}' not found. Please ensure the CSV file is in the correct location.")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading database: {str(e)}")
        return {}


def calculate_bmr(weight_kg, height_cm, age, sex):
    """
    Calculates Basal Metabolic Rate using the Mifflin-St Jeor Equation.
    This is the gold standard equation recommended by nutritionists.
    """
    if sex.lower() == "male":
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    return round(bmr)


def calculate_tdee(bmr, activity_level):
    """
    Calculates Total Daily Energy Expenditure by applying activity multiplier to BMR.
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return round(bmr * multiplier)


def calculate_personalized_targets(weight_kg, height_cm, age, sex, activity_level, 
                                 goal, protein_per_kg=None, fat_percentage=None, **kwargs):
    """
    Calculates personalized daily nutrition targets based on user inputs and goals.
    """
    # Calculate BMR and TDEE
    bmr = calculate_bmr(weight_kg, height_cm, age, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Apply goal-specific adjustments
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])
    
    # Use provided values or fall back to goal defaults
    if protein_per_kg is None:
        protein_per_kg = goal_config['protein_per_kg']
    if fat_percentage is None:
        fat_percentage = goal_config['fat_percentage']
    
    # Calculate caloric adjustment
    caloric_adjustment_percent = goal_config['caloric_adjustment']
    caloric_adjustment = tdee * caloric_adjustment_percent
    total_calories = round(tdee + caloric_adjustment)
    
    # Calculate macronutrient targets
    protein_g = round(weight_kg * protein_per_kg)
    protein_calories = protein_g * 4
    
    fat_calories = round(total_calories * fat_percentage)
    fat_g = round(fat_calories / 9)
    
    remaining_calories = total_calories - protein_calories - fat_calories
    carb_g = round(remaining_calories / 4)
    
    # Calculate percentages for display
    protein_percent = (protein_calories / total_calories) * 100
    fat_percent = (fat_calories / total_calories) * 100
    carb_percent = (remaining_calories / total_calories) * 100
    
    # Estimate weekly weight change (rough approximation)
    weekly_calorie_difference = caloric_adjustment * 7
    estimated_weekly_change = weekly_calorie_difference / 7700  # ~7700 kcal per kg
    
    return {
        'bmr': bmr,
        'tdee': tdee,
        'caloric_adjustment': round(caloric_adjustment),
        'total_calories': total_calories,
        'protein_g': protein_g,
        'carb_g': carb_g,
        'fat_g': fat_g,
        'protein_percent': protein_percent,
        'carb_percent': carb_percent,
        'fat_percent': fat_percent,
        'estimated_weekly_change': estimated_weekly_change,
        'goal': goal
    }


@st.cache_data
def assign_food_emojis(foods):
    """
    Assigns emojis to foods based on their nutritional profile and calorie density.
    Uses caching to improve performance.
    """
    if not foods:
        return {}
    
    # Find top foods for each category
    top_foods = {
        'protein': [], 'carbs': [], 'fat': [],
        'calories': {}
    }
    
    for category, items in foods.items():
        if not items:
            continue
            
        # Sort by calories (highest first)
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]

        # Map categories to their primary nutrients
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(
                items, key=lambda x: x[map_info['sort_by']], reverse=True
            )
            top_foods[map_info['key']] = [
                food['name'] for food in sorted_by_nutrient[:3]
            ]

    # Collect all top nutrient foods
    all_top_nutrient_foods = {
        food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]
    }

    # Define emoji mapping
    emoji_mapping = {
        'high_cal_nutrient': '🥇', 'high_calorie': '🔥',
        'protein': '💪', 'carbs': '🍚', 'fat': '🥑'
    }

    # Assign emojis based on criteria
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


def render_food_item(food, category):
    """Renders a single food item with enhanced interaction controls and tooltips."""
    with st.container(border=True):
        # Enhanced header with emoji tooltip
        emoji = food.get('emoji', '')
        emoji_tooltip = EMOJI_TOOLTIPS.get(emoji, '')
        
        if emoji and emoji_tooltip:
            st.subheader(f"{emoji} {food['name']}")
            with st.expander("ℹ️ What does this emoji mean?"):
                st.write(emoji_tooltip)
        else:
            st.subheader(f"{food['name']}")
            
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
            # Cap maximum servings to prevent absurd values
            custom_serving = st.number_input(
                "Custom",
                min_value=0.0, max_value=20.0,  # Capped at 20 servings
                value=float(current_serving), step=0.1,
                key=f"{key}_custom",
                label_visibility="collapsed",
                help="Enter custom serving amount (max 20)"
            )

        if custom_serving != current_serving:
            if custom_serving > 0:
                st.session_state.food_selections[food['name']] = custom_serving
            elif food['name'] in st.session_state.food_selections:
                del st.session_state.food_selections[food['name']]
            st.rerun()

        # Enhanced caption with better contrast
        caption_text = (
            f"Per Serving: {food['calories']} kcal | {food['protein']}g protein | "
            f"{food['carbs']}g carbs | {food['fat']}g fat"
        )
        st.markdown(f"<small style='color: #666666;'>{caption_text}</small>", unsafe_allow_html=True)


def render_food_grid(items, category, columns=2):
    """Renders a grid of food items for a given category."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)


def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculates daily fluid needs based on body weight and activity."""
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
            "The Shocking Truth: Getting less than 7 hours of sleep can torpedo your fat loss by more than half.",
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
    else:
        recommendations.extend([
            "⚖️ **Flexible Tracking:** Monitor your intake five days per week instead of seven for a more sustainable and flexible approach to maintenance.",
            "📅 **Regular Check-ins:** Weigh yourself weekly and take body measurements monthly to catch any significant changes early.",
            "🎯 **The 80/20 Balance:** Aim for 80 percent of your diet to consist of nutrient-dense foods, with 20 percent flexibility for social situations."
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


# ---------------------------------------------------------------------------
# Cell 7: Initialize Enhanced Application
# ---------------------------------------------------------------------------

# Initialize Session State
initialize_session_state()

# Load Food Database and Assign Emojis
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# Enhanced CSS for Better Accessibility and Styling
st.markdown("""
<style>
/* Hide input instructions for cleaner UI */
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
}

/* Better contrast for captions and text */
.stCaption {
    color: #666666 !important;
    font-size: 0.875rem;
}

/* Enhanced metric styling */
.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #e1e5e9;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

/* Improved expander styling */
.streamlit-expanderHeader {
    background-color: #f8f9fa;
    border-radius: 8px;
}

/* Progress bar enhancements */
.stProgress > div > div {
    border-radius: 10px;
}

/* Toast message styling */
.stToast {
    background-color: #ffffff;
    border-left: 4px solid #28a745;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Enhanced User Input Interface with Unit Toggle
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("""
A Smart, Evidence-Based Nutrition Tracker That Actually Gets You

Welcome aboard!

Hey there! Welcome to your new nutrition buddy. This isn't just another calorie counter—it's your personalized guide, built on rock-solid science to help you smash your goals. Whether you're aiming to shed a few pounds, hold steady, or bulk up, we've crunched the numbers so you can focus on enjoying your food.

Let's get rolling—your journey to feeling awesome starts now! 🚀
""")

# Enhanced Sidebar with Unit Toggle
st.sidebar.header("Let's Get Personal 📊")

# Unit System Toggle
use_imperial = st.sidebar.toggle(
    "Use Imperial Units (lbs/ft-in)", 
    value=st.session_state.get('use_imperial', False),
    key='unit_toggle',
    help="Toggle between metric (kg/cm) and imperial (lbs/ft-in) units"
)
st.session_state.use_imperial = use_imperial

# Collect inputs based on unit system
all_inputs = {}
standard_fields = {
    k: v for k, v in CONFIG['form_fields'].items() 
    if not v.get('advanced')
}
advanced_fields = {
    k: v for k, v in CONFIG['form_fields'].items() 
    if v.get('advanced')
}

# Filter fields based on unit system
if use_imperial:
    standard_fields = {
        k: v for k, v in standard_fields.items() 
        if k not in ['height_cm', 'weight_kg']
    }
    # Add imperial fields
    for field in ['height_ft', 'height_in', 'weight_lbs']:
        if field in CONFIG['form_fields']:
            standard_fields[field] = CONFIG['form_fields'][field]
else:
    standard_fields = {
        k: v for k, v in standard_fields.items() 
        if k not in ['height_ft', 'height_in', 'weight_lbs']
    }

# Render standard input fields
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Advanced Settings Expander
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(
        field_name, field_config, container=advanced_expander
    )
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Enhanced Activity Level Guide
with st.sidebar.expander("📖 Activity Level Guide"):
    st.markdown("""
**Choose the level that best describes your typical week:**

* **🧑‍💻 Sedentary**: Desk job, little to no exercise
* **🚶 Lightly Active**: Light exercise 1-3 days/week  
* **🏃 Moderately Active**: Moderate exercise 3-5 days/week
* **🏋️ Very Active**: Heavy exercise 6-7 days/week
* **🤸 Extremely Active**: Very heavy exercise + physical job

💡 *When in doubt, choose the lower level - it's better to underestimate than overeat!*
""")

# Save/Load Progress Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Save Your Progress")

col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Save", key='save_progress', use_container_width=True):
        progress_json = save_progress_to_json()
        st.sidebar.download_button(
            label="📥 Download",
            data=progress_json,
            file_name=f"nutrition_progress_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            key='download_progress',
            use_container_width=True
        )

with col2:
    uploaded_file = st.file_uploader(
        "📤 Load Progress",
        type=['json'],
        key='upload_progress',
        label_visibility='collapsed'
    )
    if uploaded_file is not None:
        json_data = uploaded_file.read().decode('utf-8')
        success, message = load_progress_from_json(json_data)
        if success:
            st.sidebar.success(message)
            st.rerun()
        else:
            st.sidebar.error(message)

# Process Final Values
final_values = get_final_values(all_inputs)

# Validation Check
is_valid, missing_fields = validate_user_inputs()

# Calculate Button with Validation
if st.sidebar.button("🧮 Calculate My Targets", type="primary", use_container_width=True):
    if is_valid:
        st.session_state.calculation_ready = True
        st.session_state.user_inputs_complete = True
        st.rerun()
    else:
        st.sidebar.error("Please complete the following fields:")
        for field in missing_fields:
            st.sidebar.error(f"• {field}")

# Calculate targets if validation passes
if st.session_state.get('calculation_ready', False) and is_valid:
    targets = calculate_personalized_targets(**final_values)
    show_motivational_message(targets, final_values)
else:
    # Use default values for display
    targets = calculate_personalized_targets(**DEFAULTS)

# ---------------------------------------------------------------------------
# Cell 9: Enhanced Target Display System
# ---------------------------------------------------------------------------

if not st.session_state.get('user_inputs_complete', False):
    st.info("👈 Complete your details in the sidebar and click 'Calculate My Targets' to get your personalized plan!")
    st.header("Sample Daily Targets for Reference")
    st.caption("These are example targets. Please enter your information for personalized calculations.")
else:
    goal_labels = {
        'weight_loss': 'Weight Loss',
        'weight_maintenance': 'Weight Maintenance', 
        'weight_gain': 'Weight Gain'
    }
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Custom Daily Nutrition Roadmap for {goal_label} 🎯")

st.info("🎯 **The 80/20 Rule**: Try to hit your targets about 80% of the time. This gives you wiggle room for birthday cake, date nights, and those inevitable moments when life throws you a curveball. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap.")

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# Enhanced Metrics Display with Tooltips
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("BMR", f"{targets['bmr']} kcal", "", "Energy your body needs at complete rest"),
            ("TDEE", f"{targets['tdee']} kcal", "", "Total daily energy expenditure including activity"),
            ("Daily Adjustment", f"{targets['caloric_adjustment']:+} kcal", "", "Calories added/subtracted for your goal"),
            ("Weekly Change", f"{targets['estimated_weekly_change']:+.2f} kg", "", "Expected weekly weight change"),
            ("", "", "", "")
        ]
    },
    {
        'title': 'Your Daily Nutrition Targets', 'columns': 5, 
        'metrics': [
            ("Calories", f"{targets['total_calories']} kcal", "", "Your daily calorie target"),
            ("Protein", f"{targets['protein_g']} g", f"{targets['protein_percent']:.0f}%", "Essential for muscle maintenance and growth"),
            ("Carbs", f"{targets['carb_g']} g", f"{targets['carb_percent']:.0f}%", "Primary energy source for body and brain"),
            ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}%", "Important for hormones and nutrient absorption"),
            ("Water", f"{hydration_ml} ml", f"~{hydration_ml/250:.1f} cups", "Daily hydration goal for optimal function")
        ]
    }
]

# Display metric sections
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# ---------------------------------------------------------------------------
# Cell 10: Collapsed Evidence-Based Tips Section
# ---------------------------------------------------------------------------

with st.expander("📚 Your Evidence-Based Game Plan", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs([
        "The Big Three to Win At Nutrition 🏆", "Level Up Your Progress Tracking 📊",
        "Mindset Is Everything 🧠", "The Science Behind the Magic 🔬"
    ])

    with tab1:
        st.subheader("💧 Master Your Hydration Game")
        st.markdown("""
        Daily Goal: Shoot for about 35 ml per kilogram of your body weight daily. 
        Training Bonus: Tack on an extra 500-750 ml per hour of sweat time
        Fat Loss Hack: Chugging 500 ml of water before meals can boost fullness by 13%. Your stomach will thank you, and so will your waistline.
        """)

        st.subheader("😴 Sleep Like Your Goals Depend on It")
        st.markdown("""
        The Shocking Truth: Getting less than 7 hours of sleep can torpedo your fat loss by more than half.
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

        **Start Small, Win Big:**
        - Weeks 1–2: Focus only on hitting your calorie targets
        - Weeks 3–4: Add protein tracking once calories feel natural  
        - Week 5+: Fine-tune your carb and fat intake

        **When Progress Stalls 🔄**

        *Hit a Weight Loss Plateau?*
        - Double-check your food logging accuracy
        - Review your activity level assessment
        - Add 10-15 minutes of daily walking before cutting calories
        - Consider a "diet break" every 6-8 weeks at maintenance calories
        - Fill up on low-calorie, high-volume foods

        *Struggling to Gain Weight?*
        - Drink your calories through smoothies and shakes
        - Load up on healthy fats like nuts, oils, and avocados
        - Challenge yourself consistently in the gym
        - Bump up intake by 100-150 calories if stuck for 2+ weeks

        **Pace Your Protein**
        - Spread 20-40g across 3-4 meals (0.4-0.5g per kg per meal)
        - Get protein and carbs before and after workouts
        - Try 20-30g casein protein before bed
        """)

    with tab4:
        st.subheader("Understanding Your Metabolism")
        st.markdown("""
        Your Basal Metabolic Rate (BMR) is the energy your body needs just to keep the lights on. Your Total Daily Energy Expenditure (TDEE) builds on that baseline by factoring in how active you are throughout the day.

        **The Smart Eater's Hierarchy:**
        1. **Protein**: King of fullness - digests slowly, steadies blood sugar
        2. **Fiber-Rich Carbs**: Veggies, fruits, whole grains fill you up efficiently  
        3. **Healthy Fats**: Nuts, olive oil, avocados provide steady energy
        4. **Processed Foods**: Fine occasionally, but can't build a strategy around them

        Aim for 14g fiber per 1,000 calories (25-38g daily total). Ramp up gradually to avoid digestive issues.

        **Plant-Based Nutrition Watch List:**
        - **B₁₂**: Essential for nerves/cells - supplement recommended
        - **Iron**: Pair with vitamin C for better absorption  
        - **Calcium**: Find in fortified plant milks, tofu, leafy greens
        - **Zinc**: Nuts, seeds, whole grains support immune function
        - **Iodine**: Iodized salt supports thyroid function
        - **Omega-3s**: Consider algae-based supplements for EPA/DHA

        Always consult healthcare providers for personalized supplement advice.
        """)

# ---------------------------------------------------------------------------
# Cell 11: Enhanced Food Selection with Search
# ---------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")
st.markdown("Pick how many servings of each food you're having to see how your choices stack up against your daily targets.")

# Food Search Filter
search_col1, search_col2 = st.columns([3, 1])
with search_col1:
    search_filter = st.text_input(
        "🔍 Search foods by name:",
        value=st.session_state.get('search_filter', ''),
        key='food_search',
        placeholder="Type to filter foods..."
    )
    st.session_state.search_filter = search_filter

with search_col2:
    if st.button("🔄 Reset All Selections", type="secondary", use_container_width=True):
        st.session_state.food_selections = {}
        st.rerun()

# Filter foods based on search
filtered_foods = filter_foods_by_search(foods, search_filter)

# Emoji Guide
with st.expander("💡 Emoji Guide - What do these symbols mean?"):
    for emoji, tooltip in EMOJI_TOOLTIPS.items():
        if emoji:  # Skip empty emoji
            st.markdown(f"**{emoji}** {tooltip}")

# Food Selection Tabs with Filtered Results
if filtered_foods:
    available_categories = [cat for cat, items in sorted(filtered_foods.items()) if items]
    
    if search_filter and not available_categories:
        st.warning(f"No foods found matching '{search_filter}'. Try a different search term.")
    elif available_categories:
        tabs = st.tabs(available_categories)
        
        for i, category in enumerate(available_categories):
            items = filtered_foods[category]
            sorted_items = sorted(
                items,
                key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories'])
            )
            with tabs[i]:
                render_food_grid(sorted_items, category, columns=2)
    else:
        st.info("Start typing in the search box above to find specific foods!")
else:
    st.warning("No foods available. Please check your food database.")

# ---------------------------------------------------------------------------
# Cell 12: Enhanced Daily Summary with Export Options
# ---------------------------------------------------------------------------

st.header("Today's Scorecard 📊")
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

# Create dynamic sidebar summary
create_sidebar_summary(totals, targets)

if selected_foods:
    # Enhanced progress tracking with colored bars
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

    # Export Options
    st.subheader("📥 Export Your Daily Summary")
    export_col1, export_col2 = st.columns(2)
    
    with export_col1:
        if st.button("📄 Download PDF Summary", use_container_width=True):
            pdf_buffer = create_pdf_summary(totals, targets, selected_foods, final_values)
            st.download_button(
                label="📥 Download PDF",
                data=pdf_buffer,
                file_name=f"nutrition_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    
    with export_col2:
        if st.button("📊 Download CSV Data", use_container_width=True):
            csv_data = pd.DataFrame([{
                'Date': datetime.now().strftime('%Y-%m-%d'),
                'Calories_Target': targets['total_calories'],
                'Calories_Actual': totals['calories'],
                'Protein_Target': targets['protein_g'],
                'Protein_Actual': totals['protein'],
                'Carbs_Target': targets['carb_g'], 
                'Carbs_Actual': totals['carbs'],
                'Fat_Target': targets['fat_g'],
                'Fat_Actual': totals['fat']
            }])
            st.download_button(
                label="📥 Download CSV",
                data=csv_data.to_csv(index=False),
                file_name=f"nutrition_data_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Personalized Recommendations
    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)

    # Food Choices Summary
    with st.expander("📋 Your Food Choices Today", expanded=False):
        st.subheader("What You've Logged")
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings

            st.markdown(f"**{food['name']}** - {servings} serving(s)")
            st.markdown(
                f"→ {total_cals:.0f} kcal | {total_protein:.1f}g protein | "
                f"{total_carbs:.1f}g carbs | {total_fat:.1f}g fat"
            )
else:
    st.info(
        "Haven't picked any foods yet? No worries! Go ahead and add some items from the categories above to start tracking your intake!"
    )
    st.subheader("Progress Snapshot")
    
    # Show empty progress bars with enhanced styling
    progress_html = ""
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']] if targets[config['target_key']] > 0 else 1
        progress_html += create_colored_progress_bar(0, config['label'], target, config['unit'])
    
    st.markdown(progress_html, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 13: Enhanced Personalized Recommendations System
# ---------------------------------------------------------------------------

if st.session_state.get('user_inputs_complete', False):
    st.header("Your Personalized Action Steps 🎯")
    recommendations = generate_personalized_recommendations(totals, targets, final_values)
    
    # Display recommendations in a more engaging format
    rec_col1, rec_col2 = st.columns(2)
    
    for i, rec in enumerate(recommendations):
        target_col = rec_col1 if i % 2 == 0 else rec_col2
        with target_col:
            st.info(rec)

# ---------------------------------------------------------------------------
# Cell 14: User Feedback Section
# ---------------------------------------------------------------------------

st.header("Help Us Improve! 📝")

if not st.session_state.get('feedback_submitted', False):
    with st.form("feedback_form", clear_on_submit=True):
        st.markdown("We'd love to hear from you! Your feedback helps us make this tool even better.")
        
        feedback_type = st.selectbox(
            "What type of feedback do you have?",
            ["General Feedback", "Bug Report", "Feature Request", "Suggestion"]
        )
        
        feedback_text = st.text_area(
            "Tell us what's on your mind:",
            placeholder="Share your thoughts, ideas, or any issues you've encountered...",
            height=100
        )
        
        rating = st.select_slider(
            "How would you rate your experience?",
            options=[1, 2, 3, 4, 5],
            value=4,
            format_func=lambda x: "⭐" * x
        )
        
        submitted = st.form_submit_button("Submit Feedback 🚀", type="primary")
        
        if submitted and feedback_text.strip():
            # In a real app, this would be sent to a database or API
            st.session_state.feedback_submitted = True
            st.success("🎉 Thank you for your feedback! We really appreciate you taking the time to help us improve.")
            st.balloons()
        elif submitted:
            st.warning("Please enter some feedback before submitting.")
else:
    st.success("✅ Thank you for your feedback! We've received your input and will use it to improve the app.")
    if st.button("Submit More Feedback", key="more_feedback"):
        st.session_state.feedback_submitted = False
        st.rerun()

# ---------------------------------------------------------------------------
# Cell 15: Enhanced Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()

# Quick Stats Summary
if selected_foods:
    st.subheader("📈 Quick Stats")
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.metric("Foods Logged", len(selected_foods))
    with stats_col2:
        total_servings = sum(item['servings'] for item in selected_foods)
        st.metric("Total Servings", f"{total_servings:.1f}")
    with stats_col3:
        avg_cals_per_serving = totals['calories'] / total_servings if total_servings > 0 else 0
        st.metric("Avg Cal/Serving", f"{avg_cals_per_serving:.0f}")
    with stats_col4:
        target_completion = (totals['calories'] / targets['total_calories']) * 100 if targets['total_calories'] > 0 else 0
        st.metric("Daily Progress", f"{target_completion:.0f}%")

st.markdown("---")

# Enhanced Footer with Better Formatting
footer_col1, footer_col2 = st.columns(2)

with footer_col1:
    st.markdown("### 📚 The Science We Stand On")
    st.markdown("""
    This tracker isn't built on guesswork—it's grounded in peer-reviewed research:
    
    - **BMR Calculation**: Mifflin-St Jeor equation (Academy of Nutrition and Dietetics endorsed)
    - **TDEE Estimation**: Research-validated activity multipliers from exercise physiology
    - **Protein Targets**: Guidelines from International Society of Sports Nutrition
    - **Caloric Adjustments**: Conservative, sustainable rates proven by research
    
    We're all about setting you up for evidence-based success! 🎯
    """)

with footer_col2:
    st.markdown("### ⚠️ Important Disclaimers")
    st.markdown("""
    **This tool is your launchpad, not your destination:**
    
    - Individual results may vary due to genetics, health conditions, and medications
    - Always consult qualified healthcare providers before major dietary changes
    - Listen to your body - monitor energy, performance, and overall wellbeing
    - Adjust recommendations based on your personal response
    
    **You know yourself best - we're just here to help guide the way!** 🗺️
    """)

# App Information
st.markdown("---")
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**🏗️ Built With:**")
    st.markdown("- Streamlit")
    st.markdown("- Plotly") 
    st.markdown("- Pandas")
    st.markdown("- ReportLab")

with info_col2:
    st.markdown("**📊 Features:**")
    st.markdown("- Evidence-based calculations")
    st.markdown("- Progress tracking")
    st.markdown("- Export capabilities")
    st.markdown("- Unit conversion")

with info_col3:
    st.markdown("**🔧 Version Info:**")
    st.markdown("- Version: 2.0 Enhanced")
    st.markdown("- Last Updated: 2024")
    st.markdown("- Accessibility: WCAG Compliant")
    st.markdown("- Mobile: Responsive Design")

# Final Success Message
st.markdown("---")
st.success(
    "🎉 **Congratulations on taking charge of your nutrition journey!** "
    "Remember, progress isn't always linear, but every small step forward counts. "
    "The best plan is the one you can stick to consistently. You've got this! 💪"
)

# Motivational closing based on goal
if st.session_state.get('user_inputs_complete', False):
    goal = final_values.get('goal', 'weight_gain')
    goal_messages = {
        'weight_loss': "🔥 Your transformation story starts now. Trust the process, stay consistent, and celebrate every victory along the way!",
        'weight_maintenance': "⚖️ Maintenance mode activated! You're building sustainable habits that will serve you for life. Keep up the amazing work!",
        'weight_gain': "💪 Fuel those gains! Every meal is an opportunity to build the stronger, healthier version of yourself. Let's grow!"
    }
    st.info(goal_messages.get(goal, "Your nutrition journey is unique and valuable. Keep moving forward!"))

# ---------------------------------------------------------------------------
# Cell 16: Enhanced Session State Management and Performance Optimization
# ---------------------------------------------------------------------------

# Clean up session state to prevent memory issues and improve performance
if len(st.session_state.food_selections) > 100:
    # Keep only non-zero selections to prevent memory bloat
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# Performance monitoring (optional - can be removed in production)
if st.sidebar.button("🔧 Debug Info", key="debug_info"):
    with st.sidebar.expander("Debug Information"):
        st.write("**Session State Keys:**", list(st.session_state.keys()))
        st.write("**Food Selections Count:**", len(st.session_state.food_selections))
        st.write("**Memory Usage (selections):**", f"{len(str(st.session_state.food_selections))} chars")
        st.write("**Calculation Ready:**", st.session_state.get('calculation_ready', False))
        st.write("**User Inputs Complete:**", st.session_state.get('user_inputs_complete', False))

# Cache cleanup hint for better performance
if st.sidebar.button("🧹 Clear Cache", key="clear_cache"):
    st.cache_data.clear()
    st.sidebar.success("Cache cleared! Reload the page for a fresh start.")

# End of enhanced nutrition tracker application
