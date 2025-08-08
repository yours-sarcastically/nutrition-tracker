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
   pip install streamlit pandas plotly reportlab

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

# ------ Unit Conversion Functions (defined early for use in CONFIG) ------
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
                      'placeholder': 'Enter your height', 'required': True,
                      'conversion': {
                          'system': 'imperial',
                          'label': 'Height (in inches)',
                          'to_unit': cm_to_inches,
                          'from_unit': inches_to_cm,
                          'step': 1.0
                      }},
        'weight_kg': {'type': 'number', 'label': 'Weight (in kilograms)',
                      'min': 40.0, 'max': 150.0, 'step': 0.5,
                      'placeholder': 'Enter your weight', 'required': True,
                      'conversion': {
                          'system': 'imperial',
                          'label': 'Weight (in pounds)',
                          'to_unit': kg_to_lbs,
                          'from_unit': lbs_to_kg,
                          'step': 1.0
                      }},
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
        "**Fat Loss Hack**: Chugging 500 ml of water before meals can boost fullness by by 13%. Your stomach will thank you, and so will your waistline."
    ],
    'sleep': [
        "**The Shocking Truth**: Getting less than 7 hours of sleep can torpedo your fat loss by a more than half.",
        "**Daily Goal**: Shoot for 7-9 hours and try to keep a consistent schedule.",
        "**Set the Scene**: Keep your cave dark, cool (18-20°C), and screen-free for at least an hour before lights out."
    ],
    'tracking_wins': [
        "**Morning Ritual**: Weigh yourself first thing after using the bathroom, before eating or drinking, in minimal clothing.",
        "**Look for Trends, Not Blips**: Watch your weekly average instead of getting hung up on daily fluctuations. Your weight can swing 2-3 pounds daily.",
        "**Hold the Line**: Don't tweak your plan too soon. Wait for two or more weeks of stalled progress before making changes."
    ],
    'beyond_the_scale': [
        "**The Bigger Picture**: Snap a few pics every month. Use the same pose, lighting, and time of day. The mirror doesn't lie.",
        "**Size Up Your Wins**: Measure your waist, hips, arms, and thighs monthly.",
        "**The Quiet Victories**: Pay attention to how you feel. Your energy levels, sleep quality, gym performance, and hunger patterns tell a story numbers can't."
    ],
    'protein_pacing': [
        "**Spread the Love**: Instead of cramming your protein into one or two giant meals, aim for 20-40 grams with each of your 3-4 daily meals. This works out to roughly 0.4-0.5 grams per kilogram of body weight per meal.",
        "**Frame Your Fitness**: Get some carbs and 20–40g protein before and within two hours of wrapping up your workout.",
        "**The Night Shift**: Try 20-30g of casein protein before bed for keeping your muscles fed while you snooze."
    ],
    'weight_loss_plateau': [
        "**Guess Less, Stress Less**: Before you do anything else, double-check how accurately you're logging your food. Little things can add up!",
        "**Activity Audit**: Take a fresh look at your activity level. Has it shifted?",
        "**Walk it Off**: Try adding 10-15 minutes of walking to your daily routine before cutting calories further. It's a simple way to boost progress without tightening the belt just yet.",
        "**Step Back to Leap Forward**: Consider a 'diet break' every 6-8 weeks. Eating at your maintenance calories for a week or two can give your metabolism and your mind a well-deserved reset.",
        "**Leaf Your Hunger Behind**: Load your plate with low-calorie, high-volume foods like leafy greens, cucumbers, and berries. They're light on calories but big on satisfaction."
    ],
    'weight_gain_stalls': [
        "**Drink Your Calories**: Liquid calories from smoothies, milk, and protein shakes go down way easier than another full meal.",
        "**Fat is Fuel**: Load up healthy fats like nuts, oils, and avocados.",
        "**Push Your Limits**: Give your body a reason to grow! Make sure you're consistently challenging yourself in the gym.",
        "**Turn Up the Heat**: If you've been stuck for over two weeks, bump up your intake by 100-150 calories to get the ball rolling again."
    ]
}


# ---------------------------------------------------------------------------
# Cell 4: Unit Formatting Functions
# ---------------------------------------------------------------------------

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

        min_val, max_val, step_val = field_config['min'], field_config['max'], field_config['step']
        current_value = st.session_state[session_key]
        label = field_config['label']
        
        conversion_config = field_config.get('conversion')
        if conversion_config and st.session_state.get('user_units') == conversion_config['system']:
            label = conversion_config['label']
            to_unit_func = conversion_config['to_unit']
            min_val = to_unit_func(min_val)
            max_val = to_unit_func(max_val)
            step_val = conversion_config.get('step', step_val)
            if current_value:
                current_value = to_unit_func(current_value)

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
        
        if conversion_config and st.session_state.get('user_units') == conversion_config['system'] and value:
            from_unit_func = conversion_config['from_unit']
            value = from_unit_func(value)
            
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
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
    base_needs = weight_kg * 35

    activity_bonus = {
        'sedentary': 0, 'lightly_active': 300, 'moderately_active': 500,
        'very_active': 700, 'extremely_active': 1000
    }
    climate_multiplier = {'cold': 0.9, 'temperate': 1.0, 'hot': 1.2, 'very_hot': 1.4}

    total_ml = ((base_needs + activity_bonus.get(activity_level, 500)) *
                climate_multiplier.get(climate, 1.0))
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
    if percent >= 80: return "🟢"
    elif percent >= 50: return "🟡"
    else: return "🔴"


def render_progress_bars(totals, targets):
    """Renders a set of progress bars for all nutrients."""
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals.get(nutrient, 0)
        target = targets.get(config['target_key'], 1)
        target = target if target > 0 else 1

        percent = min((actual / target) * 100, 100)
        color_indicator = get_progress_color(percent)

        st.progress(
            percent / 100,
            text=(f"{color_indicator} {config['label']}: {percent:.0f}% of your daily target "
                  f"({target:.0f} {config['unit']})")
        )


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
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
            deficits[nutrient] = {
                'amount': target - actual,
                'unit': config['unit'],
                'label': config['label'].lower(),
                'purpose': purpose_map.get(nutrient, 'for optimal nutrition')
            }

    if deficits:
        all_foods = [item for sublist in foods.values() for item in sublist]
        food_suggestions = []
        
        for food in all_foods:
            coverage_score, nutrients_helped = 0, []
            for nutrient, deficit_info in deficits.items():
                if nutrient != 'calories' and food[nutrient] > 0:
                    help_percentage = min(food[nutrient] / deficit_info['amount'], 1.0)
                    if help_percentage > 0.1:
                        coverage_score += help_percentage
                        nutrients_helped.append(nutrient)
            
            if coverage_score > 0 and len(nutrients_helped) > 1:
                food_suggestions.append({'food': food, 'nutrients_helped': nutrients_helped, 'score': coverage_score})
        
        food_suggestions.sort(key=lambda x: x['score'], reverse=True)
        top_suggestions = food_suggestions[:3]

        deficit_summary = [f"{info['amount']:.0f}g more {info['label']} {info['purpose']}" for info in deficits.values()]
        
        summary_text = f"You still need: {', '.join(deficit_summary[:-1])}, and {deficit_summary[-1]}." if len(deficit_summary) > 1 else f"You still need: {deficit_summary[0]}."
        recommendations.append(summary_text)
        
        if top_suggestions:
            for i, suggestion in enumerate(top_suggestions):
                food, nutrients_helped = suggestion['food'], suggestion['nutrients_helped']
                nutrient_benefits = [f"{food[n]:.0f}g {n}" for n in nutrients_helped]
                benefits_text = f"{', '.join(nutrient_benefits[:-1])}, and {nutrient_benefits[-1]}" if len(nutrient_benefits) > 1 else nutrient_benefits[0]
                
                if i == 0:
                    recommendations.append(f"🎯 Smart pick: One serving of {food['name']} would give you {benefits_text}, knocking out multiple targets at once!")
                else:
                    recommendations.append(f"💡 Alternative option: {food['name']} provides {benefits_text}, another great way to hit multiple goals!")
        else:
            nutrient, deficit_info = max(deficits.items(), key=lambda x: x[1]['amount'])
            best_single_food = max(all_foods, key=lambda x: x.get(nutrient, 0), default=None)
            if best_single_food and best_single_food.get(nutrient, 0) > 0:
                recommendations.append(f"💡 Try adding {best_single_food['name']} - it's packed with {best_single_food[nutrient]:.0f}g of {deficit_info['label']}.")

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
    summary_data = {'nutrition_summary': [], 'consumed_foods': []}

    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = (actual / target * 100) if target > 0 else 0
        summary_data['nutrition_summary'].append({
            'label': config['label'], 'actual': actual, 'target': target,
            'unit': config['unit'], 'percent': percent
        })

    for item in selected_foods:
        food, servings = item['food'], item['servings']
        summary_data['consumed_foods'].append({
            'name': food['name'], 'servings': servings,
            'calories': food['calories'] * servings, 'protein': food['protein'] * servings,
            'carbs': food['carbs'] * servings, 'fat': food['fat'] * servings
        })
    return summary_data


def create_pdf_summary(totals, targets, selected_foods, user_info):
    """Create a PDF summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    buffer = io.BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, height - 50, "Daily Nutrition Summary")
    
    p.setFont("Helvetica", 12)
    p.drawString(50, height - 80, f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    
    y_pos = height - 120
    p.drawString(50, y_pos, f"Age: {user_info.get('age', 'N/A')}")
    p.drawString(200, y_pos, f"Weight: {user_info.get('weight_kg', 'N/A')} kg")
    p.drawString(350, y_pos, f"Goal: {user_info.get('goal', 'N/A')}")
    
    y_pos -= 40
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y_pos, "Nutrition Summary")
    
    y_pos -= 30
    p.setFont("Helvetica", 12)
    for item in summary_data['nutrition_summary']:
        p.drawString(50, y_pos, f"{item['label']}: {item['actual']:.0f}/{item['target']:.0f} {item['unit']} ({item['percent']:.0f}%)")
        y_pos -= 20
    
    if summary_data['consumed_foods']:
        y_pos -= 20
        p.setFont("Helvetica-Bold", 14)
        p.drawString(50, y_pos, "Foods Consumed")
        
        y_pos -= 30
        p.setFont("Helvetica", 10)
        for item in summary_data['consumed_foods'][:20]:
            p.drawString(50, y_pos, f"• {item['name']}: {item['servings']} serving(s)")
            y_pos -= 15
            if y_pos < 50: break
    
    p.save()
    buffer.seek(0)
    return buffer


def create_csv_summary(totals, targets, selected_foods):
    """Create a CSV summary of the daily nutrition."""
    summary_data = prepare_summary_data(totals, targets, selected_foods)
    data = []
    
    for item in summary_data['nutrition_summary']:
        data.append({
            'Category': 'Nutrition Summary', 'Item': item['label'],
            'Actual': f"{item['actual']:.0f} {item['unit']}", 'Target': f"{item['target']:.0f} {item['unit']}",
            'Percentage': f"{item['percent']:.0f}%"
        })
    
    for item in summary_data['consumed_foods']:
        data.append({
            'Category': 'Foods Consumed', 'Item': item['name'], 'Servings': item['servings'],
            'Calories': f"{item['calories']:.0f} kcal", 'Protein': f"{item['protein']:.1f} g",
            'Carbs': f"{item['carbs']:.1f} g", 'Fat': f"{item['fat']:.1f} g"
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
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(age, height_cm, weight_kg, sex='male',
                                   activity_level='moderately_active',
                                   goal='weight_gain', protein_per_kg=None,
                                   fat_percentage=None):
    """Calculates personalized daily nutritional targets."""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])
    
    total_calories = tdee * (1 + goal_config['caloric_adjustment'])
    caloric_adjustment = total_calories - tdee

    protein_per_kg_final = protein_per_kg or goal_config['protein_per_kg']
    fat_percentage_final = fat_percentage or goal_config['fat_percentage']

    protein_g = protein_per_kg_final * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage_final
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories), 'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'fat_g': round(fat_g), 'carb_g': round(carb_g),
        'protein_calories': round(protein_calories), 'fat_calories': round(fat_calories),
        'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(calculate_estimated_weekly_change(caloric_adjustment), 3),
        'goal': goal
    }

    if total_calories > 0:
        targets['protein_percent'] = (protein_calories / total_calories) * 100
        targets['carb_percent'] = (carb_calories / total_calories) * 100
        targets['fat_percent'] = (fat_calories / total_calories) * 100
    else:
        targets.update({'protein_percent': 0, 'carb_percent': 0, 'fat_percent': 0})

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
        foods[row['category']].append({
            'name': f"{row['name']} ({row['serving_unit']})", 'calories': row['calories'],
            'protein': row['protein'], 'carbs': row['carbs'], 'fat': row['fat']
        })
    return foods


@st.cache_data
def assign_food_emojis(foods):
    """Assigns emojis to foods based on a unified ranking system."""
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'calories': {}}
    for category, items in foods.items():
        if not items: continue
        top_foods['calories'][category] = [f['name'] for f in sorted(items, key=lambda x: x['calories'], reverse=True)[:3]]
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            top_foods[map_info['key']] = [f['name'] for f in sorted(items, key=lambda x: x[map_info['sort_by']], reverse=True)[:3]]

    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]}
    emoji_mapping = {'high_cal_nutrient': '🥇', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑'}

    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])
            
            if is_high_calorie and is_top_nutrient: food['emoji'] = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: food['emoji'] = emoji_mapping['high_calorie']
            elif food_name in top_foods['protein']: food['emoji'] = emoji_mapping['protein']
            elif food_name in top_foods['carbs']: food['emoji'] = emoji_mapping['carbs']
            elif food_name in top_foods['fat']: food['emoji'] = emoji_mapping['fat']
            else: food['emoji'] = ''
    return foods


def filter_foods_by_search(foods, search_term):
    """Filter foods based on search term."""
    if not search_term: return foods
    filtered_foods = {}
    search_lower = search_term.lower()
    for category, items in foods.items():
        filtered_items = [food for food in items if search_lower in food['name'].lower()]
        if filtered_items: filtered_foods[category] = filtered_items
    return filtered_foods


def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        emoji = food.get('emoji', '')
        if emoji and emoji in EMOJI_TOOLTIPS:
            st.markdown(f"**{emoji}** {food['name']}")
            st.caption(EMOJI_TOOLTIPS[emoji])
        else:
            st.subheader(f"{emoji} {food['name']}")
            
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)

        col1, col2 = st.columns([2, 1.2])
        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k-1]:
                    if st.button(f"{k}", key=f"{key}_{k}", type="primary" if current_serving == float(k) else "secondary", help=f"Set to {k} servings", use_container_width=True):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()
        with col2:
            custom_serving = st.number_input("Custom", min_value=0.0, max_value=20.0, value=current_serving, step=0.1, key=f"{key}_custom", label_visibility="collapsed")

        if custom_serving != current_serving:
            if custom_serving > 0: st.session_state.food_selections[food['name']] = custom_serving
            elif food['name'] in st.session_state.food_selections: del st.session_state.food_selections[food['name']]
            st.rerun()

        st.caption(f"Per Serving: {food['calories']} kcal | {food['protein']}g protein | {food['carbs']}g carbs | {food['fat']}g fat")


def render_food_grid(items, category, columns=2):
    """Renders a grid of food items for a given category."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]: render_food_item(items[i + j], category)


# ---------------------------------------------------------------------------
# Cell 8: Initialize Application
# ---------------------------------------------------------------------------

initialize_session_state()
foods = assign_food_emojis(load_food_database('nutrition_results.csv'))

st.markdown("""<style>[data-testid="InputInstructions"]{display:none}.stButton>button[kind="primary"]{background-color:#ff6b6b;color:white;border:1px solid #ff6b6b}.stButton>button[kind="secondary"]{border:1px solid #ff6b6b;color:#333}.sidebar .sidebar-content{background-color:#f0f2f6}.stMetric>div>div>div>div{color:#262730}.stProgress .st-bo{background-color:#e0e0e0}.stProgress .st-bp{background-color:#ff6b6b}.stCaption{color:#555!important}</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 9: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.markdown("A Smart, Evidence-Based Nutrition Tracker That Actually Gets You\n\nWelcome aboard!\n\nHey there! Welcome to your new nutrition buddy. This isn't just another calorie counter—it's your personalized guide, built on rock-solid science to help you smash your goals. Whether you're aiming to shed a few pounds, hold steady, or bulk up, we've crunched the numbers so you can focus on enjoying your food.\n\nLet's get rolling—your journey to feeling awesome starts now! 🚀")

st.sidebar.header("Let's Get Personal 📊")
st.session_state.user_units = 'imperial' if st.sidebar.toggle("Use Imperial Units", value=(st.session_state.user_units == 'imperial'), key='units_toggle', help="Toggle for Imperial (lbs, inches) or Metric (kg, cm)") else 'metric'

all_inputs = {}
for field, config in CONFIG['form_fields'].items():
    container = st.sidebar.expander("Advanced Settings ⚙️") if config.get('advanced') else st.sidebar
    value = create_unified_input(field, config, container=container)
    if 'convert' in config: value = config['convert'](value)
    all_inputs[field] = value

if st.sidebar.button("🧮 Calculate My Targets", type="primary", key="calculate_button"):
    errors = validate_user_inputs(all_inputs)
    if errors: [st.sidebar.error(e) for e in errors]
    else:
        st.session_state.form_submitted = True
        st.session_state.show_motivational_message = True
        st.rerun()

st.sidebar.divider()
st.sidebar.subheader("💾 Save Your Progress")
if st.sidebar.button("Save", key="save_progress", type="primary"):
    st.sidebar.download_button("📥 Download", data=save_progress_to_json(st.session_state.food_selections, all_inputs), file_name=f"nutrition_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json", key="download_progress")

st.sidebar.subheader("📂 Load Progress")
uploaded_file = st.sidebar.file_uploader("Load", type="json", key="upload_progress")
if uploaded_file:
    food_selections, user_inputs = load_progress_from_json(uploaded_file.read().decode())
    st.session_state.food_selections.update(food_selections)
    for k, v in user_inputs.items():
        if f'user_{k}' in st.session_state: st.session_state[f'user_{k}'] = v
    st.sidebar.success("Progress loaded successfully!")
    st.rerun()

with st.sidebar.expander("Your Activity Level Decoded", expanded=True):
    st.markdown("* **🧑‍💻 Sedentary**: Basically married to your desk chair.\n* **🏃 Lightly Active**: Squeeze in walks or workouts 1-3x a week.\n* **🚴 Moderately Active**: Sweating it out 3-5 days a week.\n* **🏋️ Very Active**: You might be part treadmill.\n* **🤸 Extremely Active**: You live in the gym; sweat is your second skin.\n\n*💡 If torn, pick the lower level. Better to underestimate your burn than to overeat and stall.*")

final_values = get_final_values(all_inputs)
targets = calculate_personalized_targets(**final_values)
user_has_entered_info = st.session_state.form_submitted

if user_has_entered_info:
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
    st.sidebar.divider()
    st.sidebar.markdown("### 📊 Quick Summary")
    st.sidebar.metric("Calories Progress", f"{min(totals['calories']/targets['total_calories']*100,100):.0f}%" if targets['total_calories'] > 0 else "0%", f"{totals['calories']:.0f}/{targets['total_calories']:.0f} kcal")
    st.sidebar.metric("Protein Progress", f"{min(totals['protein']/targets['protein_g']*100,100):.0f}%" if targets['protein_g'] > 0 else "0%", f"{totals['protein']:.0f}/{targets['protein_g']:.0f} g")

if st.session_state.show_motivational_message and user_has_entered_info:
    goal_messages = {
        'weight_loss': f"🎉 Awesome! You're set up for success! Your plan targets a loss of approximately **{abs(targets['estimated_weekly_change']):.2f} kg/week**. Stay consistent!",
        'weight_maintenance': f"🎯 Perfect! Your maintenance plan is locked in to hold your current weight of **{format_weight(final_values['weight_kg'], st.session_state.user_units)}**.",
        'weight_gain': f"💪 Let's grow! Your journey starts now, targeting a healthy gain of about **{targets['estimated_weekly_change']:.2f} kg/week**. Fuel up and lift heavy!"
    }
    st.success(goal_messages.get(targets['goal'], "🚀 You're all set! Let's crush those goals!"))
    if st.button("✨ Got it!", key="dismiss_message"):
        st.session_state.show_motivational_message = False
        st.rerun()

# ---------------------------------------------------------------------------
# Cell 10: Unified Target Display System
# ---------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Enter your details in the sidebar and click 'Calculate My Targets' to get started.")
    st.header("Sample Daily Targets for Reference")
else:
    goal_label = {'weight_loss': 'Weight Loss', 'weight_maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}.get(targets['goal'])
    st.header(f"Your Custom Daily Nutrition Roadmap for {goal_label} 🎯")

st.info("🎯 **The 80/20 Rule**: Aim to hit your targets about 80% of the time. This gives you wiggle room for life's curveballs. Flexibility builds consistency and helps you avoid the yo-yo diet trap.")

hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
metrics_config = [
    {'title': 'Metabolic Information', 'columns': 5, 'metrics': [
        ("Weight", format_weight(final_values['weight_kg'], st.session_state.user_units)),
        ("BMR", f"{targets['bmr']} kcal"), ("TDEE", f"{targets['tdee']} kcal"),
        ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
        ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kg")
    ]},
    {'title': 'Your Daily Nutrition Targets', 'columns': 5, 'metrics': [
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
# Cell 11: Enhanced Evidence-Based Tips and Context (Collapsed by default)
# ---------------------------------------------------------------------------

with st.expander("📚 Your Evidence-Based Game Plan", expanded=False):
    tab_titles = ["The Big Three 🏆", "Progress Tracking 📊", "Mindset Is Everything 🧠", "The Science 🔬"]
    tab1, tab2, tab3, tab4 = st.tabs(tab_titles)

    with tab1:
        st.subheader("💧 Master Your Hydration Game"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['hydration']]
        st.subheader("😴 Sleep Like Your Goals Depend on It"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['sleep']]
        st.subheader("📅 Follow Your Wins"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['tracking_wins']]
    with tab2:
        st.subheader("Go Beyond the Scale 📸"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['beyond_the_scale']]
    with tab3:
        st.subheader("Mindset Is Everything 🧠")
        st.markdown("The 80/20 principle is your best defense against the perfectionist trap. Ditch the all-or-nothing mindset. Instead of mastering everything at once, build habits gradually for long-term success.\n\n**Start Small, Win Big:**\n\n* **Weeks 1–2**: Focus only on hitting your calorie targets.\n* **Weeks 3–4**: Layer in protein tracking.\n* **Week 5+**: Fine-tune your carb and fat intake.\n\n---\n**When Progress Stalls** 🔄")
        st.markdown("##### Hit a Weight Loss Plateau?"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['weight_loss_plateau']]
        st.markdown("##### Struggling to Gain Weight?"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['weight_gain_stalls']]
        st.markdown("---\n##### Pace Your Protein"); [st.markdown(f"* {tip}") for tip in TIPS_CONTENT['protein_pacing']]
    with tab4:
        st.subheader("Understanding Your Metabolism")
        st.markdown("Your **Basal Metabolic Rate (BMR)** is the energy your body needs just to keep the lights on. Your **Total Daily Energy Expenditure (TDEE)** adds the calories burned through activity.\n\n**The Smart Eater's Cheat Sheet**\n\nNot all calories are created equal. Prioritize foods that keep you full:\n\n* **Protein**: The king of fullness. Digests slowly, steadies blood sugar. *Examples: Eggs, Greek yogurt, tofu, lentils.*\n* **Fiber-Rich Carbs**: High-volume, low-calorie heroes. *Examples: Veggies, fruits, whole grains.*\n* **Healthy Fats**: Deliver steady, long-lasting energy. *Examples: Nuts, olive oil, avocados.*\n\nAim for **14g of fiber per 1,000 calories** (usually 25-38g daily). Ramp up gradually.\n\n**Your Nutritional Supporting Cast**\n\nOn a plant-based diet? Keep an eye on these micronutrients:\n\n* **B₁₂**: Essential for cell and nerve function. Usually requires a supplement.\n* **Iron**: Transports oxygen. Pair plant sources (lentils, greens) with Vitamin C (peppers, citrus) to boost absorption.\n* **Calcium**: For bones and muscles. Find it in kale, almonds, and fortified plant milks.\n* **Zinc**: Your immune system's security detail. Found in nuts, seeds, and whole grains.\n* **Iodine**: Crucial for metabolism. A pinch of iodized salt is often enough.\n* **Omega-3s (EPA/DHA)**: Premium fuel for brain and heart. Consider fortified foods or supplements.\n\nFortified foods (plant milks, cereals) and targeted supplements are your safety net. Always consult a healthcare provider to build a plan that's right for you.")

# ---------------------------------------------------------------------------
# Cell 12: [REMOVED]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Cell 13: Food Selection Interface
# ---------------------------------------------------------------------------

st.header("Track Your Daily Intake 🥗")
search_col, reset_col = st.columns([3, 1])
search_term = search_col.text_input("🔍 Search for foods", value=st.session_state.get('food_search', ''), placeholder="Type to filter...", key="food_search_input")
st.session_state.food_search = search_term
if reset_col.button("🔄 Clear Search", key="clear_search", use_container_width=True):
    st.session_state.food_search = ""
    st.rerun()

st.markdown("Select the number of servings to see how your choices stack up against your daily targets.")
with st.expander("💡 Emoji Guide"):
    st.markdown("* **🥇 Gold Medal**: Nutritional all-star; high in its target nutrient and calorie-efficient.\n* **🔥 High Calorie**: A more calorie-dense option in its group.\n* **💪 High Protein**: A protein powerhouse.\n* **🍚 High Carb**: A carbohydrate champion.\n* **🥑 High Fat**: A healthy fat hero.")

if st.button("🔄 Reset All Food Selections", type="secondary", key="reset_foods"):
    st.session_state.food_selections = {}
    st.rerun()

filtered_foods = filter_foods_by_search(foods, search_term)
if not filtered_foods and search_term:
    st.warning(f"No foods found matching '{search_term}'. Try a different search.")
elif filtered_foods:
    available_categories = [cat for cat in sorted(filtered_foods.keys()) if filtered_foods[cat]]
    tabs = st.tabs(available_categories)
    for i, category in enumerate(available_categories):
        sorted_items = sorted(filtered_foods[category], key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
        with tabs[i]: render_food_grid(sorted_items, category, columns=2)

# ---------------------------------------------------------------------------
# Cell 14: Daily Summary and Progress Tracking
# ---------------------------------------------------------------------------

st.header("Today's Scorecard 📊")
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)
    
    st.subheader("📥 Export Your Summary")
    pdf_col, csv_col = st.columns(2)
    pdf_buffer = create_pdf_summary(totals, targets, selected_foods, final_values)
    pdf_col.download_button("📄 Download PDF Report", data=pdf_buffer, file_name=f"nutrition_summary_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf", use_container_width=True)
    csv_data = create_csv_summary(totals, targets, selected_foods)
    csv_col.download_button("📊 Download CSV Data", data=csv_data, file_name=f"nutrition_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv", use_container_width=True)
    
    snap_col, macro_col = st.columns(2)
    with snap_col:
        st.subheader("Today's Nutrition Snapshot")
        summary_metrics = [("Calories", f"{totals['calories']:.0f} kcal"), ("Protein", f"{totals['protein']:.0f} g"), ("Carbs", f"{totals['carbs']:.0f} g"), ("Fat", f"{totals['fat']:.0f} g")]
        display_metrics_grid(summary_metrics, 2)
    with macro_col:
        st.subheader("Your Macronutrient Split")
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
        if sum(macro_values) > 0:
            fig = go.Figure(go.Pie(labels=['Protein', 'Carbs', 'Fat'], values=macro_values, hole=.4, marker_colors=['#ff6b6b', '#feca57', '#48dbfb'], textinfo='label+percent', insidetextorientation='radial'))
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=250)
            st.plotly_chart(fig, use_container_width=True)
        else: st.caption("Select foods to see the split.")

    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations: st.info(rec)

    with st.expander("Your Food Choices Today"):
        for item in selected_foods:
            food, servings = item['food'], item['servings']
            st.markdown(f"**{food['name']}** - {servings} serving(s)\n\n→ {food['calories']*servings:.0f} kcal | {food['protein']*servings:.1f}g protein | {food['carbs']*servings:.1f}g carbs | {food['fat']*servings:.1f}g fat")
else:
    st.info("Haven't picked any foods yet? Add items from the categories above to start tracking your intake!")
    st.subheader("Progress Snapshot")
    render_progress_bars(totals, targets)

# ---------------------------------------------------------------------------
# Cell 15: User Feedback Section
# ---------------------------------------------------------------------------

st.divider()
st.header("💬 Help Us Improve!")
with st.form("feedback_form", clear_on_submit=True):
    feedback_type = st.selectbox("What type of feedback do you have?", ["General Feedback", "Bug Report", "Feature Request", "Success Story"])
    feedback_text = st.text_area("How can we improve?", placeholder="Share your experience, suggest features, or report issues...", height=100)
    if st.form_submit_button("📤 Submit Feedback", type="primary"):
        if feedback_text.strip(): st.success(f"Thank you for your {feedback_type.lower()}! Your input helps us improve. 🙏")
        else: st.error("Please enter some feedback before submitting.")

# ---------------------------------------------------------------------------
# Cell 16: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()
st.markdown("### The Science We Stand On 📚\n\nThis tracker isn't built on guesswork—it's grounded in peer-reviewed research. We use the **Mifflin-St Jeor equation** for your Basal Metabolic Rate (BMR), endorsed by the Academy of Nutrition and Dietetics as the gold standard. Your Total Daily Energy Expenditure (TDEE) is calculated using activity multipliers from exercise physiology research. Protein targets are based on guidelines from the International Society of Sports Nutrition. Caloric adjustments are conservative and sustainable, designed for lasting results.\n\n### The Fine Print ⚠️\n\nThink of this tool as your launchpad, not a prescription. Individual results vary due to genetics, health conditions, and medications. Always consult a qualified healthcare provider before making significant dietary changes. Above all, listen to your body—track your energy, performance, and well-being, and adjust as needed.")
st.success("You've reached the end! Thanks for joining this nutrition adventure. Keep showing up for yourself—you've got this! 🥳")

# ---------------------------------------------------------------------------
# Cell 17: Session State Management and Performance
# ---------------------------------------------------------------------------

if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {k: v for k, v in st.session_state.food_selections.items() if v > 0}

temp_keys = [key for key in st.session_state if key.startswith('temp_')]
for key in temp_keys: del st.session_state[key]
