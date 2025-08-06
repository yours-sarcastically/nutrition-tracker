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
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Cell 2: Page Configuration and Initial Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Personalized Nutrition Tracker",
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
    'fat_percentage': 0.25
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
        'sex': {'type': 'selectbox', 'label': 'Sex',
                'options': ["Select Sex", "Male", "Female"],
                'required': True, 'placeholder': "Select Sex"},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level',
                           'options': [
                               ("Select Activity Level", None),
                               ("Sedentary", "sedentary"),
                               ("Lightly Active", "lightly_active"),
                               ("Moderately Active", "moderately_active"),
                               ("Very Active", "very_active"),
                               ("Extremely Active", "extremely_active")
                           ], 'required': True, 'placeholder': None},
        'goal': {'type': 'selectbox', 'label': 'Nutrition Goal',
                 'options': [
                     ("Select Goal", None),
                     ("Weight Loss", "weight_loss"),
                     ("Weight Maintenance", "weight_maintenance"),
                     ("Weight Gain", "weight_gain")
                 ], 'required': True, 'placeholder': None},
        'protein_per_kg': {'type': 'number',
                           'label': 'Protein (in grams per kilogram of '
                                    'body weight)',
                           'min': 1.2, 'max': 3.0, 'step': 0.1,
                           'help': 'Protein intake per kilogram of body '
                                   'weight',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number',
                           'label': 'Fat (as a percent of total calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Percentage of total calories from fat',
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
        ['food_selections'] +
        [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    )

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else None


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Creates an input widget based on a unified configuration."""
    session_key = f'user_{field_name}'

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
            help=field_config.get('help')
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            options = field_config['options']
            index = next(
                (i for i, (_, val) in enumerate(options) if val == current_value),
                0
            )
            selection = container.selectbox(
                field_config['label'],
                options,
                index=index,
                format_func=lambda x: x[0]
            )
            value = selection[1]
        else:
            options = field_config['options']
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(
                field_config['label'],
                options,
                index=index
            )

    st.session_state[session_key] = value
    return value


def get_final_values(user_inputs):
    """Processes all user inputs and applies default values where needed."""
    final_values = {}

    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
        elif field in ['activity_level', 'goal']:
            final_values[field] = value if value is not None else DEFAULTS[field]
        else:
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
            f"Try adding **{suggestion_servings} serving of {best_food['name']}** "
            f"(which provides approximately {best_food[nutrient]:.0f} grams of "
            f"{nutrient})."
        )
    return None


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
    recommendations = []
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")

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

        st.progress(
            percent / 100,
            text=(
                f"{config['label']}: {percent:.0f}% of daily target "
                f"({target:.0f} {config['unit']})"
            )
        )

        if actual < target:
            deficit = target - actual
            purpose = purpose_map.get(nutrient, 'for optimal nutrition')
            base_rec = (
                f"• You need **{deficit:.0f} more {config['unit']}** of "
                f"{config['label'].lower()} {purpose}."
            )

            if nutrient in ['protein', 'carbs', 'fat']:
                food_suggestion = find_best_food_for_nutrient(
                    nutrient, deficit, foods
                )
                if food_suggestion:
                    base_rec += f" {food_suggestion}"

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
        f"💧 **Daily Hydration Target:** Aim for {hydration_ml} milliliters "
        f"(approximately {hydration_ml/250:.1f} cups). Drinking 500 "
        f"milliliters before meals can help boost satiety."
    )

    if goal == 'weight_loss':
        recommendations.extend([
            "🛏️ **Sleep Priority:** Aim for seven to nine hours of sleep "
            "nightly, as poor sleep can reduce the effectiveness of fat "
            "loss by up to 55 percent.",
            "📊 **Weigh-in Strategy:** Perform daily morning weigh-ins and "
            "track weekly averages to monitor progress accurately, rather "
            "than focusing on daily fluctuations.",
            "🥗 **Volume Eating:** Prioritize high-volume, low-calorie foods "
            "such as leafy greens, cucumbers, and berries to enhance meal "
            "satisfaction."
        ])
    elif goal == 'weight_gain':
        recommendations.extend([
            "🥤 **Liquid Calories:** Include smoothies, milk, and juices to "
            "increase your overall calorie density and make it easier to "
            "reach your targets.",
            "🥑 **Healthy Fats:** Add nuts, oils, and avocados to your meals, "
            "as these are calorie-dense options that support a surplus.",
            "💪 **Progressive Overload:** Ensure you are consistently getting "
            "stronger in the gym. A caloric surplus without proper training "
            "can lead to mostly fat gain."
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
        f"⏰ **Protein Timing:** Distribute your protein intake across four "
        f"meals (approximately {protein_per_meal:.0f} grams per meal) to "
        f"optimize muscle protein synthesis."
    )

    return recommendations


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


def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        st.subheader(f"{food.get('emoji', '')} {food['name']}")
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
                min_value=0.0, max_value=10.0,
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
# Cell 7: Initialize Application
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
.stButton>button[kind="primary"] { background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b; }
.stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("A Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
This advanced nutrition tracker uses evidence-based calculations to provide
personalized daily nutrition goals for **weight loss**, **weight
maintenance**, or **weight gain**. The calculator employs the **Mifflin-St
Jeor equation** for BMR and follows a **protein-first macronutrient
strategy** that is recommended by nutrition science. 🚀
""")

# ------ Sidebar for User Input ------
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

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

# ------ Activity Level Guide in Sidebar ------
with st.sidebar.container(border=True):
    st.markdown("""
    **An Activity Level Guide:**

    • **Sedentary:** Little to no exercise, with a desk job.
    • **Lightly Active:** Light exercise one to three days per week.
    • **Moderately Active:** Moderate exercise three to five days per week.
    • **Very Active:** Heavy exercise six to seven days per week.
    • **Extremely Active:** Very heavy exercise, a physical job, or training
    two times per day.

    *💡 When in doubt, it is best to choose a lower activity level to avoid
    overestimating your calorie needs.*
    """)

# ------ Process Final Values ------
final_values = get_final_values(all_inputs)

if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_ml = calculate_hydration_needs(
        final_values['weight_kg'], final_values['activity_level']
    )
    st.sidebar.info(
        f"💧 **Daily Hydration Target:** {hydration_ml} ml "
        f"(approximately {hydration_ml/250:.1f} cups)."
    )

# ------ Check for User Input ------
required_fields = [
    field for field, config in CONFIG['form_fields'].items()
    if config.get('required')
]
user_has_entered_info = all(
    (all_inputs.get(field) is not None and
     all_inputs.get(field) != CONFIG['form_fields'][field].get('placeholder'))
    for field in required_fields
)

# ------ Calculate Personalized Targets ------
targets = calculate_personalized_targets(**final_values)

# ---------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# ---------------------------------------------------------------------------

if not user_has_entered_info:
    st.info(
        "👈 Please enter your personal information in the sidebar to view "
        "your daily nutritional targets."
    )
    st.header("Sample Daily Targets for Reference 🎯")
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
    st.header(f"Your Personalized Daily Nutritional Targets for {goal_label} 🎯")

st.info(
    "🎯 **The 80/20 Principle:** Aim for 80 percent adherence to your targets "
    "rather than perfection. This approach allows for social flexibility and "
    "prevents the all-or-nothing mentality that often leads to diet cycling."
)

hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Unified Metrics Display Configuration ------
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal per day"),
            ("Total Daily Energy Expenditure (TDEE)",
             f"{targets['tdee']} kcal per day"),
            ("Daily Caloric Adjustment",
             f"{targets['caloric_adjustment']:+} kcal per day"),
            ("Estimated Weekly Weight Change",
             f"{targets['estimated_weekly_change']:+.2f} kg per week"),
            ("", "")
        ]
    },
    {
        'title': 'Daily Macronutrient and Hydration Targets', 'columns': 5,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            ("Protein", f"{targets['protein_g']} g",
             f"{targets['protein_percent']:.0f}%"),
            ("Carbohydrates", f"{targets['carb_g']} g",
             f"{targets['carb_percent']:.0f}%"),
            ("Fat", f"{targets['fat_g']} g",
             f"{targets['fat_percent']:.0f}%"),
            ("💧 Hydration", f"{hydration_ml} ml",
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
# Cell 10: Enhanced Evidence-Based Tips and Context
# ---------------------------------------------------------------------------

st.header("📚 An Evidence-Based Playbook")
tab1, tab2, tab3, tab4 = st.tabs([
    "Foundations", "Advanced Strategies",
    "Troubleshooting", "Nutrition Science"
])

with tab1:
    st.subheader("🏆 Essential Tips for Success")
    st.markdown("""
    ### **The Foundation Trio for Success**

    **💧 A Hydration Protocol:**
    - **Target:** 35 milliliters per kilogram of body weight daily.
    - **Training Bonus:** Add 500 to 750 milliliters per hour of exercise.
    - **Fat Loss Hack:** Drinking 500 milliliters of water before meals can
      increase satiety by 13 percent.

    **😴 Sleep Quality (The Game-Changer):**
    - **Less than seven hours of sleep** can reduce the effectiveness of fat
      loss by up to 55 percent.
    - **Target:** Seven to nine hours nightly with consistent sleep and wake
      times.
    - **Optimization:** A dark, cool room (18-20°C) and no screens for one to
      two hours before bed can help.

    **⚖️ Weigh-In Best Practices:**
    - **Daily:** Weigh yourself at the same time each day, preferably in the
      morning, after using the bathroom, and with minimal clothing.
    - **Track:** Monitor weekly averages, not daily fluctuations.
    - **Adjust:** Make changes to your plan only after two or more weeks of
      stalled progress.
    """)

with tab2:
    st.subheader("📊 Advanced Monitoring and Psychology")
    st.markdown("""
    ### **Beyond the Scale: Better Progress Indicators**
    - **Progress Photos:** Take photos in the same lighting, with the same
      poses, and at the same time of day for consistency.
    - **Body Measurements:** Measure your waist, hips, arms, and thighs
      monthly.
    - **Performance Metrics:** Track your strength gains, energy levels, and
      sleep quality.

    ### **The Psychology of Sustainable Change**
    **The 80/20 Rule:** Aim for 80 percent adherence rather than perfection.
    This allows for social flexibility and prevents the "all-or-nothing"
    mentality that leads to diet cycling.

    **Progressive Implementation:**
    - **Weeks One and Two:** Focus only on hitting your calorie targets.
    - **Weeks Three and Four:** Add your protein targets to the focus.
    - **Week Five and Beyond:** Fine-tune your fat and carbohydrate
      distribution.

    **Biofeedback Awareness:** Monitor your energy levels, sleep quality,
    gym performance, and hunger patterns, not just the number on the scale.
    """)

with tab3:
    st.subheader("🔄 Plateau Prevention and Meal Timing")
    st.markdown("""
    ### **A Plateau Troubleshooting Flow**

    **Weight Loss Plateaus:**
    1. Confirm the accuracy of your food logging (within five percent of
       calories).
    2. Re-validate your selected activity multiplier.
    3. Add 10 to 15 minutes of daily walking before reducing your calories.
    4. Implement "diet breaks" by eating at maintenance for one to two
       weeks every six to eight weeks.

    **Weight Gain Plateaus:**
    1. Increase your intake of liquid calories (such as smoothies and milk).
    2. Add more healthy fats from sources like nuts, oils, and avocados.
    3. Ensure you are applying progressive overload in your training.
    4. Make gradual increases of 100 to 150 calories when you have stalled
       for two or more weeks.

    ### **Meal Timing and Distribution**

    **Protein Optimization:**
    - **Distribution:** Aim for 20 to 30 grams across three to four meals
      (0.4 to 0.5 grams per kilogram of body weight per meal).
    - **Post-Workout:** Consume 20 to 40 grams of protein within two hours
      of training.
    - **Pre-Sleep:** A serving of 20 to 30 grams of casein can support
      overnight muscle protein synthesis.

    **Performance Timing:**
    - **Pre-Workout:** Consume moderate carbohydrates and protein one to two
      hours prior to your workout.
    - **Post-Workout:** Refuel with protein and carbohydrates within two
      hours of finishing your workout.
    """)

with tab4:
    st.subheader("🔬 A Scientific Foundation and Nutrition Deep Dive")
    st.markdown(r"""
    ### **The Energy Foundation: BMR and TDEE**

    **Basal Metabolic Rate (BMR):** This is your body's energy needs at
    complete rest, calculated using the **Mifflin-St Jeor equation**, which
    is the most accurate formula recognized by the Academy of Nutrition and
    Dietetics.

    **Total Daily Energy Expenditure (TDEE):** These are your maintenance
    calories, including all daily activities, calculated by multiplying your
    BMR by scientifically validated activity factors.

    ### **A Satiety Hierarchy for Better Adherence**
    1. **Protein** (provides the highest satiety per calorie).
    2. **Fiber-Rich Carbohydrates** (vegetables, fruits, and whole grains).
    3. **Healthy Fats** (nuts, avocado, and olive oil).
    4. **Processed Foods** (provide the lowest satiety per calorie).

    **Fiber Target:** Aim for 14 grams per 1,000 kcal (approximately 25 to
    38 grams daily). It is best to increase your intake gradually to avoid
    gastrointestinal distress.

    **A Volume Eating Strategy:** Prioritize low-calorie, high-volume foods
    like leafy greens, cucumbers, and berries to create meal satisfaction
    without exceeding your calorie targets.

    ### **Micronutrient Considerations**
    **Common Shortfalls in Plant-Forward Diets:**
    - These can include **B₁₂, iron, calcium, zinc, iodine, and omega-3
      (EPA/DHA)**.
    - **Strategy:** Include fortified foods in your diet or consider
      targeted supplementation based on your lab work.
    """)

# ---------------------------------------------------------------------------
# Cell 11: Personalized Recommendations System
# ---------------------------------------------------------------------------

if user_has_entered_info:
    st.header("🎯 Your Personalized Action Plan")
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
    recommendations = generate_personalized_recommendations(
        totals, targets, final_values
    )
    for rec in recommendations:
        st.info(rec)

# ---------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# ---------------------------------------------------------------------------

st.header("Daily Food Selection and Tracking 🥗")
st.markdown(
    "Select the number of servings for each food item to track your daily "
    "nutrition intake."
)

with st.expander("💡 View the Food Emoji Guide"):
    st.markdown("""
    **A Food Emoji Guide:**

    • 🥇 **Gold Medal:** A top performer in both calories AND its primary
      nutrient.
    • 🔥 **High Calorie:** Among the most calorie-dense foods in its category.
    • 💪 **High Protein:** A top source of protein.
    • 🍚 **High Carb:** A top source of carbohydrates.
    • 🥑 **High Fat:** A top source of healthy fats.

    *Foods are ranked within each category to help you make efficient
    choices that align with your goals.*
    """)

if st.button("🔄 Reset All Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.rerun()

# ------ Food Selection with Tabs ------
available_categories = [
    cat for cat, items in sorted(foods.items()) if items
]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    sorted_items_in_category = sorted(
        items,
        key=lambda x: (
            CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']
        )
    )
    with tabs[i]:
        render_food_grid(sorted_items_in_category, category, columns=2)

# ---------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking
# ---------------------------------------------------------------------------

st.header("A Daily Nutrition Summary 📊")
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)

if selected_foods:
    recommendations = create_progress_tracking(totals, targets, foods)
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Today's Nutrition Intake")
        summary_metrics = [
            ("Calories Consumed", f"{totals['calories']:.0f} kcal"),
            ("Protein Intake", f"{totals['protein']:.0f} g"),
            ("Carbohydrates", f"{totals['carbs']:.0f} g"),
            ("Fat Intake", f"{totals['fat']:.0f} g")
        ]
        display_metrics_grid(summary_metrics, 2)

    with col2:
        st.subheader("Macronutrient Split (in grams)")
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

    with st.expander("📝 A Detailed Food Breakdown"):
        st.subheader("Foods Selected Today")
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings

            st.write(f"**{food['name']}** - {servings} serving(s)")
            st.write(
                f"  → {total_cals:.0f} kcal | {total_protein:.1f}g protein | "
                f"{total_carbs:.1f}g carbs | {total_fat:.1f}g fat"
            )
else:
    st.info(
        "No foods have been selected yet. Please choose foods from the "
        "categories above to track your daily intake."
    )
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']]
        st.progress(
            0.0,
            text=(
                f"{config['label']}: 0% of daily target ({target:.0f} "
                f"{config['unit']})"
            )
        )

# ---------------------------------------------------------------------------
# Cell 14: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()
st.markdown("""
### **📚 Evidence-Based References and Methodology**

This nutrition tracker is built on peer-reviewed research and evidence-based
guidelines from the following sources:

- **BMR Calculation:** The Mifflin-St Jeor equation, which is recommended by
  the Academy of Nutrition and Dietetics.
- **Activity Factors:** Based on validated TDEE multipliers from exercise
  physiology research.
- **Protein Targets:** Derived from the International Society of Sports
  Nutrition position stands.
- **Caloric Adjustments:** Conservative and sustainable rates based on body
  composition research.

### **⚠️ Important Disclaimers**

- This tool provides general nutrition guidance that is based on population
  averages.
- Individual needs may vary based on genetics, medical conditions, and other
  factors.
- Please consult with a qualified healthcare provider before making any
  significant dietary changes.
- It is important to monitor your biofeedback, such as your energy,
  performance, and health markers, and adjust your plan as needed.
""")

st.success(
    "You've reached the end! Thanks for using the tracker. Remember, "
    "consistency is your superpower on this journey. Keep fueling your "
    "goals! 🥳"
)

# ---------------------------------------------------------------------------
# Cell 15: Session State Management and Performance
# ---------------------------------------------------------------------------

# ------ Clean Up Session State to Prevent Memory Issues ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }
