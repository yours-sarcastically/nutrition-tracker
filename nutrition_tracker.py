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
    page_title="Your Personal Nutrition Coach",
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

# ------ Unified Configuration for All App Components ------
CONFIG = {
    'emoji_order': {'🥇': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
    },
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Total Calories',
                     'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein',
                    'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates',
                  'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    'form_fields': {
        'age': {'type': 'number', 'label': 'Age (in years)',
                'min': 16, 'max': 80, 'step': 1, 'required': True,
                'help': 'Another year wiser! How many trips around the sun have you taken?'},
        'height_cm': {'type': 'number', 'label': 'Height (in centimeters)',
                      'min': 140, 'max': 220, 'step': 1, 'required': True,
                      'help': 'Stand tall and tell us your height'},
        'weight_kg': {'type': 'number', 'label': 'Weight (in kilograms)',
                      'min': 40.0, 'max': 150.0, 'step': 0.5, 'required': True,
                      'help': 'What does the scale say today?'},
        'sex': {'type': 'selectbox', 'label': 'Sex',
                'options': ["Male", "Female"], 'required': True,
                'help': 'Please select your biological sex:'},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level',
                           'options': [
                               ("Sedentary", "sedentary"),
                               ("Lightly Active", "lightly_active"),
                               ("Moderately Active", "moderately_active"),
                               ("Very Active", "very_active"),
                               ("Extremely Active", "extremely_active")
                           ], 'required': True,
                           'help': 'Pick what sounds most like your typical week'},
        'goal': {'type': 'selectbox', 'label': 'Your Goal',
                 'options': [
                     ("Weight Loss", "weight_loss"),
                     ("Weight Maintenance", "weight_maintenance"),
                     ("Weight Gain", "weight_gain")
                 ], 'required': True,
                 'help': 'What are we working toward?'},
        'protein_per_kg': {'type': 'number', 'label': 'Protein Goal',
                           'min': 1.2, 'max': 3.0, 'step': 0.1,
                           'help': 'Define your daily protein target in grams per kilogram of body weight',
                           'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number',
                           'label': 'Fat Intake (in percentage of total calories)',
                           'min': 15, 'max': 40, 'step': 1,
                           'help': 'Set the share of your daily calories that should come from healthy fats',
                           'convert': lambda x: x / 100 if x is not None else None,
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

    if field_config.get('help'):
        container.markdown(f"**{field_config['label']}**")
        container.markdown(f"*{field_config['help']}*")
        label_for_widget = " "
    else:
        label_for_widget = field_config['label']

    if field_config['type'] == 'number':
        default_val = DEFAULTS.get(field_name)
        display_val = None
        if default_val is not None:
            display_val = (
                int(default_val * 100)
                if field_name == 'fat_percentage'
                else default_val
            )
        placeholder = f"Default: {display_val}" if field_config.get('advanced') else None

        value = container.number_input(
            label_for_widget,
            min_value=field_config['min'],
            max_value=field_config['max'],
            value=st.session_state.get(session_key),
            step=field_config['step'],
            placeholder=placeholder,
            label_visibility="collapsed" if field_config.get('help') else "visible"
        )

    elif field_config['type'] == 'selectbox':
        current_value = st.session_state.get(session_key)
        options = field_config['options']

        if isinstance(options[0], tuple): # For 'activity_level' and 'goal'
            index = next((i for i, (_, val) in enumerate(options) if val == current_value), 0)
            selection = container.selectbox(
                label_for_widget, options, index=index, format_func=lambda x: x[0],
                label_visibility="collapsed" if field_config.get('help') else "visible"
            )
            value = selection[1]
        else: # For 'sex'
            index = options.index(current_value) if current_value in options else 0
            value = container.selectbox(
                label_for_widget, options, index=index,
                label_visibility="collapsed" if field_config.get('help') else "visible"
            )

    st.session_state[session_key] = value
    return value


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


def calculate_hydration_needs(weight_kg):
    """Calculates daily fluid needs based on body weight."""
    return round(weight_kg * 35) # Baseline is 35 milliliters per kilogram


def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics in a configurable column layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                st.metric(metric_info[0], metric_info[1])
            elif len(metric_info) == 3:
                st.metric(metric_info[0], metric_info[1], metric_info[2])


def find_best_food_for_nutrient(nutrient, foods):
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
            f"Looking for a suggestion? Adding just **{suggestion_servings} "
            f"serving of {best_food['name']}** will give you a solid "
            f"{best_food[nutrient]:.0f} grams of {nutrient}."
        )
    return None


def create_progress_tracking(totals, targets, foods):
    """Creates progress bars and recommendations for nutritional targets."""
    st.subheader("Your Daily Dashboard 🎯")

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
        deficit = target - actual

        progress_text = (
            f"{config['label']}: {percent:.0f}% of your daily target "
            f"({target:.0f} {config['unit']})"
        )
        st.progress(percent / 100, text=progress_text)

        with st.container():
            status_message = ""
            if actual >= target:
                status_message = f"🎉 **You've hit your goal!**"
            elif actual / target >= 0.8:
                status_message = f"👍 **You're on track!**"
            else:
                purpose = purpose_map.get(nutrient, 'for optimal nutrition')
                status_message = (
                    f"You've got **{deficit:.0f} more {config['unit']}** of "
                    f"{config['label'].lower()} to go {purpose}."
                )

            st.info(status_message)

            if actual < target and nutrient in ['protein', 'carbs', 'fat']:
                food_suggestion = find_best_food_for_nutrient(nutrient, foods)
                if food_suggestion:
                    st.info(food_suggestion)


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
        protein_per_kg if protein_per_kg is not None else goal_config['protein_per_kg']
    )
    fat_percentage_final = (
        fat_percentage if fat_percentage is not None else goal_config['fat_percentage']
    )

    protein_g = protein_per_kg_final * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage_final
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'goal': goal
    }

    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
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
        if not items: continue
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(items, key=lambda x: x[map_info['sort_by']], reverse=True)
            top_foods[map_info['key']] = [food['name'] for food in sorted_by_nutrient[:3]]

    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]}
    emoji_mapping = {
        'high_cal_nutrient': '🥇', 'high_calorie': '🔥',
        'protein': '💪', 'carbs': '🍚', 'fat': '🥑'
    }

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


def render_food_item(food, category):
    """Renders a single food item with its interaction controls."""
    with st.container(border=True):
        st.markdown(f"**{food.get('emoji', '')} {food['name']}**")
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("Serving options: ", help="Select a preset number of servings.")
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k-1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key}_{k}", type=button_type, help=f"Set to {k} servings", use_container_width=True):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()
        with col2:
            st.markdown("Custom", help="Enter a custom number of servings.")
            custom_serving = st.number_input("Custom", min_value=0.0, max_value=10.0,
                                            value=float(current_serving), step=0.1,
                                            key=f"{key}_custom", label_visibility="collapsed")
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

initialize_session_state()
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

st.markdown("""<style>
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {font-size: 0.95rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {font-size: 1.5rem; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {font-size: 1.2rem; }
[data-testid="stSidebar"] .stNumberInput input { text-align: center; }
.stButton>button[kind="primary"] { background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b; }
.stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# ---------------------------------------------------------------------------

st.title("Your Personal Nutrition Coach 🍽️")
st.header("A Smart, Evidence-Based Nutrition Tracker That Actually Gets You")
st.markdown("""
Welcome aboard!

Hey there! Welcome to your new nutrition buddy. This isn’t just another calorie counter—it’s your personalized guide, built on rock-solid science to help you smash your goals. Whether you’re aiming to shed a few pounds, hold steady, or bulk up, we’ve crunched the numbers so you can focus on enjoying your food.

Let’s get rolling—your journey to feeling awesome starts now! 🚀
""")
st.divider()

st.sidebar.header("Let’s Get Personal 📊")
all_inputs = {}
standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config: value = field_config['convert'](value)
    all_inputs[field_name] = value

advanced_expander = st.sidebar.expander("Fine-Tune Your Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
    if 'convert' in field_config: value = field_config['convert'](value)
    all_inputs[field_name] = value

with st.sidebar.expander("Your Activity Level Decoded"):
    st.markdown("""
    Here's a quick breakdown of what these levels really mean:

    🧑‍💻 **Sedentary:** You're basically married to your desk chair

    🏃 **Lightly Active:** You squeeze in walks or workouts one to three times a week

    🚴 **Moderately Active:** You're sweating it out three to five days a week

    🏋️ **Very Active:** You might actually be part treadmill

    🤸 **Extremely Active:** You live in the gym and sweat is your second skin

    ---

    *💡 Pro tip: If you’re torn between two levels, pick the lower one. It’s better to underestimate your burn than to overeat and stall.*
    """)

final_values = get_final_values(all_inputs)
required_fields = [f for f, c in CONFIG['form_fields'].items() if c.get('required')]
user_has_entered_info = all(all_inputs.get(f) is not None for f in required_fields)
targets = calculate_personalized_targets(**final_values)
hydration_ml = calculate_hydration_needs(final_values['weight_kg'])

# ---------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# ---------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Pop your details into the sidebar to get your personalized daily targets.")
else:
    goal_labels = {
        'weight_loss': 'Weight Loss',
        'weight_maintenance': 'Weight Maintenance',
        'weight_gain': 'Weight Gain'
    }
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Custom Daily Nutrition Roadmap for {goal_label} 🎯")
    st.info(f"""
    💧 **Your Estimated Daily Hydration Goal:** {hydration_ml} ml. That's roughly {hydration_ml/250:.1f} cups of water throughout the day.

    🎯 **The 80/20 Rule:** Try to hit your targets about 80% of the time. This gives you wiggle room for birthday cake, date nights, and those inevitable moments when life throws you a curveball. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap.
    """)

st.subheader("Your Daily Nutrition Targets")
metrics_data = [
    ("Total Calories", f"{targets['total_calories']} kcal"),
    ("Protein", f"{targets['protein_g']} g", f"{targets['protein_percent']:.0f}% of your calories"),
    ("Carbohydrates", f"{targets['carb_g']} g", f"{targets['carb_percent']:.0f}% of your calories"),
    ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}% of your calories"),
    ("Water", f"{hydration_ml} ml", f"~{hydration_ml/250:.1f} cups")
]
display_metrics_grid(metrics_data, num_columns=5)
st.divider()

# ---------------------------------------------------------------------------
# Cell 10: Enhanced Evidence-Based Tips and Context
# ---------------------------------------------------------------------------

st.header("Your Evidence-Based Game Plan 📚")
tab1, tab2, tab3, tab4 = st.tabs([
    "The Big Three 🏆", "Progress & Mindset 🧠",
    "Troubleshooting & Timing 🔄", "The Science 🔬"
])

with tab1:
    st.subheader("The Big Three to Win At Nutrition 🏆")
    st.markdown("""
    ### 💧 Master Your Hydration Game:
    - **Daily Goal:** Shot for about 35 ml per kilogram of your body weight daily. 
    - **Training Bonus:** Tack on an extra 500-750 ml per hour of sweat time.
    - **Fat Loss Hack:** Chugging 500 ml of water before meals can boost fullness by by 13%. Your stomach will thank you, and so will your waistline.

    ### 😴 Sleep Like Your Goals Depend on It:
    - **The Shocking Truth:** Getting less than 7 hours of sleep can torpedo your fat loss by a more than half.
    - **Daily Goal:** Shoot for 7-9 hours and try to keep a consistent schedule.
    - **Set the Scene:** Keep your cave dark, cool (18-20°C), and screen-free for at least an hour before lights out.

    ### 📅 Follow Your Wins:
    - **Morning Ritual:** Weigh yourself first thing after using the bathroom, before eating or drinking, in minimal clothing.
    - **Look for Trends, Not Blips:** Watch your weekly average instead of getting hung up on daily fluctuations. Your weight can swing 2-3 pounds daily. 
    - **Hold the Line:** Don’t tweak your plan too soon. Wait for two or more weeks of stalled progress before making changes.
    """)

with tab2:
    st.subheader("Level Up Your Progress Tracking 📊")
    st.markdown("""
    ### Go Beyond the Scale 📸
    - **The Bigger Picture:** Snap a few pics every month. Use the same pose, lighting, and time of day. The mirror doesn't lie.
    - **Size Up Your Wins:** Measure your waist, hips, arms, and thighs monthly.
    - **The Quiet Victories:** Pay attention to how you feel. Your energy levels, sleep quality, gym performance, and hunger patterns tell a story numbers can’t.

    ### Mindset Is Everything 🧠
    The 80/20 principle is your best defense against the perfectionist trap. It's about ditching that mindset that makes you throw in the towel after one "bad" meal. Instead of trying to master everything at once, build your habits gradually and you’ll be far more likely to stick with them for the long haul.

    #### Start Small, Win Big:
    - **Weeks 1–2:** Your only job is to focus on hitting your calorie targets. Don’t worry about anything else!
    - **Weeks 3–4:** Once calories feel like second nature, start layering in protein tracking.
    - **Week 5 and Beyond:** With calories and protein in the bag, you can now fine-tune your carb and fat intake.
    """)

with tab3:
    st.subheader("When Progress Stalls 🔄")
    st.markdown("""
    ### Hit a Weight Loss Plateau?
    - **Guess Less, Stress Less:** Before you do anything else, double-check how accurately you’re logging your food. Little things can add up!
    - **Activity Audit:** Take a fresh look at your activity level. Has it shifted?
    - **Walk it Off:** Try adding 10-15 minutes of walking to your daily routine before cutting calories further. It’s a simple way to boost progress without tightening the belt just yet.
    - **Step Back to Leap Forward:** Consider a "diet break" every 6-8 weeks. Eating at your maintenance calories for a week or two can give your metabolism and your mind a well-deserved reset.
    - **Leaf Your Hunger Behind:** Load your plate with low-calorie, high-volume foods like leafy greens, cucumbers, and berries. They’re light on calories but big on satisfaction.

    ### Struggling to Gain Weight?
    - **Drink Your Calories:** Liquid calories from smoothies, milk, and protein shakes go down way easier than another full meal.
    - **Fat is Fuel:** Load up healthy fats like nuts, oils, and avocados. 
    - **Push Your Limits:** Give your body a reason to grow! Make sure you’re consistently challenging yourself in the gym.
    - **Turn Up the Heat:** If you've been stuck for over two weeks, bump up your intake by 100-150 calories to get the ball rolling again.

    ### Pace Your Protein
    - **Spread the Love:** Instead of cramming your protein into one or two giant meals, aim for 20-40 grams with each of your 3-4 daily meals. This works out to roughly 0.4-0.5 grams per kilogram of body weight per meal.
    - **Frame Your Fitness:** Get some carbs and 20–40g protein before and within two hours of wrapping up your workout.
    - **The Night Shift:** Try 20-30g of casein protein before bed for keeping your muscles fed while you snooze.
    """)

with tab4:
    st.subheader("The Science Behind the Magic 🔬")
    st.markdown("""
    ### Understanding Your Metabolism
    Your **Basal Metabolic Rate (BMR)** is the energy your body needs just to keep the lights on. Your **Total Daily Energy Expenditure (TDEE)** builds on that baseline by factoring in how active you are throughout the day.

    ### The Smart Eater's Cheat Sheet
    Not all calories are created equal. Some foods fill you up, while others leave you rummaging through the pantry an hour later. Here’s the pecking order:
    - **Protein:** Protein is the undisputed king of fullness! It digests slowly, steadies blood sugar, and even burns a few extra calories in the process. Eggs, Greek yogurt, chicken, tofu, and lentils are all your hunger-busting best friends.
    - **Fiber-Rich Carbohydrates:** Veggies, fruits, and whole grains are the unsung heroes of fullness. They fill you up, slow things down, and bulk up meals without blowing your calorie budget.
    - **Healthy Fats:** Think of nuts, olive oil, and avocados as the smooth operators delivering steady, long-lasting energy that keeps you powered throughout the day.
    - **Processed Stuff:** These foods promise the world but leave you hanging. They're fine for a cameo appearance, but you can't build a winning strategy around them.
    - As a great rule of thumb, aim for **14 grams of fibre for every 1,000 calories** you consume, which usually lands between 25 and 38 grams daily. Ramp up gradually to avoid digestive drama.

    ### Your Nutritional Supporting Cast
    Going plant-based? There are a few tiny but mighty micronutrients to keep an eye on. They may not get top billing, but they’re essential for keeping the show running smoothly.
    - **The Watch List:**
        - **B₁₂:** B₁₂ keeps your cells and nerves firing like a well-oiled machine. It’s almost exclusively found in animal products, so if you’re running a plant-powered show, you’ll need reinforcements. A trusty supplement is often the easiest way to keep your levels topped up and your brain buzzing.
        - **Iron:** Iron is the taxi service that shuttles oxygen all over your body. When it’s running low, you’ll feel like a sloth on a Monday morning. Load up on leafy greens, lentils, and fortified grains, and team them with a hit of vitamin C—think bell peppers or citrus—to supercharge absorption.
        - **Calcium:** This multitasker helps build bones, power muscles, and keeps your heart thumping to a steady beat. While dairy is the classic go-to, you can also get your fix from kale, almonds, tofu, and fortified plant milks.
        - **Zinc:** Think of zinc as your immune system's personal security detail. You’ll find it hanging out in nuts, seeds, and whole grains. Keep your zinc levels up, and you’ll be dodging colds like a ninja.
        - **Iodine:** Your thyroid is the command center for your metabolism, and iodine is its right-hand mineral. A pinch of iodized salt is usually all it takes.
        - **Omega-3s (EPA/DHA):** These healthy fats are premium fuel for your brain, heart, and emotional well-being. If fish isn’t on your plate, fortified foods or supplements can help you stay sharp and serene.
    - The good news? Fortified foods and targeted supplements have your back. Plant milks, cereals, and nutritional yeast are often spiked with B₁₂, calcium, or iodine. Supplements are a safety net, but don’t overdo it. It’s always best to chat with a doctor or dietitian to build a plan that’s right for you.
    """)
st.divider()
# ---------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# ---------------------------------------------------------------------------

st.header("Your Personalized Action Steps 🎯")
st.subheader("Track Your Daily Intake 🥗")
st.markdown("Pick how many servings of each food you’re having to see how your choices stack up against your daily targets.")

with st.expander("💡 Need a hand with food choices? Check out the emoji guide below!"):
    st.markdown("""
    🥇 **Gold Medal:** A nutritional all-star! High in its target nutrient and very calorie-efficient.

    🔥 **High Calorie:** One of the more calorie-dense options in its group.

    💪 **High Protein:** A true protein powerhouse.

    🍚 **High Carb:** A carbohydrate champion.

    🥑 **High Fat:** A healthy fat hero.
    """)

if st.button("🔄 Start Fresh: Reset All Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.rerun()

available_categories = [cat for cat, items in sorted(foods.items()) if items]
tabs = st.tabs(available_categories)
for i, category in enumerate(available_categories):
    items = foods[category]
    sorted_items = sorted(items, key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
    with tabs[i]:
        render_food_grid(sorted_items, category, columns=2)
st.divider()

# ---------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking
# ---------------------------------------------------------------------------

st.header("Today’s Scorecard 📊")
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

if selected_foods:
    create_progress_tracking(totals, targets, foods)
    st.divider()
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
                labels=['Protein', 'Carbs', 'Fat'], values=macro_values, hole=.4,
                marker_colors=['#ff6b6b', '#feca57', '#48dbfb'],
                textinfo='label+percent', insidetextorientation='radial'
            ))
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Select foods to see the macronutrient split.")

    with st.expander("Your Food Choices Today"):
        st.subheader("What You've Logged:")
        for item in selected_foods:
            food, servings = item['food'], item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings
            st.write(f"**{food['name']}** - {servings} serving(s)")
            st.write(f"→ {total_cals:.0f} kcal | {total_protein:.1f}g protein | {total_carbs:.1f}g carbs | {total_fat:.1f}g fat")
else:
    st.info("Haven't picked any foods yet? No worries! Go ahead and add some items from the categories above to start tracking your intake!")
    st.subheader("Your Daily Dashboard 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']]
        text = f"Progress Snapshot: {config['label']}: 0% of daily target ({target:.0f} {config['unit']})"
        st.progress(0.0, text=text)

# ---------------------------------------------------------------------------
# Cell 14: Footer and Additional Resources
# ---------------------------------------------------------------------------

st.divider()
st.header("The Science We Stand On 📚")
st.markdown("""
This tracker isn't built on guesswork—it's grounded in peer-reviewed research and evidence-based guidelines. We rely on the **Mifflin-St Jeor equation** to calculate your Basal Metabolic Rate (BMR). This method is widely regarded as the gold standard and is strongly endorsed by the Academy of Nutrition and Dietetics. To estimate your Total Daily Energy Expenditure (TDEE), we use well-established activity multipliers derived directly from exercise physiology research. For protein recommendations, our targets are based on official guidelines from the International Society of Sports Nutrition.

When it comes to any calorie adjustments, we stick to conservative, sustainable rates that research has consistently shown lead to lasting, meaningful results. We’re all about setting you up for success, one step at a time!
""")
st.header("The Fine Print ⚠️")
st.markdown("""
Think of this tool as your launchpad, but remember—everyone’s different. Your mileage may vary due to factors like genetics, health conditions, medications, and other factors that a calculator simply can't account for. It's always wise to consult a qualified healthcare provider before making any big dietary shifts. Above all, tune into your body—keep tabs on your energy levels, performance,and tweak things as needed. We’re here to help, but you know yourself best!
""")
st.success("You made it to the finish line! Thanks for sticking with us on this nutrition adventure. Remember, the sun doesn’t rush to rise, but it always shows up. Keep shining—you’ve got this! 🥳")

# ---------------------------------------------------------------------------
# Cell 15: Session State Management and Performance
# ---------------------------------------------------------------------------

# Clean up session state to prevent memory issues if the selection grows too large
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {k: v for k, v in st.session_state.food_selections.items() if v > 0}
