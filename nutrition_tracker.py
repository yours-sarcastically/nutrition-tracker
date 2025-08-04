# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for healthy weight gain using vegetarian food sources. It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). A caloric surplus is added to support lean bulking. Macronutrient targets follow current nutritional guidelines, with protein and fat set relative to body weight and total calories, and carbohydrates filling the remainder.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import math

# -----------------------------------------------------------------------------
# Cell 2: Page Configuration and Initial Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Personalized Nutrition Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cell 3: Default Parameter Values and Constants
# -----------------------------------------------------------------------------

# ------ Default Parameter Values Based on Published Research ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 57.5,
    'sex': "Male",
    'activity_level': "moderately_active",
    'caloric_surplus': 400,
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

# ------ Emoji Hierarchy for Food Ranking ------
EMOJI_ORDER = {'🥇': 0, '💥': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '🥦': 3, '': 4}

# ------ Nutrient Category Mapping (REFACTORED) ------
# Maps category to the nutrient column to sort by
NUTRIENT_SORT_KEY_MAP = {
    'PRIMARY PROTEIN SOURCES': 'protein',
    'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
    'PRIMARY FAT SOURCES': 'fat',
    'PRIMARY MICRONUTRIENT SOURCES': 'protein'  # Sorted by protein as a proxy for nutrient density
}
# Maps category to its key in the top_foods dictionary for ranking
CATEGORY_TO_TOP_FOODS_KEY_MAP = {
    'PRIMARY PROTEIN SOURCES': 'protein',
    'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
    'PRIMARY FAT SOURCES': 'fat',
    'PRIMARY MICRONUTRIENT SOURCES': 'micro'
}

# ------ Form Field Configurations (REFACTORED) ------
FORM_FIELD_CONFIGS = {
    'age': {
        'widget': st.sidebar.number_input, 'label': 'Age (Years)', 'session_key': 'user_age',
        'params': {'min_value': 16, 'max_value': 80, 'placeholder': "Enter your age"}
    },
    'height_cm': {
        'widget': st.sidebar.number_input, 'label': 'Height (Centimeters)', 'session_key': 'user_height',
        'params': {'min_value': 140, 'max_value': 220, 'placeholder': "Enter your height"}
    },
    'weight_kg': {
        'widget': st.sidebar.number_input, 'label': 'Weight (kg)', 'session_key': 'user_weight',
        'params': {'min_value': 40.0, 'max_value': 150.0, 'step': 0.5, 'placeholder': "Enter your weight"}
    },
    'sex': {
        'widget': st.sidebar.selectbox, 'label': 'Sex', 'session_key': 'user_sex',
        'params': {'options': ["Select Sex", "Male", "Female"]}
    },
    'activity_level': {
        'widget': st.sidebar.selectbox, 'label': 'Activity Level', 'session_key': 'user_activity',
        'params': {
            'options': [
                ("Select Activity Level", None), ("Sedentary", "sedentary"), ("Lightly Active", "lightly_active"),
                ("Moderately Active", "moderately_active"), ("Very Active", "very_active"), ("Extremely Active", "extremely_active")
            ],
            'format_func': lambda x: x[0]
        }
    }
}

# ------ Nutrient Display Configurations ------
NUTRIENT_CONFIGS = {
    'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories'},
    'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'},
    'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'},
    'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
}

# -----------------------------------------------------------------------------
# Cell 4: Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables"""
    if 'food_selections' not in st.session_state:
        st.session_state.food_selections = {}
    
    for field_config in FORM_FIELD_CONFIGS.values():
        if field_config['session_key'] not in st.session_state:
            st.session_state[field_config['session_key']] = None

def get_final_value(user_value, default_key, special_check=None):
    """Get final value using user input or default"""
    if special_check:
        return user_value if special_check(user_value) else DEFAULTS[default_key]
    return user_value if user_value is not None else DEFAULTS[default_key]

def display_metrics_grid(metrics_data, num_columns=4):
    """Display metrics in a configurable column layout"""
    columns = st.columns(num_columns)
    
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                label, value = metric_info
                st.metric(label, value)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                st.metric(label, value, delta)

def create_progress_tracking(totals, targets):
    """Create unified progress tracking with bars and recommendations"""
    recommendations = []
    
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    
    for nutrient, config in NUTRIENT_CONFIGS.items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        
        percent = min(actual / target * 100, 100) if target > 0 else 0
        st.progress(
            percent / 100,
            text=f"{config['label']}: {percent:.0f} percent of daily target ({target:.0f} {config['unit']})"
        )
        
        if actual < target:
            deficit = target - actual
            purpose_map = {
                'calories': 'to reach your weight gain target',
                'protein': 'for muscle building',
                'carbs': 'for energy and performance',
                'fat': 'for hormone production'
            }
            purpose = purpose_map.get(nutrient, 'for optimal nutrition')
            recommendations.append(f"• You need {deficit:.0f} more {config['unit']} of {config['label'].lower()} {purpose}")
    
    return recommendations

def render_food_grid(items, category, columns=2):
    """Render food items in a grid layout"""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)

def calculate_daily_totals(food_selections, foods):
    """Calculate total daily nutrition from food selections"""
    totals = {nutrient: 0 for nutrient in NUTRIENT_CONFIGS.keys()}
    selected_foods = []
    
    for category, items in foods.items():
        for food in items:
            servings = food_selections.get(food['name'], 0)
            if servings > 0:
                for nutrient in totals:
                    totals[nutrient] += food[nutrient] * servings
                selected_foods.append({'food': food, 'servings': servings})
    
    return totals, selected_foods

# -----------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation"""
    if sex.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return bmr

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure Based on Activity Level"""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_personalized_targets(
    age, height_cm, weight_kg, sex='male',
    activity_level='moderately_active',
    caloric_surplus=400, protein_per_kg=2.0, fat_percentage=0.25
):
    """Calculate Personalized Daily Nutritional Targets"""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    total_calories = tdee + caloric_surplus
    protein_g = protein_per_kg * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    return {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'target_weight_gain_per_week': round(weight_kg * 0.0025, 2)
    }

# -----------------------------------------------------------------------------
# Cell 6: Load and Process Food Database
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """Load the Vegetarian Food Database From a CSV File"""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in NUTRIENT_SORT_KEY_MAP.keys()}
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
    """Assign an Emoji to Each Food Item Based on Nutritional Hierarchy (REFACTORED)"""
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}

    # Identify top nutrient and calorie contributors in each category
    for category, items in foods.items():
        if not items:
            continue
        
        # Rank by calories
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]

        # Rank by primary nutrient using the configuration maps
        sort_key = NUTRIENT_SORT_KEY_MAP.get(category)
        top_foods_key = CATEGORY_TO_TOP_FOODS_KEY_MAP.get(category)
        if sort_key and top_foods_key:
            sorted_by_nutrient = sorted(items, key=lambda x: x.get(sort_key, 0), reverse=True)
            top_foods[top_foods_key] = [food['name'] for food in sorted_by_nutrient[:3]]

    # Identify superfoods (high rank in multiple nutrient categories)
    food_rank_counts = {}
    all_top_nutrient_foods = set(top_foods['protein'] + top_foods['carbs'] + top_foods['fat'] + top_foods['micro'])
    for food_name in all_top_nutrient_foods:
        count = sum(1 for key in ['protein', 'carbs', 'fat', 'micro'] if food_name in top_foods[key])
        food_rank_counts[food_name] = count
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    # Apply emojis based on the specified hierarchy
    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            food['emoji'] = ''
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])

            if food_name in superfoods: food['emoji'] = '🥇'
            elif is_high_calorie and is_top_nutrient: food['emoji'] = '💥'
            elif is_high_calorie: food['emoji'] = '🔥'
            elif food_name in top_foods['protein']: food['emoji'] = '💪'
            elif food_name in top_foods['carbs']: food['emoji'] = '🍚'
            elif food_name in top_foods['fat']: food['emoji'] = '🥑'
            elif food_name in top_foods['micro']: food['emoji'] = '🥦'
    return foods

def render_food_item(food, category):
    """Render a single food item with buttons and input controls"""
    st.subheader(f"{food.get('emoji', '')} {food['name']}")
    key = f"{category}_{food['name']}"
    current_serving = st.session_state.food_selections.get(food['name'], 0.0)
    
    button_cols = st.columns(5)
    for k in range(1, 6):
        with button_cols[k - 1]:
            button_type = "primary" if current_serving == float(k) else "secondary"
            if st.button(f"{k} Servings", key=f"{key}_{k}", type=button_type):
                st.session_state.food_selections[food['name']] = float(k)
                st.rerun()
    
    custom_serving = st.number_input(
        "Custom Number of Servings:", min_value=0.0, max_value=10.0,
        value=float(current_serving), step=0.1, key=f"{key}_custom"
    )
    if custom_serving != current_serving:
        if custom_serving > 0:
            st.session_state.food_selections[food['name']] = custom_serving
        elif food['name'] in st.session_state.food_selections:
            del st.session_state.food_selections[food['name']]
        st.rerun()
    
    st.caption(f"Per Serving: {food['calories']} kcal | {food['protein']} g protein | {food['carbs']} g carbs | {food['fat']} g fat")

# -----------------------------------------------------------------------------
# Cell 7: Initialize Application
# -----------------------------------------------------------------------------

initialize_session_state()
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

st.markdown("""<style>[data-testid="InputInstructions"] {display: none;}</style>""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 8: Application Title and Sidebar Parameters (REFACTORED)
# -----------------------------------------------------------------------------

st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown("Ready to turbocharge your health game? This awesome tool dishes out daily nutrition goals made just for you and makes tracking meals as easy as pie. Let's get those macros on your team! 🚀")

st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

# ------ Dynamically generate sidebar inputs and manage state ------
user_inputs = {}
for key, config in FORM_FIELD_CONFIGS.items():
    current_value = st.session_state.get(config['session_key'])
    widget_params = config['params'].copy()
    
    # Handle selectbox index logic
    if config['widget'] == st.sidebar.selectbox and current_value:
        try:
            options = widget_params['options']
            # Handle options that are (label, value) tuples
            values = [opt[1] for opt in options] if isinstance(options[0], tuple) else options
            widget_params['index'] = values.index(current_value)
        except (ValueError, IndexError):
            widget_params['index'] = 0

    # Create the widget
    input_value = config['widget'](config['label'], value=current_value, **widget_params)
    
    # For selectboxes with (label, value) tuples, get the actual value
    final_value = input_value[1] if isinstance(input_value, tuple) else input_value
    
    # Update session state and store the value
    st.session_state[config['session_key']] = final_value
    user_inputs[key] = final_value

# Unpack values for use in the script
age, height_cm, weight_kg, sex, activity_level = (
    user_inputs['age'], user_inputs['height_cm'], user_inputs['weight_kg'],
    user_inputs['sex'], user_inputs['activity_level']
)

# ------ Advanced Parameters Collapsible Section ------
with st.sidebar.expander("Advanced Settings ⚙️"):
    caloric_surplus = st.number_input(
        "Caloric Surplus (kcal Per Day)", min_value=200, max_value=800, value=None,
        placeholder=f"Default: {DEFAULTS['caloric_surplus']}", step=50, help="Additional calories above maintenance for weight gain"
    )
    protein_per_kg = st.number_input(
        "Protein (g Per Kilogram Body Weight)", min_value=1.2, max_value=3.0, value=None,
        placeholder=f"Default: {DEFAULTS['protein_per_kg']}", step=0.1, help="Protein intake per kilogram of body weight"
    )
    fat_percentage_input = st.number_input(
        "Fat (Percent of Total Calories)", min_value=15, max_value=40, value=None,
        placeholder=f"Default: {int(DEFAULTS['fat_percentage'] * 100)}", step=1, help="Percentage of total calories from fat"
    )

# ------ Use Default Values If User Has Not Entered Custom Values ------
final_values = {
    'age': get_final_value(age, 'age'),
    'height_cm': get_final_value(height_cm, 'height_cm'),
    'weight_kg': get_final_value(weight_kg, 'weight_kg'),
    'sex': get_final_value(sex, 'sex', lambda x: x != "Select Sex"),
    'activity_level': get_final_value(activity_level, 'activity_level'),
    'caloric_surplus': get_final_value(caloric_surplus, 'caloric_surplus'),
    'protein_per_kg': get_final_value(protein_per_kg, 'protein_per_kg'),
    'fat_percentage': get_final_value(fat_percentage_input / 100 if fat_percentage_input else None, 'fat_percentage')
}

user_has_entered_info = all([age, height_cm, weight_kg, sex != "Select Sex", activity_level])

targets = calculate_personalized_targets(
    age=final_values['age'], height_cm=final_values['height_cm'], weight_kg=final_values['weight_kg'],
    sex=final_values['sex'].lower(), activity_level=final_values['activity_level'],
    caloric_surplus=final_values['caloric_surplus'], protein_per_kg=final_values['protein_per_kg'],
    fat_percentage=final_values['fat_percentage']
)

# -----------------------------------------------------------------------------
# Cell 9: Display Personalized Targets and Daily Goals
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations")
else:
    st.header("Your Personalized Daily Nutritional Targets for Healthy Weight Gain 🎯")

metabolic_metrics = [
    ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal per day"),
    ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal per day"),
    ("Estimated Weekly Weight Gain", f"{targets['target_weight_gain_per_week']} kg per week")
]
display_metrics_grid(metabolic_metrics, 3)

st.subheader("Daily Nutritional Target Breakdown")
target_metrics = [
    ("Daily Calorie Target", f"{targets['total_calories']} kcal"),
    ("Protein Target", f"{targets['protein_g']} g"),
    ("Carbohydrate Target", f"{targets['carb_g']} g"),
    ("Fat Target", f"{targets['fat_g']} g")
]
display_metrics_grid(target_metrics, 4)

st.subheader("Macronutrient Distribution as Percent of Daily Calories")
protein_percent = (targets['protein_calories'] / targets['total_calories']) * 100
carb_percent = (targets['carb_calories'] / targets['total_calories']) * 100
fat_percent_display = (targets['fat_calories'] / targets['total_calories']) * 100

percentage_metrics = [
    ("Protein Contribution", f"{protein_percent:.1f} percent", f"+ {targets['protein_calories']} kcal"),
    ("Carbohydrate Contribution", f"{carb_percent:.1f} percent", f"+ {targets['carb_calories']} kcal"),
    ("Fat Contribution", f"{fat_percent_display:.1f} percent", f"+ {targets['fat_calories']} kcal")
]
display_metrics_grid(percentage_metrics, 3)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item")

available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    sorted_items = sorted(items, key=lambda x: (EMOJI_ORDER.get(x.get('emoji', ''), 4), -x['calories']))
    with tabs[i]:
        render_food_grid(sorted_items, category, 2)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 11: Calculation Button and Nutritional Results Display
# -----------------------------------------------------------------------------

if st.button("Calculate Daily Intake", type="primary", use_container_width=True):
    totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)
    
    st.header("Summary of Daily Nutritional Intake 📊")

    if selected_foods:
        st.subheader("Foods Logged for Today 🥣")
        cols = st.columns(3)
        for i, item in enumerate(selected_foods):
            with cols[i % 3]:
                st.write(f"• {item['food'].get('emoji', '')} {item['food']['name']} × {item['servings']:.1f}")
    else:
        st.info("No foods have been selected for today 🍽️")

    st.subheader("Total Nutritional Intake for the Day 📈")
    intake_metrics = [
        ("Total Calories Consumed", f"{totals['calories']:.0f} kcal"),
        ("Total Protein Consumed", f"{totals['protein']:.1f} g"),
        ("Total Carbohydrates Consumed", f"{totals['carbs']:.1f} g"),
        ("Total Fat Consumed", f"{totals['fat']:.1f} g")
    ]
    display_metrics_grid(intake_metrics, 4)

    recommendations = create_progress_tracking(totals, targets)

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    # ------ Show Surplus and Deficit Information ------
    st.subheader("Daily Caloric Balance and Weight Gain Summary ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    if cal_balance > 0:
        st.info(
            f"📈 You are consuming {cal_balance:.0f} kcal above maintenance, supporting weight gain"
        )
    else:
        st.warning(
            f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance"
        )

    if selected_foods:
        st.subheader("Detailed Food Log for Today 📋")
        food_log = [
            {
                'Food Item Name': f"{item['food'].get('emoji', '')} {item['food']['name']}",
                'Number of Servings Consumed': f"{item['servings']:.1f}",
                'Total Calories Consumed': item['food']['calories'] * item['servings'],
                'Total Protein Consumed (g)': item['food']['protein'] * item['servings'],
                'Total Carbohydrates Consumed (g)': item['food']['carbs'] * item['servings'],
                'Total Fat Consumed (g)': item['food']['fat'] * item['servings']
            }
            for item in selected_foods
        ]
        df_log = pd.DataFrame(food_log)
        st.dataframe(
            df_log.style.format({
                'Total Calories Consumed': '{:.0f}',
                'Total Protein Consumed (g)': '{:.1f}',
                'Total Carbohydrates Consumed (g)': '{:.1f}',
                'Total Fat Consumed (g)': '{:.1f}'
            }),
            use_container_width=True
        )

    st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 12: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# Cell 13: Footer Information and Application Documentation
# -----------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Activity Level Guide for Accurate TDEE 🏃‍♂️")
st.sidebar.markdown("""
- Sedentary: Little to no exercise or desk job
- Lightly Active: Light exercise or sports one to three days per week
- Moderately Active: Moderate exercise or sports three to five days per week
- Very Active: Hard exercise or sports six to seven days per week
- Extremely Active: Very hard exercise, physical job, or training twice daily
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Emoji Guide for Food Ranking 💡")
st.sidebar.markdown("""
- 🥇 **Superfood**: Excels across multiple nutrient categories
- 💥 **Nutrient and Calorie Dense**: High in both calories and its primary nutrient
- 🔥 **High-Calorie**: Among the most energy-dense options in its group
- 💪 **Top Protein Source**: A leading contributor of protein
- 🍚 **Top Carb Source**: A leading contributor of carbohydrates
- 🥑 **Top Fat Source**: A leading contributor of healthy fats
- 🥦 **Top Micronutrient Source**: Rich in vitamins and minerals
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Nutrition Calculator 📖")
st.sidebar.markdown("""
Calculations use the following methods:
- Basal Metabolic Rate (BMR): Mifflin-St Jeor equation
- Protein: 2.0 g per kilogram of body weight for muscle building
- Fat: 25 percent of total calories for hormone production
- Carbohydrates: Remaining calories after protein and fat allocation
- Weight gain target: 0.25 percent of body weight per week for lean gains
""")
