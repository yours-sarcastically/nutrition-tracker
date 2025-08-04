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
DEFAULT_AGE = 26
DEFAULT_HEIGHT_CM = 180
DEFAULT_WEIGHT_KG = 57.5
DEFAULT_GENDER = "Male"
DEFAULT_ACTIVITY_LEVEL = "moderately_active"
DEFAULT_CALORIC_SURPLUS = 400
DEFAULT_PROTEIN_PER_KG = 2.0
DEFAULT_FAT_PERCENTAGE = 0.25

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

# ------ Nutrient Category Mapping ------
NUTRIENT_MAP = {
    'PRIMARY PROTEIN SOURCES': 'protein',
    'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
    'PRIMARY FAT SOURCES': 'fat',
    'PRIMARY MICRONUTRIENT SOURCES': 'protein'
}

# -----------------------------------------------------------------------------
# Cell 4: Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """
    Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation

    Args:
        age: Age in years
        height_cm: Height in centimeters
        weight_kg: Weight in kg
        sex: 'male' or 'female'

    Returns:
        BMR in kcal per day
    """
    if sex.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return bmr

def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure Based on Activity Level

    Args:
        bmr: Basal Metabolic Rate
        activity_level: Activity level as a string

    Returns:
        TDEE in kcal per day
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_personalized_targets(
    age, height_cm, weight_kg, sex='male',
    activity_level='moderately_active',
    caloric_surplus=400, protein_per_kg=2.0, fat_percentage=0.25
):
    """
    Calculate Personalized Daily Nutritional Targets

    Args:
        age: Age in years
        height_cm: Height in centimeters
        weight_kg: Weight in kg
        sex: 'male' or 'female'
        activity_level: Activity level as a string
        caloric_surplus: Additional calories per day
        protein_per_kg: Protein g per kilogram body weight
        fat_percentage: Fraction of calories from fat

    Returns:
        Dictionary containing daily targets for calories, protein, fat, and carbohydrates
    """
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
        'bmr': round(bmr),
        'tdee': round(tdee),
        'total_calories': round(total_calories),
        'protein_g': round(protein_g),
        'protein_calories': round(protein_calories),
        'fat_g': round(fat_g),
        'fat_calories': round(fat_calories),
        'carb_g': round(carb_g),
        'carb_calories': round(carb_calories),
        'target_weight_gain_per_week': round(weight_kg * 0.0025, 2)
    }

# -----------------------------------------------------------------------------
# Cell 5: UI Helper Functions (Refactored)
# -----------------------------------------------------------------------------

def display_metrics(metrics_data, num_columns=4):
    """Renders a list of metrics in a specified number of columns."""
    cols = st.columns(num_columns)
    for i, data in enumerate(metrics_data):
        # Ensure data is a dictionary before accessing keys
        if isinstance(data, dict):
            with cols[i % num_columns]:
                st.metric(label=data.get('label', ''), value=data.get('value', ''), delta=data.get('delta'))

def display_progress(label, current, target, unit):
    """Calculates and displays a progress bar for a nutritional target."""
    if target > 0:
        percent_complete = min(current / target, 1.0)
        st.progress(
            percent_complete,
            text=f"{label}: {percent_complete*100:.0f}% of daily target ({target:.0f} {unit})"
        )

def render_food_item(food, category):
    """
    Render a single food item with buttons and input controls.
    
    Args:
        food: Food item dictionary
        category: Food category string
    """
    st.subheader(f"{food.get('emoji', '')} {food['name']}")
    key = f"{category}_{food['name']}"
    current_serving = st.session_state.food_selections.get(food['name'], 0.0)
    
    # Serving buttons
    button_cols = st.columns(5)
    for k in range(1, 6):
        with button_cols[k - 1]:
            button_type = "primary" if current_serving == float(k) else "secondary"
            if st.button(f"{k} Servings", key=f"{key}_{k}", type=button_type):
                st.session_state.food_selections[food['name']] = float(k)
                st.rerun()
    
    # Custom serving input
    custom_serving = st.number_input(
        "Custom Number of Servings:",
        min_value=0.0, max_value=10.0,
        value=float(current_serving), step=0.1,
        key=f"{key}_custom"
    )
    if custom_serving != current_serving:
        if custom_serving > 0:
            st.session_state.food_selections[food['name']] = custom_serving
        elif food['name'] in st.session_state.food_selections:
            del st.session_state.food_selections[food['name']]
        st.rerun()
    
    # Nutritional info
    st.caption(
        f"Per Serving: {food['calories']} kcal | "
        f"{food['protein']} g protein | "
        f"{food['carbs']} g carbohydrates | "
        f"{food['fat']} g fat"
    )

# -----------------------------------------------------------------------------
# Cell 6: Load and Process Food Database
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """
    Load the Vegetarian Food Database From a CSV File

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary mapping food categories to lists of food items
    """
    df = pd.read_csv(file_path)
    
    foods = {
        'PRIMARY PROTEIN SOURCES': [], 'PRIMARY FAT SOURCES': [],
        'PRIMARY CARBOHYDRATE SOURCES': [], 'PRIMARY MICRONUTRIENT SOURCES': []
    }

    for _, row in df.iterrows():
        category = row['category']
        food_item = {
            'name': f"{row['name']} ({row['serving_unit']})", 'calories': row['calories'],
            'protein': row['protein'], 'carbs': row['carbs'], 'fat': row['fat']
        }
        if category in foods:
            foods[category].append(food_item)
    
    return foods

def assign_food_emojis(foods):
    """
    Assign an Emoji to Each Food Item Based on Nutritional Hierarchy

    Args:
        foods (dict): Dictionary of categorized food items

    Returns:
        dict: Foods dictionary with an 'emoji' key added to each food item
    """
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}

    for category, items in foods.items():
        if not items: continue
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]

        nutrient = NUTRIENT_MAP.get(category)
        if nutrient:
            sorted_by_nutrient = sorted(items, key=lambda x: x[nutrient], reverse=True)
            if category == 'PRIMARY PROTEIN SOURCES': top_foods['protein'] = [food['name'] for food in sorted_by_nutrient[:3]]
            elif category == 'PRIMARY CARBOHYDRATE SOURCES': top_foods['carbs'] = [food['name'] for food in sorted_by_nutrient[:3]]
            elif category == 'PRIMARY FAT SOURCES': top_foods['fat'] = [food['name'] for food in sorted_by_nutrient[:3]]
            elif category == 'PRIMARY MICRONUTRIENT SOURCES': top_foods['micro'] = [food['name'] for food in sorted_by_nutrient[:3]]

    food_rank_counts = {}
    all_top_nutrient_foods = set(top_foods['protein'] + top_foods['carbs'] + top_foods['fat'] + top_foods['micro'])
    for food_name in all_top_nutrient_foods:
        count = sum([1 for nutrient_list in ['protein', 'carbs', 'fat', 'micro'] if food_name in top_foods[nutrient_list]])
        food_rank_counts[food_name] = count
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

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

# ------ Load Food Database and Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# -----------------------------------------------------------------------------
# Cell 7: Session State Initialization and Custom Styling (Refactored)
# -----------------------------------------------------------------------------

# ------ Initialize Session State Using a Loop ------
keys_to_initialize = [
    'food_selections', 'user_age', 'user_height', 
    'user_weight', 'user_sex', 'user_activity'
]
default_values = {'food_selections': {}}

for key in keys_to_initialize:
    if key not in st.session_state:
        st.session_state[key] = default_values.get(key, None)

# ------ Custom CSS for Enhanced Styling ------
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
.stButton > button[kind="primary"] { background-color: #ff6b6b; color: white; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 8: Application Title and Sidebar Parameters
# -----------------------------------------------------------------------------

st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown("Ready to turbocharge your health game? This awesome tool dishes out daily nutrition goals made just for you and makes tracking meals as easy as pie. Let's get those macros on your team! 🚀")

st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

age = st.sidebar.number_input("Age (Years)", min_value=16, max_value=80, value=st.session_state.user_age, placeholder="Enter your age")
height_cm = st.sidebar.number_input("Height (Centimeters)", min_value=140, max_value=220, value=st.session_state.user_height, placeholder="Enter your height")
weight_kg = st.sidebar.number_input("Weight (kg)", min_value=40.0, max_value=150.0, value=st.session_state.user_weight, step=0.5, placeholder="Enter your weight")

sex_options = ["Select Sex", "Male", "Female"]
sex_index = sex_options.index(st.session_state.user_sex) if st.session_state.user_sex in sex_options else 0
sex = st.sidebar.selectbox("Sex", sex_options, index=sex_index)

activity_options = [("Select Activity Level", None), ("Sedentary", "sedentary"), ("Lightly Active", "lightly_active"), ("Moderately Active", "moderately_active"), ("Very Active", "very_active"), ("Extremely Active", "extremely_active")]
activity_values = [opt[1] for opt in activity_options]
activity_index = activity_values.index(st.session_state.user_activity) if st.session_state.user_activity in activity_values else 0
activity_selection = st.sidebar.selectbox("Activity Level", activity_options, index=activity_index, format_func=lambda x: x[0])
activity_level = activity_selection[1]

st.session_state.update(user_age=age, user_height=height_cm, user_weight=weight_kg, user_sex=sex, user_activity=activity_level)

with st.sidebar.expander("Advanced Settings ⚙️"):
    caloric_surplus = st.number_input("Caloric Surplus (kcal Per Day)", min_value=200, max_value=800, value=None, placeholder=f"Default: {DEFAULT_CALORIC_SURPLUS}", step=50)
    protein_per_kg = st.number_input("Protein (g Per Kilogram Body Weight)", min_value=1.2, max_value=3.0, value=None, placeholder=f"Default: {DEFAULT_PROTEIN_PER_KG}", step=0.1)
    fat_percentage_input = st.number_input("Fat (Percent of Total Calories)", min_value=15, max_value=40, value=None, placeholder=f"Default: {int(DEFAULT_FAT_PERCENTAGE * 100)}", step=1)

final_age = age if age is not None else DEFAULT_AGE
final_height = height_cm if height_cm is not None else DEFAULT_HEIGHT_CM
final_weight = weight_kg if weight_kg is not None else DEFAULT_WEIGHT_KG
final_sex = sex if sex != "Select Sex" else DEFAULT_GENDER
final_activity = activity_level if activity_level is not None else DEFAULT_ACTIVITY_LEVEL
final_surplus = caloric_surplus if caloric_surplus is not None else DEFAULT_CALORIC_SURPLUS
final_protein = protein_per_kg if protein_per_kg is not None else DEFAULT_PROTEIN_PER_KG
final_fat_percent = (fat_percentage_input / 100) if fat_percentage_input is not None else DEFAULT_FAT_PERCENTAGE

user_has_entered_info = all([age, height_cm, weight_kg, sex != "Select Sex", activity_level])

targets = calculate_personalized_targets(
    age=final_age, height_cm=final_height, weight_kg=final_weight, sex=final_sex.lower(),
    activity_level=final_activity, caloric_surplus=final_surplus,
    protein_per_kg=final_protein, fat_percentage=final_fat_percent
)

# -----------------------------------------------------------------------------
# Cell 9: Display Personalized Targets and Daily Goals (Refactored)
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations")
else:
    st.header("Your Personalized Daily Nutritional Targets for Healthy Weight Gain 🎯")

# ------ Display Metrics Using Helper Function ------
metabolic_metrics = [
    {'label': "Basal Metabolic Rate (BMR)", 'value': f"{targets['bmr']} kcal per day"},
    {'label': "Total Daily Energy Expenditure (TDEE)", 'value': f"{targets['tdee']} kcal per day"},
    {'label': "Estimated Weekly Weight Gain", 'value': f"{targets['target_weight_gain_per_week']} kg per week"}
]
display_metrics(metabolic_metrics)

st.subheader("Daily Nutritional Target Breakdown")
target_metrics = [
    {'label': "Daily Calorie Target", 'value': f"{targets['total_calories']} kcal"},
    {'label': "Protein Target", 'value': f"{targets['protein_g']} g"},
    {'label': "Carbohydrate Target", 'value': f"{targets['carb_g']} g"},
    {'label': "Fat Target", 'value': f"{targets['fat_g']} g"}
]
display_metrics(target_metrics)

st.subheader("Macronutrient Distribution as Percent of Daily Calories")
protein_percent = (targets['protein_calories'] / targets['total_calories']) * 100
carb_percent = (targets['carb_calories'] / targets['total_calories']) * 100
fat_percent_display = (targets['fat_calories'] / targets['total_calories']) * 100
macro_metrics = [
    {'label': "Protein Contribution", 'value': f"{protein_percent:.1f}%", 'delta': f"+ {targets['protein_calories']} kcal"},
    {'label': "Carbohydrate Contribution", 'value': f"{carb_percent:.1f}%", 'delta': f"+ {targets['carb_calories']} kcal"},
    {'label': "Fat Contribution", 'value': f"{fat_percent_display:.1f}%", 'delta': f"+ {targets['fat_calories']} kcal"}
]
display_metrics(macro_metrics)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item")

available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    sorted_items = sorted(foods[category], key=lambda x: (EMOJI_ORDER.get(x.get('emoji', ''), 4), -x['calories']))
    with tabs[i]:
        for j in range(0, len(sorted_items), 2):
            col1, col2 = st.columns(2)
            if j < len(sorted_items):
                with col1: render_food_item(sorted_items[j], category)
            if j + 1 < len(sorted_items):
                with col2: render_food_item(sorted_items[j + 1], category)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 11: Calculation Button and Nutritional Results Display (Refactored)
# -----------------------------------------------------------------------------

if st.button("Calculate Daily Intake", type="primary", use_container_width=True):
    total_calories, total_protein, total_carbs, total_fat = 0, 0, 0, 0
    selected_foods = []
    for category, items in foods.items():
        for food in items:
            servings = st.session_state.food_selections.get(food['name'], 0)
            if servings > 0:
                total_calories += food['calories'] * servings
                total_protein += food['protein'] * servings
                total_carbs += food['carbs'] * servings
                total_fat += food['fat'] * servings
                selected_foods.append({'food': food, 'servings': servings})

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
        {'label': "Total Calories Consumed", 'value': f"{total_calories:.0f} kcal"},
        {'label': "Total Protein Consumed", 'value': f"{total_protein:.1f} g"},
        {'label': "Total Carbohydrates Consumed", 'value': f"{total_carbs:.1f} g"},
        {'label': "Total Fat Consumed", 'value': f"{total_fat:.1f} g"}
    ]
    display_metrics(intake_metrics)

    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    # ------ Display Progress Bars Using Helper Function ------
    display_progress("Calories", total_calories, targets['total_calories'], "kcal")
    display_progress("Protein", total_protein, targets['protein_g'], "g")
    display_progress("Carbohydrates", total_carbs, targets['carb_g'], "g")
    display_progress("Fat", total_fat, targets['fat_g'], "g")

    st.subheader("Personalized Recommendations for Today’s Nutrition 💡")
    # ------ Generate Recommendations Using a Loop ------
    recommendation_params = [
        {'name': 'kcal', 'current': total_calories, 'target': targets['total_calories'], 'message': 'to reach your weight gain target'},
        {'name': 'g of protein', 'current': total_protein, 'target': targets['protein_g'], 'message': 'for muscle building'},
        {'name': 'g of carbohydrates', 'current': total_carbs, 'target': targets['carb_g'], 'message': 'for energy and performance'},
        {'name': 'g of healthy fats', 'current': total_fat, 'target': targets['fat_g'], 'message': 'for hormone production'}
    ]
    recommendations = []
    for param in recommendation_params:
        if param['current'] < param['target']:
            deficit = param['target'] - param['current']
            recommendations.append(f"• You need {deficit:.0f} more {param['name']} {param['message']}")
    
    if recommendations:
        for rec in recommendations: st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    st.subheader("Daily Caloric Balance and Weight Gain Summary ⚖️")
    cal_balance = total_calories - targets['tdee']
    if cal_balance > 0:
        st.info(f"📈 You are consuming {cal_balance:.0f} kcal above maintenance, supporting weight gain")
    else:
        st.warning(f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance")

    if selected_foods:
        st.subheader("Detailed Food Log for Today 📋")
        food_log_data = [{
            'Food Item Name': f"{item['food'].get('emoji', '')} {item['food']['name']}",
            'Number of Servings Consumed': f"{item['servings']:.1f}",
            'Total Calories Consumed': item['food']['calories'] * item['servings'],
            'Total Protein Consumed (g)': item['food']['protein'] * item['servings'],
            'Total Carbohydrates Consumed (g)': item['food']['carbs'] * item['servings'],
            'Total Fat Consumed (g)': item['food']['fat'] * item['servings']
        } for item in selected_foods]
        df_log = pd.DataFrame(food_log_data)
        st.dataframe(df_log.style.format({
            'Total Calories Consumed': '{:.0f}', 'Total Protein Consumed (g)': '{:.1f}',
            'Total Carbohydrates Consumed (g)': '{:.1f}', 'Total Fat Consumed (g)': '{:.1f}'
        }), use_container_width=True)

    st.markdown("---")
    print("Daily nutritional intake calculation and summary completed successfully 📊")

# -----------------------------------------------------------------------------
# Cell 12: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()
    print("All food selections have been cleared. Ready for a fresh start! 🔄")

# -----------------------------------------------------------------------------
# Cell 13: Footer Information and Application Documentation
# -----------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### Activity Level Guide for Accurate TDEE 🏃‍♂️")
st.sidebar.markdown("- Sedentary: Little to no exercise or desk job\n- Lightly Active: Light exercise or sports one to three days per week\n- Moderately Active: Moderate exercise or sports three to five days per week\n- Very Active: Hard exercise or sports six to seven days per week\n- Extremely Active: Very hard exercise, physical job, or training twice daily")

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

print("Thank you for using the Personalized Nutrition Tracker! Eat well, feel well! 🌱")
