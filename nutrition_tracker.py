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
# Cell 3: Default Parameter Values for User Inputs
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

# -----------------------------------------------------------------------------
# Cell 4: Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """
    Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation

    Args:
        age: Age in years
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
        sex: 'male' or 'female'

    Returns:
        BMR in kilocalories per day
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
        TDEE in kilocalories per day
    """
    activity_multipliers = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extremely_active': 1.9
    }
    multiplier = activity_multipliers.get(activity_level, 1.55)
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
        weight_kg: Weight in kilograms
        sex: 'male' or 'female'
        activity_level: Activity level as a string
        caloric_surplus: Additional calories per day
        protein_per_kg: Protein grams per kilogram body weight
        fat_percentage: Fraction of calories from fat

    Returns:
        Dictionary containing daily targets for calories, protein, fat, and carbohydrates
    """
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    total_calories = tdee + caloric_surplus
    protein_grams = protein_per_kg * weight_kg
    protein_calories = protein_grams * 4
    fat_calories = total_calories * fat_percentage
    fat_grams = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_grams = carb_calories / 4

    return {
        'bmr': round(bmr),
        'tdee': round(tdee),
        'total_calories': round(total_calories),
        'protein_grams': round(protein_grams),
        'protein_calories': round(protein_calories),
        'fat_grams': round(fat_grams),
        'fat_calories': round(fat_calories),
        'carb_grams': round(carb_grams),
        'carb_calories': round(carb_calories),
        'target_weight_gain_per_week': round(weight_kg * 0.0025, 2)
    }

# -----------------------------------------------------------------------------
# Cell 5: Load and Process Food Database
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

    # ------ Map Foods To Categories for Tabbed Selection ------
    category_mapping = {
        'Eggs': 'PRIMARY PROTEIN SOURCES',
        'Greek Yogurt': 'PRIMARY PROTEIN SOURCES',
        'Protein Powder': 'PRIMARY PROTEIN SOURCES',
        'Milk': 'PRIMARY PROTEIN SOURCES',
        'Cottage Cheese': 'PRIMARY PROTEIN SOURCES',
        'Mozzarella Cheese': 'PRIMARY PROTEIN SOURCES',
        'Lentils': 'PRIMARY PROTEIN SOURCES',
        'Chickpeas': 'PRIMARY PROTEIN SOURCES',
        'Kidney Beans': 'PRIMARY PROTEIN SOURCES',
        'Hummus': 'PRIMARY PROTEIN SOURCES',
        'Cheese Tortellinis': 'PRIMARY PROTEIN SOURCES',
        'Spinach Tortellinis': 'PRIMARY PROTEIN SOURCES',
        'Olive Oil': 'PRIMARY FAT SOURCES',
        'Peanut Butter': 'PRIMARY FAT SOURCES',
        'Almonds': 'PRIMARY FAT SOURCES',
        'Mixed Nuts': 'PRIMARY FAT SOURCES',
        'Avocados': 'PRIMARY FAT SOURCES',
        'Sunflower Seeds': 'PRIMARY FAT SOURCES',
        'Chia Seeds': 'PRIMARY FAT SOURCES',
        'Tahini': 'PRIMARY FAT SOURCES',
        'Heavy Cream': 'PRIMARY FAT SOURCES',
        'Trail Mix': 'PRIMARY FAT SOURCES',
        'Oats': 'PRIMARY CARBOHYDRATE SOURCES',
        'Potatoes': 'PRIMARY CARBOHYDRATE SOURCES',
        'White Rice': 'PRIMARY CARBOHYDRATE SOURCES',
        'Multigrain Bread': 'PRIMARY CARBOHYDRATE SOURCES',
        'Pasta': 'PRIMARY CARBOHYDRATE SOURCES',
        'Bananas': 'PRIMARY CARBOHYDRATE SOURCES',
        'Couscous': 'PRIMARY CARBOHYDRATE SOURCES',
        'Corn': 'PRIMARY CARBOHYDRATE SOURCES',
        'Green Peas': 'PRIMARY CARBOHYDRATE SOURCES',
        'Pizza': 'PRIMARY CARBOHYDRATE SOURCES',
        'Mixed Vegetables': 'PRIMARY MICRONUTRIENT SOURCES',
        'Spinach': 'PRIMARY MICRONUTRIENT SOURCES',
        'Broccoli': 'PRIMARY MICRONUTRIENT SOURCES',
        'Berries': 'PRIMARY MICRONUTRIENT SOURCES',
        'Carrots': 'PRIMARY MICRONUTRIENT SOURCES',
        'Tomatoes': 'PRIMARY MICRONUTRIENT SOURCES',
        'Mushrooms': 'PRIMARY MICRONUTRIENT SOURCES',
        'Cauliflower': 'PRIMARY MICRONUTRIENT SOURCES',
        'Green Beans': 'PRIMARY MICRONUTRIENT SOURCES',
        'Orange Juice': 'PRIMARY MICRONUTRIENT SOURCES',
        'Apple Juice': 'PRIMARY MICRONUTRIENT SOURCES',
        'Fruit Juice': 'PRIMARY MICRONUTRIENT SOURCES'
    }

    foods = {
        'PRIMARY PROTEIN SOURCES': [],
        'PRIMARY FAT SOURCES': [],
        'PRIMARY CARBOHYDRATE SOURCES': [],
        'PRIMARY MICRONUTRIENT SOURCES': []
    }

    for _, row in df.iterrows():
        food_name = row['name']
        category = category_mapping.get(food_name, 'PRIMARY MICRONUTRIENT SOURCES')
        food_item = {
            'name': f"{food_name} ({row['serving_unit']})",
            'calories': row['calories'],
            'protein': row['protein'],
            'carbs': row['carbs'],
            'fat': row['fat']
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
    nutrient_map = {
        'PRIMARY PROTEIN SOURCES': 'protein',
        'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
        'PRIMARY FAT SOURCES': 'fat',
        'PRIMARY MICRONUTRIENT SOURCES': 'protein'
    }

    # Identify top three nutrient and calorie contributors in each category
    for category, items in foods.items():
        if not items:
            continue
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]

        nutrient = nutrient_map.get(category)
        if nutrient:
            sorted_by_nutrient = sorted(items, key=lambda x: x[nutrient], reverse=True)
            if category == 'PRIMARY PROTEIN SOURCES':
                top_foods['protein'] = [food['name'] for food in sorted_by_nutrient[:3]]
            elif category == 'PRIMARY CARBOHYDRATE SOURCES':
                top_foods['carbs'] = [food['name'] for food in sorted_by_nutrient[:3]]
            elif category == 'PRIMARY FAT SOURCES':
                top_foods['fat'] = [food['name'] for food in sorted_by_nutrient[:3]]
            elif category == 'PRIMARY MICRONUTRIENT SOURCES':
                top_foods['micro'] = [food['name'] for food in sorted_by_nutrient[:3]]

    # Identify superfoods (high rank in multiple nutrient categories)
    food_rank_counts = {}
    all_top_nutrient_foods = set(top_foods['protein'] + top_foods['carbs'] + top_foods['fat'] + top_foods['micro'])
    for food_name in all_top_nutrient_foods:
        count = sum([
            1 for nutrient_list in ['protein', 'carbs', 'fat', 'micro']
            if food_name in top_foods[nutrient_list]
        ])
        food_rank_counts[food_name] = count
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    # Apply emojis based on the specified hierarchy
    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            food['emoji'] = ''
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])

            if food_name in superfoods:
                food['emoji'] = '🥇'
            elif is_high_calorie and is_top_nutrient:
                food['emoji'] = '💥'
            elif is_high_calorie:
                food['emoji'] = '🔥'
            elif food_name in top_foods['protein']:
                food['emoji'] = '💪'
            elif food_name in top_foods['carbs']:
                food['emoji'] = '🍚'
            elif food_name in top_foods['fat']:
                food['emoji'] = '🥑'
            elif food_name in top_foods['micro']:
                food['emoji'] = '🥦'
    return foods

# ------ Load Food Database and Assign Emojis ------
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# -----------------------------------------------------------------------------
# Cell 6: Session State Initialization and Custom Styling
# -----------------------------------------------------------------------------

# ------ Initialize Session State for Food Selections ------
if 'food_selections' not in st.session_state:
    st.session_state.food_selections = {}

# ------ Initialize Session State for User Inputs ------
if 'user_age' not in st.session_state:
    st.session_state.user_age = None
if 'user_height' not in st.session_state:
    st.session_state.user_height = None
if 'user_weight' not in st.session_state:
    st.session_state.user_weight = None
if 'user_sex' not in st.session_state:
    st.session_state.user_sex = None
if 'user_activity' not in st.session_state:
    st.session_state.user_activity = None

# ------ Custom CSS for Enhanced Styling and Hiding Input Instructions ------
st.markdown("""
<style>
[data-testid="InputInstructions"] {
    display: none;
}
.active-button {
    background-color: #ff6b6b !important;
    color: white !important;
    border: 2px solid #ff6b6b !important;
}
.button-container {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
}
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}
.stNumberInput > div > div > input[value] {
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 7: Application Title and Sidebar Parameters
# -----------------------------------------------------------------------------

st.title("Personalized Nutrition Tracker With Evidence-Based Vegetarian Macros 🍽️")
st.markdown("""
Ready to fuel your journey to better health? This tool creates personalized recommendations tailored just for you and makes tracking your meals a breeze. Let's get your macros working in your favor! 🚀
""")

# ------ Sidebar for Personal Parameters ------
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

# ------ Personal Information Inputs With Placeholder Text ------
age = st.sidebar.number_input(
    "Age (Years)",
    min_value=16,
    max_value=80,
    value=st.session_state.user_age,
    placeholder="Enter your age"
)

height_cm = st.sidebar.number_input(
    "Height (Centimeters)",
    min_value=140,
    max_value=220,
    value=st.session_state.user_height,
    placeholder="Enter your height"
)

weight_kg = st.sidebar.number_input(
    "Weight (Kilograms)",
    min_value=40.0,
    max_value=150.0,
    value=st.session_state.user_weight,
    step=0.5,
    placeholder="Enter your weight"
)

sex_options = ["Select Sex", "Male", "Female"]
sex_index = 0
if st.session_state.user_sex:
    try:
        sex_index = sex_options.index(st.session_state.user_sex)
    except ValueError:
        sex_index = 0

sex = st.sidebar.selectbox("Sex", sex_options, index=sex_index)

activity_options = [
    ("Select Activity Level", None),
    ("Sedentary", "sedentary"),
    ("Lightly Active", "lightly_active"),
    ("Moderately Active", "moderately_active"),
    ("Very Active", "very_active"),
    ("Extremely Active", "extremely_active")
]
activity_index = 0
if st.session_state.user_activity:
    for i, (label, value) in enumerate(activity_options):
        if value == st.session_state.user_activity:
            activity_index = i
            break

activity_selection = st.sidebar.selectbox(
    "Activity Level",
    activity_options,
    index=activity_index,
    format_func=lambda x: x[0]
)
activity_level = activity_selection[1]

# ------ Update Session State With User Inputs ------
st.session_state.user_age = age
st.session_state.user_height = height_cm
st.session_state.user_weight = weight_kg
st.session_state.user_sex = sex
st.session_state.user_activity = activity_level

# ------ Advanced Parameters Collapsible Section ------
with st.sidebar.expander("Advanced Settings ⚙️"):
    caloric_surplus = st.number_input(
        "Caloric Surplus (Kilocalories Per Day)",
        min_value=200, max_value=800,
        value=None,
        placeholder=f"Default: {DEFAULT_CALORIC_SURPLUS}",
        step=50,
        help="Additional calories above maintenance for weight gain"
    )
    protein_per_kg = st.number_input(
        "Protein (Grams Per Kilogram Body Weight)",
        min_value=1.2, max_value=3.0,
        value=None,
        placeholder=f"Default: {DEFAULT_PROTEIN_PER_KG}",
        step=0.1,
        help="Protein intake per kilogram of body weight"
    )
    fat_percentage_input = st.number_input(
        "Fat (Percent of Total Calories)",
        min_value=15, max_value=40,
        value=None,
        placeholder=f"Default: {int(DEFAULT_FAT_PERCENTAGE * 100)}",
        step=1,
        help="Percentage of total calories from fat"
    )

# ------ Use Default Values If User Has Not Entered Custom Values ------
final_age = age if age is not None else DEFAULT_AGE
final_height = height_cm if height_cm is not None else DEFAULT_HEIGHT_CM
final_weight = weight_kg if weight_kg is not None else DEFAULT_WEIGHT_KG
final_sex = sex if sex != "Select Sex" else DEFAULT_GENDER
final_activity = activity_level if activity_level is not None else DEFAULT_ACTIVITY_LEVEL
final_surplus = caloric_surplus if caloric_surplus is not None else DEFAULT_CALORIC_SURPLUS
final_protein = protein_per_kg if protein_per_kg is not None else DEFAULT_PROTEIN_PER_KG
final_fat_percent = (fat_percentage_input / 100) if fat_percentage_input is not None else DEFAULT_FAT_PERCENTAGE

# ------ Check If User Has Entered All Required Information ------
user_has_entered_info = (
    age is not None and
    height_cm is not None and
    weight_kg is not None and
    sex != "Select Sex" and
    activity_level is not None
)

# ------ Calculate Personalized Targets for the User ------
targets = calculate_personalized_targets(
    age=final_age,
    height_cm=final_height,
    weight_kg=final_weight,
    sex=final_sex.lower(),
    activity_level=final_activity,
    caloric_surplus=final_surplus,
    protein_per_kg=final_protein,
    fat_percentage=final_fat_percent
)

# -----------------------------------------------------------------------------
# Cell 8: Display Personalized Targets and Daily Goals
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations")
else:
    st.header("Your Personalized Daily Nutritional Targets for Healthy Weight Gain 🎯")

# ------ Display Metabolic Information In Four Columns ------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kilocalories per day")
with col2:
    st.metric("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kilocalories per day")
with col3:
    st.metric("Estimated Weekly Weight Gain", f"{targets['target_weight_gain_per_week']} kilograms per week")
with col4:
    pass  # Empty column for alignment

# ------ Display Daily Nutritional Targets In Four Columns ------
st.subheader("Daily Nutritional Target Breakdown")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Daily Calorie Target", f"{targets['total_calories']} kilocalories")
with col2:
    st.metric("Protein Target", f"{targets['protein_grams']} grams")
with col3:
    st.metric("Carbohydrate Target", f"{targets['carb_grams']} grams")
with col4:
    st.metric("Fat Target", f"{targets['fat_grams']} grams")

# ------ Display Macronutrient Percentages In Four Columns ------
st.subheader("Macronutrient Distribution as Percent of Daily Calories")
protein_percent = (targets['protein_calories'] / targets['total_calories']) * 100
carb_percent = (targets['carb_calories'] / targets['total_calories']) * 100
fat_percent_display = (targets['fat_calories'] / targets['total_calories']) * 100
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Protein Contribution", f"{protein_percent:.1f} percent", f"+ {targets['protein_calories']} kilocalories")
with col2:
    st.metric("Carbohydrate Contribution", f"{carb_percent:.1f} percent", f"+ {targets['carb_calories']} kilocalories")
with col3:
    st.metric("Fat Contribution", f"{fat_percent_display:.1f} percent", f"+ {targets['fat_calories']} kilocalories")
with col4:
    pass  # Empty column for alignment

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 9: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item")

# ------ Create Category Tabs for Food Organization ------
available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]

    # ------ Define Sort Order and Sort Items by Emoji Hierarchy ------
    emoji_order = {'🥇': 0, '💥': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '🥦': 3, '': 4}
    sorted_items = sorted(items, key=lambda x: emoji_order.get(x.get('emoji', ''), 4))

    with tabs[i]:
        # ------ Display Foods In Two-Column Layout ------
        for j in range(0, len(sorted_items), 2):
            col1, col2 = st.columns(2)

            # ------ First Food Item In Row ------
            if j < len(sorted_items):
                with col1:
                    food = sorted_items[j]
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
                    st.caption(
                        f"Per Serving: {food['calories']} kilocalories | "
                        f"{food['protein']} grams protein | "
                        f"{food['carbs']} grams carbohydrates | "
                        f"{food['fat']} grams fat"
                    )

            # ------ Second Food Item In Row ------
            if j + 1 < len(sorted_items):
                with col2:
                    food = sorted_items[j + 1]
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
                    st.caption(
                        f"Per Serving: {food['calories']} kilocalories | "
                        f"{food['protein']} grams protein | "
                        f"{food['carbs']} grams carbohydrates | "
                        f"{food['fat']} grams fat"
                    )

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Calculation Button and Nutritional Results Display
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
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Calories Consumed", f"{total_calories:.0f} kilocalories")
    with col2:
        st.metric("Total Protein Consumed", f"{total_protein:.1f} grams")
    with col3:
        st.metric("Total Carbohydrates Consumed", f"{total_carbs:.1f} grams")
    with col4:
        st.metric("Total Fat Consumed", f"{total_fat:.1f} grams")

    st.subheader("Progress Toward Daily Nutritional Targets 🎯")

    # ------ Calculate Percentages for Progress Bars ------
    cal_percent = (min(total_calories / targets['total_calories'] * 100, 100)
                   if targets['total_calories'] > 0 else 0)
    st.progress(
        cal_percent / 100,
        text=f"Calories: {cal_percent:.0f} percent of daily target ({targets['total_calories']} kilocalories)"
    )
    prot_percent = (min(total_protein / targets['protein_grams'] * 100, 100)
                    if targets['protein_grams'] > 0 else 0)
    st.progress(
        prot_percent / 100,
        text=f"Protein: {prot_percent:.0f} percent of daily target ({targets['protein_grams']} grams)"
    )
    carb_percent = (min(total_carbs / targets['carb_grams'] * 100, 100)
                    if targets['carb_grams'] > 0 else 0)
    st.progress(
        carb_percent / 100,
        text=f"Carbohydrates: {carb_percent:.0f} percent of daily target ({targets['carb_grams']} grams)"
    )
    fat_percent_progress = (min(total_fat / targets['fat_grams'] * 100, 100)
                            if targets['fat_grams'] > 0 else 0)
    st.progress(
        fat_percent_progress / 100,
        text=f"Fat: {fat_percent_progress:.0f} percent of daily target ({targets['fat_grams']} grams)"
    )

    st.subheader("Personalized Recommendations for Today’s Nutrition 💡")
    recommendations = []
    if total_calories < targets['total_calories']:
        deficit = targets['total_calories'] - total_calories
        recommendations.append(
            f"• You need {deficit:.0f} more kilocalories to reach your weight gain target"
        )
    if total_protein < targets['protein_grams']:
        deficit = targets['protein_grams'] - total_protein
        recommendations.append(
            f"• You need {deficit:.0f} more grams of protein for muscle building"
        )
    if total_carbs < targets['carb_grams']:
        deficit = targets['carb_grams'] - total_carbs
        recommendations.append(
            f"• You need {deficit:.0f} more grams of carbohydrates for energy and performance"
        )
    if total_fat < targets['fat_grams']:
        deficit = targets['fat_grams'] - total_fat
        recommendations.append(
            f"• You need {deficit:.0f} more grams of healthy fats for hormone production"
        )
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    # ------ Show Surplus and Deficit Information ------
    st.subheader("Daily Caloric Balance and Weight Gain Summary ⚖️")
    cal_balance = total_calories - targets['tdee']
    if cal_balance > 0:
        st.info(
            f"📈 You are consuming {cal_balance:.0f} kilocalories above maintenance, supporting weight gain"
        )
    else:
        st.warning(
            f"📉 You are consuming {abs(cal_balance):.0f} kilocalories below maintenance"
        )

    if selected_foods:
        st.subheader("Detailed Food Log for Today 📋")
        food_log = [
            {
                'Food Item Name': f"{item['food'].get('emoji', '')} {item['food']['name']}",
                'Number of Servings Consumed': f"{item['servings']:.1f}",
                'Total Calories Consumed': item['food']['calories'] * item['servings'],
                'Total Protein Consumed (Grams)': item['food']['protein'] * item['servings'],
                'Total Carbohydrates Consumed (Grams)': item['food']['carbs'] * item['servings'],
                'Total Fat Consumed (Grams)': item['food']['fat'] * item['servings']
            }
            for item in selected_foods
        ]
        df_log = pd.DataFrame(food_log)
        st.dataframe(
            df_log.style.format({
                'Total Calories Consumed': '{:.0f}',
                'Total Protein Consumed (Grams)': '{:.1f}',
                'Total Carbohydrates Consumed (Grams)': '{:.1f}',
                'Total Fat Consumed (Grams)': '{:.1f}'
            }),
            use_container_width=True
        )

    st.markdown("---")
    print("Daily nutritional intake calculation and summary completed successfully 📊")

# -----------------------------------------------------------------------------
# Cell 11: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()
    print("All food selections have been cleared. Ready for a fresh start! 🔄")

# -----------------------------------------------------------------------------
# Cell 12: Footer Information and Application Documentation
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
- Protein: 2.0 grams per kilogram of body weight for muscle building
- Fat: 25 percent of total calories for hormone production
- Carbohydrates: Remaining calories after protein and fat allocation
- Weight gain target: 0.25 percent of body weight per week for lean gains
""")

print("Thank you for using the Personalized Nutrition Tracker! Eat well, feel well! 🌱")
