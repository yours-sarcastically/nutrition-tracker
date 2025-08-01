# -----------------------------------------------------------------------------
# Personalized Nutrition Tracker for Healthy Weight Gain
# -----------------------------------------------------------------------------

"""
This application provides a comprehensive tool for tracking daily nutritional
intake using vegetarian food sources. The application calculates personalized
daily targets based on individual parameters including age, height, weight,
and activity level using the Mifflin-St Jeor equation and evidence-based
nutritional guidelines.

The targets are calculated using the following methods:
- BMR: Mifflin-St Jeor equation for metabolic rate calculation
- TDEE: BMR multiplied by activity-specific multiplier
- Caloric surplus: TDEE plus 400 calories for lean bulking approach
- Protein: 2.0 grams per kilogram of body weight for muscle building
- Fat: 25% of total calories for hormone production and nutrient absorption
- Carbohydrates: Remaining calories after protein and fat allocation

Usage:
1. Enter personal parameters in the sidebar (age, height, weight, gender,
   activity level)
2. Optionally adjust advanced settings (caloric surplus, protein ratio,
   fat percentage)
3. Select foods from categorized tabs using quick-select buttons or custom
   serving amounts
4. Click "Calculate Daily Intake" to view nutritional analysis and
   personalized recommendations
5. Use "Clear All Selections" to reset food choices

The application provides real-time progress tracking against personalized
targets, detailed food logging, and specific recommendations for achieving
healthy weight gain goals.
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
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cell 3: Hidden Default Values Not Displayed on Website
# -----------------------------------------------------------------------------

# ------ Default Parameter Values from Research Article ------
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

def calculate_bmr(age, height_cm, weight_kg, gender='male'):
    """
    Calculate Basal Metabolic Rate using Mifflin-St Jeor equation.
    
    Args:
        age: Age in years
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
        gender: 'male' or 'female'
    
    Returns:
        BMR in kcal/day
    """
    if gender.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    
    return bmr


def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure.
    
    Args:
        bmr: Basal Metabolic Rate
        activity_level: Activity multiplier string
    
    Returns:
        TDEE in kcal/day
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


def calculate_personalized_targets(age, height_cm, weight_kg, gender='male',
                                   activity_level='moderately_active',
                                   caloric_surplus=400, protein_per_kg=2.0,
                                   fat_percentage=0.25):
    """
    Calculate personalized daily nutritional targets.
    
    Args:
        age: Age in years
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
        gender: 'male' or 'female'
        activity_level: Activity level string
        caloric_surplus: Additional calories for weight gain
        protein_per_kg: Protein grams per kg body weight
        fat_percentage: Percentage of calories from fat
    
    Returns:
        Dictionary with daily targets
    """
    # ------ Calculate BMR and TDEE ------
    bmr = calculate_bmr(age, height_cm, weight_kg, gender)
    tdee = calculate_tdee(bmr, activity_level)
    
    # ------ Calculate Total Daily Calories for Weight Gain ------
    total_calories = tdee + caloric_surplus
    
    # ------ Calculate Protein Target in Grams and Calories ------
    protein_grams = protein_per_kg * weight_kg
    protein_calories = protein_grams * 4
    
    # ------ Calculate Fat Target in Calories and Grams ------
    fat_calories = total_calories * fat_percentage
    fat_grams = fat_calories / 9
    
    # ------ Calculate Carbohydrate Target from Remaining Calories ------
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
    """Load food database from CSV file."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # ------ Create Sample Database if CSV File Not Found ------
        sample_data = {
            'name': ['Eggs', 'Greek Yogurt', 'Protein Powder', 'Milk',
                     'Cottage Cheese', 'Lentils', 'Chickpeas', 'Olive Oil',
                     'Peanut Butter', 'Almonds', 'Oats', 'White Rice',
                     'Bananas', 'Pasta', 'Potatoes', 'Spinach', 'Broccoli',
                     'Mixed Vegetables'],
            'serving_unit': ['2 large', '1 cup', '1 scoop', '1 cup', '1 cup',
                             '1 cup cooked', '1 cup cooked', '1 tbsp',
                             '2 tbsp', '1 oz', '1 cup cooked',
                             '1 cup cooked', '1 medium', '1 cup cooked',
                             '1 medium', '1 cup', '1 cup', '1 cup'],
            'calories': [140, 130, 120, 150, 220, 230, 270, 120, 190, 160,
                         150, 205, 105, 220, 160, 7, 25, 25],
            'protein': [12, 20, 25, 8, 25, 18, 15, 0, 8, 6, 5, 4, 1, 8, 4,
                        1, 3, 2],
            'carbs': [1, 9, 2, 12, 8, 40, 45, 0, 8, 6, 27, 45, 27, 43, 37,
                      1, 5, 5],
            'fat': [10, 0, 1, 8, 10, 1, 4, 14, 16, 14, 3, 0, 0, 1, 0, 0,
                    0, 0]
        }
        df = pd.DataFrame(sample_data)
    
    # ------ Category Mapping for Food Organization ------
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
        'Cheese Tortellini': 'PRIMARY PROTEIN SOURCES',
        'Spinach Tortellini': 'PRIMARY PROTEIN SOURCES',
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
        'Oats': 'CARBOHYDRATE SOURCES',
        'Potatoes': 'CARBOHYDRATE SOURCES',
        'White Rice': 'CARBOHYDRATE SOURCES',
        'Multigrain Bread': 'CARBOHYDRATE SOURCES',
        'Pasta': 'CARBOHYDRATE SOURCES',
        'Bananas': 'CARBOHYDRATE SOURCES',
        'Couscous': 'CARBOHYDRATE SOURCES',
        'Corn': 'CARBOHYDRATE SOURCES',
        'Green Peas': 'CARBOHYDRATE SOURCES',
        'Pizza': 'CARBOHYDRATE SOURCES',
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

    # ------ Create Food Dictionary Structure ------
    foods = {
        'PRIMARY PROTEIN SOURCES': [],
        'PRIMARY FAT SOURCES': [],
        'CARBOHYDRATE SOURCES': [],
        'PRIMARY MICRONUTRIENT SOURCES': []
    }

    # ------ Process All Foods from CSV File ------
    for _, row in df.iterrows():
        food_name = row['name']
        category = category_mapping.get(food_name,
                                        'PRIMARY MICRONUTRIENT SOURCES')
        
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


# ------ Load Food Database ------
foods = load_food_database('nutrition_results.csv')

# -----------------------------------------------------------------------------
# Cell 6: Session State Initialization and Custom Styling
# -----------------------------------------------------------------------------

# ------ Initialize Session State for Food Selections ------
if 'food_selections' not in st.session_state:
    st.session_state.food_selections = {}

# ------ Initialize Session State for User Inputs with Empty Values ------
if 'user_age' not in st.session_state:
    st.session_state.user_age = None
if 'user_height' not in st.session_state:
    st.session_state.user_height = None
if 'user_weight' not in st.session_state:
    st.session_state.user_weight = None
if 'user_gender' not in st.session_state:
    st.session_state.user_gender = None
if 'user_activity' not in st.session_state:
    st.session_state.user_activity = None

# ------ Custom CSS for Enhanced Button Styling ------
st.markdown("""
<style>
/* Active button styling */
.active-button {
    background-color: #ff6b6b !important;
    color: white !important;
    border: 2px solid #ff6b6b !important;
}

/* Custom button container */
.button-container {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
}

/* Sidebar styling */
.sidebar .sidebar-content {
    background-color: #f0f2f6;
}

/* Hide default values in number inputs */
.stNumberInput > div > div > input[value] {
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 7: Application Title and Sidebar Parameters
# -----------------------------------------------------------------------------

st.title("Personalized Nutrition Tracker 🥗")
st.markdown("""
Welcome to your personalized nutrition tracking application! This tool calculates
your individual daily nutritional targets based on your personal parameters and
helps you track your food intake for healthy weight gain.
""")

# ------ Sidebar for Personal Parameters ------
st.sidebar.header("Personal Parameters 📊")

# ------ Personal Information Inputs with Placeholder Text ------
age = st.sidebar.number_input(
    "Age (years)",
    min_value=16,
    max_value=80,
    value=st.session_state.user_age,
    placeholder="Enter your age"
)

height_cm = st.sidebar.number_input(
    "Height (cm)",
    min_value=140,
    max_value=220,
    value=st.session_state.user_height,
    placeholder="Enter your height in cm"
)

weight_kg = st.sidebar.number_input(
    "Weight (kg)",
    min_value=40.0,
    max_value=150.0,
    value=st.session_state.user_weight,
    step=0.5,
    placeholder="Enter your weight in kg"
)

gender_options = ["Select Gender", "Male", "Female"]
gender_index = 0
if st.session_state.user_gender:
    try:
        gender_index = gender_options.index(st.session_state.user_gender)
    except ValueError:
        gender_index = 0

gender = st.sidebar.selectbox("Gender", gender_options, index=gender_index)

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

# ------ Update Session State ------
st.session_state.user_age = age
st.session_state.user_height = height_cm
st.session_state.user_weight = weight_kg
st.session_state.user_gender = gender
st.session_state.user_activity = activity_level

# ------ Advanced Parameters Collapsible Section ------
with st.sidebar.expander("Advanced Settings ⚙️"):
    caloric_surplus = st.number_input(
        "Caloric Surplus (kcal/day)",
        min_value=200, max_value=800,
        value=None,
        placeholder=f"Default: {DEFAULT_CALORIC_SURPLUS}",
        step=50,
        help="Additional calories above maintenance for weight gain"
    )
    
    protein_per_kg = st.number_input(
        "Protein (g/kg body weight)",
        min_value=1.2, max_value=3.0,
        value=None,
        placeholder=f"Default: {DEFAULT_PROTEIN_PER_KG}",
        step=0.1,
        help="Protein intake per kg of body weight"
    )
    
    fat_percentage_input = st.number_input(
        "Fat (% of total calories)",
        min_value=15, max_value=40,
        value=None,
        placeholder=f"Default: {int(DEFAULT_FAT_PERCENTAGE * 100)}",
        step=1,
        help="Percentage of total calories from fat"
    )

# ------ Use Default Values if User Has Not Entered Custom Values ------
final_age = age if age is not None else DEFAULT_AGE
final_height = height_cm if height_cm is not None else DEFAULT_HEIGHT_CM
final_weight = weight_kg if weight_kg is not None else DEFAULT_WEIGHT_KG
final_gender = gender if gender != "Select Gender" else DEFAULT_GENDER
final_activity = (activity_level if activity_level is not None
                  else DEFAULT_ACTIVITY_LEVEL)
final_surplus = (caloric_surplus if caloric_surplus is not None
                 else DEFAULT_CALORIC_SURPLUS)
final_protein = (protein_per_kg if protein_per_kg is not None
                 else DEFAULT_PROTEIN_PER_KG)
final_fat_percent = ((fat_percentage_input / 100)
                     if fat_percentage_input is not None
                     else DEFAULT_FAT_PERCENTAGE)

# ------ Check if User Has Entered All Required Information ------
user_has_entered_info = (
    age is not None and
    height_cm is not None and
    weight_kg is not None and
    gender != "Select Gender" and
    activity_level is not None
)

# ------ Calculate Personalized Targets ------
targets = calculate_personalized_targets(
    age=final_age,
    height_cm=final_height,
    weight_kg=final_weight,
    gender=final_gender.lower(),
    activity_level=final_activity,
    caloric_surplus=final_surplus,
    protein_per_kg=final_protein,
    fat_percentage=final_fat_percent
)

# -----------------------------------------------------------------------------
# Cell 8: Display Personalized Targets (FIXED ALIGNMENT)
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to see your personalized nutritional targets.")
    st.header("Sample Daily Targets 🎯")
    st.caption("*These are example targets. Enter your information in the sidebar for personalized calculations.*")
else:
    st.header("Your Personalized Daily Targets 🎯")

# ------ Display ALL Metrics in Consistent 4-Column Grid ------
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("BMR (Basal Metabolic Rate)", f"{targets['bmr']} kcal")
with col2:
    st.metric("TDEE (Maintenance)", f"{targets['tdee']} kcal")
with col3:
    st.metric("Target Weight Gain", f"{targets['target_weight_gain_per_week']} kg/week")
with col4:
    st.metric("Daily Calorie Target", f"{targets['total_calories']} kcal")

# ------ Display Daily Nutritional Targets in Same 4-Column Grid ------
st.subheader("Daily Nutritional Targets")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Protein", f"{targets['protein_grams']}g", f"{targets['protein_calories']} kcal")
with col2:
    st.metric("Carbohydrates", f"{targets['carb_grams']}g", f"{targets['carb_calories']} kcal")
with col3:
    st.metric("Fat", f"{targets['fat_grams']}g", f"{targets['fat_calories']} kcal")
with col4:
    # Calculate and display fiber target (recommended 14g per 1000 kcal)
    fiber_target = round((targets['total_calories'] / 1000) * 14)
    st.metric("Fiber Target", f"{fiber_target}g", "Daily recommendation")

# ------ Display Macronutrient Percentages in Same 4-Column Grid ------
st.subheader("Macronutrient Distribution")
protein_percent = (targets['protein_calories'] / targets['total_calories']) * 100
carb_percent = (targets['carb_calories'] / targets['total_calories']) * 100
fat_percent_display = (targets['fat_calories'] / targets['total_calories']) * 100

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Protein", f"{protein_percent:.1f}%", "of total calories")
with col2:
    st.metric("Carbohydrates", f"{carb_percent:.1f}%", "of total calories")
with col3:
    st.metric("Fat", f"{fat_percent_display:.1f}%", "of total calories")
with col4:
    # Calculate calories per gram for reference
    cal_per_gram = targets['total_calories'] / (targets['protein_grams'] + targets['carb_grams'] + targets['fat_grams'])
    st.metric("Energy Density", f"{cal_per_gram:.1f}", "kcal/g average")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 9: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Your Foods 📝")
st.markdown("Choose foods using the buttons (0-5 servings) or enter custom serving amounts.")

# ------ Create Category Tabs for Food Organization ------
available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    with tabs[i]:
        # ------ Create Columns for Better Layout ------
        for j in range(0, len(items), 2):
            col1, col2 = st.columns(2)

            # ------ First Item in Row ------
            if j < len(items):
                with col1:
                    food = items[j]
                    st.subheader(food['name'])

                    key = f"{category}_{food['name']}"
                    current_serving = st.session_state.food_selections.get(food['name'], 0.0)

                    button_cols = st.columns(5)
                    for k in range(1, 6):
                        with button_cols[k-1]:
                            button_type = "primary" if current_serving == float(k) else "secondary"
                            if st.button(f"{k}×", key=f"{key}_{k}", type=button_type):
                                st.session_state.food_selections[food['name']] = float(k)
                                st.rerun()

                    custom_serving = st.number_input(
                        "Custom servings:",
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
                    
                    st.caption(f"Per Serving: {food['calories']} kcal | "
                               f"{food['protein']}g Protein | "
                               f"{food['carbs']}g Carbs | "
                               f"{food['fat']}g Fat")

            # ------ Second Item in Row ------
            if j + 1 < len(items):
                with col2:
                    food = items[j + 1]
                    st.subheader(food['name'])
                    
                    key = f"{category}_{food['name']}"
                    current_serving = st.session_state.food_selections.get(food['name'], 0.0)

                    button_cols = st.columns(5)
                    for k in range(1, 6):
                        with button_cols[k-1]:
                            button_type = "primary" if current_serving == float(k) else "secondary"
                            if st.button(f"{k}×", key=f"{key}_{k}", type=button_type):
                                st.session_state.food_selections[food['name']] = float(k)
                                st.rerun()

                    custom_serving = st.number_input(
                        "Custom servings:",
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

                    st.caption(f"Per Serving: {food['calories']} kcal | "
                               f"{food['protein']}g Protein | "
                               f"{food['carbs']}g Carbs | "
                               f"{food['fat']}g Fat")

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

    st.header("Daily Intake Summary 📊")

    if selected_foods:
        st.subheader("Foods Consumed Today 🍽️")
        cols = st.columns(3)
        for i, item in enumerate(selected_foods):
            with cols[i % 3]:
                st.write(f"• {item['food']['name']} x {item['servings']:.1f}")
    else:
        st.info("No foods selected 🥗")

    st.subheader("Total Nutritional Intake 📈")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calories", f"{total_calories:.0f} kcal")
    with col2:
        st.metric("Protein", f"{total_protein:.1f}g")
    with col3:
        st.metric("Carbohydrates", f"{total_carbs:.1f}g")
    with col4:
        st.metric("Fat", f"{total_fat:.1f}g")

    st.subheader("Target Achievement 🎯")

    # ------ Calculate Percentages Based on Personalized Targets ------
    cal_percent = (min(total_calories / targets['total_calories'] * 100, 100)
                   if targets['total_calories'] > 0 else 0)
    st.progress(cal_percent / 100,
                text=f"Calories: {cal_percent:.0f}% of target ({targets['total_calories']} kcal)")

    prot_percent = (min(total_protein / targets['protein_grams'] * 100, 100)
                    if targets['protein_grams'] > 0 else 0)
    st.progress(prot_percent / 100,
                text=f"Protein: {prot_percent:.0f}% of target ({targets['protein_grams']}g)")

    carb_percent = (min(total_carbs / targets['carb_grams'] * 100, 100)
                    if targets['carb_grams'] > 0 else 0)
    st.progress(carb_percent / 100,
                text=f"Carbohydrates: {carb_percent:.0f}% of target ({targets['carb_grams']}g)")

    fat_percent_progress = (min(total_fat / targets['fat_grams'] * 100, 100)
                            if targets['fat_grams'] > 0 else 0)
    st.progress(fat_percent_progress / 100,
                text=f"Fat: {fat_percent_progress:.0f}% of target ({targets['fat_grams']}g)")

    st.subheader("Personalized Recommendations 💡")
    recommendations = []
    
    if total_calories < targets['total_calories']:
        deficit = targets['total_calories'] - total_calories
        recommendations.append(f"• You need {deficit:.0f} more calories to reach your weight gain target")
    
    if total_protein < targets['protein_grams']:
        deficit = targets['protein_grams'] - total_protein
        recommendations.append(f"• You need {deficit:.0f}g more protein for muscle building")
    
    if total_carbs < targets['carb_grams']:
        deficit = targets['carb_grams'] - total_carbs
        recommendations.append(f"• You need {deficit:.0f}g more carbohydrates for energy and performance")
    
    if total_fat < targets['fat_grams']:
        deficit = targets['fat_grams'] - total_fat
        recommendations.append(f"• You need {deficit:.0f}g more healthy fats for hormone production")

    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("Outstanding work! You have met all your personalized daily nutritional targets 🎉")

    # ------ Show Surplus and Deficit Information ------
    st.subheader("Daily Balance Summary ⚖️")
    cal_balance = total_calories - targets['tdee']
    if cal_balance > 0:
        st.info(f"📈 You are consuming {cal_balance:.0f} calories above maintenance (surplus for weight gain)")
    else:
        st.warning(f"📉 You are consuming {abs(cal_balance):.0f} calories below maintenance")

    if selected_foods:
        st.subheader("Detailed Food Log 📋")
        food_log = [
            {
                'Food Item Name': item['food']['name'],
                'Number of Servings': f"{item['servings']:.1f}",
                'Total Calories': item['food']['calories'] * item['servings'],
                'Total Protein Grams': item['food']['protein'] * item['servings'],
                'Total Carbohydrate Grams': item['food']['carbs'] * item['servings'],
                'Total Fat Grams': item['food']['fat'] * item['servings']
            }
            for item in selected_foods
        ]
        df_log = pd.DataFrame(food_log)
        st.dataframe(df_log.style.format({
            'Total Calories': '{:.0f}',
            'Total Protein Grams': '{:.1f}',
            'Total Carbohydrate Grams': '{:.1f}',
            'Total Fat Grams': '{:.1f}'
        }), use_container_width=True)

    st.markdown("---")
    print("Daily nutritional intake calculation completed successfully 📊")

# -----------------------------------------------------------------------------
# Cell 11: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()
    print("All food selections have been cleared and reset 🔄")

# -----------------------------------------------------------------------------
# Cell 12: Footer Information and Application Documentation
# -----------------------------------------------------------------------------

st.sidebar.markdown("---")
st.sidebar.markdown("### About This Calculator 📖")
st.sidebar.markdown("""
**Calculations are based on the following methods:**
- **BMR**: Mifflin-St Jeor equation for metabolic rate
- **Protein**: 2.0g per kg body weight for muscle building
- **Fat**: 25% of total calories for hormone production
- **Carbohydrates**: Remaining calories after protein and fat allocation
- **Weight gain target**: 0.25% of body weight per week for lean gains
""")

st.sidebar.markdown("### Activity Level Guide 🏃‍♂️")
st.sidebar.markdown("""
- **Sedentary**: Little to no exercise or desk job
- **Lightly Active**: Light exercise or sports 1-3 days per week
- **Moderately Active**: Moderate exercise or sports 3-5 days per week
- **Very Active**: Hard exercise or sports 6-7 days per week
- **Extremely Active**: Very hard exercise, physical job, or training twice daily
""")

print("Thank you for using the Personalized Nutrition Tracker! Eat well, feel well! 🌱")
