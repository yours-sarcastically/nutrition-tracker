# -----------------------------------------------------------------------------
# Personalized Evidence-Based Fitness & Nutrition Tracker (v3)
# -----------------------------------------------------------------------------

"""
This script implements a comprehensive, interactive nutrition and fitness tracking 
application. It is designed to guide users through weight loss, maintenance, or 
gain phases based on the most current, high-validity scientific evidence.

---------------------------------
CORE SCIENTIFIC PRINCIPLES APPLIED
---------------------------------
1.  BMR Calculation (Highest Validity): 
    - Utilizes the Mifflin-St Jeor equation, widely recognized as the most 
      accurate formula for estimating Basal Metabolic Rate in healthy adults.

2.  Energy Balance (Highest Validity):
    - The entire framework is built on the principle of energy balance (calories 
      in vs. calories out), the fundamental driver of weight change.

3.  Percentage-Based Caloric Targets (High Validity):
    - Caloric deficits/surpluses are calculated as a percentage of TDEE (e.g., 
      -20% for fat loss, +10% for muscle gain). This is superior to a fixed 
      number as it scales the diet's intensity to the individual's metabolic 
      rate, enhancing sustainability and safety.

4.  Optimized & Goal-Specific Protein Intake (Highest & High Validity):
    - Protein targets are automatically set based on the user's goal, aligning 
      with consensus recommendations from leading sports nutrition bodies (ISSN, ACSM).
      - Fat Loss: 1.8 g/kg to maximize lean muscle preservation.
      - Muscle Gain: 2.0 g/kg to maximize muscle protein synthesis.
      - Maintenance: 1.6 g/kg to support recovery for active individuals.

5.  Hormone-Supporting Fat Intake (High Validity):
    - Fat intake is maintained at or above 20-25% of total calories to support
      the endocrine system and production of essential hormones like testosterone.

6.  Integration of Resistance Training (Highest Validity):
    - The application explicitly documents that nutritional strategies are most
      effective when paired with a consistent resistance training program, which
      provides the necessary stimulus for muscle preservation and growth.

7.  Dynamic Monitoring & Adjustment (High Validity):
    - The tool calculates and displays the estimated weekly rate of weight change,
      prompting users to track their real-world progress and adjust their plan
      as their TDEE adapts over time.
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
    page_title="Evidence-Based Fitness & Nutrition Tracker",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cell 3: Unified Configuration & Scientific Constants
# -----------------------------------------------------------------------------

# ------ Activity Level Multipliers for TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Goal-Specific Scientific Presets (Highest & High Validity) ------
# These presets define the evidence-based starting points for calories and protein.
GOAL_PRESETS = {
    "Weight Loss": {
        "caloric_adjustment_percent": -0.20,  # A 20% deficit is effective and sustainable.
        "protein_per_kg": 1.8,               # Higher protein to preserve muscle mass.
        "fat_percent": 0.25,                 # Ensures hormonal health.
        "purpose_text": "to maximize fat loss while preserving muscle",
        "rate_of_change_label": "Est. Weekly Fat Loss"
    },
    "Weight Maintenance": {
        "caloric_adjustment_percent": 0.0,
        "protein_per_kg": 1.6,               # Sufficient protein for active individuals.
        "fat_percent": 0.30,
        "purpose_text": "to maintain your current weight and performance",
        "rate_of_change_label": "Est. Weekly Weight Change"
    },
    "Weight Gain": {
        "caloric_adjustment_percent": 0.10,  # A conservative 10% surplus minimizes fat gain.
        "protein_per_kg": 2.0,               # Higher protein to fuel muscle growth.
        "fat_percent": 0.25,
        "purpose_text": "to maximize lean muscle gain",
        "rate_of_change_label": "Est. Weekly Weight Gain"
    }
}

# ------ General Application Configuration ------
CONFIG = {
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    'form_fields': {
        'goal': {'type': 'selectbox', 'label': 'Primary Goal', 'options': ["Weight Loss", "Weight Maintenance", "Weight Gain"], 'required': True},
        'age': {'type': 'number', 'label': 'Age (Years)', 'min': 16, 'max': 80, 'step': 1, 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (cm)', 'min': 140, 'max': 220, 'step': 1, 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (kg)', 'min': 40.0, 'max': 150.0, 'step': 0.5, 'required': True},
        'sex': {'type': 'selectbox', 'label': 'Sex', 'options': ["Male", "Female"], 'required': True},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level', 'options': [
            ("Sedentary", "sedentary"), ("Lightly Active", "lightly_active"),
            ("Moderately Active", "moderately_active"), ("Very Active", "very_active"), 
            ("Extremely Active", "extremely_active")
        ], 'required': True},
    }
}

# -----------------------------------------------------------------------------
# Cell 4: Core Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex):
    """
    (Highest Validity) Calculates Basal Metabolic Rate using the Mifflin-St Jeor Equation.
    This is the most accurate predictive equation for healthy adults.
    """
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """
    (Highest Validity) Calculates Total Daily Energy Expenditure (TDEE).
    TDEE = BMR * Activity Multiplier. This estimates your daily maintenance calories.
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_personalized_targets(age, height_cm, weight_kg, sex, activity_level, goal):
    """
    (Highest & High Validity) Calculates all nutritional targets based on goal.
    This function integrates multiple high-validity principles:
    - Uses Mifflin-St Jeor for BMR.
    - Applies a percentage-based caloric adjustment for the selected goal.
    - Sets protein and fat based on evidence-based presets for that goal.
    - Calculates carbohydrates as the remaining caloric balance.
    """
    presets = GOAL_PRESETS.get(goal, GOAL_PRESETS["Weight Maintenance"])

    # 1. Calculate BMR and TDEE (Core Energy Balance)
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)

    # 2. Apply Percentage-Based Caloric Adjustment
    caloric_adjustment = tdee * presets["caloric_adjustment_percent"]
    total_calories = tdee + caloric_adjustment

    # 3. Set Macronutrients based on Goal-Specific Presets
    protein_g = presets["protein_per_kg"] * weight_kg
    protein_calories = protein_g * 4

    fat_calories = total_calories * presets["fat_percent"]
    fat_g = fat_calories / 9

    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    # 4. Estimate Weekly Weight Change (1 kg body fat ≈ 7700 kcal)
    # This encourages dynamic monitoring and adjustment.
    weekly_cal_change = caloric_adjustment * 7
    target_weight_change_kg = weekly_cal_change / 7700

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'fat_g': round(fat_g), 'carb_g': round(carb_g),
        'protein_calories': round(protein_calories), 'fat_calories': round(fat_calories), 'carb_calories': round(carb_calories),
        'target_weight_change_per_week': round(target_weight_change_kg, 2)
    }
    
    # Calculate macronutrient percentages for display
    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
    else:
        targets['protein_percent'] = targets['carb_percent'] = targets['fat_percent'] = 0
        
    return targets

# -----------------------------------------------------------------------------
# Cell 5: Food Database and UI Helper Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """Loads the food database from a CSV file."""
    df = pd.read_csv(file_path)
    # Convert to a dictionary of lists for easier processing
    foods = {cat: group.to_dict('records') for cat, group in df.groupby('category')}
    for category in foods:
        for food in foods[category]:
            food['name'] = f"{food['name']} ({food['serving_unit']})"
    return foods

def calculate_daily_totals(food_selections, foods):
    """Calculates total nutrition from selected foods."""
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

def display_metrics_grid(metrics_data, num_columns=4):
    """Displays metrics in a clean, configurable grid layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                st.metric(label=metric_info[0], value=metric_info[1])
            elif len(metric_info) == 3:
                st.metric(label=metric_info[0], value=metric_info[1], delta=metric_info[2])

# -----------------------------------------------------------------------------
# Cell 6: Main Application UI and Logic
# -----------------------------------------------------------------------------

# --- Page Title and Introduction ---
st.title("🔬 Evidence-Based Fitness & Nutrition Tracker")
st.markdown("""
Welcome! This tool translates complex nutritional science into a simple, actionable plan. 
Enter your details in the sidebar to receive personalized targets for calories and macronutrients, 
all calculated using scientifically-validated methods to help you achieve your fitness goals effectively.
""")

# --- Sidebar for User Inputs ---
st.sidebar.header("Step 1: Your Personal Parameters")

user_inputs = {}
for field_name, field_config in CONFIG['form_fields'].items():
    if field_config['type'] == 'selectbox':
        if field_name == 'activity_level':
            # Handle tuple format for activity level options
            options = field_config['options']
            user_inputs[field_name] = st.sidebar.selectbox(field_config['label'], options, format_func=lambda x: x[0])[1]
        else:
            options = field_config['options']
            user_inputs[field_name] = st.sidebar.selectbox(field_config['label'], options)
    elif field_config['type'] == 'number':
        user_inputs[field_name] = st.sidebar.number_input(
            field_config['label'], min_value=field_config['min'], 
            max_value=field_config['max'], step=field_config['step']
        )

# --- Calculate Targets Based on Input ---
targets = calculate_personalized_targets(**user_inputs)

# --- Main Content Area: Display Targets and Documentation ---
st.header(f"Step 2: Your Personalized Daily Targets for {user_inputs['goal']}")

# --- Section 1: Core Metabolic & Caloric Targets with Documentation ---
with st.expander("Your Metabolic & Caloric Targets Explained", expanded=True):
    st.markdown("""
    Your body's energy needs are the foundation of your nutrition plan. We start by calculating your **Basal Metabolic Rate (BMR)**—the energy you burn at complete rest—using the highly accurate **Mifflin-St Jeor equation**. We then multiply this by an activity factor to find your **Total Daily Energy Expenditure (TDEE)**, which is your estimated daily maintenance calorie level.

    Your final **Daily Calorie Target** is adjusted from your TDEE based on your goal:
    - **Weight Loss:** A **20% caloric deficit** is applied. This is a sustainable rate that promotes fat loss while minimizing muscle loss and metabolic slowdown.
    - **Weight Gain:** A conservative **10% caloric surplus** is applied. This provides enough energy to build muscle while minimizing unnecessary fat gain.
    - **Maintenance:** Your target is set equal to your TDEE.
    """)
    
    weight_change_label = GOAL_PRESETS[user_inputs['goal']]['rate_of_change_label']
    weight_change_val = f"{targets['target_weight_change_per_week']:+.2f} kg" if user_inputs['goal'] != 'Weight Maintenance' else f"~0.00 kg"

    metrics_data_calories = [
        ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal/day"),
        ("Maintenance Calories (TDEE)", f"{targets['tdee']} kcal/day"),
        ("Daily Calorie Target", f"{targets['total_calories']} kcal"),
        (weight_change_label, weight_change_val)
    ]
    display_metrics_grid(metrics_data_calories, 4)

# --- Section 2: Macronutrient Targets with Documentation ---
with st.expander("Your Macronutrient Targets Explained", expanded=True):
    st.markdown(f"""
    While calories determine the *quantity* of weight change, macronutrients determine the *quality* (muscle vs. fat). Your targets are set based on the most robust scientific evidence for your goal of **{user_inputs['goal']}**.

    - **Protein (The Builder):** Your target of **{targets['protein_g']}g** is based on **{GOAL_PRESETS[user_inputs['goal']]['protein_per_kg']}g per kg** of body weight. This elevated intake is crucial for preserving muscle during a deficit or building new muscle in a surplus. It is the most important macronutrient for body composition.
    
    - **Fat (The Regulator):** Your target of **{targets['fat_g']}g** ensures you consume at least **{GOAL_PRESETS[user_inputs['goal']]['fat_percent']:.0%}** of your calories from fat. This is vital for supporting hormone production and overall health.
    
    - **Carbohydrates (The Fuel):** Your target of **{targets['carb_g']}g** is calculated from the calories remaining after protein and fat are accounted for. Carbohydrates are your body's primary energy source, fueling your workouts and replenishing glycogen stores.
    """)
    
    metrics_data_macros = [
        ("Protein Target", f"{targets['protein_g']} g", f"{targets['protein_percent']:.1f}% of calories"),
        ("Fat Target", f"{targets['fat_g']} g", f"{targets['fat_percent']:.1f}% of calories"),
        ("Carbohydrate Target", f"{targets['carb_g']} g", f"{targets['carb_percent']:.1f}% of calories")
    ]
    display_metrics_grid(metrics_data_macros, 3)

st.markdown("---")

# --- Section 3: The Indispensable Role of Resistance Training ---
with st.expander("CRITICAL: The Role of Physical Fitness 🏋️‍♀️", expanded=False):
    st.markdown("""
    **(Highest Validity)** A nutrition plan provides the building materials, but **resistance training provides the architectural blueprint.** It is the single most important stimulus that tells your body how to use the nutrients you consume.

    - **During Fat Loss:** Resistance training signals your body to **preserve muscle tissue**. Without it, a significant portion of weight lost will come from metabolically active muscle, slowing your metabolism and making weight regain more likely.
    - **During Muscle Gain:** Resistance training is the **non-negotiable trigger for muscle growth**. A caloric surplus without a strong training stimulus will primarily result in fat gain.

    #### Minimum Effective Dose (ACSM Guidelines)
    To get the best results from this nutrition plan, aim for the following:
    - **Resistance Training:** Train each major muscle group **2-3 times per week**.
    - **Cardiovascular Exercise:** Accumulate **150-300 minutes** of moderate-intensity activity per week to support heart health and increase energy expenditure.
    """)

st.markdown("---")

# --- Section 4: Interactive Food Logging ---
st.header("Step 3: Log Your Daily Food Intake")

# Load food database
foods = load_food_database('nutrition_results.csv')
if 'food_selections' not in st.session_state:
    st.session_state.food_selections = {}

# Create tabs for each food category
available_categories = list(foods.keys())
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    with tabs[i]:
        # Display food items in a grid
        items = foods[category]
        cols = st.columns(2)
        for j, food in enumerate(items):
            with cols[j % 2]:
                st.subheader(food['name'])
                st.caption(f"Per Serving: {food['calories']} kcal | {food['protein']}g P | {food['carbs']}g C | {food['fat']}g F")
                
                # Input for number of servings
                servings = st.number_input(f"Servings:", min_value=0.0, max_value=10.0, step=0.5, key=f"{category}_{food['name']}")
                if servings > 0:
                    st.session_state.food_selections[food['name']] = servings
                elif food['name'] in st.session_state.food_selections:
                    # Remove from selections if servings are set to 0
                    del st.session_state.food_selections[food['name']]

st.markdown("---")

# --- Section 5: Results and Analysis ---
if st.button("Calculate & Analyze My Intake", type="primary", use_container_width=True):
    totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)
    
    st.header("Your Daily Intake Analysis")

    if not selected_foods:
        st.warning("You haven't logged any foods yet. Please select your servings above and click the button again.")
    else:
        # Display total intake
        st.subheader("Total Nutritional Intake")
        intake_metrics = [
            (f"Total {config['label']}", f"{totals[nutrient]:.0f} {config['unit']}")
            for nutrient, config in CONFIG['nutrient_configs'].items()
        ]
        display_metrics_grid(intake_metrics, 4)

        # Display progress towards targets
        st.subheader("Progress Toward Daily Targets")
        for nutrient, config in CONFIG['nutrient_configs'].items():
            actual = totals[nutrient]
            target = targets[config['target_key']]
            percent = min(actual / target * 100, 100) if target > 0 else 0
            st.progress(
                percent / 100,
                text=f"{config['label']}: {actual:.0f} / {target:.0f} {config['unit']} ({percent:.0f}%)"
            )

        # Display caloric balance analysis
        st.subheader("Caloric Balance Summary")
        cal_balance = totals['calories'] - targets['tdee']
        if cal_balance > 0:
            st.info(f"📈 You consumed **{cal_balance:.0f} kcal above** your maintenance level (TDEE).")
        elif cal_balance < 0:
            st.warning(f"📉 You consumed **{abs(cal_balance):.0f} kcal below** your maintenance level (TDEE).")
        else:
            st.success("✅ You consumed at your exact maintenance level (TDEE).")
        st.caption(f"This intake aligns with your goal of **{user_inputs['goal']}**.")

# --- Footer and Clear Button ---
if st.button("Clear All Food Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()
