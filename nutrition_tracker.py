# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive, evidence-based nutrition tracking application. It now supports multiple user goals (weight loss, maintenance, or weight gain) and provides personalized daily targets for calories and macronutrients. The calculations are founded on the highest-validity scientific principles: the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR), TDEE calculation via activity multipliers, and goal-specific caloric and macronutrient targets based on peer-reviewed research. This ensures the recommendations are safe, effective, and tailored to the individual's metabolic needs and fitness objectives.
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
# Cell 3: Unified Configuration Constants
# -----------------------------------------------------------------------------

# ------ MODIFIED: Default parameters updated for multi-goal functionality ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 75, # Adjusted to a more average starting weight
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': "Weight Maintenance" # New default parameter for user goal
}

# ------ Activity Level Multipliers for TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Unified Configuration for All App Components ------
CONFIG = {
    # MODIFIED: Removed '🥇' and '🥦' from emoji ranking order
    'emoji_order': {'🥇': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
        # REMOVED: 'PRIMARY MICRONUTRIENT SOURCES' no longer needs a special mapping for ranking
    },
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    # MODIFIED: Form fields updated for multi-goal functionality.
    # Caloric surplus, protein per kg, and fat percentage fields are removed as they are now calculated automatically based on the user's goal.
    'form_fields': {
        'goal': {'type': 'selectbox', 'label': 'Primary Goal', 'options': ["Select Goal", "Weight Loss", "Weight Maintenance", "Weight Gain"], 'required': True, 'placeholder': "Select Goal"},
        'age': {'type': 'number', 'label': 'Age (Years)', 'min': 16, 'max': 80, 'step': 1, 'placeholder': 'Enter your age', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (Centimeters)', 'min': 140, 'max': 220, 'step': 1, 'placeholder': 'Enter your height', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (kg)', 'min': 40.0, 'max': 150.0, 'step': 0.5, 'placeholder': 'Enter your weight', 'required': True},
        'sex': {'type': 'selectbox', 'label': 'Sex', 'options': ["Select Sex", "Male", "Female"], 'required': True, 'placeholder': "Select Sex"},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level', 'options': [
            ("Select Activity Level", None),
            ("Sedentary", "sedentary"),
            ("Lightly Active", "lightly_active"),
            ("Moderately Active", "moderately_active"),
            ("Very Active", "very_active"),
            ("Extremely Active", "extremely_active")
        ], 'required': True, 'placeholder': None},
    }
}

# -----------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables using unified approach"""
    session_vars = ['food_selections'] + [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else None

def create_unified_input(field_name, field_config, container=st.sidebar):
    """Create input widgets using unified configuration, now handling advanced fields."""
    session_key = f'user_{field_name}'
    
    if field_config['type'] == 'number':
        # Dynamically create placeholder for advanced fields
        if field_config.get('advanced'):
            default_val = DEFAULTS.get(field_name, 0)
            # Handle percentage display for fat
            display_val = int(default_val * 100) if field_name == 'fat_percentage' else default_val
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
        if field_name == 'activity_level':
            index = next((i for i, (_, val) in enumerate(field_config['options']) if val == current_value), 0)
            selection = container.selectbox(field_config['label'], field_config['options'], index=index, format_func=lambda x: x[0])
            value = selection[1]
        else:
            index = field_config['options'].index(current_value) if current_value in field_config['options'] else 0
            value = container.selectbox(field_config['label'], field_config['options'], index=index)
    
    st.session_state[session_key] = value
    return value

def get_final_values(user_inputs):
    """Process all user inputs and apply defaults using unified approach"""
    final_values = {}
    
    for field, value in user_inputs.items():
        if field == 'sex' or field == 'goal':
            final_values[field] = value if value and "Select" not in value else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None else DEFAULTS[field]
    
    return final_values

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
    
    # MODIFIED: Purpose map is now dynamic based on the user's goal
    goal_purpose = {
        'Weight Loss': 'to reach your weight loss target',
        'Weight Maintenance': 'to maintain your current weight',
        'Weight Gain': 'to reach your weight gain target'
    }
    
    purpose_map = {
        'calories': goal_purpose.get(targets.get('goal'), 'for your energy needs'),
        'protein': 'for muscle preservation and growth',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and health'
    }
    
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        
        percent = min(actual / target * 100, 100) if target > 0 else 0
        st.progress(
            percent / 100,
            text=f"{config['label']}: {percent:.0f}% of daily target ({target:.0f} {config['unit']})"
        )
        
        if actual < target:
            deficit = target - actual
            purpose = purpose_map.get(nutrient, 'for optimal nutrition')
            recommendations.append(f"• You need {deficit:.0f} more {config['unit']} of {config['label'].lower()} {purpose}.")
    
    return recommendations

def calculate_daily_totals(food_selections, foods):
    """Calculate total daily nutrition from food selections"""
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

# -----------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """
    Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation.
    This equation is considered by the Academy of Nutrition and Dietetics to be the most accurate for predicting BMR in healthy adults.
    """
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure Based on Activity Level.
    TDEE represents the total calories burned in a day, accounting for both resting metabolism and physical activity.
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

# MODIFIED: This function is completely overhauled to implement the evidence-based, multi-goal strategy.
def calculate_personalized_targets(age, height_cm, weight_kg, sex, activity_level, goal):
    """
    Calculate Personalized Daily Nutritional Targets based on Evidence-Based Principles.
    This function dynamically sets caloric and macronutrient goals based on the user's primary objective (loss, maintenance, gain),
    ensuring a scientifically-sound approach to nutrition planning.
    """
    # Principle 1 & 2: Calculate BMR (Mifflin-St Jeor) and TDEE
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)

    # Principle 3 & 4: Define goal-specific parameters for calories, protein, and fat
    goal_params = {
        'Weight Loss':      {'adjustment': -0.20, 'protein_kg': 1.8, 'fat_pct': 0.25},
        'Weight Maintenance': {'adjustment':  0.00, 'protein_kg': 1.6, 'fat_pct': 0.30},
        'Weight Gain':      {'adjustment': +0.10, 'protein_kg': 2.0, 'fat_pct': 0.25}
    }
    params = goal_params.get(goal, goal_params['Weight Maintenance'])

    # Calculate Goal-Specific Target Calories
    daily_caloric_adjustment = tdee * params['adjustment']
    total_calories = tdee + daily_caloric_adjustment

    # Calculate Macronutrients based on a "Protein-First" strategy
    protein_g = params['protein_kg'] * weight_kg
    protein_calories = protein_g * 4
    
    fat_calories = total_calories * params['fat_pct']
    fat_g = fat_calories / 9

    # Carbohydrates fill the remaining caloric budget
    carb_calories = max(0, total_calories - protein_calories - fat_calories)
    carb_g = carb_calories / 4

    # Principle 5: Estimate the rate of weight change based on the caloric surplus/deficit
    # Based on the approximation that 1 kg of body mass change requires a 7700 kcal adjustment.
    est_weekly_change_kg = (daily_caloric_adjustment * 7) / 7700
    
    targets = {
        'goal': goal, 'bmr': round(bmr), 'tdee': round(tdee), 
        'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'est_weekly_change_kg': round(est_weekly_change_kg, 2),
        'daily_caloric_adjustment': round(daily_caloric_adjustment)
    }

    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
    else:
        targets['protein_percent'] = targets['carb_percent'] = targets['fat_percent'] = 0
        
    return targets

# -----------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """Load the Vegetarian Food Database From a CSV File"""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in df['category'].unique()} # Use unique categories from CSV

    for _, row in df.iterrows():
        category = row['category']
        if category in foods:
            foods[category].append({
                'name': f"{row['name']} ({row['serving_unit']})",
                'calories': row['calories'], 'protein': row['protein'],
                'carbs': row['carbs'], 'fat': row['fat']
            })
    return foods

# MODIFIED: assign_food_emojis function updated to remove flawed logic
def assign_food_emojis(foods):
    """Assign emojis to foods using a unified ranking system."""
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'calories': {}}
    
    # Identify top performers in each category
    for category, items in foods.items():
        if not items: continue
            
        # Rank top 3 most calorie-dense foods within each category
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:3]]
        
        # Rank top 3 foods by their primary macronutrient (if applicable)
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(items, key=lambda x: x[map_info['sort_by']], reverse=True)
            top_foods[map_info['key']] = [food['name'] for food in sorted_by_nutrient[:3]]

    # Create a set of all foods that are top nutrient performers
    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]}

    # Define the emoji mapping
    emoji_mapping = {'high_cal_nutrient': '🥇', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑'}
    
    # Assign emojis based on the rankings
    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])
            
            # REMOVED: Superfood logic ('🥇') as it was non-functional
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
            # REMOVED: Micronutrient ranking ('🥦') as it was misleading
            else:
                food['emoji'] = ''
    return foods


def render_food_item(food, category):
    """Render a single food item with unified interaction controls"""
    st.subheader(f"{food.get('emoji', '')} {food['name']}")
    key = f"{category}_{food['name']}"
    current_serving = st.session_state.food_selections.get(food['name'], 0.0)
    
    button_cols = st.columns(5)
    for k in range(1, 6):
        with button_cols[k - 1]:
            button_type = "primary" if current_serving == float(k) else "secondary"
            if st.button(f"{k}", key=f"{key}_{k}", type=button_type, help=f"Set to {k} servings"):
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

def render_food_grid(items, category, columns=2):
    """Render food items in a grid layout"""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)

# -----------------------------------------------------------------------------
# Cell 7: Initialize Application
# -----------------------------------------------------------------------------

# Initialize session state
initialize_session_state()

# Load food database and assign emojis
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# Custom CSS for enhanced styling
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
.stButton>button[kind="primary"] { background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b; }
.stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
This tracker generates personalized daily nutrition targets based on your unique goals and physiology. Input your details in the sidebar, and let's optimize your nutrition! 🚀
""")

# ------ MODIFIED: Sidebar inputs streamlined for new goal-oriented approach ------
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Render all input fields, which are now standard fields
for field_name, field_config in CONFIG['form_fields'].items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# ------ Process Final Values Using Unified Approach ------
final_values = get_final_values(all_inputs)

# ------ REFACTORED: Check User Input Completeness Dynamically ------
required_fields = [
    field for field, config in CONFIG['form_fields'].items() if config.get('required')
]
user_has_entered_info = all(
    (all_inputs.get(field) and "Select" not in str(all_inputs.get(field)))
    for field in required_fields
)

# ------ MODIFIED: calculate_personalized_targets call updated with new parameters ------
targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your personalized nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets using default values. Enter your information for personalized calculations.")
else:
    # MODIFIED: Header is now dynamic based on the user's selected goal
    st.header(f"Your Personalized Daily Targets for {targets['goal']} 🎯")

# NEW: Educational expander explaining the science behind the plan
with st.expander("🔬 The Science Behind Your Plan & The Role of Fitness", expanded=False):
    st.markdown("""
        #### **1. Energy Foundation: BMR & TDEE**
        Your plan starts by calculating your **Basal Metabolic Rate (BMR)** using the **Mifflin-St Jeor equation**, recognized by the scientific community as the most accurate formula for healthy adults. This is the energy you burn at complete rest. We then calculate your **Total Daily Energy Expenditure (TDEE)** by multiplying your BMR by an activity factor. TDEE is your "maintenance" calorie level.
        
        $$TDEE = BMR \\times \\text{Activity Multiplier}$$
        
        #### **2. Goal-Specific Caloric Targets**
        Instead of a fixed number, we adjust your calories by a percentage of your TDEE. This scales the plan to your specific metabolism, making it safer and more effective.
        - **Weight Loss:** A **20% caloric deficit** from TDEE to promote fat loss while preserving muscle.
        - **Weight Maintenance:** A **0% adjustment** to balance energy in and out.
        - **Weight Gain:** A **10% caloric surplus** over TDEE to fuel muscle growth while minimizing fat gain.

        #### **3. Macronutrient Architecture**
        Calories determine the *quantity* of weight change, but macros determine its *quality* (muscle vs. fat). We use a protein-first approach:
        - **Protein (The Builder):** Set first based on your body weight and goal to maximize muscle retention/growth and satiety.
        - **Fat (The Regulator):** Set as a percentage of total calories to ensure healthy hormone function.
        - **Carbohydrates (The Fuel):** Fill the remaining calorie budget to power your training and daily life.

        #### **Principle 6: The Indispensable Role of Fitness**
        Nutrition provides the building materials, but **resistance training provides the stimulus** that tells your body what to do with them.
        - **During Fat Loss:** It signals the body to preserve precious, metabolically active muscle tissue.
        - **During Weight Gain:** It is the non-negotiable trigger for muscle protein synthesis. A caloric surplus without resistance training will result primarily in fat gain.
        
        **Minimum Effective Guidelines (ACSM):**
        - **Resistance Training:** Train each major muscle group **2-3 times per week**.
        - **Cardiovascular Exercise:** Include **150-300 minutes** of moderate-intensity cardio per week for heart health.
    """)
    


# MODIFIED: Metrics configuration updated to display new, evidence-based metrics
weekly_change_val = targets['est_weekly_change_kg']
if weekly_change_val < 0:
    weekly_change_label = "Est. Weekly Weight Loss"
    weekly_change_display = f"{abs(weekly_change_val)} kg"
elif weekly_change_val > 0:
    weekly_change_label = "Est. Weekly Weight Gain"
    weekly_change_display = f"{weekly_change_val} kg"
else:
    weekly_change_label = "Est. Weekly Change"
    weekly_change_display = "0.0 kg"

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 4,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal/day"),
            ("Maintenance Calories (TDEE)", f"{targets['tdee']} kcal/day"),
            (weekly_change_label, weekly_change_display),
            ("Daily Caloric Adj.", f"{targets['daily_caloric_adjustment']:.0f} kcal")
        ]
    },
    {
        'title': 'Daily Nutritional Target Breakdown', 'columns': 4,
        'metrics': [
            ("Daily Calorie Target", f"{targets['total_calories']} kcal"),
            ("Protein Target", f"{targets['protein_g']} g"),
            ("Carbohydrate Target", f"{targets['carb_g']} g"),
            ("Fat Target", f"{targets['fat_g']} g")
        ]
    },
    {
        'title': 'Macronutrient Distribution (% of Daily Calories)', 'columns': 4,
        'metrics': [
            ("Protein", f"{targets['protein_percent']:.1f}%", f"({targets['protein_calories']} kcal)"),
            ("Carbohydrates", f"{targets['carb_percent']:.1f}%", f"({targets['carb_calories']} kcal)"),
            ("Fat", f"{targets['fat_percent']:.1f}%", f"({targets['fat_calories']} kcal)"),
            ("", "") # Empty placeholder for layout
        ]
    }
]

# Display all metrics using unified system
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item.")

# ------ Create Category Tabs for Food Organization ------
available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    sorted_items = sorted(items, key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
    with tabs[i]:
        render_food_grid(sorted_items, category, 2)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 11: Unified Results Display and Analysis
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
        st.info("No foods have been selected for today. 🍽️")

    # Refactored: Dynamically generate intake metrics from CONFIG
    st.subheader("Total Nutritional Intake for the Day 📈")
    intake_metrics = []
    for nutrient, config in CONFIG['nutrient_configs'].items():
        label = f"Total {config['label']} Consumed"
        value_format = "{:.0f}" if nutrient == 'calories' else "{:.1f}"
        value_str = f"{value_format.format(totals[nutrient])} {config['unit']}"
        intake_metrics.append((label, value_str))
    
    display_metrics_grid(intake_metrics, 4)

    # Unified progress tracking
    recommendations = create_progress_tracking(totals, targets)

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    # MODIFIED: Caloric balance analysis is now goal-aware
    st.subheader("Daily Caloric Balance Summary ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    goal = targets['goal']

    if goal == 'Weight Loss':
        if cal_balance >= 0:
            st.warning(f"⚠️ Your intake is {abs(cal_balance):.0f} kcal above your maintenance level (TDEE), which may hinder weight loss.")
        else:
            st.info(f"✅ Your intake is {abs(cal_balance):.0f} kcal below your maintenance level (TDEE), supporting your weight loss goal.")
    elif goal == 'Weight Gain':
        if cal_balance <= 0:
            st.warning(f"⚠️ Your intake is {abs(cal_balance):.0f} kcal below your maintenance level (TDEE), which may prevent weight gain.")
        else:
            st.info(f"✅ Your intake is {abs(cal_balance):.0f} kcal above your maintenance level (TDEE), supporting your weight gain goal.")
    else: # Maintenance
        if abs(cal_balance) > 50: # Allow a small buffer
            direction = "above" if cal_balance > 0 else "below"
            st.info(f"Your intake is {abs(cal_balance):.0f} kcal {direction} your maintenance level (TDEE). Adjust intake to stay closer to your TDEE for weight maintenance.")
        else:
            st.info("✅ Your intake is aligned with your maintenance calories (TDEE), supporting your goal.")


    # Detailed food log
    if selected_foods:
        st.subheader("Detailed Food Log for Today 📋")
        food_log_data = [{
            'Food Item Name': f"{item['food'].get('emoji', '')} {item['food']['name']}",
            'Servings': item['servings'],
            'Calories': item['food']['calories'] * item['servings'],
            'Protein (g)': item['food']['protein'] * item['servings'],
            'Carbs (g)': item['food']['carbs'] * item['servings'],
            'Fat (g)': item['food']['fat'] * item['servings']
        } for item in selected_foods]
        
        df_log = pd.DataFrame(food_log_data)
        st.dataframe(
            df_log.style.format({
                'Servings': '{:.1f}', 'Calories': '{:.0f}', 'Protein (g)': '{:.1f}',
                'Carbs (g)': '{:.1f}', 'Fat (g)': '{:.1f}'
            }),
            use_container_width=True
        )
    st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 12: Clear Selections and Footer
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()

# MODIFIED: Info sections updated to reflect new scientific basis and remove flawed/misleading info
info_sections = [
    {
        'title': "Activity Level Guide for Accurate TDEE 🏃‍♂️",
        'content': """
- **Sedentary**: Little to no exercise or desk job.
- **Lightly Active**: Light exercise/sports 1-3 days/week.
- **Moderately Active**: Moderate exercise/sports 3-5 days/week.
- **Very Active**: Hard exercise/sports 6-7 days/week.
- **Extremely Active**: Very hard exercise, physical job, or training twice daily.
"""
    },
    {
        'title': "Emoji Guide for Food Ranking 💡",
        'content': """
- 🥇 **Nutrient & Calorie Dense**: High in both calories and its primary nutrient.
- 🔥 **High-Calorie**: Among the most energy-dense options in its group.
- 💪 **Top Protein Source**: A leading contributor of protein.
- 🍚 **Top Carb Source**: A leading contributor of carbohydrates.
- 🥑 **Top Fat Source**: A leading contributor of healthy fats.
"""
    },
    {
        'title': "About This Nutrition Calculator 📖",
        'content': """
This tool uses evidence-based formulas to create your plan:
- **BMR**: Mifflin-St Jeor equation.
- **TDEE**: BMR x Activity Multiplier.
- **Caloric Target**: TDEE adjusted by a percentage based on your goal (e.g., -20% for Loss, +10% for Gain).
- **Macronutrients**: Protein is set based on g/kg of body weight for your goal. Fat is set as a % of calories. Carbs fill the rest.
"""
    }
]

for section in info_sections:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {section['title']}")
    st.sidebar.markdown(section['content'])
