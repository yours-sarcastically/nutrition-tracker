# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application based on the highest 
standards of evidence-based nutrition science. It supports multi-goal functionality 
(weight loss, maintenance, gain) using scientifically validated calculations and 
methodologies as outlined in the Evidence-Based Nutrition Tracker Blueprint.

The application implements all six core principles:
1. BMR calculation using the Mifflin-St Jeor equation (highest validity)
2. TDEE calculation with validated activity multipliers (highest validity)
3. Goal-specific caloric targets using percentage-based approach (high validity)
4. Protein-first macronutrient architecture (highest validity)
5. Dynamic monitoring with estimated rate of change (high validity)
6. Fitness integration with resistance training emphasis (highest validity)
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
    page_title="Evidence-Based Nutrition Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cell 3: Evidence-Based Configuration Constants
# -----------------------------------------------------------------------------

# PRINCIPLE 1 & 2: BMR and TDEE Configuration (Highest Validity)
"""
Scientific Rationale: The Mifflin-St Jeor equation is recognized by the Academy of 
Nutrition and Dietetics as the most accurate predictive formula for estimating BMR 
in healthy adults. Activity multipliers are based on validated research for TDEE calculation.
"""
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,           # Little to no exercise, desk job
    'lightly_active': 1.375,    # Light exercise or sports 1-3 days/week
    'moderately_active': 1.55,  # Moderate exercise or sports 3-5 days/week
    'very_active': 1.725,       # Hard exercise or sports 6-7 days/week
    'extremely_active': 1.9     # Very hard exercise, physical job, or training twice daily
}

# PRINCIPLE 3: Goal-Specific Caloric Targets (High Validity)
"""
Scientific Rationale: Percentage-based adjustments scale the diet's intensity to the 
individual's metabolic reality, preventing overly aggressive deficits for smaller 
individuals and insufficient surpluses for larger individuals.
"""
GOAL_ADJUSTMENTS = {
    'weight_loss': -0.20,      # -20% from TDEE for effective, sustainable fat loss
    'maintenance': 0.00,       # 0% from TDEE to maintain current weight
    'weight_gain': 0.10        # +10% over TDEE for muscle growth with minimal fat gain
}

# PRINCIPLE 4: Protein-First Macronutrient Architecture (Highest Validity)
"""
Scientific Rationale: Protein needs are set first based on body weight and goal; 
fat is set to ensure hormonal health; carbohydrates fill remaining energy needs.
"""
PROTEIN_TARGETS = {
    'weight_loss': 1.8,        # 1.8g/kg for muscle preservation during deficit
    'maintenance': 1.6,        # 1.6g/kg for general health and maintenance
    'weight_gain': 2.0         # 2.0g/kg for muscle building during surplus
}

FAT_PERCENTAGES = {
    'weight_loss': 0.25,       # 25% of calories for hormone production
    'maintenance': 0.30,       # 30% of calories for optimal health
    'weight_gain': 0.25        # 25% of calories to maximize carbs for performance
}

# Default values for demonstration
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 70.0,
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': "weight_gain"
}

# UI Configuration
CONFIG = {
    'emoji_order': {'🥇': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
    },
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    'form_fields': {
        'age': {'type': 'number', 'label': 'Age (Years)', 'min': 16, 'max': 80, 'step': 1, 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (Centimeters)', 'min': 140, 'max': 220, 'step': 1, 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (kg)', 'min': 40.0, 'max': 150.0, 'step': 0.5, 'required': True},
        'sex': {'type': 'selectbox', 'label': 'Sex', 'options': ["Select Sex", "Male", "Female"], 'required': True},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level', 'options': [
            ("Select Activity Level", None),
            ("Sedentary", "sedentary"),
            ("Lightly Active", "lightly_active"),
            ("Moderately Active", "moderately_active"),
            ("Very Active", "very_active"),
            ("Extremely Active", "extremely_active")
        ], 'required': True},
        'goal': {'type': 'selectbox', 'label': 'Nutrition Goal', 'options': [
            ("Select Goal", None),
            ("Weight Loss", "weight_loss"),
            ("Weight Maintenance", "maintenance"),
            ("Weight Gain", "weight_gain")
        ], 'required': True}
    }
}

# -----------------------------------------------------------------------------
# Cell 4: Evidence-Based Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """
    PRINCIPLE 1: Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation
    
    Scientific Rationale: The Mifflin-St Jeor equation is the most accurate predictive 
    formula for estimating BMR in healthy adults, consistently outperforming older 
    equations like the Harris-Benedict.
    
    Equations:
    - For Men: BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age in years) + 5
    - For Women: BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age in years) - 161
    
    Args:
        age (int): Age in years
        height_cm (float): Height in centimeters
        weight_kg (float): Weight in kilograms
        sex (str): 'male' or 'female'
    
    Returns:
        float: Basal Metabolic Rate in kcal/day
    """
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """
    PRINCIPLE 2: Calculate Total Daily Energy Expenditure Based on Activity Level
    
    Scientific Rationale: TDEE represents total "maintenance" calories—the energy 
    required to maintain current weight with lifestyle. Calculated by multiplying 
    BMR by scientifically validated activity factors.
    
    Activity Multipliers:
    - Sedentary: 1.2 (Little to no exercise, desk job)
    - Lightly Active: 1.375 (Light exercise or sports 1-3 days/week)
    - Moderately Active: 1.55 (Moderate exercise or sports 3-5 days/week)
    - Very Active: 1.725 (Hard exercise or sports 6-7 days/week)
    - Extremely Active: 1.9 (Very hard exercise, physical job, or training twice daily)
    
    Args:
        bmr (float): Basal Metabolic Rate in kcal/day
        activity_level (str): Activity level key
    
    Returns:
        float: Total Daily Energy Expenditure in kcal/day
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_goal_specific_calories(tdee, goal):
    """
    PRINCIPLE 3: Calculate Goal-Specific Caloric Targets Using Percentage-Based Approach
    
    Scientific Rationale: Using a percentage of TDEE to set caloric surplus or deficit 
    is superior to using a fixed number. A percentage-based adjustment scales the diet's 
    intensity to the individual's metabolic reality.
    
    Goal Adjustments:
    - Weight Loss: -20% from TDEE (effective and sustainable rate)
    - Weight Maintenance: 0% from TDEE (balances energy in with energy out)
    - Weight Gain: +10% over TDEE (conservative surplus for muscle growth)
    
    Args:
        tdee (float): Total Daily Energy Expenditure
        goal (str): Goal type ('weight_loss', 'maintenance', 'weight_gain')
    
    Returns:
        float: Target calories for the specified goal
    """
    adjustment = GOAL_ADJUSTMENTS.get(goal, 0.0)
    return tdee * (1 + adjustment)

def calculate_macronutrient_targets(target_calories, weight_kg, goal):
    """
    PRINCIPLE 4: Calculate Macronutrients Using Protein-First, Goal-Specific Strategy
    
    Scientific Rationale: A "protein-first" approach is the most effective way to 
    structure a diet. Protein needs are set first based on body weight and goal; 
    fat is set to ensure hormonal health; carbohydrates fill remaining energy needs.
    
    Protein Targets:
    - Weight Loss: 1.8g/kg (muscle preservation during deficit)
    - Weight Maintenance: 1.6g/kg (general health and maintenance)
    - Weight Gain: 2.0g/kg (muscle building during surplus)
    
    Fat Targets:
    - Weight Loss: 25% of calories (hormone production)
    - Weight Maintenance: 30% of calories (optimal health)
    - Weight Gain: 25% of calories (maximize carbs for performance)
    
    Carbohydrates: Fill remaining calories after protein and fat
    
    Args:
        target_calories (float): Daily calorie target
        weight_kg (float): Body weight in kilograms
        goal (str): Goal type
    
    Returns:
        dict: Macronutrient targets in grams and calories
    """
    # Protein calculation (4 kcal/g)
    protein_g_per_kg = PROTEIN_TARGETS.get(goal, 1.6)
    protein_g = protein_g_per_kg * weight_kg
    protein_calories = protein_g * 4
    
    # Fat calculation (9 kcal/g)
    fat_percentage = FAT_PERCENTAGES.get(goal, 0.25)
    fat_calories = target_calories * fat_percentage
    fat_g = fat_calories / 9
    
    # Carbohydrate calculation (4 kcal/g)
    carb_calories = target_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4
    
    return {
        'protein_g': round(protein_g, 1),
        'protein_calories': round(protein_calories),
        'fat_g': round(fat_g, 1),
        'fat_calories': round(fat_calories),
        'carb_g': round(carb_g, 1),
        'carb_calories': round(carb_calories)
    }

def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """
    PRINCIPLE 5: Calculate Estimated Rate of Change for Dynamic Monitoring
    
    Scientific Rationale: Calculating and displaying the estimated rate of change 
    encourages monitoring of actual weekly weight change. If progress stalls or 
    deviates significantly, it's a cue to re-evaluate inputs and adjust.
    
    Formula: Weekly Change (kg) = (Daily Caloric Adjustment × 7) / 7700
    Note: Based on approximation that 1 kg of body fat contains ~7700 kcal
    
    Args:
        daily_caloric_adjustment (float): Daily caloric surplus/deficit
    
    Returns:
        float: Estimated weekly weight change in kg
    """
    return (daily_caloric_adjustment * 7) / 7700

def calculate_personalized_targets(age, height_cm, weight_kg, sex='male', 
                                 activity_level='moderately_active', goal='weight_gain'):
    """
    Comprehensive calculation implementing all evidence-based principles.
    
    This function integrates all six principles from the Evidence-Based Nutrition 
    Tracker Blueprint to provide scientifically validated nutritional targets.
    
    Returns:
        dict: Complete nutritional targets and metabolic information
    """
    # PRINCIPLE 1: Calculate BMR using Mifflin-St Jeor equation
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    
    # PRINCIPLE 2: Calculate TDEE with activity multipliers
    tdee = calculate_tdee(bmr, activity_level)
    
    # PRINCIPLE 3: Calculate goal-specific calories using percentage approach
    target_calories = calculate_goal_specific_calories(tdee, goal)
    
    # PRINCIPLE 4: Calculate macronutrients using protein-first strategy
    macros = calculate_macronutrient_targets(target_calories, weight_kg, goal)
    
    # PRINCIPLE 5: Calculate estimated rate of change for monitoring
    daily_adjustment = target_calories - tdee
    estimated_weekly_change = calculate_estimated_weekly_change(daily_adjustment)
    
    # Compile comprehensive results
    targets = {
        'bmr': round(bmr),
        'tdee': round(tdee),
        'total_calories': round(target_calories),
        'daily_caloric_adjustment': round(daily_adjustment),
        'estimated_weekly_change_kg': round(estimated_weekly_change, 3),
        **macros
    }
    
    # Calculate macronutrient percentages
    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
    else:
        targets['protein_percent'] = targets['carb_percent'] = targets['fat_percent'] = 0
    
    return targets

# -----------------------------------------------------------------------------
# Cell 5: Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables"""
    session_vars = ['food_selections'] + [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else None

def create_unified_input(field_name, field_config, container=st.sidebar):
    """Create input widgets using unified configuration"""
    session_key = f'user_{field_name}'
    
    if field_config['type'] == 'number':
        value = container.number_input(
            field_config['label'],
            min_value=field_config['min'],
            max_value=field_config['max'],
            value=st.session_state[session_key],
            step=field_config['step']
        )
    elif field_config['type'] == 'selectbox':
        current_value = st.session_state[session_key]
        if field_name in ['activity_level', 'goal']:
            index = next((i for i, (_, val) in enumerate(field_config['options']) if val == current_value), 0)
            selection = container.selectbox(field_config['label'], field_config['options'], index=index, format_func=lambda x: x[0])
            value = selection[1]
        else:
            index = field_config['options'].index(current_value) if current_value in field_config['options'] else 0
            value = container.selectbox(field_config['label'], field_config['options'], index=index)
    
    st.session_state[session_key] = value
    return value

def get_final_values(user_inputs):
    """Process all user inputs and apply defaults"""
    final_values = {}
    
    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
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
    """Create progress tracking with bars and recommendations"""
    recommendations = []
    
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    
    purpose_map = {
        'calories': 'to reach your goal',
        'protein': 'for muscle building/preservation',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production'
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
# Cell 6: Food Database Processing Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """Load the Vegetarian Food Database From a CSV File"""
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
    """Assign emojis to foods using a unified ranking system"""
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

    emoji_mapping = {'high_cal_nutrient': '🥇', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑'}
    
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
    """Render a single food item with interaction controls"""
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

initialize_session_state()
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
.stButton>button[kind="primary"] { background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b; }
.stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 8: Application Title and Input Interface
# -----------------------------------------------------------------------------

st.title("Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
**Multi-Goal Nutrition Planning Based on Scientific Evidence**

This application implements the highest standards of evidence-based nutrition science 
to provide personalized targets for weight loss, maintenance, or gain. All calculations 
are based on peer-reviewed research and validated methodologies.
""")

st.sidebar.header("Personal Parameters 📊")

all_inputs = {}

for field_name, field_config in CONFIG['form_fields'].items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    all_inputs[field_name] = value

final_values = get_final_values(all_inputs)

required_fields = [field for field, config in CONFIG['form_fields'].items() if config.get('required')]
user_has_entered_info = all(
    (all_inputs.get(field) is not None and all_inputs.get(field) != CONFIG['form_fields'][field].get('placeholder'))
    for field in required_fields
)

targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Evidence-Based Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your personalized nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
else:
    goal_labels = {'weight_loss': 'Weight Loss', 'maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}
    current_goal = goal_labels.get(final_values['goal'], 'Weight Gain')
    st.header(f"Your Personalized Daily Targets for {current_goal} 🎯")

# Evidence-Based Metrics Display
st.subheader("Metabolic Foundation (Mifflin-St Jeor Equation)")
st.caption("**PRINCIPLE 1 & 2**: BMR calculated using the most accurate predictive formula; TDEE using validated activity multipliers")

metabolic_metrics = [
    ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal/day"),
    ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal/day"),
    ("Daily Caloric Adjustment", f"{targets['daily_caloric_adjustment']:+.0f} kcal"),
    ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change_kg']:+.3f} kg")
]
display_metrics_grid(metabolic_metrics, 4)

st.subheader("Goal-Specific Caloric Target (Percentage-Based Approach)")
st.caption("**PRINCIPLE 3**: Caloric targets scaled to individual metabolic reality using percentage adjustments")

caloric_metrics = [
    ("Daily Calorie Target", f"{targets['total_calories']} kcal"),
    ("", ""), ("", ""), ("", "")
]
display_metrics_grid(caloric_metrics, 4)

st.subheader("Protein-First Macronutrient Architecture")
st.caption("**PRINCIPLE 4**: Protein set first for goal; fat for hormonal health; carbs fill remaining energy needs")

macro_metrics = [
    ("Protein Target", f"{targets['protein_g']} g"),
    ("Carbohydrate Target", f"{targets['carb_g']} g"),
    ("Fat Target", f"{targets['fat_g']} g"),
    ("", "")
]
display_metrics_grid(macro_metrics, 4)

st.subheader("Macronutrient Distribution (% of Daily Calories)")
distribution_metrics = [
    ("Protein", f"{targets['protein_percent']:.1f}%", f"{targets['protein_calories']} kcal"),
    ("Carbohydrates", f"{targets['carb_percent']:.1f}%", f"{targets['carb_calories']} kcal"),
    ("Fat", f"{targets['fat_percent']:.1f}%", f"{targets['fat_calories']} kcal"),
    ("", "")
]
display_metrics_grid(distribution_metrics, 4)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item.")

available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    sorted_items = sorted(items, key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
    with tabs[i]:
        render_food_grid(sorted_items, category, 2)

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 11: Results Display and Dynamic Monitoring
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

    # Total intake metrics
    st.subheader("Total Nutritional Intake for the Day 📈")
    intake_metrics = []
    for nutrient, config in CONFIG['nutrient_configs'].items():
        label = f"Total {config['label']} Consumed"
        value_format = "{:.0f}" if nutrient == 'calories' else "{:.1f}"
        value_str = f"{value_format.format(totals[nutrient])} {config['unit']}"
        intake_metrics.append((label, value_str))
    
    display_metrics_grid(intake_metrics, 4)

    # PRINCIPLE 5: Dynamic Monitoring - Progress tracking with recommendations
    st.markdown("### **PRINCIPLE 5**: Dynamic Monitoring System")
    st.caption("**Scientific Rationale**: Progress tracking encourages monitoring of actual results vs. predicted outcomes")
    
    recommendations = create_progress_tracking(totals, targets)

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("🎉 All daily nutritional targets have been met! Excellent work following your evidence-based plan.")

    # Enhanced caloric balance analysis with goal context
    st.subheader("Daily Caloric Balance and Goal Progress ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    goal_labels = {'weight_loss': 'weight loss', 'maintenance': 'weight maintenance', 'weight_gain': 'weight gain'}
    current_goal_label = goal_labels.get(final_values['goal'], 'your goal')
    
    if final_values['goal'] == 'weight_loss':
        target_deficit = targets['tdee'] * 0.20  # 20% deficit target
        if cal_balance <= -target_deficit * 0.8:  # Within 80% of target deficit
            st.success(f"✅ You are in an appropriate caloric deficit of {abs(cal_balance):.0f} kcal, supporting {current_goal_label}.")
        elif cal_balance > 0:
            st.warning(f"⚠️ You are consuming {cal_balance:.0f} kcal above maintenance. Consider reducing intake for {current_goal_label}.")
        else:
            st.info(f"📊 You have a {abs(cal_balance):.0f} kcal deficit. Target deficit for optimal {current_goal_label} is {target_deficit:.0f} kcal.")
    
    elif final_values['goal'] == 'maintenance':
        if abs(cal_balance) <= 100:  # Within 100 kcal of maintenance
            st.success(f"✅ You are consuming near maintenance calories ({cal_balance:+.0f} kcal), perfect for {current_goal_label}.")
        else:
            direction = "above" if cal_balance > 0 else "below"
            st.info(f"📊 You are consuming {abs(cal_balance):.0f} kcal {direction} maintenance. Adjust intake for precise {current_goal_label}.")
    
    else:  # weight_gain
        target_surplus = targets['tdee'] * 0.10  # 10% surplus target
        if cal_balance >= target_surplus * 0.8:  # Within 80% of target surplus
            st.success(f"✅ You are in an appropriate caloric surplus of {cal_balance:.0f} kcal, supporting {current_goal_label}.")
        elif cal_balance < 0:
            st.warning(f"⚠️ You are consuming {abs(cal_balance):.0f} kcal below maintenance. Increase intake for {current_goal_label}.")
        else:
            st.info(f"📊 You have a {cal_balance:.0f} kcal surplus. Target surplus for optimal {current_goal_label} is {target_surplus:.0f} kcal.")

    # Weekly change prediction based on current intake
    if totals['calories'] > 0:
        actual_weekly_change = calculate_estimated_weekly_change(cal_balance)
        st.metric(
            "Predicted Weekly Weight Change (Based on Today's Intake)",
            f"{actual_weekly_change:+.3f} kg",
            f"Target: {targets['estimated_weekly_change_kg']:+.3f} kg"
        )

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
# Cell 12: Fitness Integration and Educational Content
# -----------------------------------------------------------------------------

# PRINCIPLE 6: The Fitness Component - Resistance Training Integration
st.header("🏋️‍♂️ **PRINCIPLE 6**: The Indispensable Role of Resistance Training")
st.markdown("""
**Scientific Rationale**: Nutrition provides the building materials, but **resistance training provides the stimulus** 
that tells your body what to do with them. It is a powerful nutrient-partitioning agent.

**Key Benefits by Goal**:
- **During Fat Loss**: Signals the body to preserve precious, metabolically active muscle tissue
- **During Weight Gain**: Non-negotiable trigger for muscle growth (surplus without training = primarily fat gain)
- **During Maintenance**: Maintains muscle mass and metabolic health
""")

fitness_cols = st.columns(2)

with fitness_cols[0]:
    st.subheader("Minimum Effective Training Guidelines")
    st.markdown("""
    **ACSM Recommendations**:
    - Train each major muscle group **2-3 times per week**
    - Focus on compound movements (squats, deadlifts, presses)
    - Progressive overload is essential for continued adaptation
    
    **Cardio Recommendations**:
    - **150-300 minutes** of moderate-intensity cardio per week
    - Supports heart health and assists with energy expenditure
    - Can be adjusted based on individual goals and preferences
    """)

with fitness_cols[1]:
    st.subheader("Training Considerations by Goal")
    
    goal_training_advice = {
        'weight_loss': """
        **Weight Loss Focus**:
        - Maintain training intensity to preserve muscle
        - Resistance training prevents metabolic slowdown
        - Cardio can increase caloric deficit
        - Recovery may be slower due to caloric restriction
        """,
        'maintenance': """
        **Maintenance Focus**:
        - Balanced approach to training
        - Focus on strength and skill development
        - Adequate recovery with sufficient calories
        - Opportunity to refine technique and form
        """,
        'weight_gain': """
        **Weight Gain Focus**:
        - Progressive overload is critical
        - Focus on compound movements for mass
        - Adequate rest between sessions for growth
        - Surplus calories support training intensity
        """
    }
    
    current_advice = goal_training_advice.get(final_values['goal'], goal_training_advice['weight_gain'])
    st.markdown(current_advice)

st.markdown("---")

# Clear selections button
if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()

# -----------------------------------------------------------------------------
# Cell 13: Educational Sidebar Content
# -----------------------------------------------------------------------------

# Enhanced educational content in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 **Evidence-Based Guide**")

# Activity Level Guide with Scientific Context
st.sidebar.markdown("#### **PRINCIPLE 2**: Activity Level Guide for Accurate TDEE")
st.sidebar.markdown("""
**Scientific Rationale**: Accurate activity assessment is crucial for TDEE calculation.

- **Sedentary (1.2)**: Little to no exercise, desk job
- **Lightly Active (1.375)**: Light exercise/sports 1-3 days/week  
- **Moderately Active (1.55)**: Moderate exercise/sports 3-5 days/week
- **Very Active (1.725)**: Hard exercise/sports 6-7 days/week
- **Extremely Active (1.9)**: Very hard exercise, physical job, or training twice daily

*Multipliers based on validated research for energy expenditure estimation.*
""")

# Goal-Specific Information
st.sidebar.markdown("#### **PRINCIPLE 3**: Goal-Specific Caloric Strategies")
st.sidebar.markdown("""
**Weight Loss (-20% TDEE)**:
- Sustainable fat loss rate
- Preserves metabolic health
- Minimizes muscle loss

**Maintenance (0% TDEE)**:
- Energy balance for stable weight
- Optimal for body recomposition
- Sustainable long-term approach

**Weight Gain (+10% TDEE)**:
- Conservative surplus for lean gains
- Minimizes excess fat accumulation
- Supports muscle growth when combined with training
""")

# Macronutrient Education
st.sidebar.markdown("#### **PRINCIPLE 4**: Protein-First Macronutrient Strategy")
st.sidebar.markdown("""
**Protein (The Builder)**:
- Set first based on body weight and goal
- Essential for muscle protein synthesis
- High thermic effect aids metabolism

**Fat (The Regulator)**:
- Minimum 20% of calories for hormone production
- Essential for vitamin absorption
- Provides satiety and flavor

**Carbohydrates (The Fuel)**:
- Fill remaining caloric needs
- Primary fuel for high-intensity training
- Protein-sparing effect during exercise
""")

# Monitoring and Adjustment Guide
st.sidebar.markdown("#### **PRINCIPLE 5**: Dynamic Monitoring System")
st.sidebar.markdown("""
**Key Monitoring Points**:
- Weekly weight changes vs. predicted
- Energy levels and training performance
- Hunger and satiety cues
- Body composition changes

**When to Adjust**:
- Progress stalls for 2-3 weeks
- Excessive fatigue or hunger
- Training performance declines
- Weight change exceeds predictions by >50%
""")

# Food Ranking System
st.sidebar.markdown("#### Emoji Guide for Food Ranking 💡")
st.sidebar.markdown("""
- 🥇 **Nutrient & Calorie Dense**: High in both calories and primary nutrient
- 🔥 **High-Calorie**: Among most energy-dense in category
- 💪 **Top Protein Source**: Leading protein contributor
- 🍚 **Top Carb Source**: Leading carbohydrate contributor  
- 🥑 **Top Fat Source**: Leading healthy fat contributor
""")

# Scientific Foundation
st.sidebar.markdown("#### About This Evidence-Based Calculator 📖")
st.sidebar.markdown("""
**Scientific Methods Used**:
- **BMR**: Mifflin-St Jeor equation (highest accuracy)
- **TDEE**: Validated activity multipliers
- **Goals**: Percentage-based caloric adjustments
- **Protein**: Goal-specific g/kg recommendations
- **Fat**: Minimum percentages for hormonal health
- **Monitoring**: 7700 kcal/kg body fat approximation

*All recommendations based on peer-reviewed research and professional guidelines.*
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Built on evidence-based nutrition science for optimal results.*")
