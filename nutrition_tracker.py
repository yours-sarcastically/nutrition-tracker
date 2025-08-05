# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive, evidence-based nutrition tracking application. It supports multiple user goals (Weight Loss, Maintenance, Weight Gain) and calculates personalized daily targets for calories and macronutrients.

Core scientific principles applied:
1.  **BMR Calculation:** Uses the Mifflin-St Jeor equation, recognized by the Academy of Nutrition and Dietetics as the most accurate formula for estimating Basal Metabolic Rate (BMR).
2.  **TDEE Calculation:** Determines Total Daily Energy Expenditure (TDEE) by multiplying BMR with scientifically validated activity factors.
3.  **Goal-Specific Calories:** Applies a percentage-based caloric adjustment to the TDEE (+10% for gain, 0% for maintenance, -20% for loss) for a scalable and sustainable approach.
4.  **Macronutrient Strategy:** Implements a "protein-first," goal-specific architecture. Protein and fat targets are set based on the user's goal and body weight to optimize body composition and hormonal health, with carbohydrates filling the remaining energy needs.
5.  **Dynamic Monitoring:** Calculates and displays the estimated weekly rate of weight change based on the caloric surplus or deficit, enabling users to track progress against a scientifically derived estimate.
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

# ------ MODIFIED: Default values updated to include a 'goal' and remove advanced parameters ------
# Default parameters are now streamlined, as complex settings (protein/fat ratios) are determined by the user's goal.
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 57.5,
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': "Weight Gain" # Added 'goal' as a primary user input.
}

# ------ Activity Level Multipliers for TDEE Calculation ------
# These multipliers are based on established scientific standards for estimating TDEE from BMR.
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
    # MODIFIED: Input fields reconfigured to support goal-based calculations.
    # The 'goal' selector is added, and the manual override fields for caloric surplus,
    # protein, and fat are removed to align with the evidence-based guide.
    'form_fields': {
        'age': {'type': 'number', 'label': 'Age (Years)', 'min': 16, 'max': 80, 'step': 1, 'placeholder': 'Enter your age', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (Centimeters)', 'min': 140, 'max': 220, 'step': 1, 'placeholder': 'Enter your height', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (kg)', 'min': 40.0, 'max': 150.0, 'step': 0.5, 'placeholder': 'Enter your weight', 'required': True},
        'sex': {'type': 'selectbox', 'label': 'Sex', 'options': ["Select Sex", "Male", "Female"], 'required': True, 'placeholder': "Select Sex"},
        'goal': {'type': 'selectbox', 'label': 'Primary Goal', 'options': ["Select Goal", "Weight Loss", "Maintenance", "Weight Gain"], 'required': True, 'placeholder': "Select Goal"},
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
        else: # Handles 'sex' and the new 'goal' field
            index = field_config['options'].index(current_value) if current_value in field_config['options'] else 0
            value = container.selectbox(field_config['label'], field_config['options'], index=index)
    
    st.session_state[session_key] = value
    return value

def get_final_values(user_inputs):
    """Process all user inputs and apply defaults using unified approach"""
    final_values = {}
    
    for field, value in user_inputs.items():
        # MODIFIED: Logic updated to handle placeholder text for all select boxes, including the new 'goal' field.
        if field in ['sex', 'goal']:
            final_values[field] = value if value not in ["Select Sex", "Select Goal"] else DEFAULTS[field]
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

# MODIFIED: Function now accepts the user's 'goal' to provide more contextual recommendations.
def create_progress_tracking(totals, targets, goal):
    """Create unified progress tracking with bars and recommendations"""
    recommendations = []
    
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    
    # MODIFIED: Recommendation text is now more scientific and dynamic based on the user's goal.
    purpose_map = {
        'calories': f'to reach your {goal.lower()} target',
        'protein': 'for muscle synthesis and preservation',
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
    Calculate Basal Metabolic Rate (BMR) using the Mifflin-St Jeor Equation.
    
    This equation is implemented as per Principle 1 of the guide, recognized as the
    most accurate method for estimating resting energy expenditure in healthy adults.
    """
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure (TDEE) based on BMR and activity level.
    
    This function implements Principle 2, multiplying the BMR by a scientifically
    validated activity factor to estimate total maintenance calories.
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

# MODIFIED: This function is completely overhauled to align with the evidence-based guide.
def calculate_personalized_targets(age, height_cm, weight_kg, sex='male', activity_level='moderately_active', goal='Weight Gain'):
    """
    Calculate Personalized Daily Nutritional Targets based on user goal.

    This function integrates the core scientific principles from the guide:
    - Principle 3 (Caloric Targets): Applies a percentage-based adjustment to TDEE.
      - Weight Gain: +10%
      - Maintenance: 0%
      - Weight Loss: -20%
    - Principle 4 (Macronutrients): Sets protein and fat based on goal-specific, evidence-based targets.
      - Protein (g/kg): 2.0 (Gain), 1.6 (Maintenance), 1.8 (Loss)
      - Fat (% of calories): 25% (Gain/Loss), 30% (Maintenance)
    - Principle 5 (Monitoring): Calculates the estimated weekly weight change based on the caloric surplus/deficit.
    """
    # Define goal-specific parameter sets based on the guide
    GOAL_ADJUSTMENTS = {
        'Weight Loss': {'cal_adj': -0.20, 'prot_g_kg': 1.8, 'fat_pct': 0.25},
        'Maintenance': {'cal_adj': 0.0, 'prot_g_kg': 1.6, 'fat_pct': 0.30},
        'Weight Gain': {'cal_adj': 0.10, 'prot_g_kg': 2.0, 'fat_pct': 0.25}
    }
    
    params = GOAL_ADJUSTMENTS.get(goal, GOAL_ADJUSTMENTS['Weight Gain']) # Default to 'Weight Gain' if goal is invalid
    
    # Principle 1 & 2: Calculate BMR and TDEE
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Principle 3: Apply percentage-based caloric adjustment
    total_calories = tdee * (1 + params['cal_adj'])
    
    # Principle 4: Apply protein-first, goal-specific macronutrient strategy
    protein_g = params['prot_g_kg'] * weight_kg
    protein_calories = protein_g * 4
    
    fat_calories = total_calories * params['fat_pct']
    fat_g = fat_calories / 9
    
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    # Principle 5: Estimate weekly weight change based on caloric surplus/deficit (7700 kcal ≈ 1 kg)
    daily_caloric_adjustment = total_calories - tdee
    est_weekly_change_kg = (daily_caloric_adjustment * 7) / 7700

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'est_weekly_change_kg': round(est_weekly_change_kg, 2)
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
Ready to achieve your health goals? This tool provides personalized daily nutrition targets based on your chosen objective (weight loss, maintenance, or gain) and makes tracking simple. Let's get started! 🚀
""")

# ------ MODIFIED: Sidebar inputs simplified to align with the new goal-oriented approach. ------
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# The 'Advanced Settings' expander is removed, as all inputs are now primary.
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
# MODIFIED: Check now includes the 'goal' field and correctly handles placeholder text.
user_has_entered_info = all(
    (all_inputs.get(field) is not None and all_inputs.get(field) not in ["Select Sex", "Select Goal", None])
    for field in required_fields
)

# ------ Calculate Personalized Targets ------
# The 'goal' is now a key input for the calculation function.
targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information and select a goal in the sidebar to view your personalized nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
else:
    # MODIFIED: Header text is now dynamic based on the user's selected goal.
    st.header(f"Your Personalized Daily Nutritional Targets for {final_values['goal']} 🎯")

# MODIFIED: Metrics display updated to show the new calculated values from the guide.
# The 'est_weekly_change_kg' is added, and delta values are updated.
goal_word = "Gain" if targets['est_weekly_change_kg'] > 0 else "Loss"
change_label = f"Est. Weekly Weight {goal_word}" if final_values['goal'] != 'Maintenance' else "Est. Weekly Change"
caloric_adjustment = targets['total_calories'] - targets['tdee']

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 4,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal per day"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal per day"),
            (change_label, f"{targets['est_weekly_change_kg']} kg"),
            ("", "") # Empty placeholder for layout
        ]
    },
    {
        'title': 'Daily Nutritional Target Breakdown', 'columns': 4,
        'metrics': [
            # The delta now reflects the calculated surplus or deficit for the goal.
            ("Daily Calorie Target", f"{targets['total_calories']} kcal", f"{caloric_adjustment:+.0f} vs TDEE"),
            ("Protein Target", f"{targets['protein_g']} g"),
            ("Carbohydrate Target", f"{targets['carb_g']} g"),
            ("Fat Target", f"{targets['fat_g']} g")
        ]
    },
    {
        'title': 'Macronutrient Distribution (% of Daily Calories)', 'columns': 4,
        'metrics': [
            ("Protein", f"{targets['protein_percent']:.1f}%", f"{targets['protein_calories']} kcal"),
            ("Carbohydrates", f"{targets['carb_percent']:.1f}%", f"{targets['carb_calories']} kcal"),
            ("Fat", f"{targets['fat_percent']:.1f}%", f"{targets['fat_calories']} kcal"),
            ("", "") # Empty placeholder for layout
        ]
    }
]

# Display all metrics using unified system
for config in metrics_config:
    if config['title'] != 'Metabolic Information':
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

    # MODIFIED: The user's goal is passed to provide more specific feedback.
    recommendations = create_progress_tracking(totals, targets, final_values['goal'])

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    # MODIFIED: Caloric balance analysis is now goal-aware and provides more nuanced feedback.
    st.subheader("Daily Caloric Balance and Goal Analysis ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    target_balance = targets['total_calories'] - targets['tdee']

    if final_values['goal'] == 'Weight Loss':
        if cal_balance < 0:
            st.info(f"📉 You are in a caloric deficit of {abs(cal_balance):.0f} kcal, which is aligned with your weight loss goal. Your target deficit is {abs(target_balance):.0f} kcal.")
        else:
            st.warning(f"📈 You are in a caloric surplus of {cal_balance:.0f} kcal, which is not aligned with your weight loss goal of a {abs(target_balance):.0f} kcal deficit.")
    elif final_values['goal'] == 'Weight Gain':
        if cal_balance > 0:
            st.info(f"📈 You are in a caloric surplus of {cal_balance:.0f} kcal, supporting your weight gain goal. Your target surplus is {target_balance:.0f} kcal.")
        else:
            st.warning(f"📉 You are in a caloric deficit of {abs(cal_balance):.0f} kcal, which will not support weight gain. Aim for a surplus of {target_balance:.0f} kcal.")
    else: # Maintenance
        if abs(cal_balance) <= 50: # Allow a small buffer for maintenance
            st.info(f"⚖️ Your caloric intake is balanced with your maintenance needs (TDEE), with only a {cal_balance:+.0f} kcal difference.")
        elif cal_balance > 0:
            st.warning(f"📈 You are in a caloric surplus of {cal_balance:.0f} kcal. To maintain weight, aim to eat closer to your TDEE of {targets['tdee']} kcal.")
        else:
            st.warning(f"📉 You are in a caloric deficit of {abs(cal_balance):.0f} kcal. To maintain weight, aim to eat closer to your TDEE of {targets['tdee']} kcal.")


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

# MODIFIED: Info sections updated to include fitness recommendations and reflect the new goal-based calculation logic.
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
        'title': "The Role of Fitness in Body Composition 🏋️‍♀️",
        'content': """
**Principle 6: Nutrition provides the materials, but resistance training provides the stimulus.**
- **During Fat Loss:** It signals the body to preserve metabolically active muscle tissue.
- **During Weight Gain:** It is the non-negotiable trigger for muscle growth. A caloric surplus without resistance training will primarily result in fat gain.

**Guidelines:**
- **Resistance Training (ACSM):** Train each major muscle group **2-3 times per week**.
- **Cardio (for health):** Include **150-300 minutes** of moderate-intensity cardio per week.
"""
    },
    {
        'title': "About This Nutrition Calculator 📖",
        'content': """
This tool uses evidence-based formulas for its calculations:
- **BMR**: Mifflin-St Jeor equation.
- **Caloric Target**: TDEE is adjusted based on your goal (`+10%` for Gain, `-20%` for Loss).
- **Protein**: Set by goal (`1.6-2.0 g/kg`) for muscle management.
- **Fat**: Set by goal (`25-30% of kcal`) for hormonal health.
- **Carbohydrates**: Fills the remaining calories for energy.
- **Est. Weight Change**: Based on the ~7700 kcal per kg rule.
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
]

for section in info_sections:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {section['title']}")
    st.sidebar.markdown(section['content'])
