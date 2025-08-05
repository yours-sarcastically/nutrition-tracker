# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive, multi-goal nutrition tracking application. It calculates personalized daily targets for calories, protein, fat, and carbohydrates for weight loss, maintenance, or gain based on user-specific attributes. The calculations are founded on evidence-based principles, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and goal-specific, percentage-based adjustments to determine Total Daily Energy Expenditure (TDEE). Macronutrient targets are set using a protein-first methodology to support the user's specific physiological goal.
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

# ------ MODIFIED: Goal-Specific Parameters based on the Blueprint ------
GOAL_PARAMETERS = {
    'weight_loss':      {'adjustment': -0.20, 'protein_per_kg': 1.8, 'fat_percentage': 0.25},
    'weight_maintenance': {'adjustment':  0.00, 'protein_per_kg': 1.6, 'fat_percentage': 0.30},
    'weight_gain':      {'adjustment':  0.10, 'protein_per_kg': 2.0, 'fat_percentage': 0.25}
}

# ------ MODIFIED: Default Parameter Values updated for multi-goal functionality ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 75.0, # Adjusted for a more common starting point
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': 'weight_maintenance', # New default goal
    # Note: 'caloric_surplus' is removed as it's replaced by a percentage-based system.
    # Defaults for protein and fat are now derived from the selected goal via GOAL_PARAMETERS.
    'protein_per_kg': 1.6,
    'fat_percentage': 0.30
}

# ------ Activity Level Multipliers for TDEE Calculation (Unchanged) ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
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
        'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    # MODIFIED: form_fields updated for multi-goal functionality
    'form_fields': {
        'goal': {'type': 'selectbox', 'label': 'Your Primary Goal', 'options': [
            ("Select Goal", None),
            ("Weight Loss", "weight_loss"),
            ("Weight Maintenance", "weight_maintenance"),
            ("Weight Gain", "weight_gain")
        ], 'required': True, 'placeholder': None},
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
        # REMOVED: 'caloric_surplus' is no longer needed; replaced by goal-based percentage adjustment.
        'protein_per_kg': {'type': 'number', 'label': 'Protein (g Per Kilogram Body Weight)', 'min': 1.2, 'max': 3.0, 'step': 0.1, 'help': 'Overrides the default protein target for your selected goal.', 'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number', 'label': 'Fat (Percent of Total Calories)', 'min': 15, 'max': 40, 'step': 1, 'help': 'Overrides the default fat target for your selected goal.', 'convert': lambda x: x / 100 if x is not None else None, 'advanced': True, 'required': False}
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

def create_unified_input(field_name, field_config, container=st.sidebar, goal_based_default=None):
    """Create input widgets using unified configuration, now handling advanced fields."""
    session_key = f'user_{field_name}'
    
    if field_config['type'] == 'number':
        if field_config.get('advanced'):
            # Display goal-based default in placeholder for advanced fields
            display_val = int(goal_based_default * 100) if field_name == 'fat_percentage' else goal_based_default
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
        
        # Handle selectboxes that use tuples (label, value)
        if isinstance(field_config['options'][0], tuple):
            index = next((i for i, (_, val) in enumerate(field_config['options']) if val == current_value), 0)
            selection = container.selectbox(field_config['label'], field_config['options'], index=index, format_func=lambda x: x[0])
            value = selection[1]
        else: # Handle simple list of options
            index = field_config['options'].index(current_value) if current_value in field_config['options'] else 0
            value = container.selectbox(field_config['label'], field_config['options'], index=index)
    
    st.session_state[session_key] = value
    return value


def get_final_values(user_inputs):
    """Process all user inputs, applying standard and goal-based defaults."""
    final_values = {}

    # 1. Determine the goal first, as it dictates other defaults.
    goal = user_inputs.get('goal') or DEFAULTS['goal']
    final_values['goal'] = goal
    
    # 2. Get the specific parameters for that goal to use as defaults for advanced fields.
    goal_params = GOAL_PARAMETERS[goal]

    # 3. Iterate through all form fields to build the final arguments dict.
    for field, config in CONFIG['form_fields'].items():
        if field in final_values:  # Skip 'goal' which is already set
            continue

        user_value = user_inputs.get(field)
        is_advanced_macro = field in ['protein_per_kg', 'fat_percentage']

        if user_value is not None and user_value not in ["Select Sex", "Select Goal", None]:
            # Use the value provided by the user.
            final_values[field] = user_value
        else:
            # No valid user value, apply a default.
            if is_advanced_macro:
                # For advanced macros, the default is based on the selected goal.
                final_values[field] = goal_params[field]
            elif field in DEFAULTS:
                # For other fields, use the standard default.
                final_values[field] = DEFAULTS[field]
                
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

def create_progress_tracking(totals, targets, goal):
    """Create unified progress tracking with bars and recommendations, adapted for any goal."""
    recommendations = []
    
    goal_text_map = {
        'weight_loss': 'to reach your weight loss target',
        'weight_maintenance': 'to maintain your weight',
        'weight_gain': 'to reach your weight gain target'
    }
    
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    
    purpose_map = {
        'calories': goal_text_map.get(goal, 'to meet your energy needs'),
        'protein': 'for muscle repair and growth',
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
    """Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation"""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure Based on Activity Level"""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

# MODIFIED: Reworked function to implement all principles from the evidence-based blueprint.
def calculate_personalized_targets(age, height_cm, weight_kg, sex, activity_level, goal, protein_per_kg, fat_percentage):
    """Calculate Personalized Daily Nutritional Targets based on selected goal."""
    # Principle 1 & 2: BMR (Mifflin-St Jeor) and TDEE
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)

    # Principle 3: Goal-Specific Caloric Targets (Percentage-Based)
    adjustment_percentage = GOAL_PARAMETERS[goal]['adjustment']
    daily_caloric_adjustment = tdee * adjustment_percentage
    total_calories = tdee + daily_caloric_adjustment

    # Principle 4: Macronutrient Architecture (Protein-First)
    protein_g = protein_per_kg * weight_kg
    protein_calories = protein_g * 4
    
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / 9
    
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    # Principle 5: Dynamic Monitoring - Estimating Rate of Change
    est_weekly_change_kg = (daily_caloric_adjustment * 7) / 7700

    targets = {
        'bmr': round(bmr), 
        'tdee': round(tdee), 
        'total_calories': round(total_calories),
        'protein_g': round(protein_g), 
        'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 
        'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 
        'carb_calories': round(carb_calories),
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
This tool provides personalized daily nutrition targets based on your unique profile and goals. Select your goal, enter your details in the sidebar, and start logging your meals to track your progress! 🚀
""")

# MODIFIED: Addition of educational expanders to provide scientific context
with st.expander("💡 How Your Caloric Needs Are Calculated (BMR & TDEE)"):
    st.markdown("""
    Your body's energy needs are the foundation of your nutrition plan. We use the most accurate, scientifically-validated formulas to estimate them:
    - **Basal Metabolic Rate (BMR):** This is the energy your body burns at complete rest. We use the **Mifflin-St Jeor equation**, which is recognized by the scientific community (including the Academy of Nutrition and Dietetics) as the most accurate predictive formula for BMR.
    $$BMR_{male} = (10 \\times \\text{weight in kg}) + (6.25 \\times \\text{height in cm}) - (5 \\times \\text{age}) + 5$$
    $$BMR_{female} = (10 \\times \\text{weight in kg}) + (6.25 \\times \\text{height in cm}) - (5 \\times \\text{age}) - 161$$
    - **Total Daily Energy Expenditure (TDEE):** This is your "maintenance" calorie level—the total energy you burn in a day, including all your activities. It's calculated by multiplying your BMR by a validated activity factor. This TDEE value is the baseline from which your goal-specific targets are set.
    """)

with st.expander("🎯 How Your Goal-Specific Targets Are Set (The Science of Change)"):
    st.markdown("""
    Instead of using a fixed number of calories (e.g., +/- 500), we use a **percentage of your TDEE** to set your calorie target. This method is safer and more sustainable because it scales the diet's intensity to your personal metabolism.
    - **Weight Loss:** A **20% deficit** from TDEE promotes effective fat loss while helping to preserve muscle.
    - **Weight Maintenance:** Your target is set to your TDEE to balance energy intake and expenditure.
    - **Weight Gain:** A **10% surplus** over TDEE provides enough energy for muscle growth while minimizing fat gain.
    """)

with st.expander("🧱 The Blueprint for Your Macros (Protein, Fat, & Carbs)"):
    st.markdown("""
    While calories determine the *quantity* of weight change, macronutrients determine its *quality* (i.e., muscle vs. fat). We use a **protein-first** approach:
    1.  **Protein (The Builder):** Your protein target is set first based on your body weight and goal to maximize muscle retention or growth and enhance satiety.
    2.  **Fat (The Regulator):** Your fat target is set to support hormonal health (e.g., testosterone production).
    3.  **Carbohydrates (The Fuel):** Carbohydrates fill the remaining calorie budget to provide the primary energy source for performance and daily activities.
    """)

st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Separate standard and advanced fields
standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields = {k, v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

# Determine the current goal to set correct defaults for advanced fields
# This ensures the 'Default:' placeholder in the advanced inputs is accurate
temp_goal = st.session_state.get('user_goal') or DEFAULTS['goal']
goal_macro_defaults = GOAL_PARAMETERS[temp_goal]

# 1. Render the standard (primary) input fields first
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# 2. Render the advanced fields inside an expander
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    # Pass the goal-specific default to the input creator for placeholder text
    goal_default = goal_macro_defaults.get(field_name)
    value = create_unified_input(field_name, field_config, container=advanced_expander, goal_based_default=goal_default)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value


# MODIFIED: New logic to get final values and calculate targets
final_values = get_final_values(all_inputs)
targets = calculate_personalized_targets(**final_values)

# Check User Input Completeness
required_fields = [f for f, c in CONFIG['form_fields'].items() if c.get('required')]
user_has_entered_info = all(
    (all_inputs.get(field) is not None and all_inputs.get(field) not in ["Select Sex", "Select Goal"])
    for field in required_fields
)

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

# MODIFIED: Header and text are now dynamic based on the selected goal
goal_text_map = {
    'weight_loss': 'Weight Loss',
    'weight_maintenance': 'Weight Maintenance',
    'weight_gain': 'Lean Weight Gain'
}
goal_display_text = goal_text_map.get(final_values['goal'])

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to calculate your personalized targets.")
    st.header(f"Sample Daily Targets for {goal_display_text} 🎯")
    st.caption("These are example targets based on default values. Enter your information for accurate calculations.")
else:
    st.header(f"Your Personalized Daily Targets for {goal_display_text} 🎯")


# MODIFIED: Metrics display updated to show new estimated weekly change
est_change_val = targets['est_weekly_change_kg']
if est_change_val > 0:
    est_change_label = "Est. Weekly Weight Gain"
    est_change_str = f"+{est_change_val} kg"
elif est_change_val < 0:
    est_change_label = "Est. Weekly Weight Loss"
    est_change_str = f"{est_change_val} kg"
else:
    est_change_label = "Est. Weekly Weight Change"
    est_change_str = "0 kg"

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 3,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal/day"),
            ("Maintenance Calories (TDEE)", f"{targets['tdee']} kcal/day"),
            (est_change_label, est_change_str),
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
        'title': 'Macronutrient Distribution (% of Daily Calories)', 'columns': 3,
        'metrics': [
            ("Protein", f"{targets['protein_percent']:.1f}%", f"{targets['protein_calories']} kcal"),
            ("Carbohydrates", f"{targets['carb_percent']:.1f}%", f"{targets['carb_calories']} kcal"),
            ("Fat", f"{targets['fat_percent']:.1f}%", f"{targets['fat_calories']} kcal"),
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

    st.subheader("Total Nutritional Intake for the Day 📈")
    intake_metrics = []
    for nutrient, config in CONFIG['nutrient_configs'].items():
        label = f"Total {config['label']} Consumed"
        value_format = "{:.0f}" if nutrient == 'calories' else "{:.1f}"
        value_str = f"{value_format.format(totals[nutrient])} {config['unit']}"
        intake_metrics.append((label, value_str))
    
    display_metrics_grid(intake_metrics, 4)

    # MODIFIED: Pass goal to progress tracking for contextual recommendations
    recommendations = create_progress_tracking(totals, targets, final_values['goal'])

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    # MODIFIED: Caloric balance analysis now works for all goals
    st.subheader("Daily Caloric Balance Summary ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    
    if final_values['goal'] == 'weight_loss':
        if cal_balance < 0:
            st.info(f"✅ You are in a caloric deficit of {abs(cal_balance):.0f} kcal, which supports weight loss.")
        else:
            st.warning(f"⚠️ You are consuming {cal_balance:.0f} kcal above your maintenance level.")
    elif final_values['goal'] == 'weight_gain':
        if cal_balance > 0:
            st.info(f"✅ You are in a caloric surplus of {cal_balance:.0f} kcal, which supports weight gain.")
        else:
            st.warning(f"⚠️ You are consuming {abs(cal_balance):.0f} kcal below your maintenance level.")
    else: # Maintenance
        st.info(f"Your current intake is {cal_balance:+.0f} kcal relative to your maintenance target.")

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

# MODIFIED: Info sections updated with new fitness guide and calculator methodology
info_sections = [
    {
        'title': "The Role of Fitness 🏋️‍♀️",
        'content': """
Nutrition provides the building materials, but **resistance training provides the stimulus** that tells your body what to do with them.
- **For Fat Loss:** It signals your body to preserve precious muscle.
- **For Muscle Gain:** It is the non-negotiable trigger for growth. A caloric surplus without resistance training will primarily result in fat gain.
**ACSM Recommendations:**
- **Resistance Training:** Train each major muscle group **2-3 times per week**.
- **Cardio:** Include **150-300 minutes** of moderate-intensity cardio per week for heart health.
"""
    },
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
This calculator uses evidence-based formulas and principles:
- **BMR Calculation**: Mifflin-St Jeor Equation.
- **Caloric Target**: TDEE +/- a percentage based on your goal.
- **Macronutrients**: Protein-first approach with goal-specific targets for protein (g/kg) and fat (% of calories).
"""
    }
]

for section in info_sections:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {section['title']}")
    st.sidebar.markdown(section['content'])
