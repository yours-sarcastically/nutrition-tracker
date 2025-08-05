# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for personalized nutrition goals (weight loss, maintenance, and gain) using vegetarian food sources. It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied to support the selected objective. Macronutrient targets follow evidence-based nutritional guidelines with a protein-first approach.
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

# ------ Goal-Specific Targets Based on Evidence-Based Guide ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # -20% from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,   # 0% from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # +10% over TDEE
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
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
    'form_fields': {
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
        'goal': {'type': 'selectbox', 'label': 'Nutrition Goal', 'options': [
            ("Select Goal", None),
            ("Weight Loss", "weight_loss"),
            ("Weight Maintenance", "weight_maintenance"),
            ("Weight Gain", "weight_gain")
        ], 'required': True, 'placeholder': None},
        'protein_per_kg': {'type': 'number', 'label': 'Protein (g Per Kilogram Body Weight)', 'min': 1.2, 'max': 3.0, 'step': 0.1, 'help': 'Protein intake per kilogram of body weight', 'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number', 'label': 'Fat (Percent of Total Calories)', 'min': 15, 'max': 40, 'step': 1, 'help': 'Percentage of total calories from fat', 'convert': lambda x: x / 100 if x is not None else None, 'advanced': True, 'required': False}
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
    """Process all user inputs and apply defaults using unified approach"""
    final_values = {}
    
    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
        elif field in ['activity_level', 'goal']:
            final_values[field] = value if value is not None else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None else DEFAULTS[field]
    
    # Apply goal-specific defaults for advanced settings
    if final_values['goal'] in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[final_values['goal']]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']
    
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
    
    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation/building',
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

def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Calculate estimated weekly weight change based on caloric adjustment"""
    # Based on approximation that 1 kg of body fat contains ~7700 kcal
    return (daily_caloric_adjustment * 7) / 7700

def calculate_personalized_targets(age, height_cm, weight_kg, sex='male', activity_level='moderately_active', 
                                   goal='weight_gain', protein_per_kg=None, fat_percentage=None):
    """Calculate Personalized Daily Nutritional Targets Based on Evidence-Based Guidelines"""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Get goal-specific configuration
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])
    
    # Apply goal-specific caloric adjustment
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment
    
    # Use provided values or goal-specific defaults
    protein_per_kg = protein_per_kg if protein_per_kg is not None else goal_config['protein_per_kg']
    fat_percentage = fat_percentage if fat_percentage is not None else goal_config['fat_percentage']
    
    protein_g = protein_per_kg * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    # Calculate estimated weekly weight change
    estimated_weekly_change = calculate_estimated_weekly_change(caloric_adjustment)

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 3),
        'goal': goal
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
This advanced nutrition tracker uses evidence-based calculations to provide personalized daily nutrition goals for **weight loss**, **weight maintenance**, or **weight gain**. The calculator employs the **Mifflin-St Jeor equation** for BMR and follows a **protein-first macronutrient strategy** recommended by nutrition science. 🚀
""")

# Educational Context Box
with st.expander("📚 **Scientific Foundation & Evidence-Based Approach**", expanded=False):
    st.markdown("""
    ### **Energy Foundation: BMR & TDEE**
    
    **Basal Metabolic Rate (BMR):** Your body's energy needs at complete rest, calculated using the **Mifflin-St Jeor equation** - the most accurate formula recognized by the Academy of Nutrition and Dietetics.
    
    **Total Daily Energy Expenditure (TDEE):** Your maintenance calories including daily activities, calculated by multiplying BMR by scientifically validated activity factors.
    
    ### **Goal-Specific Approach**
    
    Rather than using arbitrary caloric adjustments, this tracker uses **percentage-based adjustments** that scale appropriately to your individual metabolism:
    
    - **Weight Loss:** -20% from TDEE (sustainable fat loss while preserving muscle)
    - **Weight Maintenance:** 0% from TDEE (energy balance)  
    - **Weight Gain:** +10% over TDEE (lean muscle growth with minimal fat gain)
    
    ### **Protein-First Macronutrient Strategy**
    
    This evidence-based approach prioritizes protein needs first, then allocates fat for hormonal health (minimum 20% of calories), with carbohydrates filling remaining energy needs:
    
    - **Weight Loss:** 1.8g protein/kg body weight, 25% fat
    - **Weight Maintenance:** 1.6g protein/kg body weight, 30% fat
    - **Weight Gain:** 2.0g protein/kg body weight, 25% fat
    """)

# ------ MODIFIED: Reordered Sidebar for Improved User Experience ------
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Separate standard and advanced fields to control their display order
standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

# 1. Render the standard (primary) input fields first
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# 2. Render the advanced fields inside an expander placed at the bottom
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
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
    (all_inputs.get(field) is not None and all_inputs.get(field) != CONFIG['form_fields'][field].get('placeholder'))
    for field in required_fields
)

# ------ Calculate Personalized Targets ------
targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
else:
    goal_labels = {'weight_loss': 'Weight Loss', 'weight_maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Personalized Daily Nutritional Targets for {goal_label} 🎯")

# ------ Unified Metrics Display Configuration ------
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 4,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal per day"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal per day"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
            ("Est. Weekly Weight Change", f"{targets['estimated_weekly_change']:+.3f} kg")
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
            ("Protein", f"{targets['protein_percent']:.1f}%", f"↑ {targets['protein_calories']} kcal"),
            ("Carbohydrates", f"{targets['carb_percent']:.1f}%", f"↑ {targets['carb_calories']} kcal"),
            ("Fat", f"{targets['fat_percent']:.1f}%", f"↑ {targets['fat_calories']} kcal"),
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

    # Unified progress tracking
    recommendations = create_progress_tracking(totals, targets)

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    # Enhanced caloric balance analysis with goal-specific messaging
    st.subheader("Daily Caloric Balance and Goal Progress Summary ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    goal_labels = {'weight_loss': 'weight loss', 'weight_maintenance': 'weight maintenance', 'weight_gain': 'weight gain'}
    current_goal = goal_labels.get(targets['goal'], 'your goal')
    
    if targets['goal'] == 'weight_loss':
        if cal_balance < -100:  # Sufficient deficit
            st.success(f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance, supporting {current_goal}.")
        else:
            st.warning(f"⚠️ Your deficit of {abs(cal_balance):.0f} kcal may be insufficient for effective weight loss.")
    elif targets['goal'] == 'weight_maintenance':
        if abs(cal_balance) <= 100:  # Within maintenance range
            st.success(f"⚖️ You are within {abs(cal_balance):.0f} kcal of maintenance, supporting {current_goal}.")
        else:
            direction = "above" if cal_balance > 0 else "below"
            st.info(f"📊 You are consuming {abs(cal_balance):.0f} kcal {direction} maintenance.")
    else:  # weight_gain
        if cal_balance > 100:  # Sufficient surplus
            st.success(f"📈 You are consuming {cal_balance:.0f} kcal above maintenance, supporting {current_goal}.")
        else:
            st.warning(f"⚠️ Your surplus of {cal_balance:.0f} kcal may be insufficient for effective weight gain.")

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

# Educational Documentation and Fitness Integration
with st.expander("🏋️ **The Critical Role of Resistance Training**", expanded=False):
    st.markdown("""
    ### **Why Exercise Matters: The Nutrient Partitioning Agent**
    
    **Nutrition provides the building materials, but resistance training provides the stimulus** that tells your body what to do with them. Exercise acts as a powerful "nutrient-partitioning agent":
    
    **During Fat Loss:**
    - Signals the body to preserve metabolically active muscle tissue
    - Increases metabolic rate both during and after exercise
    - Improves insulin sensitivity for better nutrient utilization
    
    **During Weight Gain:**
    - **Non-negotiable trigger for muscle growth** - without it, surplus calories become primarily fat
    - Enhances protein synthesis and muscle recovery
    - Optimizes the muscle-to-fat ratio of weight gained
    
    ### **Evidence-Based Exercise Recommendations**
    
    **Resistance Training (Essential):**
    - Train each major muscle group **2-3 times per week**
    - Focus on compound movements (squats, deadlifts, bench press, rows)
    - Progressive overload: gradually increase weight, reps, or sets over time
    
    **Cardiovascular Exercise (Recommended):**
    - **150-300 minutes** of moderate-intensity cardio per week for heart health
    - Assists with energy expenditure during fat loss phases
    - Improves recovery between resistance training sessions
    
    💡 **Key Insight:** A caloric surplus without resistance training results primarily in fat gain, not muscle growth.
    """)

# Updated info sections with evidence-based content
info_sections = [
    {
        'title': "Activity Level Guide for Accurate TDEE 🏃‍♂️",
        'content': """
**Choose the description that best matches your lifestyle:**

- **Sedentary (1.2x BMR)**: Little to no exercise, desk job, minimal daily movement
- **Lightly Active (1.375x BMR)**: Light exercise/sports 1-3 days/week, some walking
- **Moderately Active (1.55x BMR)**: Moderate exercise/sports 3-5 days/week
- **Very Active (1.725x BMR)**: Hard exercise/sports 6-7 days/week, active job
- **Extremely Active (1.9x BMR)**: Very hard exercise, physical job, or training twice daily

💡 **Tip:** If unsure between two levels, start with the lower one and monitor your progress over 2-3 weeks.
"""
    },
    {
        'title': "Emoji Guide for Food Ranking 💡",
        'content': """
Foods are ranked based on their nutritional density within each category:

- 🥇 **Nutrient & Calorie Dense**: High in both calories and its primary nutrient
- 🔥 **High-Calorie**: Among the most energy-dense options in its group
- 💪 **Top Protein Source**: A leading contributor of protein for muscle building
- 🍚 **Top Carb Source**: A leading contributor of carbohydrates for energy
- 🥑 **Top Fat Source**: A leading contributor of healthy fats for hormones

These rankings help you prioritize foods that efficiently meet your macro targets.
"""
    },
    {
        'title': "Scientific Methodology 📖",
        'content': """
**This calculator uses evidence-based formulas:**

- **BMR Calculation**: Mifflin-St Jeor equation (most accurate for healthy adults)
- **TDEE Calculation**: BMR × scientifically validated activity multipliers
- **Caloric Targets**: Percentage-based adjustments that scale to your metabolism
- **Macronutrient Strategy**: Protein-first approach based on body weight and goals
- **Weight Change Estimation**: Based on the principle that ~7700 kcal = 1 kg body fat

**Goal-Specific Targets:**
- **Weight Loss**: -20% TDEE, 1.8g protein/kg, 25% fat calories
- **Maintenance**: 0% TDEE, 1.6g protein/kg, 30% fat calories  
- **Weight Gain**: +10% TDEE, 2.0g protein/kg, 25% fat calories
"""
    },
    {
        'title': "Dynamic Monitoring Tips 📊",
        'content': """
**Your TDEE adapts as you lose or gain weight. Monitor and adjust:**

**Track Weekly:**
- Body weight (same time, same conditions)
- Progress photos and measurements
- Energy levels and workout performance

**When to Reassess:**
- Weight change stalls for 2+ weeks
- Losing/gaining weight faster/slower than estimated
- Significant changes in activity level or body weight

**Red Flags:**
- Rapid weight loss (>1% body weight/week) = increase calories
- No weight change after 3 weeks on weight loss = decrease calories or increase activity
- Excessive fatigue or poor recovery = may need diet break

💡 **Remember**: This calculator provides your starting point. Real-world results guide adjustments.
"""
    }
]

for section in info_sections:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {section['title']}")
    st.sidebar.markdown(section['content'])
