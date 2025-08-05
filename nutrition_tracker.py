# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker - Enhanced Version
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for personalized nutrition goals (weight loss, maintenance, and gain) using vegetarian food sources. It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied to support the selected objective. Macronutrient targets follow evidence-based nutritional guidelines with a protein-first approach.

Enhanced with comprehensive evidence-based tips for long-term success.
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

# ------ Activity Level Descriptions ------
ACTIVITY_DESCRIPTIONS = {
    'sedentary': "Little to no exercise, desk job",
    'lightly_active': "Light exercise 1-3 days per week",
    'moderately_active': "Moderate exercise 3-5 days per week",
    'very_active': "Heavy exercise 6-7 days per week",
    'extremely_active': "Very heavy exercise, physical job, or 2x/day training"
}

# ------ Goal-Specific Targets Based on Evidence-Based Guide ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # -20% from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,   # 0% from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # +10% over TDEE
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
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

def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculate daily fluid needs based on body weight and activity"""
    base_needs = weight_kg * 35  # 35ml per kg baseline
    
    activity_bonus = {
        'sedentary': 0,
        'lightly_active': 300,
        'moderately_active': 500,
        'very_active': 700,
        'extremely_active': 1000
    }
    
    climate_multiplier = {
        'cold': 0.9,
        'temperate': 1.0,
        'hot': 1.2,
        'very_hot': 1.4
    }
    
    total_ml = (base_needs + activity_bonus.get(activity_level, 500)) * climate_multiplier.get(climate, 1.0)
    return round(total_ml)

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

def generate_personalized_recommendations(totals, targets, final_values):
    """Generate personalized recommendations based on current intake and goals"""
    recommendations = []
    goal = final_values['goal']
    
    # Hydration recommendation
    hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
    recommendations.append(f"💧 **Daily Hydration Target:** {hydration_ml} ml ({hydration_ml/250:.1f} cups) - drink 500ml before meals to boost satiety")
    
    # Goal-specific recommendations
    if goal == 'weight_loss':
        recommendations.extend([
            "🛏️ **Sleep Priority:** Aim for 7-9 hours nightly - poor sleep reduces fat loss effectiveness by up to 55%",
            "📊 **Weigh-in Strategy:** Daily morning weigh-ins, track weekly averages instead of daily fluctuations",
            "🥗 **Volume Eating:** Prioritize high-volume, low-calorie foods (leafy greens, cucumbers, berries) for meal satisfaction"
        ])
    elif goal == 'weight_gain':
        recommendations.extend([
            "🥤 **Liquid Calories:** Include smoothies, milk, and juices to increase calorie density",
            "🥑 **Healthy Fats:** Add nuts, oils, and avocados - calorie-dense options for easier surplus",
            "💪 **Progressive Overload:** Ensure you're getting stronger in the gym - surplus without training = mostly fat gain"
        ])
    else:  # maintenance
        recommendations.extend([
            "⚖️ **Flexible Tracking:** Monitor intake 5 days/week instead of 7 for sustainable maintenance",
            "📅 **Regular Check-ins:** Weigh weekly, measure monthly to catch changes early",
            "🎯 **80/20 Balance:** 80% nutrient-dense foods, 20% flexibility for social situations"
        ])
    
    # Protein timing recommendations
    protein_per_meal = targets['protein_g'] / 4
    recommendations.append(f"⏰ **Protein Timing:** Distribute protein across meals (~{protein_per_meal:.0f}g per meal) for optimal muscle protein synthesis")
    
    return recommendations

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
This advanced nutrition tracker uses evidence-based calculations to provide personalized daily nutrition goals for **weight loss**, **weight maintenance**, or **weight gain**. The calculator employs the **Mifflin-St Jeor equation** for BMR and follows a **protein-first macronutrient strategy** recommended by nutrition science. 🚀
""")

# Enhanced Educational Context Box
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

# ------ Sidebar for Improved User Experience ------
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

# ------ Activity Level Guide in Sidebar ------
with st.sidebar.expander("📋 **Activity Level Guide**"):
    st.markdown("""
    **Choose the level that best describes your weekly activity:**
    
    • **Sedentary:** Little to no exercise, desk job
    • **Lightly Active:** Light exercise 1-3 days per week  
    • **Moderately Active:** Moderate exercise 3-5 days per week
    • **Very Active:** Heavy exercise 6-7 days per week
    • **Extremely Active:** Very heavy exercise, physical job, or 2x/day training
    
    *💡 Tip: When in doubt, choose a lower activity level to avoid overestimating your calorie needs.*
    """)

# ------ Emoji-Based Food Ranking System Explanation in Sidebar ------
with st.sidebar.expander("🏆 **Food Emoji Guide**"):
    st.markdown("""
    **Our foods are ranked by nutritional density:**
    
    🥇 **Gold Medal:** Top performer in both calories AND primary nutrient
    🔥 **High Calorie:** Among the most calorie-dense in its category
    💪 **High Protein:** Top protein source
    🍚 **High Carb:** Top carbohydrate source  
    🥑 **High Fat:** Top healthy fat source
    
    *Focus on emoji-marked foods to efficiently meet your macro targets!*
    """)

# 2. Render the advanced fields inside an expander placed at the bottom
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# ------ Process Final Values Using Unified Approach ------
final_values = get_final_values(all_inputs)

# Display hydration recommendation in sidebar
if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
    st.sidebar.info(f"💧 **Daily Hydration Target:** {hydration_ml} ml ({hydration_ml/250:.1f} cups)")

# ------ Check User Input Completeness Dynamically ------
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
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal per day"),
            ("Est. Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kg per week")
        ]
    },
    {
        'title': 'Daily Macronutrient Targets', 'columns': 4,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            ("Protein", f"{targets['protein_g']} g ({targets['protein_percent']:.0f}%)"),
            ("Carbohydrates", f"{targets['carb_g']} g ({targets['carb_percent']:.0f}%)"),
            ("Fat", f"{targets['fat_g']} g ({targets['fat_percent']:.0f}%)")
        ]
    }
]

# ------ Display All Metric Sections ------
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Enhanced Evidence-Based Tips & Context
# -----------------------------------------------------------------------------

# ------ Foundation Tips (Always Visible) ------
with st.expander("🏆 **Essential Tips for Success**", expanded=True):
    st.markdown("""
    ### **The Foundation Trio for Success**
    
    **💧 Hydration Protocol:**
    - **Target:** 35ml per kg body weight daily
    - **Training bonus:** +500-750ml per hour of exercise
    - **Fat loss hack:** 500ml water before meals increases satiety by 13%
    
    **😴 Sleep Quality (The Game-Changer):**
    - **< 7 hours sleep** reduces fat loss effectiveness by up to 55%
    - **Target:** 7-9 hours nightly with consistent sleep/wake times
    - **Optimization:** Dark, cool room (18-20°C), no screens 1-2 hours before bed
    
    **⚖️ Weigh-In Best Practices:**
    - **Daily:** Same time (morning, post-bathroom, minimal clothing)
    - **Track:** Weekly averages, not daily fluctuations
    - **Adjust:** Only after 2+ weeks of stalled progress
    """)

# ------ Advanced Monitoring & Psychology ------
with st.expander("📊 **Advanced Monitoring & Psychology**"):
    st.markdown("""
    ### **Beyond the Scale: Better Progress Indicators**
    - **Progress photos:** Same lighting, poses, time of day
    - **Body measurements:** Waist, hips, arms, thighs (monthly)
    - **Performance metrics:** Strength gains, energy levels, sleep quality
    
    ### **The Psychology of Sustainable Change**
    **80/20 Rule:** Aim for 80% adherence rather than perfection - this allows for social flexibility and prevents the "all-or-nothing" mentality that leads to diet cycling.
    
    **Progressive Implementation:**
    - **Week 1-2:** Focus only on hitting calorie targets
    - **Week 3-4:** Add protein targets
    - **Week 5+:** Fine-tune fat and carb distribution
    
    **Biofeedback Awareness:** Monitor energy levels, sleep quality, gym performance, and hunger patterns—not just the scale.
    """)

# ------ Plateau Prevention & Meal Timing ------
with st.expander("🔄 **Plateau Prevention & Meal Timing**"):
    st.markdown("""
    ### **Plateau Troubleshooting Flow**
    **Weight Loss Plateaus:**
    1. Confirm logging accuracy (±5% calories)
    2. Re-validate activity multiplier
    3. Add 10-15 minutes daily walking before reducing calories
    4. Implement "diet breaks": 1-2 weeks at maintenance every 6-8 weeks
    
    **Weight Gain Plateaus:**
    1. Increase liquid calories (smoothies, milk)
    2. Add healthy fats (nuts, oils, avocados)
    3. Ensure progressive overload in training
    4. Gradual increases: +100-150 calories when stalled 2+ weeks
    
    ### **Meal Timing & Distribution**
    **Protein Optimization:**
    - **Distribution:** 20-30g across 3-4 meals (0.4-0.5g per kg body weight per meal)
    - **Post-workout:** 20-40g within 2 hours of training
    - **Pre-sleep:** 20-30g casein for overnight muscle protein synthesis
    
    **Performance Timing:**
    - **Pre-workout:** Moderate carbs + protein 1-2 hours prior
    - **Post-workout:** Protein + carbs within 2 hours
    """)

# ------ Food Quality & Micronutrients ------
with st.expander("🌱 **Food Quality & Micronutrient Optimization**"):
    st.markdown("""
    ### **Satiety Hierarchy (for Better Adherence)**
    1. **Protein** (highest satiety per calorie)
    2. **Fiber-rich carbs** (vegetables, fruits, whole grains)
    3. **Healthy fats** (nuts, avocado, olive oil)
    4. **Processed foods** (lowest satiety per calorie)
    
    **Fiber Target:** 14g per 1,000 kcal (≈25-38g daily) - gradually increase to avoid GI distress
    
    **Volume Eating Strategy:** Prioritize low-calorie, high-volume foods (leafy greens, cucumbers, berries) to create meal satisfaction without exceeding calorie targets.
    
    ### **Micronutrient Considerations**
    **Common Shortfalls in Plant-Forward Diets:**
    - **B₁₂, iron, calcium, zinc, iodine, omega-3 (EPA/DHA)**
    - **Strategy:** Include fortified foods or consider targeted supplementation based on lab work
    """)

# ------ Long-Term Success Framework ------
with st.expander("🎯 **Long-Term Success Framework**"):
    st.markdown("""
    ### **Maintenance Practice Concept**
    - Spend time at maintenance calories between diet phases
    - Practice maintaining goal weight for several months before further changes
    - **Reverse dieting:** Gradually increase calories post-weight loss (50-100 kcal/week)
    
    ### **Metabolic Adaptation Awareness**
    - BMR decreases 10-25% during prolonged restriction
    - Recalculate targets every 4-6 weeks as weight changes
    - TDEE naturally adjusts with body weight changes
    
    ### **Red Flag Adjustments**
    - **Too fast weight loss (>1% body weight/week):** Increase calories to prevent muscle loss
    - **Extreme fatigue/irritability:** Consider maintenance break
    - **No change for 3+ weeks:** Adjust by 5-10%
    """)

# ------ Dynamic Monitoring Tips ------
with st.expander("📈 **Dynamic Monitoring Tips**"):
    st.markdown("""
    ### **Important:** Your targets will change as you progress!
    
    **Why targets change:**
    - As you lose weight, your BMR and TDEE naturally decrease (smaller body = less energy needed)
    - As you gain weight, your TDEE increases (larger body = more energy needed)
    - This is normal metabolic adaptation, not a sign of failure
    
    **When to recalculate:**
    - **Every 4-6 weeks** or after **2-3 kg weight change**
    - Update your weight in the sidebar and generate new targets
    - Compare your actual vs. predicted weight change weekly
    
    **Patience is key:**
    - Weight can fluctuate 1-3 kg daily (water, glycogen, digestion)
    - Focus on weekly trends, not daily changes
    - True body composition changes take 2-4 weeks to become apparent
    """)

# -----------------------------------------------------------------------------
# Cell 11: Personalized Recommendations System
# -----------------------------------------------------------------------------

if user_has_entered_info:
    st.header("🎯 **Your Personalized Action Plan**")
    
    # Calculate current totals for recommendations
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
    recommendations = generate_personalized_recommendations(totals, targets, final_values)
    
    for rec in recommendations:
        st.info(rec)

# -----------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Daily Food Selection & Tracking 🥗")
st.markdown("Select the number of servings for each food item to track your daily nutrition intake.")

# ------ Reset Selection Button ------
if st.button("🔄 Reset All Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.rerun()

# ------ Sort and Display Food Categories ------
sorted_items = sorted(foods.items())

for category, items in sorted_items:
    if not items:
        continue
        
    with st.expander(f"🍽️ **{category}** ({len(items)} options)", expanded=True):
        # Sort items within each category by emoji priority first, then by calories
        sorted_items_in_category = sorted(
            items, 
            key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories'])
        )
        render_food_grid(sorted_items_in_category, category, columns=2)

# -----------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking
# -----------------------------------------------------------------------------

st.header("Daily Nutrition Summary 📊")

# Calculate current daily totals
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

if selected_foods:
    # Progress tracking with recommendations
    recommendations = create_progress_tracking(totals, targets)
    
    # Daily summary metrics
    st.subheader("Today's Nutrition Intake")
    summary_metrics = [
        ("Calories Consumed", f"{totals['calories']:.0f} kcal"),
        ("Protein Intake", f"{totals['protein']:.0f} g"),
        ("Carbohydrates", f"{totals['carbs']:.0f} g"),
        ("Fat Intake", f"{totals['fat']:.0f} g")
    ]
    display_metrics_grid(summary_metrics, 4)
    
    # Recommendations based on current intake
    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)
    
    # Detailed food breakdown
    with st.expander("📝 **Detailed Food Breakdown**"):
        st.subheader("Foods Selected Today")
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings
            
            st.write(f"**{food['name']}** - {servings} serving(s)")
            st.write(f"  → {total_cals:.0f} kcal | {total_protein:.1f}g protein | {total_carbs:.1f}g carbs | {total_fat:.1f}g fat")
else:
    st.info("No foods selected yet. Choose foods from the categories above to track your daily intake.")
    
    # Show sample progress bars with zero values
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']]
        st.progress(
            0.0,
            text=f"{config['label']}: 0% of daily target ({target:.0f} {config['unit']})"
        )

# -----------------------------------------------------------------------------
# Cell 14: Footer and Additional Resources
# -----------------------------------------------------------------------------

st.markdown("---")
st.markdown("""
### **📚 Evidence-Based References & Methodology**

This nutrition tracker is built on peer-reviewed research and evidence-based guidelines:

- **BMR Calculation:** Mifflin-St Jeor equation (Academy of Nutrition and Dietetics recommended)
- **Activity Factors:** Based on validated TDEE multipliers from exercise physiology research
- **Protein Targets:** International Society of Sports Nutrition position stands
- **Caloric Adjustments:** Conservative, sustainable rates based on body composition research

### **⚠️ Important Disclaimers**

- This tool provides general nutrition guidance based on population averages
- Individual needs may vary based on genetics, medical conditions, and other factors
- Consult with a qualified healthcare provider before making significant dietary changes
- Monitor your biofeedback (energy, performance, health markers) and adjust as needed

### **🔬 Continuous Improvement**

This tracker incorporates the latest nutrition science. As research evolves, recommendations may be updated to reflect current best practices.

**Remember:** The best nutrition plan is one you can follow consistently. Focus on sustainable habits over perfect adherence.
""")

# -----------------------------------------------------------------------------
# Cell 15: Session State Management and Performance
# -----------------------------------------------------------------------------

# Clean up session state if needed (prevent memory issues)
if len(st.session_state.food_selections) > 100:  # Arbitrary limit
    # Keep only non-zero selections
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# Add debugging info for development (can be removed in production)
if st.sidebar.checkbox("Show Debug Info", value=False):
    with st.expander("Debug Information"):
        st.write("**Final Values:**", final_values)
        st.write("**Current Selections:**", st.session_state.food_selections)
        st.write("**Calculated Targets:**", targets)
        st.write("**Current Totals:**", totals)
