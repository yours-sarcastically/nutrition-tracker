# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker Using Protein-First Macronutrient Strategy and Mifflin-St Jeor Equation
# -----------------------------------------------------------------------------

"""
This script provides an interactive nutrition tracking application for personalized dietary goals using vegetarian food sources. It calculates individualized daily targets for calories, protein, fat, and carbohydrates, based on user attributes and activity level, by employing the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and activity multipliers for Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied to support objectives such as weight loss, maintenance, or gain. Macronutrient targets follow evidence-based nutritional guidelines, prioritizing protein intake.

The script incorporates a protein-first approach to macronutrient distribution, and provides hydration recommendations, daily food selection tools, and evidence-based behavioral strategies for sustainable nutrition. Users can log servings of vegetarian foods, visualize daily intake, and receive actionable recommendations to support their goals.

Usage Documentation:
- The application is designed for use within a Streamlit environment.
- User parameters (age, height, weight, sex, activity level, and nutrition goal) are entered via the sidebar.
- Advanced settings for protein and fat targets are available under an expander.
- Food selection and serving tracking are provided in categorized tabs.
- The application displays real-time progress toward daily nutrition targets and provides personalized recommendations.
- Evidence-based tips, scientific context, and references are accessible from within the app.
- No command-line interface is provided; all interaction occurs via the Streamlit web interface.

Implementation Steps:
1. Import all required libraries and configure Streamlit.
2. Define constants and configuration dictionaries for defaults, activity multipliers, goal targets, and form fields.
3. Initialize session state for persistent user input and food selections.
4. Implement helper functions for input handling, nutritional calculations, hydration estimation, progress tracking, and food suggestion logic.
5. Load and process the vegetarian food database from CSV, assigning nutrients and emojis for user guidance.
6. Render the user interface: sidebar for parameters, main area for targets, food selection, daily summary, and evidence-based recommendations.
7. Provide comprehensive documentation, references, and disclaimers.
8. Manage session state for performance and memory optimization.

All section and comment headings follow Chicago-style title case. Output elements, plot titles, and print statements use descriptive, full-sentence phrasing. Emojis are used to enhance readability and user engagement where appropriate.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import math
import plotly.graph_objects as go

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
        'caloric_adjustment': -0.20,
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,
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
    """Initialize all session state variables for user inputs and food selections"""
    session_vars = ['food_selections'] + [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else None

def create_unified_input(field_name, field_config, container=st.sidebar):
    """Create input widgets using the unified configuration. Handles advanced fields and default values."""
    session_key = f'user_{field_name}'
    if field_config['type'] == 'number':
        if field_config.get('advanced'):
            default_val = DEFAULTS.get(field_name, 0)
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
    """Process all user inputs and apply default values and goal-specific settings"""
    final_values = {}
    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
        elif field in ['activity_level', 'goal']:
            final_values[field] = value if value is not None else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None else DEFAULTS[field]
    if final_values['goal'] in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[final_values['goal']]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']
    return final_values

def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculate daily fluid needs based on body weight and activity level"""
    base_needs = weight_kg * 35
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

def find_best_food_for_nutrient(nutrient, deficit, foods):
    """Find a food that is a good source for the needed nutrient"""
    best_food = None
    highest_nutrient_val = 0
    all_foods = [item for sublist in foods.values() for item in sublist]
    for food in all_foods:
        if food[nutrient] > highest_nutrient_val:
            highest_nutrient_val = food[nutrient]
            best_food = food
    if best_food and highest_nutrient_val > 0:
        servings_needed = deficit / highest_nutrient_val
        suggestion_servings = 1
        return f"Try adding **{suggestion_servings} serving of {best_food['name']}** (approximately {best_food[nutrient] * suggestion_servings:.0f} grams {nutrient})."
    return None

def create_progress_tracking(totals, targets, foods):
    """Create progress tracking with bars and actionable recommendations"""
    recommendations = []
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    purpose_map = {
        'calories': 'to reach your target',
        'protein': 'for muscle preservation or building',
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
            base_rec = f"• You need **{deficit:.0f} more {config['unit']}** of {config['label'].lower()} {purpose}."
            if nutrient in ['protein', 'carbs', 'fat']:
                food_suggestion = find_best_food_for_nutrient(nutrient, deficit, foods)
                if food_suggestion:
                    base_rec += f" {food_suggestion}"
            recommendations.append(base_rec)
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
    """Generate personalized recommendations based on intake and selected goal"""
    recommendations = []
    goal = final_values['goal']
    hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
    recommendations.append(f"💧 **Daily Hydration Target:** {hydration_ml} milliliters (approximately {hydration_ml/250:.1f} cups). Drink 500 milliliters before meals to boost satiety.")
    if goal == 'weight_loss':
        recommendations.extend([
            "🛏️ **Sleep Priority:** Aim for 7 to 9 hours nightly. Poor sleep reduces fat loss effectiveness by up to 55 percent.",
            "📊 **Weigh-In Strategy:** Daily morning weigh-ins. Track weekly averages instead of daily fluctuations.",
            "🥗 **Volume Eating:** Prioritize high-volume, low-calorie foods such as leafy greens, cucumbers, and berries for meal satisfaction."
        ])
    elif goal == 'weight_gain':
        recommendations.extend([
            "🥤 **Liquid Calories:** Include smoothies, milk, and juices to increase calorie density.",
            "🥑 **Healthy Fats:** Add nuts, oils, and avocados. These are calorie-dense options for easier surplus.",
            "💪 **Progressive Overload:** Ensure you are getting stronger in the gym. A surplus without training results in mostly fat gain."
        ])
    else:
        recommendations.extend([
            "⚖️ **Flexible Tracking:** Monitor intake five days per week instead of seven for sustainable maintenance.",
            "📅 **Regular Check-Ins:** Weigh weekly and measure monthly to catch changes early.",
            "🎯 **Eighty-Twenty Balance:** Eat eighty percent nutrient-dense foods and allow twenty percent flexibility for social situations."
        ])
    protein_per_meal = targets['protein_g'] / 4
    recommendations.append(f"⏰ **Protein Timing:** Distribute protein across meals (approximately {protein_per_meal:.0f} grams per meal) for optimal muscle protein synthesis.")
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
    """Estimate weekly weight change based on daily caloric adjustment"""
    return (daily_caloric_adjustment * 7) / 7700

def calculate_personalized_targets(age, height_cm, weight_kg, sex='male', activity_level='moderately_active',
                                   goal='weight_gain', protein_per_kg=None, fat_percentage=None):
    """Calculate Personalized Daily Nutritional Targets Based on Evidence-Based Guidelines"""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment
    protein_per_kg = protein_per_kg if protein_per_kg is not None else goal_config['protein_per_kg']
    fat_percentage = fat_percentage if fat_percentage is not None else goal_config['fat_percentage']
    protein_g = protein_per_kg * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4
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
        if not items:
            continue
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
    with st.container(border=True):
        st.subheader(f"{food.get('emoji', '')} {food['name']}")
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food['name'], 0.0)
        col1, col2 = st.columns([2, 1.2])
        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key}_{k}", type=button_type, help=f"Set to {k} servings", use_container_width=True):
                        st.session_state.food_selections[food['name']] = float(k)
                        st.rerun()
        with col2:
            custom_serving = st.number_input(
                "Custom",
                min_value=0.0, max_value=10.0,
                value=float(current_serving), step=0.1,
                key=f"{key}_custom",
                label_visibility="collapsed"
            )
        if custom_serving != current_serving:
            if custom_serving > 0:
                st.session_state.food_selections[food['name']] = custom_serving
            elif food['name'] in st.session_state.food_selections:
                del st.session_state.food_selections[food['name']]
            st.rerun()
        caption_text = (
            f"Per Serving: {food['calories']} kilocalories | {food['protein']} grams protein | "
            f"{food['carbs']} grams carbohydrates | {food['fat']} grams fat"
        )
        st.caption(caption_text)

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
This advanced nutrition tracker uses evidence-based calculations to provide personalized daily nutrition goals for weight loss, weight maintenance, or weight gain. The calculator employs the Mifflin-St Jeor equation for Basal Metabolic Rate and follows a protein-first macronutrient strategy recommended by nutrition science. 🚀
""")

# ------ Sidebar for Improved User Experience ------
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Separate standard and advanced fields for display order
standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

# 1. Render standard (primary) input fields first
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# 2. Render advanced fields inside an expander at the bottom
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# ------ Activity Level Guide in Sidebar ------
with st.sidebar.container(border=True):
    st.markdown("""
    **Activity Level Guide:**

    • **Sedentary:** Little to no exercise, desk job
    • **Lightly Active:** Light exercise one to three days per week  
    • **Moderately Active:** Moderate exercise three to five days per week
    • **Very Active:** Heavy exercise six to seven days per week
    • **Extremely Active:** Very heavy exercise, physical job, or two times per day training

    *💡 When in doubt, choose a lower activity level to avoid overestimating your calorie needs.*
    """)

# ------ Process Final Values Using Unified Approach ------
final_values = get_final_values(all_inputs)

# Display hydration recommendation in sidebar
if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
    st.sidebar.info(f"💧 **Daily Hydration Target:** {hydration_ml} milliliters (approximately {hydration_ml/250:.1f} cups)")

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

# ------ Eighty-Twenty Principle Info Box ------
st.info("🎯 Eighty-Twenty Principle: Aim for eighty percent adherence to your targets rather than perfection. This allows for social flexibility and prevents the all-or-nothing mentality that leads to diet cycling.")

# Calculate hydration for the metrics grid
hydration_ml = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])

# ------ Unified Metrics Display Configuration ------
metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 5,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kilocalories per day"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kilocalories per day"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kilocalories per day"),
            ("Estimated Weekly Weight Change", f"{targets['estimated_weekly_change']:+.2f} kilograms per week"),
            ("", "")
        ]
    },
    {
        'title': 'Daily Macronutrient and Hydration Targets', 'columns': 5,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kilocalories"),
            ("Protein", f"{targets['protein_g']} grams", f"{targets['protein_percent']:.0f}%"),
            ("Carbohydrates", f"{targets['carb_g']} grams", f"{targets['carb_percent']:.0f}%"),
            ("Fat", f"{targets['fat_g']} grams", f"{targets['fat_percent']:.0f}%"),
            ("💧 Hydration", f"{hydration_ml} milliliters", f"Approximately {hydration_ml/250:.1f} cups")
        ]
    }
]

# ------ Display All Metric Sections ------
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()

# -----------------------------------------------------------------------------
# Cell 10: Enhanced Evidence-Based Tips and Context
# -----------------------------------------------------------------------------

st.header("📚 Evidence-Based Playbook")
tab1, tab2, tab3, tab4 = st.tabs(["Foundations", "Advanced Strategies", "Troubleshooting", "Nutrition Science"])

with tab1:
    st.subheader("🏆 Essential Tips for Success")
    st.markdown("""
    ### The Foundation Trio for Success

    **💧 Hydration Protocol:**
    - **Target:** Thirty-five milliliters per kilogram body weight daily
    - **Training bonus:** Add five hundred to seven hundred fifty milliliters per hour of exercise
    - **Fat loss hack:** Five hundred milliliters of water before meals increases satiety by thirteen percent

    **😴 Sleep Quality (The Game-Changer):**
    - Less than seven hours of sleep reduces fat loss effectiveness by up to fifty-five percent
    - **Target:** Seven to nine hours nightly with consistent sleep and wake times
    - **Optimization:** Dark, cool room (eighteen to twenty degrees Celsius), no screens one to two hours before bed

    **⚖️ Weigh-In Best Practices:**
    - **Daily:** Same time (morning, post-bathroom, minimal clothing)
    - **Track:** Weekly averages, not daily fluctuations
    - **Adjust:** Only after two or more weeks of stalled progress
    """)

with tab2:
    st.subheader("📊 Advanced Monitoring and Psychology")
    st.markdown("""
    ### Beyond the Scale: Better Progress Indicators

    - **Progress photos:** Same lighting, poses, time of day
    - **Body measurements:** Waist, hips, arms, thighs (monthly)
    - **Performance metrics:** Strength gains, energy levels, sleep quality

    ### The Psychology of Sustainable Change

    **Eighty-Twenty Rule:** Aim for eighty percent adherence rather than perfection. This allows for social flexibility and prevents the all-or-nothing mentality that leads to diet cycling.

    **Progressive Implementation:**
    - **Week 1-2:** Focus only on hitting calorie targets
    - **Week 3-4:** Add protein targets
    - **Week 5+:** Fine-tune fat and carbohydrate distribution

    **Biofeedback Awareness:** Monitor energy levels, sleep quality, gym performance, and hunger patterns, not just the scale.
    """)

with tab3:
    st.subheader("🔄 Plateau Prevention and Meal Timing")
    st.markdown("""
    ### Plateau Troubleshooting Flow

    **Weight Loss Plateaus:**
    1. Confirm logging accuracy (plus or minus five percent calories)
    2. Re-validate activity multiplier
    3. Add ten to fifteen minutes daily walking before reducing calories
    4. Implement "diet breaks": one to two weeks at maintenance every six to eight weeks

    **Weight Gain Plateaus:**
    1. Increase liquid calories (smoothies, milk)
    2. Add healthy fats (nuts, oils, avocados)
    3. Ensure progressive overload in training
    4. Gradual increases: add one hundred to one hundred fifty calories when stalled for two or more weeks

    ### Meal Timing and Distribution

    **Protein Optimization:**
    - **Distribution:** Twenty to thirty grams across three to four meals (zero point four to zero point five grams per kilogram body weight per meal)
    - **Post-workout:** Twenty to forty grams within two hours of training
    - **Pre-sleep:** Twenty to thirty grams casein for overnight muscle protein synthesis

    **Performance Timing:**
    - **Pre-workout:** Moderate carbohydrates and protein one to two hours prior
    - **Post-workout:** Protein and carbohydrates within two hours
    """)

with tab4:
    st.subheader("🔬 Scientific Foundation and Nutrition Deep Dive")
    st.markdown("""
    ### Energy Foundation: BMR and TDEE

    **Basal Metabolic Rate (BMR):** Your body's energy needs at complete rest, calculated using the Mifflin-St Jeor equation, which is the most accurate formula recognized by the Academy of Nutrition and Dietetics.

    **Total Daily Energy Expenditure (TDEE):** Your maintenance calories including daily activities, calculated by multiplying BMR by scientifically validated activity factors.

    ### Satiety Hierarchy (for Better Adherence)
    1. **Protein** (highest satiety per calorie)
    2. **Fiber-rich carbohydrates** (vegetables, fruits, whole grains)
    3. **Healthy fats** (nuts, avocado, olive oil)
    4. **Processed foods** (lowest satiety per calorie)

    **Fiber Target:** Fourteen grams per one thousand kilocalories (approximately twenty-five to thirty-eight grams daily). Gradually increase to avoid gastrointestinal distress.

    **Volume Eating Strategy:** Prioritize low-calorie, high-volume foods such as leafy greens, cucumbers, and berries to create meal satisfaction without exceeding calorie targets.

    ### Micronutrient Considerations

    **Common Shortfalls in Plant-Forward Diets:**
    - Vitamin B₁₂, iron, calcium, zinc, iodine, and omega-3 fatty acids (EPA and DHA)
    - **Strategy:** Include fortified foods or consider targeted supplementation based on laboratory work
    """)

# -----------------------------------------------------------------------------
# Cell 11: Personalized Recommendations System
# -----------------------------------------------------------------------------

if user_has_entered_info:
    st.header("🎯 Your Personalized Action Plan")
    # Calculate current totals for recommendations
    totals, _ = calculate_daily_totals(st.session_state.food_selections, foods)
    recommendations = generate_personalized_recommendations(totals, targets, final_values)
    for rec in recommendations:
        st.info(rec)

# -----------------------------------------------------------------------------
# Cell 12: Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Daily Food Selection and Tracking 🥗")
st.markdown("Select the number of servings for each food item to track your daily nutrition intake.")

with st.expander("💡 View Food Emoji Guide"):
    st.markdown("""
    **Food Emoji Guide:**

    • 🥇 Gold Medal: Top performer in both calories and primary nutrient
    • 🔥 High Calorie: Among the most calorie-dense in its category
    • 💪 High Protein: Top protein source
    • 🍚 High Carbohydrate: Top carbohydrate source  
    • 🥑 High Fat: Top healthy fat source

    Foods are ranked within each category to help you make efficient choices for your goals.
    """)

# ------ Reset Selection Button ------
if st.button("🔄 Reset All Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.rerun()

# ------ Food Selection with Tabs ------
available_categories = [cat for cat, items in sorted(foods.items()) if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    # Sort items within each category by emoji priority first, then by calories
    sorted_items_in_category = sorted(
        items,
        key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories'])
    )
    with tabs[i]:
        render_food_grid(sorted_items_in_category, category, columns=2)

# -----------------------------------------------------------------------------
# Cell 13: Daily Summary and Progress Tracking
# -----------------------------------------------------------------------------

st.header("Daily Nutrition Summary 📊")

# Calculate current daily totals
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

if selected_foods:
    # Progress tracking with recommendations
    recommendations = create_progress_tracking(totals, targets, foods)

    # Daily summary metrics
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Today's Nutrition Intake")
        summary_metrics = [
            ("Calories Consumed", f"{totals['calories']:.0f} kilocalories"),
            ("Protein Intake", f"{totals['protein']:.0f} grams"),
            ("Carbohydrate Intake", f"{totals['carbs']:.0f} grams"),
            ("Fat Intake", f"{totals['fat']:.0f} grams")
        ]
        display_metrics_grid(summary_metrics, 2)

    with col2:
        st.subheader("Macronutrient Split (Grams)")
        # Donut chart for macronutrient split
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
        if sum(macro_values) > 0:
            fig = go.Figure(go.Pie(
                labels=['Protein', 'Carbohydrates', 'Fat'],
                values=macro_values,
                hole=.4,
                marker_colors=['#ff6b6b', '#feca57', '#48dbfb'],
                textinfo='label+percent',
                insidetextorientation='radial'
            ))
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=250
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Select foods to see the macronutrient split.")

    # Recommendations based on current intake
    if recommendations:
        st.subheader("Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)

    # Detailed food breakdown
    with st.expander("📝 Detailed Food Breakdown"):
        st.subheader("Foods Selected Today")
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings

            st.write(f"**{food['name']}** – {servings} serving(s)")
            st.write(f"  → {total_cals:.0f} kilocalories | {total_protein:.1f} grams protein | {total_carbs:.1f} grams carbohydrates | {total_fat:.1f} grams fat")
else:
    st.info("No foods selected yet. Choose foods from the categories above to track your daily intake. 🥗")

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

st.divider()
st.markdown("""
### 📚 Evidence-Based References and Methodology

This nutrition tracker is built on peer-reviewed research and evidence-based guidelines:

- **BMR Calculation:** Mifflin-St Jeor equation (Academy of Nutrition and Dietetics recommended)
- **Activity Factors:** Based on validated TDEE multipliers from exercise physiology research
- **Protein Targets:** International Society of Sports Nutrition position stands
- **Caloric Adjustments:** Conservative, sustainable rates based on body composition research

### ⚠️ Important Disclaimers

- This tool provides general nutrition guidance based on population averages
- Individual needs may vary based on genetics, medical conditions, and other factors
- Consult with a qualified healthcare provider before making significant dietary changes
- Monitor your biofeedback (energy, performance, health markers) and adjust as needed

### 🔬 Continuous Improvement

This tracker incorporates the latest nutrition science. As research evolves, recommendations may be updated to reflect current best practices.

**Remember:** The best nutrition plan is one you can follow consistently. Focus on sustainable habits over perfect adherence.
""")

# -----------------------------------------------------------------------------
# Cell 15: Session State Management and Performance
# -----------------------------------------------------------------------------

# Clean up session state if needed to prevent memory issues
if len(st.session_state.food_selections) > 100:
    # Keep only non-zero selections
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# Uncomment for debugging during development
# if st.sidebar.checkbox("Show Debug Info", value=False):
#     with st.expander("Debug Information"):
#         st.write("**Final Values:**", final_values)
#         st.write("**Current Selections:**", st.session_state.food_selections)
#         st.write("**Calculated Targets:**", targets)
#         st.write("**Current Totals:**", totals)

# -----------------------------------------------------------------------------
# Cell 16: Closing Statement
# -----------------------------------------------------------------------------

st.success("Thank you for using the Personalized Evidence-Based Nutrition Tracker! 🥗 Fuel your goals, one meal at a time. Stay curious, stay consistent, and remember: progress is built plate by plate. See you at your next check-in!")
