# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for multiple health goals (weight loss, maintenance, gain) using vegetarian food sources. It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied to support the user's objectives. Macronutrient targets follow current nutritional guidelines based on evidence-based research.
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
# Cell 3: Unified Configuration Constants
# -----------------------------------------------------------------------------

# ------ Default Parameter Values Based on Published Research ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 57.5,
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': "weight_gain"
}

# ------ Activity Level Multipliers for TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Goal-Specific Configurations Based on Evidence ------
GOAL_CONFIGS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # -20% from TDEE
        'protein_per_kg': 1.8,       # 1.8g per kg body weight
        'fat_percentage': 0.25,      # 25% of total calories
        'label': 'Weight Loss',
        'description': 'Sustainable fat loss while preserving muscle mass'
    },
    'maintenance': {
        'caloric_adjustment': 0.0,   # 0% from TDEE
        'protein_per_kg': 1.6,       # 1.6g per kg body weight
        'fat_percentage': 0.30,      # 30% of total calories
        'label': 'Weight Maintenance',
        'description': 'Maintain current weight and body composition'
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # +10% over TDEE
        'protein_per_kg': 2.0,       # 2.0g per kg body weight
        'fat_percentage': 0.25,      # 25% of total calories
        'label': 'Weight Gain',
        'description': 'Lean muscle building with minimal fat gain'
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
        'goal': {'type': 'selectbox', 'label': 'Health Goal', 'options': [
            ("Select Your Goal", None),
            ("Weight Loss", "weight_loss"),
            ("Weight Maintenance", "maintenance"),
            ("Weight Gain", "weight_gain")
        ], 'required': True, 'placeholder': None}
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
    """Create input widgets using unified configuration"""
    session_key = f'user_{field_name}'
    
    if field_config['type'] == 'number':
        value = container.number_input(
            field_config['label'],
            min_value=field_config['min'],
            max_value=field_config['max'],
            value=st.session_state[session_key],
            step=field_config['step'],
            placeholder=field_config.get('placeholder'),
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

def create_progress_tracking(totals, targets, goal):
    """Create unified progress tracking with bars and recommendations"""
    recommendations = []
    
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    
    purpose_map = {
        'weight_loss': {
            'calories': 'to achieve your weight loss target',
            'protein': 'to preserve muscle during fat loss',
            'carbs': 'for energy and performance',
            'fat': 'for hormone production and satiety'
        },
        'maintenance': {
            'calories': 'to maintain your current weight',
            'protein': 'for muscle maintenance',
            'carbs': 'for energy and performance',
            'fat': 'for hormone production'
        },
        'weight_gain': {
            'calories': 'to reach your weight gain target',
            'protein': 'for muscle building',
            'carbs': 'for energy and performance',
            'fat': 'for hormone production'
        }
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
            purpose = purpose_map.get(goal, {}).get(nutrient, 'for optimal nutrition')
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
    weekly_caloric_change = daily_caloric_adjustment * 7
    return weekly_caloric_change / 7700  # 1 kg body fat ≈ 7700 kcal

def calculate_personalized_targets(age, height_cm, weight_kg, sex='male', activity_level='moderately_active', goal='weight_gain'):
    """Calculate Personalized Daily Nutritional Targets Based on Goal"""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Get goal-specific configuration
    goal_config = GOAL_CONFIGS.get(goal, GOAL_CONFIGS['weight_gain'])
    
    # Calculate target calories based on goal
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment
    
    # Calculate macronutrients using goal-specific ratios
    protein_g = goal_config['protein_per_kg'] * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * goal_config['fat_percentage']
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'estimated_weekly_change': round(calculate_estimated_weekly_change(caloric_adjustment), 2),
        'goal_config': goal_config
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
.evidence-box { 
    background-color: #f0f8ff; 
    padding: 15px; 
    border-radius: 10px; 
    border-left: 4px solid #4CAF50; 
    margin: 10px 0; 
}
.science-tip { 
    background-color: #fff3cd; 
    padding: 10px; 
    border-radius: 5px; 
    border-left: 3px solid #ffc107; 
    margin: 8px 0; 
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 8: Application Title and Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
<div class="evidence-box">
<h4>🧬 Science-Based Nutrition Planning</h4>
This tool uses the <strong>Mifflin-St Jeor equation</strong> (the gold standard for BMR calculation) and evidence-based macronutrient ratios to create personalized nutrition plans for weight loss, maintenance, or gain. All recommendations follow current research from the Academy of Nutrition and Dietetics and ACSM guidelines.
</div>
""", unsafe_allow_html=True)

# ------ Sidebar Input Interface ------
st.sidebar.header("Personal Parameters for Evidence-Based Calculations 📊")

all_inputs = {}

# Render all input fields
for field_name, field_config in CONFIG['form_fields'].items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    all_inputs[field_name] = value

# ------ Process Final Values Using Unified Approach ------
final_values = get_final_values(all_inputs)

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
# Cell 9: Educational Content and Scientific Context
# -----------------------------------------------------------------------------

# Display educational content based on selected goal
if user_has_entered_info and final_values['goal'] in GOAL_CONFIGS:
    goal_config = GOAL_CONFIGS[final_values['goal']]
    
    st.markdown(f"""
    <div class="science-tip">
    <h4>📚 Scientific Context for {goal_config['label']}</h4>
    <p><strong>Your Goal:</strong> {goal_config['description']}</p>
    <p><strong>Caloric Strategy:</strong> {'+' if goal_config['caloric_adjustment'] > 0 else ''}{goal_config['caloric_adjustment']*100:.0f}% from maintenance calories</p>
    <p><strong>Protein Target:</strong> {goal_config['protein_per_kg']} g/kg body weight - optimal for your goal</p>
    <p><strong>Fat Allocation:</strong> {goal_config['fat_percentage']*100:.0f}% of calories for hormone production</p>
    </div>
    """, unsafe_allow_html=True)

# Add fitness integration information
st.markdown("""
<div class="evidence-box">
<h4>💪 The Critical Role of Resistance Training</h4>
<p><strong>Why Exercise Matters:</strong> Nutrition provides the building materials, but <strong>resistance training provides the stimulus</strong> that tells your body what to do with them.</p>
<ul>
<li><strong>During Fat Loss:</strong> Preserves muscle tissue and maintains metabolic rate</li>
<li><strong>During Weight Gain:</strong> Directs nutrients toward muscle growth rather than fat storage</li>
<li><strong>ACSM Recommendation:</strong> Train each major muscle group 2-3 times per week</li>
<li><strong>Cardio Guideline:</strong> 150-300 minutes of moderate-intensity cardio per week for heart health</li>
</ul>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 10: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your personalized nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
else:
    goal_label = targets['goal_config']['label']
    st.header(f"Your Personalized Daily Targets for {goal_label} 🎯")

# ------ Unified Metrics Display Configuration ------
metrics_config = [
    {
        'title': 'Metabolic Foundation (Mifflin-St Jeor Equation)', 'columns': 4,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal/day"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal/day"),
            ("Daily Caloric Adjustment", f"{'+' if targets['caloric_adjustment'] >= 0 else ''}{targets['caloric_adjustment']} kcal"),
            ("Est. Weekly Weight Change", f"{'+' if targets['estimated_weekly_change'] >= 0 else ''}{targets['estimated_weekly_change']} kg")
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
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 11: Interactive Food Selection Interface
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
# Cell 12: Unified Results Display and Analysis
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

    # Dynamically generate intake metrics from CONFIG
    st.subheader("Total Nutritional Intake for the Day 📈")
    intake_metrics = []
    for nutrient, config in CONFIG['nutrient_configs'].items():
        label = f"Total {config['label']} Consumed"
        value_format = "{:.0f}" if nutrient == 'calories' else "{:.1f}"
        value_str = f"{value_format.format(totals[nutrient])} {config['unit']}"
        intake_metrics.append((label, value_str))
    
    display_metrics_grid(intake_metrics, 4)

    # Unified progress tracking with goal-specific recommendations
    recommendations = create_progress_tracking(totals, targets, final_values['goal'])

    st.subheader("Personalized Recommendations for Today's Nutrition 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the excellent work! 🎉")

    # Goal-specific analysis
    st.subheader("Daily Progress Analysis ⚖️")
    goal_config = targets['goal_config']
    cal_balance = totals['calories'] - targets['tdee']
    
    if final_values['goal'] == 'weight_loss':
        if cal_balance < 0:
            st.success(f"📉 You're in a {abs(cal_balance):.0f} kcal deficit, promoting fat loss while preserving muscle.")
        else:
            st.warning(f"📈 You're consuming {cal_balance:.0f} kcal above maintenance. Consider reducing portions for weight loss.")
    elif final_values['goal'] == 'maintenance':
        if abs(cal_balance) <= 100:
            st.success(f"⚖️ You're within {abs(cal_balance):.0f} kcal of maintenance - perfect for weight stability.")
        else:
            if cal_balance > 0:
                st.info(f"📈 You're {cal_balance:.0f} kcal above maintenance. This may lead to gradual weight gain.")
            else:
                st.info(f"📉 You're {abs(cal_balance):.0f} kcal below maintenance. This may lead to gradual weight loss.")
    else:  # weight_gain
        if cal_balance > 0:
            st.success(f"📈 You're consuming {cal_balance:.0f} kcal above maintenance, supporting lean muscle growth.")
        else:
            st.warning(f"📉 You're {abs(cal_balance):.0f} kcal below maintenance. Increase portions to support weight gain goals.")

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
# Cell 13: Clear Selections and Footer
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()

# Educational sidebar content with evidence-based information
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔬 Scientific Foundation")
st.sidebar.markdown("""
**BMR Calculation:** Uses the Mifflin-St Jeor equation, recognized by the Academy of Nutrition and Dietetics as the most accurate formula for healthy adults.

**TDEE Multipliers:** Based on validated activity factors from exercise physiology research.

**Goal Strategies:** Evidence-based percentage adjustments that optimize results while maintaining safety.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🏃‍♂️ Activity Level Guide")
st.sidebar.markdown("""
**Sedentary (1.2x):** Little to no exercise, desk job

**Lightly Active (1.375x):** Light exercise 1-3 days/week

**Moderately Active (1.55x):** Moderate exercise 3-5 days/week

**Very Active (1.725x):** Hard exercise 6-7 days/week

**Extremely Active (1.9x):** Very hard exercise, physical job, or training twice daily
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Goal-Specific Science")
st.sidebar.markdown("""
**Weight Loss (-20% TDEE):**
- Sustainable fat loss rate
- 1.8g protein/kg preserves muscle
- 25% fat maintains hormones

**Maintenance (0% TDEE):**
- Energy balance for stability
- 1.6g protein/kg maintains muscle
- 30% fat for optimal health

**Weight Gain (+10% TDEE):**
- Conservative surplus for lean gains
- 2.0g protein/kg builds muscle
- 25% fat supports growth
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Emoji Guide")
st.sidebar.markdown("""
🥇 **Nutrient & Calorie Dense:** High in both calories and primary nutrient

🔥 **High-Calorie:** Among most energy-dense in category

💪 **Top Protein:** Leading protein contributor

🍚 **Top Carb:** Leading carbohydrate contributor

🥑 **Top Fat:** Leading healthy fat contributor
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Monitoring Your Progress")
st.sidebar.markdown("""
**Dynamic Tracking:** Your TDEE changes as you lose or gain weight. Monitor your actual weekly weight change and adjust activity level if progress stalls.

**Rate of Change:** The estimated weekly change is based on the approximation that 1 kg of body fat contains ~7700 kcal.

**Adjustment Cues:** If actual progress differs significantly from estimates, re-evaluate your activity level input.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Practical Tips")
st.sidebar.markdown("""
**Protein Priority:** Set protein targets first - they're the most important for body composition.

**Meal Timing:** While total daily intake matters most, spreading protein throughout the day optimizes muscle protein synthesis.

**Hydration:** Aim for 35ml per kg of body weight daily, plus extra during exercise.

**Sleep:** 7-9 hours supports recovery and hormone optimization.
""")
