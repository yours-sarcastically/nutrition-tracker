# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker - Enhanced Version
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for personalized nutrition goals (weight loss, maintenance, and gain) using vegetarian food sources. It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied to support the selected objective. Macronutrient targets follow evidence-based nutritional guidelines with a protein-first approach.

Enhanced with hydration calculations, sleep/stress guidance, plateau troubleshooting, and comprehensive monitoring strategies.
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
# Cell 4: Enhanced Helper Functions
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

def calculate_hydration_needs(weight_kg, activity_level):
    """Calculate daily fluid needs based on body weight and activity"""
    base_needs = weight_kg * 35  # 35ml per kg baseline
    
    activity_bonus = {
        'sedentary': 0,
        'lightly_active': 300,
        'moderately_active': 500,
        'very_active': 700,
        'extremely_active': 1000
    }
    
    total_ml = base_needs + activity_bonus.get(activity_level, 500)
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
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 8: Application Title and Enhanced Input Interface
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

# Enhanced Sleep & Stress Impact Section
with st.expander("😴 **Sleep & Stress: The Hidden Variables**", expanded=False):
    st.markdown("""
    ### **Sleep's Critical Impact on Body Composition**
    
    **Poor sleep (<7 hours) can reduce fat loss effectiveness by up to 55%** even with identical caloric deficits. Here's why:
    
    - **Hormonal disruption:** Increases hunger hormone (ghrelin), decreases satiety hormone (leptin)
    - **Muscle protein synthesis:** Drops 18-20% with poor sleep quality
    - **Cortisol elevation:** Promotes fat storage, especially abdominal
    - **Recovery impairment:** Reduces workout performance and muscle building
    
    ### **Stress Management for Better Results**
    
    **Chronic stress elevates cortisol, which:**
    - Promotes abdominal fat storage
    - Impairs muscle building even with adequate protein
    - Increases appetite and cravings for high-calorie foods
    - Reduces insulin sensitivity
    
    ### **Optimization Strategies**
    
    **Sleep Optimization:**
    - 7-9 hours nightly with consistent sleep/wake times
    - Dark, cool room (18-20°C)
    - Morning sunlight exposure
    - Limit screens 1-2 hours before bed
    
    **Stress Reduction:**
    - Regular meditation or deep breathing
    - Nature walks or light cardio
    - Hobby time and social connection
    - Professional help if chronic stress persists
    """)

# Enhanced sidebar with hydration and optimization tips
st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Separate standard and advanced fields
standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

# Render standard input fields
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Add hydration calculator after activity level
if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_needs = calculate_hydration_needs(all_inputs['weight_kg'], all_inputs['activity_level'])
    st.sidebar.info(f"💧 **Daily Fluid Target:** {hydration_needs} ml ({hydration_needs/250:.1f} cups)")

# Render advanced fields in expander
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Enhanced Meal Timing & Optimization Tips
st.sidebar.markdown("---")
st.sidebar.markdown("### ⏰ **Meal Timing for Optimization**")
st.sidebar.markdown("""
**Protein Distribution:**
- **Target:** ≥0.4g per kg body weight per meal, 3-4 meals daily
- **Post-workout:** 20-40g protein within 2 hours of training
- **Evening:** 20-30g casein protein enhances overnight muscle synthesis

**Pre-Workout Nutrition:**
- 1-2g carbs per kg body weight, 1-2 hours before training
- Enhances high-intensity performance
""")

# Process final values
final_values = get_final_values(all_inputs)

# Check user input completeness
required_fields = [
    field for field, config in CONFIG['form_fields'].items() if config.get('required')
]
user_has_entered_info = all(
    (all_inputs.get(field) is not None and all_inputs.get(field) != CONFIG['form_fields'][field].get('placeholder'))
    for field in required_fields
)

# Calculate personalized targets
targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Enhanced Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
else:
    goal_labels = {'weight_loss': 'Weight Loss', 'weight_maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(f"Your Personalized Daily Nutritional Targets for {goal_label} 🎯")

# Enhanced metrics display with 80/20 principle
st.info("🎯 **80/20 Principle:** Aim for 80% adherence to your targets rather than perfection. This allows for social flexibility and prevents the all-or-nothing mentality that leads to diet cycling.")

# Unified metrics display
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
            ("Total Energy", f"100%", f"↑ {targets['total_calories']} kcal")
        ]
    }
]

for metric_group in metrics_config:
    st.subheader(metric_group['title'])
    display_metrics_grid(metric_group['metrics'], metric_group['columns'])
    st.markdown("---")

# Enhanced Dynamic Monitoring Tips with Plateau Troubleshooting
st.sidebar.markdown("### 📊 **Dynamic Monitoring Tips**")
st.sidebar.markdown("""
**Scale Weight Protocol:**
- Daily morning weigh-ins (post-bathroom, pre-food)
- **Compare weekly averages**, not daily readings
- Expect 1-3 day delays between caloric changes and scale response

**Better Progress Indicators:**
- Progress photos (same lighting/time/clothing)
- Body measurements (waist, chest, arms, thighs)
- Performance metrics (strength gains, energy levels)
- How clothes fit
""")

# Plateau Troubleshooting Section
with st.sidebar.expander("🔄 **Plateau Troubleshooting**"):
    st.markdown("""
    **4-6 Week Check-in Protocol:**
    
    1. **Confirm logging accuracy** (±5% of calories)
    2. **Re-validate activity multiplier** (habit drift is common)
    3. **Adjust calories by 5-10%** only after two consecutive "stalled" weeks
    4. **Update your weight** in the calculator (BMR/TDEE shifts as you change)
    
    **Red flags requiring adjustment:**
    - Weight loss >1% body weight/week (increase calories)
    - No change for 3+ weeks (reassess targets)
    - Persistent fatigue despite adherence (4+ weeks)
    
    **When to seek help:** Persistent plateaus despite adherence
    """)

# Micronutrient Awareness Section
with st.sidebar.expander("🌱 **Vegetarian Nutrition Considerations**"):
    st.markdown("""
    **Common shortfalls to monitor:**
    - **Vitamin B₁₂** (supplement recommended)
    - **Iron** (combine with vitamin C for absorption)
    - **Zinc, Calcium, Iodine** (include fortified foods)
    - **Omega-3 EPA/DHA** (consider algae-based supplements)
    
    **Fiber target:** 14g per 1,000 kcal (≈25-38g daily)
    - Increase gradually to avoid GI distress
    - Adequate water intake is crucial
    """)

# -----------------------------------------------------------------------------
# Cell 10: Enhanced Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Your Daily Food Intake 🥗")

# Calculate current totals
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

# Enhanced progress tracking with recommendations
recommendations = create_progress_tracking(totals, targets)

if recommendations:
    st.subheader("Personalized Recommendations 💡")
    for rec in recommendations:
        st.markdown(rec)

# Food categories with enhanced organization
category_order = [
    'PRIMARY PROTEIN SOURCES',
    'PRIMARY CARBOHYDRATE SOURCES', 
    'PRIMARY FAT SOURCES',
    'PRIMARY MICRONUTRIENT SOURCES'
]

# Sort foods within categories by emoji priority
for category in foods:
    foods[category] = sorted(foods[category], key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), x['name']))

# Display food categories with enhanced interface
for category in category_order:
    if category in foods and foods[category]:
        with st.expander(f"**{category}** ({len(foods[category])} options)", expanded=True):
            render_food_grid(foods[category], category, columns=2)

# -----------------------------------------------------------------------------
# Cell 11: Enhanced Daily Summary and Insights
# -----------------------------------------------------------------------------

st.header("Daily Nutritional Summary 📋")

if selected_foods:
    # Enhanced summary table
    summary_data = []
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        summary_data.append({
            'Food Item': food['name'],
            'Servings': f"{servings:.1f}",
            'Calories': f"{food['calories'] * servings:.0f}",
            'Protein (g)': f"{food['protein'] * servings:.1f}",
            'Carbs (g)': f"{food['carbs'] * servings:.1f}",
            'Fat (g)': f"{food['fat'] * servings:.1f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Enhanced totals display
    st.subheader("Daily Totals vs Targets 🎯")
    
    comparison_data = []
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        difference = actual - target
        percent_of_target = (actual / target * 100) if target > 0 else 0
        
        comparison_data.append({
            'Nutrient': config['label'],
            'Actual': f"{actual:.0f} {config['unit']}",
            'Target': f"{target:.0f} {config['unit']}",
            'Difference': f"{difference:+.0f} {config['unit']}",
            'Progress': f"{percent_of_target:.0f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Enhanced insights section
    st.subheader("Nutritional Insights & Recommendations 🔍")
    
    # Caloric balance insight
    caloric_difference = totals['calories'] - targets['total_calories']
    if abs(caloric_difference) <= 50:
        st.success(f"✅ **Excellent caloric balance!** You're within 50 kcal of your target.")
    elif caloric_difference > 50:
        st.warning(f"⚠️ **Above target by {caloric_difference:.0f} kcal.** Consider reducing portions or choosing lower-calorie options.")
    else:
        st.info(f"📊 **Below target by {abs(caloric_difference):.0f} kcal.** Add more food to reach your energy needs.")
    
    # Protein adequacy insight
    protein_difference = totals['protein'] - targets['protein_g']
    if protein_difference >= 0:
        st.success(f"💪 **Protein target achieved!** Excellent for muscle maintenance/building.")
    else:
        st.warning(f"💪 **Need {abs(protein_difference):.0f}g more protein.** Consider adding protein-rich foods.")
    
    # Macronutrient balance insight
    if totals['calories'] > 0:
        actual_protein_percent = (totals['protein'] * 4 / totals['calories']) * 100
        actual_fat_percent = (totals['fat'] * 9 / totals['calories']) * 100
        actual_carb_percent = (totals['carbs'] * 4 / totals['calories']) * 100
        
        st.info(f"📊 **Current macro split:** {actual_protein_percent:.0f}% protein, {actual_carb_percent:.0f}% carbs, {actual_fat_percent:.0f}% fat")

else:
    st.info("👆 Select foods above to see your daily nutritional summary and personalized recommendations.")

# -----------------------------------------------------------------------------
# Cell 12: Enhanced Educational Resources
# -----------------------------------------------------------------------------

# Sustainability and Long-term Success Section
with st.expander("🎯 **Sustainable Approach & Long-term Success**", expanded=False):
    st.markdown("""
    ### **The 80/20 Rule for Long-term Success**
    
    **Why 80% adherence beats 100% perfection:**
    - Allows for social flexibility and prevents restrictive eating patterns
    - Reduces psychological stress and food obsession
    - Creates sustainable habits rather than temporary changes
    - Prevents the all-or-nothing mentality that leads to diet cycling
    
    **Practical application:**
    - 80% of calories from nutrient-dense whole foods
    - 20% flexibility for treats and social situations
    - Focus on getting back on track with next meal after an "off" choice
    - Think in weeks and months, not individual days
    
    ### **Building Food Flexibility**
    
    **Learn to estimate portions without always measuring:**
    - Practice portion visualization (palm for protein, fist for carbs, thumb for fats)
    - Develop go-to meals for different situations
    - Practice navigating social eating situations
    
    **The "Maintenance Practice" Concept:**
    - Spend time at maintenance calories between diet phases
    - This prevents metabolic adaptation and psychological burnout
    - Practice maintaining your goal weight for several months before attempting further changes
    """)

# Enhanced Progress Monitoring Section
with st.expander("📏 **Beyond the Scale: Smart Progress Monitoring**", expanded=False):
    st.markdown("""
    ### **Why the Scale Can Be Misleading**
    
    **Daily weight fluctuations (1-3 kg) are normal due to:**
    - Water retention from hormones, sodium, carbs, stress
    - Digestive contents and timing of meals
    - Muscle tissue is denser than fat tissue
    - Beginners can gain muscle while losing fat simultaneously ("recomposition")
    
    ### **Better Progress Indicators**
    
    **Visual Progress:**
    - **Progress photos:** Same lighting, poses, time of day, minimal clothing
    - **How clothes fit:** Often changes before scale weight
    - **Mirror assessment:** Focus on body shape changes
    
    **Measurements:**
    - **Body measurements:** Waist, chest, arms, thighs monthly
    - **Waist-to-hip ratio:** Important health indicator
    
    **Performance Metrics:**
    - **Strength gains:** Progressive overload in the gym
    - **Endurance improvements:** Cardio performance
    - **Energy levels:** Daily vitality and mood
    - **Sleep quality:** Recovery and hormonal health
    
    ### **Scale Weight Guidelines**
    
    **Best practices:**
    - Weigh daily at the same time (morning, post-bathroom, minimal clothing)
    - Focus on weekly averages, not daily fluctuations
    - Expect 1-3 day delays between caloric changes and scale responses
    - Compare weekly means when deciding to adjust calories
    """)

# Evidence-Based Supplement Hierarchy
with st.expander("💊 **Evidence-Based Supplement Hierarchy**", expanded=False):
    st.markdown("""
    ### **Tier 1 (Strong Evidence)**
    
    **Creatine Monohydrate:**
    - **Dosage:** 3-5g daily, any time
    - **Benefits:** Improves strength, power, and muscle mass
    - **Safety:** Extensively researched, very safe
    
    **Whey/Plant Protein Powder:**
    - **When needed:** Only if struggling to meet protein targets through food
    - **Timing:** Post-workout or between meals
    - **Quality:** Look for third-party tested products
    
    **Vitamin D3:**
    - **Dosage:** 1000-2000 IU daily if deficient
    - **Testing:** Get blood levels checked (aim for 30-50 ng/mL)
    - **Importance:** Bone health, immune function, hormone production
    
    ### **Tier 2 (Moderate Evidence)**
    
    **Caffeine:**
    - **Dosage:** 3-6mg/kg body weight, 30-60 min pre-workout
    - **Benefits:** Enhanced performance and focus
    - **Caution:** Avoid late in day to prevent sleep disruption
    
    **Omega-3 (EPA/DHA):**
    - **Dosage:** 1-3g daily if fish intake is low
    - **Source:** Fish oil or algae-based for vegetarians
    - **Benefits:** Heart health, inflammation reduction, brain function
    
    **Beta-Alanine:**
    - **Dosage:** 3-5g daily for muscular endurance
    - **Note:** Causes harmless tingling sensation
    - **Best for:** High-intensity, repeated efforts
    
    ### **Tier 3 (Limited Evidence)**
    
    **Multivitamin:** Insurance policy if diet variety is limited
    **Magnesium:** If experiencing sleep issues or muscle cramps
    **Zinc:** Important for immune function and hormone production
    
    ### **❌ Generally Unnecessary**
    
    - **Fat burners:** No magic pills for fat loss
    - **Testosterone boosters:** Rarely effective in healthy individuals
    - **BCAAs:** Unnecessary if protein intake is adequate
    - **Detox products:** Your liver and kidneys handle detoxification
    """)

# Advanced Strategies Section
with st.expander("🔄 **Advanced Strategies for Breaking Plateaus**", expanded=False):
    st.markdown("""
    ### **Understanding Metabolic Adaptation**
    
    **What happens during extended dieting:**
    - BMR can decrease by 10-25% during prolonged caloric restriction
    - Hormones like leptin, thyroid hormones (T3/T4), and testosterone may decline
    - Non-exercise activity thermogenesis (NEAT) often decreases unconsciously
    
    **Signs you need a diet break:**
    - Weight loss stalls despite adherence for 2+ weeks
    - Persistent fatigue, poor sleep, or mood changes
    - Decreased workout performance or recovery
    - Constant hunger or food obsession
    
    ### **For Weight Loss Plateaus**
    
    **Strategic approaches:**
    1. **Recalculate targets** (your TDEE decreases as you lose weight)
    2. **Increase NEAT** (take stairs, park farther, fidget more)
    3. **Add 1-2 cardio sessions** (150-200 calories burned)
    4. **Implement refeed days** (1 day at maintenance every 7-14 days)
    5. **Check food logging accuracy** (weigh foods, track condiments/oils)
    
    **Diet breaks:** 1-2 weeks at maintenance calories every 6-8 weeks can help restore hormonal balance
    
    ### **For Weight Gain Plateaus**
    
    **Effective strategies:**
    1. **Increase liquid calories** (smoothies, milk, juices)
    2. **Add healthy fats** (nuts, oils, avocados - calorie-dense)
    3. **Increase meal frequency** (5-6 smaller meals vs 3 large)
    4. **Time carbs around workouts** (pre/post training for better utilization)
    5. **Reduce excessive cardio** (if doing >300 min/week)
    
    ### **Long-term Maintenance Strategies**
    
    **The maintenance mindset shift:**
    - **From restriction to balance:** Focus on sustainable eating patterns
    - **From perfection to consistency:** Aim for good choices most of the time
    - **From short-term to lifestyle:** Think in years, not weeks
    
    **Reverse dieting (for post-weight loss):**
    - **Gradually increase calories:** Add 50-100 kcal per week
    - **Monitor weight stability:** Expect 1-2 kg regain (normal glycogen/water restoration)
    - **Prioritize strength training:** Maintain muscle mass during metabolic recovery
    - **Be patient:** Full metabolic recovery can take 3-6 months
    """)

# Final Tips and Reminders
st.markdown("---")
st.markdown("""
### **🏆 Remember: Consistency Over Perfection**

Your calculator provides the **what** (your targets); focus on the **how** (achieving them consistently). 
It's more effective to hit your targets 80-90% of the time over months than to be 100% perfect for two weeks and then burn out.

**Key takeaways:**
- Trust the process and focus on weekly average progress
- Listen to your body's biofeedback (energy, sleep, performance)
- Adjust targets every 4-6 weeks as your body changes
- Prioritize whole foods while allowing flexibility
- Remember that sustainable change is a marathon, not a sprint

**Questions or concerns?** Consider consulting with a registered dietitian or qualified nutrition professional for personalized guidance.
""")

# -----------------------------------------------------------------------------
# End of Enhanced Script
# -----------------------------------------------------------------------------
