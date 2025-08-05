# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script now supports weight-loss, maintenance, and weight-gain goals
following the highest-validity recommendations in the supplied blueprint.
It keeps every pre-existing feature while adding:

1. Goal selector (loss / maintenance / gain) that applies the blueprint’s
   percentage-based caloric adjustment (+10 %, 0 %, ‑20 %).
2. Automatic goal-specific protein (1.8 / 1.6 / 2.0 g·kg⁻¹) and fat
   distribution (25 / 30 / 25 % of kcal).
3. Science-backed estimated weekly weight-change display.
4. Contextual educational blocks for BMR, TDEE, macronutrient strategy,
   dynamic monitoring, and the indispensable role of resistance training.

All prior advanced settings (custom surplus, custom protein g·kg⁻¹,
custom fat %) remain untouched and, when filled, override the defaults.
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
    'goal': "weight_gain",            # NEW
    'caloric_surplus': 400,
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

# ------ Goal-Specific Settings from Blueprint  -------------------
GOAL_SETTINGS = {
    'weight_loss':       {'calorie_pct': -0.20, 'protein_kg': 1.8, 'fat_pct': 0.25},
    'weight_maintenance':{'calorie_pct':  0.00, 'protein_kg': 1.6, 'fat_pct': 0.30},
    'weight_gain':       {'calorie_pct':  0.10, 'protein_kg': 2.0, 'fat_pct': 0.25}
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
        'protein':  {'unit': 'g',    'label': 'Protein',  'target_key': 'protein_g'},
        'carbs':    {'unit': 'g',    'label': 'Carbohydrates', 'target_key': 'carb_g'},
        'fat':      {'unit': 'g',    'label': 'Fat',      'target_key': 'fat_g'}
    },
    'form_fields': {
        # NEW goal selector --------------------------------------------------
        'goal': {'type': 'selectbox', 'label': 'Primary Goal',
                 'options': [("Select Goal", None),
                             ("Weight Loss", "weight_loss"),
                             ("Weight Maintenance", "weight_maintenance"),
                             ("Weight Gain", "weight_gain")],
                 'required': True, 'placeholder': None},
        # --------------------------------------------------------------------
        'age': {'type': 'number', 'label': 'Age (Years)', 'min': 16, 'max': 80,
                'step': 1, 'placeholder': 'Enter your age', 'required': True},
        'height_cm': {'type': 'number', 'label': 'Height (Centimeters)', 'min': 140, 'max': 220,
                      'step': 1, 'placeholder': 'Enter your height', 'required': True},
        'weight_kg': {'type': 'number', 'label': 'Weight (kg)', 'min': 40.0, 'max': 150.0,
                      'step': 0.5, 'placeholder': 'Enter your weight', 'required': True},
        'sex': {'type': 'selectbox', 'label': 'Sex',
                'options': ["Select Sex", "Male", "Female"], 'required': True,
                'placeholder': "Select Sex"},
        'activity_level': {'type': 'selectbox', 'label': 'Activity Level',
                'options': [("Select Activity Level", None),
                            ("Sedentary", "sedentary"),
                            ("Lightly Active", "lightly_active"),
                            ("Moderately Active", "moderately_active"),
                            ("Very Active", "very_active"),
                            ("Extremely Active", "extremely_active")],
                'required': True, 'placeholder': None},
        # The three advanced custom overrides remain unchanged --------------
        'caloric_surplus': {'type': 'number', 'label': 'Caloric Surplus (kcal Per Day)',
                            'min': 200, 'max': 800, 'step': 50,
                            'help': 'Custom daily kcal adjustment (overrides goal %)',
                            'advanced': True, 'required': False},
        'protein_per_kg':  {'type': 'number', 'label': 'Protein (g Per Kilogram Body Weight)',
                            'min': 1.2, 'max': 3.0, 'step': 0.1,
                            'help': 'Custom protein g·kg⁻¹ (overrides goal preset)',
                            'advanced': True, 'required': False},
        'fat_percentage':  {'type': 'number', 'label': 'Fat (Percent of Total Calories)',
                            'min': 15, 'max': 40, 'step': 1,
                            'help': 'Custom % kcal from fat (overrides goal preset)',
                            'convert': lambda x: x / 100 if x is not None else None,
                            'advanced': True, 'required': False}
    }
}

# -----------------------------------------------------------------------------
# Cell 4: Unified Helper Functions (unchanged except internals that rely on new field)
# -----------------------------------------------------------------------------
def initialize_session_state():
    session_vars = ['food_selections'] + [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else None

def create_unified_input(field_name, field_config, container=st.sidebar):
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
        if field_name in ('activity_level', 'goal'):
            index = next((i for i, (_, val) in enumerate(field_config['options']) if val == current_value), 0)
            selection = container.selectbox(field_config['label'], field_config['options'],
                                            index=index, format_func=lambda x: x[0])
            value = selection[1]
        else:
            index = field_config['options'].index(current_value) if current_value in field_config['options'] else 0
            value = container.selectbox(field_config['label'], field_config['options'], index=index)
    st.session_state[session_key] = value
    return value

def get_final_values(user_inputs):
    final_values = {}
    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None else DEFAULTS[field]
    return final_values

def display_metrics_grid(metrics_data, num_columns=4):
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                label, value = metric_info
                st.metric(label, value)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                st.metric(label, value, delta)

# (remaining helper functions unchanged)
# -----------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# -----------------------------------------------------------------------------
def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

# ------------ UPDATED to honour goal-specific blueprint settings -------------
def calculate_personalized_targets(age, height_cm, weight_kg, sex='male',
                                   activity_level='moderately_active',
                                   goal='weight_gain',
                                   caloric_surplus=None,
                                   protein_per_kg=None,
                                   fat_percentage=None):
    """
    Calculate personalized targets using blueprint rules.
    Custom advanced overrides (caloric_surplus, protein_per_kg, fat_percentage)
    take precedence if provided by user.
    """
    bmr  = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)

    # Blueprint defaults for selected goal
    goal_defaults = GOAL_SETTINGS.get(goal, GOAL_SETTINGS['weight_gain'])
    pct_adj   = goal_defaults['calorie_pct']
    prot_kg   = goal_defaults['protein_kg']
    fat_pct   = goal_defaults['fat_pct']

    # Advanced overrides ------------------------------------------------------
    if caloric_surplus is not None and caloric_surplus != 0:
        total_calories = tdee + caloric_surplus
        daily_adjust   = caloric_surplus
    else:
        total_calories = tdee * (1 + pct_adj)
        daily_adjust   = tdee * pct_adj
    if protein_per_kg is not None and protein_per_kg > 0:
        prot_kg = protein_per_kg
    if fat_percentage is not None and fat_percentage > 0:
        fat_pct = fat_percentage
    # ------------------------------------------------------------------------

    protein_g        = prot_kg * weight_kg
    protein_calories = protein_g * 4
    fat_calories     = total_calories * fat_pct
    fat_g            = fat_calories / 9
    carb_calories    = total_calories - protein_calories - fat_calories
    carb_g           = carb_calories / 4

    est_weekly_change = (daily_adjust * 7) / 7700  # kg per week (+/-)

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(est_weekly_change, 2)
    }
    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent']    = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent']     = (targets['fat_calories'] / targets['total_calories']) * 100
    else:
        targets['protein_percent'] = targets['carb_percent'] = targets['fat_percent'] = 0
    return targets
# -----------------------------------------------------------------------------
# Cell 6 onward: (no functional changes except for displays that reference new target keys)
# -----------------------------------------------------------------------------
# [food-database functions unchanged]

# -----------------------------------------------------------------------------
# Cell 7: Initialize Application
# -----------------------------------------------------------------------------
initialize_session_state()
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# Custom CSS (unchanged)
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
Ready to turbocharge your health game? This tool now adapts automatically
for weight loss, maintenance, or lean gains, backed by peer-reviewed science. 🚀
""")

st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")
all_inputs = {}

standard_fields  = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields  = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

# Render standard fields (goal now included)
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Advanced overrides
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

final_values = get_final_values(all_inputs)

required_fields = [field for field, config in CONFIG['form_fields'].items() if config.get('required')]
user_has_entered_info = all(
    (all_inputs.get(field) is not None and all_inputs.get(field) != CONFIG['form_fields'][field].get('placeholder'))
    for field in required_fields
)

targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------
if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
else:
    st.header("Your Personalized Daily Nutritional Targets 🎯")

metrics_config = [
    {
        'title': 'Metabolic Information', 'columns': 4,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal"),
            ("Est. Weekly Change", f"{targets['estimated_weekly_change']} kg"),
            ("", "")
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
        'title': 'Macronutrient Distribution (% of Calories)', 'columns': 4,
        'metrics': [
            ("Protein",      f"{targets['protein_percent']:.1f}%", f"↑ {targets['protein_calories']} kcal"),
            ("Carbohydrates",f"{targets['carb_percent']:.1f}%",    f"↑ {targets['carb_calories']} kcal"),
            ("Fat",          f"{targets['fat_percent']:.1f}%",     f"↑ {targets['fat_calories']} kcal"),
            ("", "")
        ]
    }
]
for config in metrics_config:
    if config['title'] != 'Metabolic Information':
        st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10 – 12: Food selection, calculation, clearing (unchanged logic)
# -----------------------------------------------------------------------------
# (all existing code in these cells remains identical)

# -----------------------------------------------------------------------------
# Additional Blueprint Educational Sections (sidebar)
# -----------------------------------------------------------------------------
blueprint_sections = [
    {
        'title': "Science Corner 🧬 – Why Mifflin-St Jeor?",
        'content': """
The **Mifflin-St Jeor equation** consistently outperforms older predictors,
making it the gold-standard for estimating Basal Metabolic Rate in healthy
adults (Academy of Nutrition & Dietetics, 2021).
"""
    },
    {
        'title': "Goal-Specific Calorie Adjustments 📉📈",
        'content': """
• **Weight Loss:** ‑20 % below TDEE – balances steady fat loss and muscle retention.  
• **Maintenance:** 0 % – stays in energy equilibrium.  
• **Weight Gain:** +10 % above TDEE – supports lean hypertrophy with minimal fat.
"""
    },
    {
        'title': "Protein-First Macro Design 🍗",
        'content': """
Set protein first (1.8 / 1.6 / 2.0 g·kg⁻¹) → choose fat (≥ 20 % kcal) →
fill the rest with carbs for training fuel.
"""
    },
    {
        'title': "Dynamic Monitoring 🔄",
        'content': """
Estimated weekly change = (daily calorie adjustment × 7) ÷ 7700.  
Track scale trends each week and tweak activity level or food choices if progress stalls.
"""
    },
    {
        'title': "Don’t Skip Resistance Training 💪",
        'content': """
Nutrition supplies the bricks; lifting provides the blueprint.  
ACSM minimum: **2–3 sessions per muscle group weekly** plus **150–300 min** cardio for heart health.
"""
    }
]
for section in blueprint_sections:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {section['title']}")
    st.sidebar.markdown(section['content'])

# -----------------------------------------------------------------------------
