# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for personalized nutrition goals (weight loss, maintenance, and gain). It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied as evidence-based defaults, with advanced options for user customization.
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
    # MODIFIED: These are now fallbacks if a goal isn't selected
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
        'caloric_adjustment': -0.20,  # -20% from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,    # 0% from TDEE
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
        'protein_per_kg': {'type': 'number', 'label': 'Protein (g Per Kilogram Body Weight)', 'min': 1.2, 'max': 3.0, 'step': 0.1, 'help': 'Overrides the default for your selected goal.', 'advanced': True, 'required': False},
        'fat_percentage': {'type': 'number', 'label': 'Fat (Percent of Total Calories)', 'min': 15, 'max': 40, 'step': 1, 'help': 'Overrides the default for your selected goal.', 'convert': lambda x: x / 100 if x is not None else None, 'advanced': True, 'required': False}
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
        # MODIFIED: Placeholder for advanced fields now shows the goal-specific default
        if field_config.get('advanced'):
            goal = st.session_state.get('user_goal') or DEFAULTS['goal']
            goal_config = GOAL_TARGETS.get(goal, {})
            default_val = goal_config.get(field_name, DEFAULTS.get(field_name, 0))
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
    """Process all user inputs and apply defaults using a hybrid approach."""
    final_values = {}
    
    # Process primary fields first
    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
        elif field in ['activity_level', 'goal']:
            final_values[field] = value if value is not None else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None else DEFAULTS[field]
    
    # Apply goal-specific defaults ONLY if advanced settings are not manually entered
    selected_goal = final_values.get('goal')
    if selected_goal in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[selected_goal]
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
            if not metric_info or not metric_info[0]: continue
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

def calculate_personalized_targets(age, height_cm, weight_kg, sex, activity_level, 
                                     goal, protein_per_kg, fat_percentage):
    """Calculate Personalized Daily Nutritional Targets Based on Final Inputs"""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_maintenance'])
    
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment
    
    protein_g = protein_per_kg * weight_kg
    protein_calories = protein_g * 4
    
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / 9
    
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    est_weekly_change_kg = (caloric_adjustment * 7) / 7700

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': est_weekly_change_kg,
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
    
    custom_serving = st.number_input(
        "Custom Number of Servings:", min_value=0.0, max_value=10.0,
        value=float(current_serving), step=0.1, key=f"{key}_custom"
    )
    
    if custom_serving != current_serving:
        if custom_serving > 0:
            st.session_state.food_selections[food['name']] = custom_serving
        elif food['name'] in st.session_state.food_selections:
            del st.session_state.food_selections[food['name']]
        st.rerun()
    
    st.caption(
        f"Per Serving: {food['calories']} kcal | {food['protein']} g protein | "
        f"{food['carbs']} g carbohydrates | {food['fat']} g fat"
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
# Cell 8: Application Title and Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
This advanced tracker provides personalized nutrition goals for **weight loss**, **maintenance**, or **gain**. It uses evidence-based defaults but allows for advanced customization. Enter your details, log your meals, and use the educational guides below to optimize your journey. 🚀
""")

with st.expander("📚 **Click here to understand the Scientific Foundation of this tracker**"):
    st.markdown("""
    #### Energy Foundation: BMR & TDEE
    * **Basal Metabolic Rate (BMR):** Your body's energy needs at complete rest, calculated using the **Mifflin-St Jeor equation**—the most accurate formula recognized by the Academy of Nutrition and Dietetics.
    * **Total Daily Energy Expenditure (TDEE):** Your total "maintenance" calories, calculated by multiplying your BMR by a scientifically validated activity factor.

    #### Goal-Specific Caloric Targets
    This tracker uses **percentage-based adjustments** from your TDEE, which scale appropriately to your individual metabolism for sustainable, effective results:
    * **Weight Loss:** **-20%** from TDEE (promotes fat loss while minimizing muscle loss).
    * **Weight Maintenance:** **0%** from TDEE (balances energy in with energy out).
    * **Weight Gain:** **+10%** over TDEE (provides a conservative surplus for lean muscle growth).

    #### Protein-First Macronutrient Strategy
    This evidence-based approach prioritizes protein needs first to support muscle tissue, allocates dietary fat for hormonal health, and uses carbohydrates to fill the remaining energy needs.
    * **Weight Loss:** 1.8g protein/kg body weight, 25% of calories from fat.
    * **Weight Maintenance:** 1.6g protein/kg, 30% fat.
    * **Weight Gain:** 2.0g protein/kg, 25% fat.
    """)

st.sidebar.header("Personal Parameters 📊")
all_inputs = {}

standard_fields = {k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')}
advanced_fields = {k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')}

for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config: value = field_config['convert'](value)
    all_inputs[field_name] = value

if advanced_fields:
    advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
    for field_name, field_config in advanced_fields.items():
        value = create_unified_input(field_name, field_config, container=advanced_expander)
        if 'convert' in field_config: value = field_config['convert'](value)
        all_inputs[field_name] = value

final_values = get_final_values(all_inputs)

required_fields = [f for f, c in CONFIG['form_fields'].items() if c.get('required')]
user_has_entered_info = all(all_inputs.get(field) is not None for field in required_fields)

targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please complete your personal information in the sidebar to calculate your daily nutritional targets.")
else:
    goal_labels = {'weight_loss': 'Weight Loss', 'weight_maintenance': 'Weight Maintenance', 'weight_gain': 'Weight Gain'}
    goal_label = goal_labels.get(targets['goal'], 'Your Goal')
    st.header(f"Your Personalized Daily Nutritional Targets for {goal_label} 🎯")

    metrics_config = [
        {
            'title': 'Metabolic Information', 'columns': 4,
            'metrics': [
                ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal"),
                ("Maintenance Calories (TDEE)", f"{targets['tdee']} kcal"),
                ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
                ("Est. Weekly Change", f"{targets['estimated_weekly_change']:.2f} kg")
            ]
        },
        {
            'title': 'Daily Macronutrient Target Breakdown', 'columns': 4,
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
                ("Protein", f"{targets['protein_percent']:.1f}%", f"{targets['protein_calories']} kcal"),
                ("Carbohydrates", f"{targets['carb_percent']:.1f}%", f"{targets['carb_calories']} kcal"),
                ("Fat", f"{targets['fat_percent']:.1f}%", f"{targets['fat_calories']} kcal"),
                ("", "")
            ]
        }
    ]

    for config in metrics_config:
        st.subheader(config['title'])
        display_metrics_grid(config['metrics'], config['columns'])

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
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
        label = f"Total {config['label']}"
        value_str = f"{totals[nutrient]:.0f} {config['unit']}"
        intake_metrics.append((label, value_str))
    display_metrics_grid(intake_metrics, 4)

    recommendations = create_progress_tracking(totals, targets)

    st.subheader("Personalized Recommendations 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("🎉 All daily nutritional targets have been met. Keep up the good work!")

    st.subheader("Daily Caloric Balance Summary ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    if abs(cal_balance) <= 100:
        st.success(f"⚖️ You are within {abs(cal_balance):.0f} kcal of your maintenance calories. Perfect for weight maintenance!")
    elif cal_balance > 100:
        st.info(f"📈 You are in a {cal_balance:.0f} kcal surplus, which supports weight gain.")
    else:
        st.info(f"📉 You are in a {abs(cal_balance):.0f} kcal deficit, which supports weight loss.")

    if selected_foods:
        st.subheader("Detailed Food Log 📋")
        food_log_data = [{'Food': f"{item['food'].get('emoji', '')} {item['food']['name']}", 'Servings': item['servings'],
                          'Calories': item['food']['calories'] * item['servings'], 'Protein (g)': item['food']['protein'] * item['servings'],
                          'Carbs (g)': item['food']['carbs'] * item['servings'], 'Fat (g)': item['food']['fat'] * item['servings']} for item in selected_foods]
        df_log = pd.DataFrame(food_log_data)
        st.dataframe(df_log.style.format({'Servings': '{:.1f}', 'Calories': '{:.0f}', 'Protein (g)': '{:.1f}', 'Carbs (g)': '{:.1f}', 'Fat (g)': '{:.1f}'}), use_container_width=True)
    st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 12: Clear Selections and Educational Footer
# -----------------------------------------------------------------------------

if st.button("Clear All Food Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()

st.markdown("---")
st.header("Guides for Long-Term Success")

with st.expander("🧠 The Psychology of Sustainable Nutrition"):
    st.markdown("""
    * **Start Small:** Don't change your entire diet overnight. Focus on hitting your calorie and protein targets first.
    * **Environment Design:** Keep healthy, whole foods visible and accessible. Keep hyper-palatable junk foods out of sight (or out of the house).
    * **Consistency over Perfection (The 80/20 Rule):** Adhering to your plan 80% of the time is more sustainable and effective than a perfect-but-short-lived attempt. If you have an off meal, get right back on track with the next one.
    * **Avoid All-or-Nothing Thinking:** One "bad" meal doesn't ruin your progress. The goal is a positive trend over weeks and months.
    * **Self-Compassion:** Treat yourself with the same kindness and patience you'd show a friend on this journey.
    """)

with st.expander("🔄 Plateau-Breaking Strategies"):
    st.markdown("""
    A plateau is 2-3+ weeks of no progress despite adherence. Here’s a troubleshooting flow:
    1.  **Confirm Logging Accuracy:** Are you weighing your food and tracking oils, sauces, and drinks? These often contain hidden calories.
    2.  **Re-validate Activity Level:** Has your daily activity (NEAT) or exercise frequency decreased? Be honest.
    3.  **For Weight Loss:**
        * Increase daily activity (e.g., add a 15-minute walk).
        * If still stalled after 1-2 weeks, decrease daily calories by 100-150 kcal.
        * Consider a 1-2 week "diet break" at your new maintenance (TDEE) calories to reduce diet fatigue and restore hormonal balance.
    4.  **For Weight Gain:**
        * Ensure you are applying progressive overload in your training.
        * Increase daily calories by 150-200 kcal.
        * Prioritize sleep, as it's a major limiting factor in muscle growth.
    """)

with st.expander("💊 Evidence-Based Supplement Guide"):
    st.markdown("""
    Supplements are not magic; they only work if your nutrition, training, and sleep are in order.
    * **Tier 1 (Strong Evidence & Generally Useful):**
        * **Creatine Monohydrate:** 3-5g daily. Improves strength, power, and muscle mass. The most studied sports supplement.
        * **Protein Powder (Whey/Casein/Plant-based):** A convenient way to meet your daily protein targets. Not superior to whole food protein.
        * **Vitamin D3:** If you have limited sun exposure. Crucial for hormonal and immune health.
    * **Tier 2 (Moderate Evidence for Specific Uses):**
        * **Caffeine:** 3-6mg/kg body weight, 30-60 min pre-workout can improve performance.
        * **Omega-3 (EPA/DHA):** 1-3g daily if your diet is low in fatty fish. Supports cardiovascular health.
    * **❌ Generally Unnecessary:** Fat burners, testosterone boosters, BCAAs (if protein intake is adequate).
    """)

# ------ MODIFIED: Reinstated all guides in a new, organized sidebar structure ------
st.sidebar.markdown("---")
st.sidebar.subheader("💡 Practical Tips")
with st.sidebar.container(border=True):
    st.markdown("""
    **Weigh-In Best Practices:**
    * Weigh yourself daily, in the morning, after using the bathroom, and before eating/drinking.
    * **Focus on the 7-day average**, not daily numbers. This smooths out fluctuations from water, salt, and carbs.
    """)
with st.sidebar.container(border=True):
    st.markdown("""
    **Dynamic Monitoring:**
    * Your TDEE is a moving target. **Re-enter your new weight** in the calculator every 4-6 weeks or after every 5kg of weight change to update your targets.
    * If progress stalls, re-evaluate your selected **Activity Level**. It's the most common source of miscalculation.
    """)
with st.sidebar.container(border=True):
    st.markdown("""
    **Sleep & Stress:**
    * Aim for **7-9 hours of quality sleep**. Less than 6 hours can significantly impair fat loss and muscle gain.
    * Chronic stress elevates cortisol, which can increase fat storage. Prioritize stress management.
    """)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Reference Guides")
with st.sidebar.container(border=True):
    st.markdown("""
    **Activity Level Guide:**
    * **Sedentary**: Little to no exercise, desk job.
    * **Lightly Active**: Light exercise/sports 1-3 days/week.
    * **Moderately Active**: Moderate exercise/sports 3-5 days/week.
    * **Very Active**: Hard exercise/sports 6-7 days/week.
    * **Extremely Active**: Very hard exercise, physical job, or training twice a day.
    """)
with st.sidebar.container(border=True):
    st.markdown("""
    **Emoji Food Ranking:**
    * 🥇 **Top Tier**: High in both calories and its primary nutrient.
    * 🔥 **Calorie Dense**: Among the most energy-dense options.
    * 💪 **High Protein**: A leading source of protein.
    * 🍚 **High Carb**: A leading source of carbohydrates.
    * 🥑 **High Fat**: A leading source of healthy fats.
    """)
with st.sidebar.container(border=True):
    st.markdown("""
    **The Role of Fitness 🏋️‍♀️:**
    Nutrition provides the building materials, but **resistance training provides the stimulus.**
    * **For Fat Loss**: It signals the body to preserve muscle.
    * **For Muscle Gain**: It's the non-negotiable trigger for growth. A caloric surplus without training results in fat gain.
    * **Recommendation**: Train each major muscle group **2-3 times per week**.
    """)
