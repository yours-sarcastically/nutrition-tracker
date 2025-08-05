# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application based on evidence-based nutritional science. 
It calculates personalized daily targets for calories, protein, fat, and carbohydrates using the Mifflin-St Jeor 
equation for Basal Metabolic Rate (BMR) and scientifically validated activity multipliers for Total Daily Energy 
Expenditure (TDEE). The application supports multiple goals (weight loss, maintenance, gain) using percentage-based 
caloric adjustments and follows a protein-first macronutrient distribution approach as recommended by current 
nutritional research.

Scientific Foundation:
- BMR calculation uses the Mifflin-St Jeor equation (highest validity for healthy adults)
- TDEE calculation employs evidence-based activity multipliers
- Goal-specific caloric adjustments use percentage-based approach for optimal sustainability
- Macronutrient distribution follows protein-first strategy for body composition optimization
- Estimated rate of change calculation based on thermodynamic principles (7700 kcal ≈ 1 kg body fat)
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
    'protein_per_kg': None,  # Will be set based on goal
    'fat_percentage': None   # Will be set based on goal
}

# ------ Activity Level Multipliers for TDEE Calculation (Evidence-Based) ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Goal-Specific Targets Based on Scientific Literature ------
GOAL_CONFIGS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # -20% from TDEE
        'protein_per_kg': 1.8,       # Higher protein for muscle preservation
        'fat_percentage': 0.25,      # 25% of total calories
        'label': 'Weight Loss',
        'description': 'Sustainable fat loss while preserving muscle mass'
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,   # 0% from TDEE
        'protein_per_kg': 1.6,       # Maintenance protein needs
        'fat_percentage': 0.30,      # 30% of total calories for hormone health
        'label': 'Weight Maintenance',
        'description': 'Maintain current weight and body composition'
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # +10% over TDEE
        'protein_per_kg': 2.0,       # Higher protein for muscle building
        'fat_percentage': 0.25,      # 25% of total calories
        'label': 'Weight Gain',
        'description': 'Conservative surplus for lean muscle growth'
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
        'goal': {'type': 'selectbox', 'label': 'Primary Goal', 'options': [
            ("Select Goal", None),
            ("Weight Loss", "weight_loss"),
            ("Weight Maintenance", "weight_maintenance"),
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
    """Create input widgets using unified configuration, now handling advanced fields."""
    session_key = f'user_{field_name}'
    
    if field_config['type'] == 'number':
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
        'protein': 'for muscle building and preservation',
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
# Cell 5: Evidence-Based Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """
    Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation
    
    Scientific Rationale: The Mifflin-St Jeor equation is recognized by the Academy of Nutrition 
    and Dietetics as the most accurate predictive formula for estimating BMR in healthy adults. 
    It consistently outperforms older equations like the Harris-Benedict.
    
    Equations:
    - For Men: BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age in years) + 5
    - For Women: BMR = (10 × weight in kg) + (6.25 × height in cm) - (5 × age in years) - 161
    """
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """
    Calculate Total Daily Energy Expenditure Based on Activity Level
    
    Scientific Rationale: TDEE represents your total "maintenance" calories—the energy required 
    to maintain your current weight with your lifestyle. It's calculated by multiplying BMR by 
    a scientifically validated activity factor.
    """
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """
    Calculate Estimated Weekly Weight Change
    
    Scientific Rationale: Based on the approximation that 1 kg of body fat contains ~7700 kcal.
    This calculation provides an estimate of expected rate of change for monitoring purposes.
    
    Equation: Est. Weekly Change (kg) = (Daily Caloric Adjustment × 7) / 7700
    """
    weekly_caloric_change = daily_caloric_adjustment * 7
    return weekly_caloric_change / 7700

def calculate_personalized_targets(age, height_cm, weight_kg, sex='male', activity_level='moderately_active', goal='weight_gain'):
    """
    Calculate Personalized Daily Nutritional Targets Using Evidence-Based Methods
    
    Scientific Approach:
    1. Calculate BMR using Mifflin-St Jeor equation (highest validity)
    2. Calculate TDEE using validated activity multipliers
    3. Apply percentage-based caloric adjustment based on goal
    4. Use protein-first macronutrient strategy with goal-specific targets
    5. Calculate estimated rate of change for monitoring
    """
    # Step 1 & 2: Calculate BMR and TDEE
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Step 3: Apply goal-specific caloric adjustment (percentage-based approach)
    goal_config = GOAL_CONFIGS.get(goal, GOAL_CONFIGS['weight_gain'])
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment
    
    # Step 4: Protein-first macronutrient distribution
    # Protein (The Builder): Set first based on body weight and goal
    protein_g = goal_config['protein_per_kg'] * weight_kg
    protein_calories = protein_g * 4
    
    # Fat (The Regulator): Essential for hormone production, set as percentage of total calories
    fat_calories = total_calories * goal_config['fat_percentage']
    fat_g = fat_calories / 9
    
    # Carbohydrates (The Fuel): Fill remaining energy needs
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4
    
    # Step 5: Calculate estimated rate of change
    estimated_weekly_change = calculate_estimated_weekly_change(caloric_adjustment)

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 3),
        'goal_label': goal_config['label']
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
# Cell 8: Application Title and Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
**Multi-Goal Nutrition Planning Based on Scientific Research**

This advanced nutrition tracker uses evidence-based calculations to create personalized daily targets for **weight loss**, **maintenance**, or **weight gain**. Built on the Mifflin-St Jeor BMR equation and scientifically validated activity multipliers, it employs a protein-first macronutrient strategy for optimal body composition results. 🚀
""")

st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Render all input fields
for field_name, field_config in CONFIG['form_fields'].items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Process final values using unified approach
final_values = get_final_values(all_inputs)

# Check user input completeness dynamically
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
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets.")
    st.header("Sample Daily Targets for Reference 🎯")
    st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
else:
    st.header(f"Your Personalized Daily Targets for **{targets['goal_label']}** 🎯")

# Unified Metrics Display Configuration
metrics_config = [
    {
        'title': 'Metabolic Information & Rate of Change', 'columns': 4,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal/day"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal/day"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+.0f} kcal"),
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
            ("", "")
        ]
    }
]

# Display all metrics using unified system
for config in metrics_config:
    if config['title'] != 'Metabolic Information & Rate of Change':
        st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 10: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Foods and Log Servings for Today 📝")
st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item.")

# Create Category Tabs for Food Organization
available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories + ["📚 Scientific Background"])

for i, category in enumerate(available_categories):
    items = foods[category]
    sorted_items = sorted(items, key=lambda x: (CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']))
    with tabs[i]:
        render_food_grid(sorted_items, category, 2)

# Scientific Background Tab
with tabs[-1]:
    st.header("Evidence-Based Nutrition Science")
    
    info_cols = st.columns(2)
    
    with info_cols[0]:
        with st.expander("🧬 **Metabolic Calculations**", expanded=True):
            st.markdown("""
            **BMR Calculation (Mifflin-St Jeor Equation)**
            - **Men:** BMR = (10 × kg) + (6.25 × cm) - (5 × age) + 5
            - **Women:** BMR = (10 × kg) + (6.25 × cm) - (5 × age) - 161

            **TDEE Calculation**
            - TDEE = BMR × Activity Multiplier
            - Most accurate method for estimating maintenance calories

            **Rate of Change Estimation**
            - Based on ~7700 kcal per kg of body fat
            - Weekly Change = (Daily Adjustment × 7) ÷ 7700
            """)
        
        with st.expander("🎯 **Goal-Specific Strategy**", expanded=True):
            st.markdown("""
            **Weight Loss (-20% TDEE)**
            - Sustainable rate: 0.5-1% body weight/week
            - Higher protein (1.8g/kg) preserves muscle
            - 25% calories from fat for hormone health

            **Weight Maintenance (0% TDEE)**
            - Energy balance: calories in = calories out
            - Moderate protein (1.6g/kg) maintains muscle
            - 30% calories from fat for optimal health

            **Weight Gain (+10% TDEE)**
            - Conservative surplus minimizes fat gain
            - Higher protein (2.0g/kg) supports muscle growth
            - 25% calories from fat balances performance
            """)
        
        with st.expander("🥗 **Macronutrient Strategy**", expanded=True):
            st.markdown("""
            **Protein-First Approach:**
            1. **Protein** set based on body weight and goal
            2. **Fat** set for hormonal health (20-30% calories)
            3. **Carbohydrates** fill remaining energy needs

            **Scientific Rationale:**
            - Protein: muscle preservation/building, satiety
            - Fat: hormone production, nutrient absorption  
            - Carbs: primary fuel for high-intensity exercise
            """)
    
    with info_cols[1]:
        with st.expander("💪 **Fitness Integration**", expanded=True):
            st.markdown("""
            **The Critical Role of Resistance Training:**

            **During Fat Loss:** Preserves metabolically active muscle tissue and prevents metabolic slowdown.

            **During Weight Gain:** The non-negotiable stimulus for muscle growth. Without it, surplus calories become fat.

            **Minimum Effective Guidelines:**
            - **Resistance Training:** Each muscle group 2-3×/week
            - **Cardiovascular Exercise:** 150-300 min/week moderate intensity
            - **Recovery:** Adequate sleep and rest between sessions

            *Note: Nutrition provides building materials; exercise provides the blueprint.*
            """)
        
        with st.expander("🏃‍♂️ **Activity Level Guide**", expanded=True):
            st.markdown("""
            **Accurate TDEE depends on honest activity assessment:**

            - **Sedentary (1.2×):** Desk job, minimal exercise
            - **Lightly Active (1.375×):** Light exercise 1-3 days/week
            - **Moderately Active (1.55×):** Moderate exercise 3-5 days/week  
            - **Very Active (1.725×):** Hard exercise 6-7 days/week
            - **Extremely Active (1.9×):** Very hard exercise + physical job

            *Tip: Most people overestimate their activity level. When in doubt, choose the lower option.*
            """)
        
        with st.expander("📊 **Dynamic Monitoring**", expanded=True):
            st.markdown("""
            **Your TDEE adapts as you lose/gain weight:**

            1. **Track Progress:** Compare estimated vs. actual weekly changes
            2. **Adjust Inputs:** If progress stalls, reassess activity level
            3. **Stay Flexible:** Metabolic adaptation requires plan adjustments
            4. **Focus on Trends:** Daily fluctuations are normal; track weekly averages

            *This transforms a static plan into a responsive, adaptive system.*
            """)
        
        with st.expander("🏆 **Food Ranking System**", expanded=True):
            st.markdown("""
            **Emoji Guide for Optimal Food Selection:**

            - 🥇 **Nutrient & Calorie Dense:** Top performer in both categories
            - 🔥 **High-Calorie:** Most energy-dense in its category
            - 💪 **Top Protein Source:** Leading protein contributor
            - 🍚 **Top Carb Source:** Leading carbohydrate contributor  
            - 🥑 **Top Fat Source:** Leading healthy fat contributor

            *Rankings help identify the most nutritionally efficient foods for your goals.*
            """)

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

    # Dynamically generate intake metrics from CONFIG
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

    # Enhanced caloric balance analysis with goal context
    st.subheader("Daily Caloric Balance and Goal Progress Summary ⚖️")
    cal_balance = totals['calories'] - targets['tdee']
    
    if final_values['goal'] == 'weight_loss':
        if cal_balance < 0:
            st.info(f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance, supporting your weight loss goal.")
        else:
            st.warning(f"📈 You are consuming {cal_balance:.0f} kcal above maintenance, which may slow weight loss progress.")
    elif final_values['goal'] == 'weight_maintenance':
        if abs(cal_balance) <= 50:
            st.success(f"⚖️ You are within {abs(cal_balance):.0f} kcal of maintenance - excellent for weight stability!")
        elif cal_balance > 50:
            st.info(f"📈 You are consuming {cal_balance:.0f} kcal above maintenance.")
        else:
            st.info(f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance.")
    elif final_values['goal'] == 'weight_gain':
        if cal_balance > 0:
            st.info(f"📈 You are consuming {cal_balance:.0f} kcal above maintenance, supporting your weight gain goal.")
        else:
            st.warning(f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance, which may impede weight gain progress.")

    # Display target vs actual comparison
    target_surplus_deficit = targets['caloric_adjustment']
    actual_surplus_deficit = cal_balance
    difference = actual_surplus_deficit - target_surplus_deficit
    
    if abs(difference) <= 50:
        st.success(f"🎯 Your intake is within {abs(difference):.0f} kcal of your target adjustment!")
    else:
        direction = "above" if difference > 0 else "below"
        st.info(f"📊 Your intake is {abs(difference):.0f} kcal {direction} your target adjustment of {target_surplus_deficit:+.0f} kcal.")

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
# Cell 12: Clear Selections
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()
