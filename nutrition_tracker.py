# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker - Enhanced Hybrid UI Model
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application based on evidence-based nutritional science. 
It uses a hybrid information architecture that provides contextual learning through tooltips, comprehensive 
education through a dedicated guide, and quick references through streamlined sidebar content.

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

# ------ Contextual Help Content for Tooltips ------
CONTEXTUAL_HELP = {
    'bmr': {
        'title': 'Basal Metabolic Rate (BMR)',
        'content': '''**BMR** is the energy your body burns at complete rest to maintain basic physiological functions.

**Calculation (Mifflin-St Jeor Equation):**
- **Men:** BMR = (10 × kg) + (6.25 × cm) - (5 × age) + 5  
- **Women:** BMR = (10 × kg) + (6.25 × cm) - (5 × age) - 161

**Why this equation?** Recognized by the Academy of Nutrition and Dietetics as the most accurate for healthy adults.'''
    },
    'tdee': {
        'title': 'Total Daily Energy Expenditure (TDEE)',
        'content': '''**TDEE** represents your total "maintenance" calories—the energy required to maintain your current weight with your lifestyle.

**Calculation:** TDEE = BMR × Activity Multiplier

**Scientific Rationale:** This method accounts for all energy expenditure beyond basic metabolic functions, including exercise, work activities, and daily movement.'''
    },
    'caloric_adjustment': {
        'title': 'Daily Caloric Adjustment',
        'content': '''**Caloric Adjustment** is the daily calorie modification from your maintenance level to achieve your goal.

**Evidence-Based Targets:**
- **Weight Loss:** -20% from TDEE (sustainable 0.5-1% body weight loss/week)
- **Maintenance:** 0% from TDEE (energy balance)
- **Weight Gain:** +10% over TDEE (conservative surplus minimizes fat gain)'''
    },
    'weekly_change': {
        'title': 'Estimated Weekly Weight Change',
        'content': '''**Weekly Change Estimate** is based on the thermodynamic principle that ~7700 kcal equals 1 kg of body fat.

**Formula:** (Daily Caloric Adjustment × 7) ÷ 7700 = kg/week

**Important:** This is an estimate. Actual results vary due to water retention, metabolic adaptation, and individual differences.'''
    },
    'protein_target': {
        'title': 'Protein Target Rationale',
        'content': '''**Protein-First Strategy** sets protein based on body weight and goal for optimal body composition.

**Evidence-Based Targets:**
- **Weight Loss:** 1.8g/kg (preserves muscle during calorie deficit)
- **Maintenance:** 1.6g/kg (maintains muscle mass and function)  
- **Weight Gain:** 2.0g/kg (supports muscle protein synthesis)'''
    },
    'macronutrient_distribution': {
        'title': 'Macronutrient Distribution Strategy',
        'content': '''**Scientific Approach:**
1. **Protein** set first based on body weight and goal
2. **Fat** set for hormonal health (20-30% of calories)
3. **Carbohydrates** fill remaining energy needs

**Rationale:** This ensures adequate protein for body composition goals while meeting essential fat requirements for health.'''
    }
}

# -----------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables using unified approach"""
    session_vars = ['food_selections', 'show_help'] + [f'user_{field}' for field in CONFIG['form_fields'].keys()]
    
    for var in session_vars:
        if var not in st.session_state:
            if var == 'food_selections':
                st.session_state[var] = {}
            elif var == 'show_help':
                st.session_state[var] = {}
            else:
                st.session_state[var] = None

def show_contextual_help(help_key, container=st):
    """Display contextual help using modal-like expander"""
    help_content = CONTEXTUAL_HELP.get(help_key)
    if help_content:
        help_expander = container.expander(f"ℹ️ {help_content['title']}", expanded=False)
        help_expander.markdown(help_content['content'])

def create_metric_with_help(label, value, delta=None, help_key=None, container=st):
    """Create metric with optional contextual help"""
    col1, col2 = container.columns([4, 1])
    
    with col1:
        if delta:
            st.metric(label, value, delta)
        else:
            st.metric(label, value)
    
    with col2:
        if help_key and st.button("ℹ️", key=f"help_{help_key}", help=f"Learn about {label}"):
            show_contextual_help(help_key, container)

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

def display_metrics_grid_with_help(metrics_data, num_columns=4):
    """Display metrics in a configurable column layout with contextual help"""
    columns = st.columns(num_columns)
    
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                label, value = metric_info
                st.metric(label, value)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                st.metric(label, value, delta)
            elif len(metric_info) == 4:  # New format with help
                label, value, delta, help_key = metric_info
                create_metric_with_help(label, value, delta, help_key, st)

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
# Cell 7: Science Guide Page Function
# -----------------------------------------------------------------------------

def render_science_guide():
    """Render the comprehensive science guide as a separate page component"""
    st.header("Evidence-Based Nutrition Science Guide 🧬")
    st.markdown("""
    This comprehensive guide explains the scientific foundation behind your personalized nutrition targets. 
    Each section is based on peer-reviewed research and established nutritional science principles.
    """)
    
    # Create two main columns for organized content
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Metabolic Calculations")
        with st.expander("Basal Metabolic Rate (BMR)"):
            st.markdown("""
            **BMR** represents the energy your body burns at complete rest to maintain basic physiological functions like breathing, circulation, and cellular maintenance.
            
            **Mifflin-St Jeor Equation (Most Accurate):**
            - **Men:** BMR = (10 × weight kg) + (6.25 × height cm) - (5 × age) + 5
            - **Women:** BMR = (10 × weight kg) + (6.25 × height cm) - (5 × age) - 161
            
            **Why This Equation?** The Academy of Nutrition and Dietetics recognizes it as the most accurate predictor for healthy adults, consistently outperforming older formulas like Harris-Benedict.
            """)
        
        with st.expander("Total Daily Energy Expenditure (TDEE)"):
            st.markdown("""
            **TDEE** is your total "maintenance" calories—the energy needed to maintain your current weight with your lifestyle.
            
            **Calculation:** TDEE = BMR × Activity Multiplier
            
            **Activity Multipliers (Evidence-Based):**
            - **Sedentary (1.2×):** Desk job, minimal exercise
            - **Lightly Active (1.375×):** Light exercise 1-3 days/week
            - **Moderately Active (1.55×):** Moderate exercise 3-5 days/week
            - **Very Active (1.725×):** Hard exercise 6-7 days/week
            - **Extremely Active (1.9×):** Very hard exercise + physical job
            """)
        
        st.subheader("🎯 Goal-Specific Strategies")
        with st.expander("Weight Loss Strategy"):
            st.markdown("""
            **Target:** -20% below TDEE for sustainable fat loss
            
            **Scientific Rationale:**
            - Creates moderate deficit of ~0.5-1% body weight loss per week
            - Preserves metabolically active muscle tissue
            - Higher protein (1.8g/kg) prevents muscle loss during deficit
            - Fat at 25% of calories maintains hormone production
            
            **Key Success Factors:**
            - Resistance training to preserve muscle mass
            - Adequate sleep (7-9 hours) for recovery and hormones
            - Consistent meal timing for metabolic regulation
            """)
        
        with st.expander("Weight Maintenance Strategy"):
            st.markdown("""
            **Target:** 0% adjustment from TDEE (energy balance)
            
            **Scientific Rationale:**
            - Matches energy intake to expenditure
            - Protein at 1.6g/kg maintains muscle mass and function
            - Fat at 30% of calories optimizes hormone production
            - Flexible approach allows for body recomposition
            
            **Applications:**
            - Transitioning between cutting/bulking phases
            - Long-term health maintenance
            - Body recomposition (simultaneous fat loss + muscle gain)
            """)
        
        with st.expander("Weight Gain Strategy"):
            st.markdown("""
            **Target:** +10% above TDEE for conservative muscle growth
            
            **Scientific Rationale:**
            - Small surplus minimizes fat accumulation
            - Higher protein (2.0g/kg) maximizes muscle protein synthesis
            - Moderate fat (25%) allows more calories for carbohydrates
            - Sustainable approach prevents excessive fat gain
            
            **Optimization Tips:**
            - Progressive resistance training is essential
            - Focus on compound movements (squats, deadlifts, presses)
            - Adequate recovery between training sessions
            """)
    
    with col2:
        st.subheader("🥗 Macronutrient Science")
        with st.expander("Protein-First Strategy"):
            st.markdown("""
            **Why Protein First?**
            Protein needs are based on body weight and goal, not percentages. This ensures optimal body composition regardless of total caloric intake.
            
            **Evidence-Based Targets:**
            - **Weight Loss:** 1.8g/kg (muscle preservation during deficit)
            - **Maintenance:** 1.6g/kg (optimal health and function)
            - **Weight Gain:** 2.0g/kg (maximizes muscle protein synthesis)
            
            **Benefits:**
            - Higher thermic effect (burns more calories to digest)
            - Greater satiety (helps control hunger)
            - Preserves lean body mass during weight changes
            - Supports immune function and recovery
            """)
        
        with st.expander("Carbohydrate Strategy"):
            st.markdown("""
            **Role:** Primary fuel source for brain and muscles
            
            **Calculation:** Fills remaining calories after protein and fat are set
            
            **Optimization:**
            - **Pre-workout:** Fast-digesting carbs for energy
            - **Post-workout:** Combined with protein for recovery
            - **Throughout day:** Focus on fiber-rich, nutrient-dense sources
            
            **Quality Sources:**
            - Whole grains, legumes, fruits, vegetables
            - Minimize processed sugars and refined carbohydrates
            """)
        
        with st.expander("Fat Requirements"):
            st.markdown("""
            **Essential Functions:**
            - Hormone production (testosterone, estrogen, growth hormone)
            - Vitamin absorption (A, D, E, K)
            - Cell membrane integrity
            - Inflammatory regulation
            
            **Target Range:** 20-35% of total calories
            - **Weight Loss/Gain:** 25% (allows more flexibility for protein/carbs)
            - **Maintenance:** 30% (optimal for hormone health)
            
            **Quality Sources:**
            - Unsaturated fats: olive oil, nuts, avocados, fatty fish
            - Limit saturated fats: <10% of total calories
            """)
        
        st.subheader("💪 Exercise Integration")
        with st.expander("Resistance Training Priority"):
            st.markdown("""
            **Why Resistance Training is Essential:**
            - Preserves/builds muscle mass during all phases
            - Increases metabolic rate (muscle burns more calories at rest)
            - Improves insulin sensitivity and nutrient partitioning
            - Enhances bone density and functional strength
            
            **Minimum Effective Dose:**
            - **Frequency:** 2-3 sessions per week
            - **Volume:** 10-20 sets per muscle group per week
            - **Intensity:** 6-12 rep range for hypertrophy
            - **Progression:** Gradually increase weight, reps, or sets
            """)
        
        with st.expander("Cardiovascular Exercise"):
            st.markdown("""
            **Role in Nutrition Strategy:**
            - Increases total daily energy expenditure
            - Improves cardiovascular health and endurance
            - Enhances recovery between resistance sessions
            - Can aid appetite regulation
            
            **Implementation:**
            - **Weight Loss:** 150-300 minutes moderate intensity per week
            - **Maintenance/Gain:** 150 minutes moderate intensity per week
            - **Types:** Walking, cycling, swimming, sports activities
            - **Timing:** Separate from resistance training when possible
            """)
        
        with st.expander("Recovery and Sleep"):
            st.markdown("""
            **Critical for Results:**
            - **Sleep:** 7-9 hours per night for hormone optimization
            - **Stress Management:** Chronic stress impairs body composition
            - **Hydration:** ~35ml per kg body weight daily
            - **Rest Days:** Allow 48-72 hours between training same muscles
            
            **Sleep's Impact on Nutrition:**
            - Poor sleep increases hunger hormones (ghrelin)
            - Decreases satiety hormones (leptin)
            - Impairs insulin sensitivity
            - Reduces recovery and muscle protein synthesis
            """)
    
    st.subheader("📊 Monitoring and Adjustments")
    with st.expander("Dynamic Monitoring Strategy"):
        st.markdown("""
        **Key Metrics to Track:**
        1. **Body Weight:** Daily, same time, track weekly averages
        2. **Body Measurements:** Weekly (waist, arms, thighs)
        3. **Performance:** Strength, endurance, energy levels
        4. **Subjective Measures:** Sleep quality, hunger, mood
        
        **When to Adjust:**
        - **No change for 2-3 weeks:** Modify calories by 5-10%
        - **Too rapid change:** Adjust in opposite direction
        - **Plateau in strength:** Consider maintenance break
        - **Poor adherence:** Simplify approach or reassess goals
        
        **Individual Variation:**
        Remember that formulas provide starting points. Your actual needs may vary based on genetics, medical history, medications, and individual metabolism.
        """)

# -----------------------------------------------------------------------------
# Cell 8: Main Application Interface
# -----------------------------------------------------------------------------

def main():
    """Main application interface with hybrid information architecture"""
    
    # Initialize session state
    initialize_session_state()
    
    # Page title and introduction
    st.title("🍽️ Personalized Evidence-Based Nutrition Tracker")
    st.markdown("""
    **Transform your nutrition with science-backed personalized targets.** This tool uses peer-reviewed research 
    to calculate your optimal daily intake for sustainable results.
    """)
    
    # Load food database
    try:
        foods = load_food_database('vegetarian_foods.csv')
        foods = assign_food_emojis(foods)
    except FileNotFoundError:
        st.error("❌ Food database not found. Please ensure 'vegetarian_foods.csv' is in the same directory.")
        st.stop()
    
    # Create main tabs with Science Guide as dedicated education space
    main_tabs = st.tabs(["🎯 Daily Targets", "📝 Food Selection", "📚 Science Guide", "📊 Results"])
    
    # ===============================
    # TAB 1: DAILY TARGETS
    # ===============================
    with main_tabs[0]:
        st.header("Personal Information & Daily Targets 🎯")
        
        # Sidebar for user inputs with streamlined content
        st.sidebar.markdown("### Personal Information 👤")
        
        # Collect user inputs using unified approach
        user_inputs = {}
        for field_name, field_config in CONFIG['form_fields'].items():
            user_inputs[field_name] = create_unified_input(field_name, field_config)
        
        # Streamlined sidebar - only essential references
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Quick Reference 📋")
        
        with st.sidebar.expander("🏃‍♂️ Activity Level Guide"):
            st.markdown("""
            **Sedentary:** Desk job, minimal exercise
            **Lightly Active:** Light exercise 1-3 days/week  
            **Moderately Active:** Moderate exercise 3-5 days/week
            **Very Active:** Hard exercise 6-7 days/week
            **Extremely Active:** Very hard exercise + physical job
            """)
        
        with st.sidebar.expander("🏆 Food Rankings Guide"):
            st.markdown("""
            **🥇 Elite:** High calories + top macronutrient source
            **🔥 High-Calorie:** Energy-dense options
            **💪 Protein:** Top protein sources  
            **🍚 Carbs:** Top carbohydrate sources
            **🥑 Fats:** Top fat sources
            """)
        
        # Process inputs and calculate targets
        final_values = get_final_values(user_inputs)
        
        # Check if all required fields are provided
        required_check = all(
            final_values[field] not in [None, "Select Sex", "Select Activity Level", "Select Goal"] 
            for field in ['age', 'height_cm', 'weight_kg', 'sex', 'activity_level', 'goal']
        )
        
        if required_check:
            # Calculate personalized targets
            targets = calculate_personalized_targets(**final_values)
            
            # Display results with contextual help
            st.success("✅ Your personalized daily nutrition targets have been calculated!")
            
            # Metabolic calculations with contextual help
            st.subheader("Metabolic Calculations 🔬")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                create_metric_with_help("BMR", f"{targets['bmr']:,} kcal/day", help_key='bmr')
            with col2:
                create_metric_with_help("TDEE", f"{targets['tdee']:,} kcal/day", help_key='tdee')
            with col3:
                sign = "+" if targets['caloric_adjustment'] > 0 else ""
                create_metric_with_help(
                    "Daily Adjustment", 
                    f"{sign}{targets['caloric_adjustment']:,} kcal", 
                    help_key='caloric_adjustment'
                )
            
            # Goal-specific targets
            st.subheader(f"Daily Targets for {targets['goal_label']} 🎯")
            
            # Contextual education moment
            st.info("ℹ️ **Calculation Method**: Using the Mifflin-St Jeor equation (most accurate for healthy adults) with evidence-based macronutrient distribution.")
            
            # Main nutritional targets
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Calories", f"{targets['total_calories']:,} kcal")
            with col2:
                create_metric_with_help(
                    "Protein", 
                    f"{targets['protein_g']}g ({targets['protein_percent']:.0f}%)", 
                    help_key='protein_target'
                )
            with col3:
                st.metric("Carbohydrates", f"{targets['carb_g']}g ({targets['carb_percent']:.0f}%)")
            with col4:
                st.metric("Fat", f"{targets['fat_g']}g ({targets['fat_percent']:.0f}%)")
            
            # Weekly change estimate with contextual help
            st.subheader("Expected Progress 📈")
            col1, col2 = st.columns(2)
            
            with col1:
                change_direction = "gain" if targets['estimated_weekly_change'] > 0 else "loss"
                create_metric_with_help(
                    "Est. Weekly Change", 
                    f"{abs(targets['estimated_weekly_change']):.2f} kg {change_direction}",
                    help_key='weekly_change'
                )
            
            # Progressive disclosure for calculation explanation
            with st.expander("🔬 How were these targets calculated?"):
                st.markdown(f"""
                **Your Personalized Calculation:**
                1. **BMR** = {targets['bmr']} kcal (using Mifflin-St Jeor equation)
                2. **TDEE** = BMR × {ACTIVITY_MULTIPLIERS[final_values['activity_level']]} = {targets['tdee']} kcal
                3. **Goal Adjustment** = {GOAL_CONFIGS[final_values['goal']]['caloric_adjustment']*100:+.0f}% = {targets['caloric_adjustment']:+} kcal
                4. **Target Calories** = {targets['total_calories']} kcal/day
                
                **Macronutrient Distribution:**
                - **Protein**: {GOAL_CONFIGS[final_values['goal']]['protein_per_kg']}g/kg × {final_values['weight_kg']}kg = {targets['protein_g']}g
                - **Fat**: {GOAL_CONFIGS[final_values['goal']]['fat_percentage']*100:.0f}% of calories = {targets['fat_g']}g  
                - **Carbohydrates**: Remaining calories = {targets['carb_g']}g
                """)
            
            # Store targets in session state for other tabs
            st.session_state.targets = targets
            
        else:
            st.warning("⚠️ Please complete all personal information fields to calculate your personalized targets.")
    
    # ===============================
    # TAB 2: FOOD SELECTION
    # ===============================
    with main_tabs[1]:
        st.header("Food Selection & Meal Planning 📝")
        
        if 'targets' not in st.session_state:
            st.warning("⚠️ Please complete your personal information in the 'Daily Targets' tab first.")
        else:
            st.success(f"✅ Building meals for your **{st.session_state.targets['goal_label']}** goal")
            
            # Display current targets for reference
            with st.expander("📋 Your Daily Targets (for reference)"):
                col1, col2, col3, col4 = st.columns(4)
                targets = st.session_state.targets
                col1.metric("Calories", f"{targets['total_calories']} kcal")
                col2.metric("Protein", f"{targets['protein_g']}g")  
                col3.metric("Carbs", f"{targets['carb_g']}g")
                col4.metric("Fat", f"{targets['fat_g']}g")
            
            # Food selection interface
            st.subheader("Select Your Foods 🥘")
            st.markdown("Choose foods and serving sizes to build your daily meal plan. Rankings help identify the most efficient choices for your goals.")
            
            # Create tabs for each food category
            category_tabs = st.tabs(list(foods.keys()))
            
            for i, (category, items) in enumerate(foods.items()):
                with category_tabs[i]:
                    if items:
                        # Sort foods by emoji ranking for better UX
                        sorted_items = sorted(items, key=lambda x: CONFIG['emoji_order'].get(x.get('emoji', ''), 5))
                        render_food_grid(sorted_items, category, columns=2)
                    else:
                        st.info(f"No foods available in {category} category.")
    
    # ===============================
    # TAB 3: SCIENCE GUIDE  
    # ===============================
    with main_tabs[2]:
        render_science_guide()
    
    # ===============================
    # TAB 4: RESULTS & PROGRESS
    # ===============================
    with main_tabs[3]:
        st.header("Daily Progress & Results 📊")
        
        if 'targets' not in st.session_state:
            st.warning("⚠️ Please complete your personal information in the 'Daily Targets' tab first.")
        elif not st.session_state.food_selections:
            st.info("🍽️ Select some foods in the 'Food Selection' tab to see your daily progress.")
        else:
            targets = st.session_state.targets
            totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)
            
            # Current daily intake summary
            st.subheader("Today's Nutrition Summary 📋")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Calories", f"{totals['calories']:.0f} kcal")
            col2.metric("Protein", f"{totals['protein']:.0f}g")
            col3.metric("Carbohydrates", f"{totals['carbs']:.0f}g")  
            col4.metric("Fat", f"{totals['fat']:.0f}g")
            
            # Progress tracking with recommendations
            recommendations = create_progress_tracking(totals, targets)
            
            # Contextual education about macronutrient ratios
            with st.expander("💡 Why these macronutrient ratios?"):
                st.markdown(f"""
                Your **protein target of {targets['protein_g']}g** follows the evidence-based recommendation of 
                **{GOAL_CONFIGS[final_values['goal']]['protein_per_kg']}g per kg body weight** for {targets['goal_label'].lower()}.
                
                **Fat is set at {GOAL_CONFIGS[final_values['goal']]['fat_percentage']*100:.0f}% of calories** to ensure adequate hormone production 
                and vitamin absorption, while **carbohydrates fill the remaining energy needs** to fuel your activities and workouts.
                """)
            
            # Recommendations for improvement
            if recommendations:
                st.subheader("Personalized Recommendations 💡")
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("🎉 Congratulations! You've met all your daily nutritional targets!")
            
            # Selected foods summary  
            if selected_foods:
                st.subheader("Selected Foods Summary 📝")
                
                foods_df = pd.DataFrame([
                    {
                        'Food': item['food']['name'],
                        'Servings': f"{item['servings']:.1f}",
                        'Calories': f"{item['food']['calories'] * item['servings']:.0f}",
                        'Protein (g)': f"{item['food']['protein'] * item['servings']:.1f}",
                        'Carbs (g)': f"{item['food']['carbs'] * item['servings']:.1f}",
                        'Fat (g)': f"{item['food']['fat'] * item['servings']:.1f}"
                    }
                    for item in selected_foods
                ])
                
                st.dataframe(foods_df, use_container_width=True, hide_index=True)
                
                # Reset button
                if st.button("🔄 Reset All Food Selections", type="secondary"):
                    st.session_state.food_selections = {}
                    st.rerun()

# -----------------------------------------------------------------------------
# Cell 9: Application Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
