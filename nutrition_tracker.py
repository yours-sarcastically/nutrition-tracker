# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker - Enhanced Hybrid UI Model
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application based on evidence-based nutritional science. 
It uses a hybrid information architecture that provides contextual learning through pop-ups, comprehensive 
education through a dedicated guide tab, and quick references through streamlined sidebar content.

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
            ("Weight Gain", "weight_gain"),
            ("Weight Maintenance", "weight_maintenance"),
            ("Weight Loss", "weight_loss"),
        ], 'required': True, 'placeholder': None}
    }
}

# ------ Contextual Help Content for Popovers ------
CONTEXTUAL_HELP = {
    'bmr': {
        'title': 'Basal Metabolic Rate (BMR)',
        'content': '''**BMR** is the energy your body burns at complete rest. We use the **Mifflin-St Jeor equation**, recognized by the Academy of Nutrition and Dietetics as the most accurate for healthy adults.'''
    },
    'tdee': {
        'title': 'Total Daily Energy Expenditure (TDEE)',
        'content': '''**TDEE**, or your "maintenance calories," is your BMR multiplied by a scientifically validated **activity factor**. It's the energy needed to maintain your current weight.'''
    },
    'caloric_adjustment': {
        'title': 'Daily Caloric Adjustment',
        'content': '''This is the calorie modification from your TDEE to achieve your goal. We use a **percentage-based approach** for safety and sustainability (-20% for loss, +10% for gain).'''
    },
    'weekly_change': {
        'title': 'Estimated Weekly Weight Change',
        'content': '''This estimate is based on the principle that a **~7700 kcal** deficit or surplus results in ~1 kg of weight change. It's a guide; actual results will vary.'''
    },
    'protein_target': {
        'title': 'Protein Target Rationale',
        'content': '''Your protein target is set based on your **body weight and goal**, not percentages. This ensures optimal muscle preservation (loss) or growth (gain).'''
    }
}

# -----------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize all session state variables."""
    session_vars = ['food_selections', 'show_results']
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else False

def create_metric_with_help(label, value, help_key=None, container=st):
    """Create a metric with an optional popover for contextual help."""
    if help_key and help_key in CONTEXTUAL_HELP:
        with container:
            col1, col2 = st.columns([0.85, 0.15])
            col1.metric(label, value)
            with col2:
                # Use st.popover for a clean, non-disruptive help tooltip
                with st.popover("ℹ️", use_container_width=True):
                    st.markdown(f"**{CONTEXTUAL_HELP[help_key]['title']}**")
                    st.markdown(CONTEXTUAL_HELP[help_key]['content'])
    else:
        container.metric(label, value)

def create_unified_input(field_name, field_config, container=st.sidebar):
    """Create input widgets using unified configuration."""
    session_key = f'user_{field_name}'
    if session_key not in st.session_state:
        st.session_state[session_key] = None

    if field_config['type'] == 'number':
        value = container.number_input(field_config['label'], min_value=field_config['min'], max_value=field_config['max'], value=st.session_state[session_key], step=field_config['step'], placeholder=field_config.get('placeholder'))
    elif field_config['type'] == 'selectbox':
        options = field_config['options']
        # Handle tuple format (label, value)
        if isinstance(options[0], tuple):
            index = next((i for i, (_, val) in enumerate(options) if val == st.session_state[session_key]), 0)
            selection = container.selectbox(field_config['label'], options, index=index, format_func=lambda x: x[0])
            value = selection[1]
        else:
            index = options.index(st.session_state[session_key]) if st.session_state[session_key] in options else 0
            value = container.selectbox(field_config['label'], options, index=index)
    
    st.session_state[session_key] = value
    return value

def get_final_values(user_inputs):
    """Process all user inputs and apply defaults."""
    final_values = {}
    for field, value in user_inputs.items():
        if field == 'sex' and value == "Select Sex":
            final_values[field] = DEFAULTS[field]
        elif value is None:
            final_values[field] = DEFAULTS[field]
        else:
            final_values[field] = value
    return final_values

def create_progress_tracking(totals, targets, goal_label):
    """Create unified progress tracking with bars and recommendations."""
    st.subheader("Progress Toward Daily Targets 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]
        percent = min(actual / target * 100, 100) if target > 0 else 0
        st.progress(percent / 100, text=f"{config['label']}: {actual:.0f} / {target:.0f} {config['unit']} ({percent:.0f}%)")

# -----------------------------------------------------------------------------
# Cell 5: Evidence-Based Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)

def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure Based on Activity Level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_personalized_targets(age, height_cm, weight_kg, sex, activity_level, goal):
    """Calculate Personalized Daily Nutritional Targets Using Evidence-Based Methods."""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    
    goal_config = GOAL_CONFIGS.get(goal, GOAL_CONFIGS['weight_gain'])
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment
    
    protein_g = goal_config['protein_per_kg'] * weight_kg
    protein_calories = protein_g * 4
    
    fat_calories = total_calories * goal_config['fat_percentage']
    fat_g = fat_calories / 9
    
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4
    
    estimated_weekly_change = (caloric_adjustment * 7) / 7700

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 2),
        'goal_label': goal_config['label']
    }
    
    if targets['total_calories'] > 0:
        targets.update({
            'protein_percent': (protein_calories / total_calories) * 100,
            'carb_percent': (carb_calories / total_calories) * 100,
            'fat_percent': (fat_calories / total_calories) * 100
        })
    else:
        targets.update({'protein_percent': 0, 'carb_percent': 0, 'fat_percent': 0})
        
    return targets

# -----------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path):
    """Load the Food Database From a CSV File."""
    try:
        df = pd.read_csv(file_path)
        foods = {cat: [] for cat in df['category'].unique()}
        for _, row in df.iterrows():
            foods[row['category']].append(row.to_dict())
        return foods
    except FileNotFoundError:
        st.error(f"❌ Food database not found. Please ensure '{file_path}' is in the correct directory.")
        return None

def assign_food_emojis(foods):
    """Assign emojis to foods using a unified ranking system."""
    if not foods: return None
    # This logic remains complex but is preserved as per original functionality.
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'calories': {}}
    for category, items in foods.items():
        if not items: continue
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [d['name'] for d in sorted_by_calories[:3]]
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(items, key=lambda x: x[map_info['sort_by']], reverse=True)
            top_foods[map_info['key']] = [d['name'] for d in sorted_by_nutrient[:3]]
    
    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat'] for food in top_foods[key]}
    emoji_mapping = {'high_cal_nutrient': '🥇', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑'}

    for category, items in foods.items():
        for food in items:
            is_top_nutrient = food['name'] in all_top_nutrient_foods
            is_high_calorie = food['name'] in top_foods['calories'].get(category, [])
            if is_high_calorie and is_top_nutrient: food['emoji'] = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: food['emoji'] = emoji_mapping['high_calorie']
            elif food['name'] in top_foods['protein']: food['emoji'] = emoji_mapping['protein']
            elif food['name'] in top_foods['carbs']: food['emoji'] = emoji_mapping['carbs']
            elif food['name'] in top_foods['fat']: food['emoji'] = emoji_mapping['fat']
            else: food['emoji'] = ''
    return foods

def render_food_item(food, category):
    """Render a single food item with interaction controls."""
    food_name_unit = f"{food['name']} ({food['serving_unit']})"
    st.subheader(f"{food.get('emoji', '')} {food_name_unit}")
    
    current_serving = st.session_state.food_selections.get(food_name_unit, 0.0)
    
    # Custom serving input
    custom_serving = st.number_input("Servings:", min_value=0.0, max_value=10.0, value=float(current_serving), step=0.5, key=f"num_{category}_{food['name']}")
    
    if custom_serving != current_serving:
        if custom_serving > 0:
            st.session_state.food_selections[food_name_unit] = custom_serving
        elif food_name_unit in st.session_state.food_selections:
            del st.session_state.food_selections[food_name_unit]
        st.rerun()
    
    st.caption(f"Per Serving: {food['calories']} kcal | P:{food['protein']}g | C:{food['carbs']}g | F:{food['fat']}g")

def render_food_grid(items, category, columns=2):
    """Render food items in a grid layout."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)

# -----------------------------------------------------------------------------
# Cell 7: UI Rendering Functions for Main Tabs
# -----------------------------------------------------------------------------

def render_targets_tab(user_inputs, final_values):
    """Renders the UI for the 'My Plan & Targets' tab."""
    st.header("Your Personalized Nutrition Plan 🎯")
    
    required_check = all(final_values.get(field) not in [None, "Select Sex"] for field in ['age', 'height_cm', 'weight_kg', 'sex', 'activity_level', 'goal'])
    
    if not required_check:
        st.info("👈 Please complete all fields in the sidebar to generate your plan.")
        return

    targets = calculate_personalized_targets(**final_values)
    st.session_state.targets = targets
    
    st.success(f"**Plan Generated for {targets['goal_label']}**")
    
    # --- METRICS DISPLAY ---
    st.subheader("Metabolic & Caloric Targets")
    col1, col2, col3 = st.columns(3)
    create_metric_with_help("BMR", f"{targets['bmr']:,} kcal", help_key='bmr', container=col1)
    create_metric_with_help("TDEE", f"{targets['tdee']:,} kcal", help_key='tdee', container=col2)
    create_metric_with_help("Daily Target", f"{targets['total_calories']:,} kcal", help_key='caloric_adjustment', container=col3)

    st.subheader("Daily Macronutrient Targets")
    col1, col2, col3, col4 = st.columns(4)
    create_metric_with_help("Protein", f"{targets['protein_g']}g", help_key='protein_target', container=col1)
    col2.metric("Carbohydrates", f"{targets['carb_g']}g")
    col3.metric("Fat", f"{targets['fat_g']}g")
    create_metric_with_help("Est. Weekly Change", f"{targets['estimated_weekly_change']:+.2f} kg", help_key='weekly_change', container=col4)
    
    # --- ENHANCEMENT: Progressive Disclosure with User's Data ---
    with st.expander("🔬 Show me my exact calculation breakdown"):
        st.markdown(f"""
        Your personalized plan was calculated as follows:
        - **BMR:** `{targets['bmr']}` kcal (based on your age, height, weight, and sex).
        - **TDEE:** `{targets['bmr']}` × `{ACTIVITY_MULTIPLIERS[final_values['activity_level']]}` (your activity level) = `{targets['tdee']}` kcal.
        - **Goal Adjustment:** `{targets['tdee']}` × `{GOAL_CONFIGS[final_values['goal']]['caloric_adjustment']*100:+.0f}%` = `{targets['caloric_adjustment']:+}` kcal.
        - **Final Calorie Target:** `{targets['tdee']}` + `{targets['caloric_adjustment']}` = **`{targets['total_calories']}` kcal.**
        - **Protein Target:** `{final_values['weight_kg']}` kg × `{GOAL_CONFIGS[final_values['goal']]['protein_per_kg']}` g/kg = **`{targets['protein_g']}` g.**
        """)

def render_logging_tab(foods):
    """Renders the UI for the 'Log My Meals' tab."""
    st.header("Log Your Meals for Today 📝")

    if 'targets' not in st.session_state:
        st.warning("⚠️ Please generate your plan in the 'My Plan & Targets' tab first.")
        return
        
    targets = st.session_state.targets
    
    # --- FOOD SELECTION ---
    st.subheader("Select Foods")
    category_tabs = st.tabs(list(foods.keys()))
    for i, (category, items) in enumerate(foods.items()):
        with category_tabs[i]:
            sorted_items = sorted(items, key=lambda x: CONFIG['emoji_order'].get(x.get('emoji', ''), 5))
            render_food_grid(sorted_items, category, columns=3)
            
    st.markdown("---")
    
    # --- RESULTS DISPLAY (DYNAMIC) ---
    if not st.session_state.food_selections:
        st.info("Select some foods above to see your progress.")
    else:
        totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)
        create_progress_tracking(totals, targets, targets['goal_label'])
        
        with st.expander("View Detailed Food Log", expanded=False):
            food_log_data = []
            for item in selected_foods:
                food_name_unit = item['food']
                servings = item['servings']
                # Find the original food dict to get nutrient values
                original_food = next((f for cat_items in foods.values() for f in cat_items if f"{f['name']} ({f['serving_unit']})" == food_name_unit), None)
                if original_food:
                    food_log_data.append({
                        'Food': food_name_unit, 'Servings': servings,
                        'Calories': original_food['calories'] * servings, 'Protein (g)': original_food['protein'] * servings,
                        'Carbs (g)': original_food['carbs'] * servings, 'Fat (g)': original_food['fat'] * servings
                    })
            df_log = pd.DataFrame(food_log_data)
            st.dataframe(df_log.style.format({'Servings': '{:.1f}', 'Calories': '{:.0f}', 'Protein (g)': '{:.1f}', 'Carbs (g)': '{:.1f}', 'Fat (g)': '{:.1f}'}), use_container_width=True, hide_index=True)
            
        if st.button("Clear All Selections"):
            st.session_state.food_selections.clear()
            st.rerun()

def render_science_guide_tab():
    """Renders the comprehensive science guide in its own tab."""
    st.header("The Science Behind Your Plan 📚")
    st.markdown("This guide explains the evidence-based principles used to generate your personalized nutrition targets.")
    
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🎯 Goal-Specific Strategies", expanded=True):
            st.markdown("""
            A one-size-fits-all approach is ineffective. Your needs change based on your goal.
            - **Weight Loss (-20% TDEE):** A sustainable deficit to promote fat loss while preserving muscle. Protein is set high (1.8g/kg).
            - **Weight Maintenance (0% TDEE):** Balances energy to maintain weight. Protein is set at a standard 1.6g/kg.
            - **Weight Gain (+10% TDEE):** A conservative surplus to promote muscle growth while minimizing fat gain. Protein is set highest (2.0g/kg).
            """)
        with st.expander("📊 Dynamic Monitoring & Adaptation"):
            st.markdown("""
            Your metabolism adapts. Compare the app's *estimated* weekly change to your *actual* weekly weight average. If you plateau for 2-3 weeks, your TDEE has likely changed. Re-evaluating your inputs, especially your **activity level**, is key.
            """)
    with col2:
        with st.expander("💪 The Role of Fitness", expanded=True):
            st.markdown("""
            Nutrition provides the bricks, but **resistance training is the architect** that tells your body where to put them.
            - **For Fat Loss:** It signals your body to keep muscle.
            - **For Weight Gain:** It's the non-negotiable trigger for muscle growth.
            - **Recommendation:** Train each major muscle group **2-3 times per week**.
            """)
        with st.expander("🥗 Macronutrient Science"):
            st.markdown("""
            We use a **Protein-First** approach. Protein is set based on your body weight and goal. Fat is set as a percentage of calories for hormone health. Carbohydrates, your primary fuel, fill the rest.
            """)

# -----------------------------------------------------------------------------
# Cell 8: Main Application Logic
# -----------------------------------------------------------------------------

def main():
    """Main application function to orchestrate the UI."""
    initialize_session_state()

    st.title("Evidence-Based Nutrition Tracker")
    
    # --- Sidebar for Inputs ---
    st.sidebar.header("Your Personal Parameters 👤")
    user_inputs = {field: create_unified_input(field, config) for field, config in CONFIG['form_fields'].items()}
    final_values = get_final_values(user_inputs)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick References")
    with st.sidebar.expander("🏃‍♂️ Activity Level Guide"):
        st.markdown("- **Sedentary**: Desk job, little exercise\n- **Lightly Active**: Light exercise 1-3 days/wk\n- **Moderately Active**: Moderate exercise 3-5 days/wk\n- **Very Active**: Hard exercise 6-7 days/wk\n- **Extremely Active**: Very hard exercise + physical job")
    with st.sidebar.expander("💡 Food Emoji Guide"):
        st.markdown("- 🥇 Elite (High Calorie & Macro)\n- 🔥 High-Calorie\n- 💪 Top Protein\n- 🍚 Top Carbs\n- 🥑 Top Fats")

    # --- Main Content with Logical Tabs ---
    foods_db = load_food_database('nutrition_results.csv')
    if foods_db:
        foods = assign_food_emojis(foods_db)
        
        # ENHANCEMENT: A logical tab structure for a clear user journey
        plan_tab, logging_tab, guide_tab = st.tabs(["🎯 My Plan & Targets", "📝 Log My Meals", "📚 The Science Guide"])
        
        with plan_tab:
            render_targets_tab(user_inputs, final_values)
        
        with logging_tab:
            render_logging_tab(foods)
            
        with guide_tab:
            render_science_guide_tab()

if __name__ == "__main__":
    main()
