# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker - Enhanced Version
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application for personalized nutrition goals (weight loss, maintenance, and gain) using vegetarian food sources. It calculates personalized daily targets for calories, protein, fat, and carbohydrates based on user-specific attributes and activity levels, using the Mifflin-St Jeor equation for Basal Metabolic Rate (BMR) and multiplies by an activity factor to estimate Total Daily Energy Expenditure (TDEE). Goal-specific caloric adjustments are applied to support the selected objective. Macronutrient targets follow evidence-based nutritional guidelines with a protein-first approach.

Enhanced with comprehensive educational content, hydration tracking, and evidence-based guidance for long-term success.
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
# Cell 7: Educational Content Definitions
# -----------------------------------------------------------------------------

EDUCATIONAL_CONTENT = {
    'monitoring_tips': """
### 📊 **Smart Progress Tracking**

**Weigh-in best practices:**
- Daily morning weigh-ins (after bathroom, before food)
- Compare weekly averages, not daily readings
- Expect 1-3 day delays between caloric changes and scale responses

**Better progress indicators:**
- **Progress photos** (same lighting, poses, time of day)
- **Body measurements** (waist, chest, arms, thighs)
- **Performance metrics** (strength gains, endurance improvements)
- **How clothes fit** and energy levels

**When to adjust targets:**
- Confirm logging accuracy (±5% of calories)
- Re-validate activity multiplier - habit drift is common
- Adjust calories by 5-10% only after two consecutive stalled weeks
""",

    'lifestyle_factors': """
### 😴 **The Hidden Variables: Sleep & Stress**

**Sleep's impact on body composition:**
- **<7 hours sleep**: Increases cortisol, decreases leptin, increases ghrelin
- **Poor sleep quality**: Can reduce muscle protein synthesis by 18-20%
- **Sleep debt**: Makes fat loss 55% less effective even with identical caloric deficits

**Stress management for better results:**
- **Chronic stress**: Elevates cortisol, promotes abdominal fat storage
- **High cortisol**: Increases appetite and cravings for high-calorie foods
- **Recovery practices**: Meditation, yoga, deep breathing, nature walks

**Optimization strategies:**
- 7-9 hours sleep nightly with consistent sleep/wake times
- Dark, cool sleeping environment (18-20°C)
- Limit screens 1-2 hours before bed
- Morning sunlight exposure for circadian rhythm regulation
""",

    'plateau_strategies': """
### 🔄 **Breaking Through Plateaus**

**For Weight Loss Plateaus:**
1. **Recalculate targets** (your TDEE decreases as you lose weight)
2. **Increase NEAT** (take stairs, park farther, fidget more)
3. **Add 1-2 cardio sessions** (150-200 calories burned)
4. **Implement refeed days** (1 day at maintenance every 7-14 days)
5. **Check food logging accuracy** (weigh foods, track condiments/oils)

**For Weight Gain Plateaus:**
1. **Increase liquid calories** (smoothies, milk, juices)
2. **Add healthy fats** (nuts, oils, avocados - calorie-dense)
3. **Increase meal frequency** (5-6 smaller meals vs 3 large)
4. **Time carbs around workouts** (pre/post training for better utilization)
5. **Reduce excessive cardio** (if doing >300 min/week)

**When to seek professional help:**
- Persistent plateaus despite consistent adherence (4+ weeks)
- Extreme fatigue, hair loss, or mood changes
- Disordered eating patterns developing
- Medical conditions affecting metabolism
""",

    'metabolic_adaptation': """
### 🧬 **Understanding Metabolic Adaptation**

**What happens during extended dieting:**
- Your BMR can decrease by 10-25% during prolonged caloric restriction
- Hormones like leptin, thyroid hormones (T3/T4), and testosterone may decline
- Non-exercise activity thermogenesis (NEAT) often decreases unconsciously

**Signs you need a diet break:**
- Weight loss stalls despite adherence for 2+ weeks
- Persistent fatigue, poor sleep, or mood changes
- Decreased workout performance or recovery
- Constant hunger or food obsession

**Strategic diet breaks:** 1-2 weeks at maintenance calories every 6-8 weeks can help restore hormonal balance and improve long-term adherence.

**Audit your progress every 4-6 weeks:**
- Re-enter updated weight in the tracker (BMR and TDEE will shift)
- Re-check waist/hip measurements or body composition scans
- Assess energy levels, sleep quality, and workout performance
""",

    'protein_timing': """
### ⏰ **Optimizing Protein Distribution**

**Research-backed timing strategies:**
- **Post-workout window**: 20-40g protein within 2 hours of training
- **Even distribution**: Spread protein across 3-4 meals (20-30g per meal)
- **Pre-sleep**: 20-30g casein protein can enhance overnight muscle protein synthesis
- **Leucine threshold**: Each meal should contain ~2.5-3g leucine to maximize muscle protein synthesis

**Practical application:** 
- Aim for ≥0.4g protein per kg body weight per meal
- Distribute across 3-5 feedings daily for optimal muscle protein synthesis
- Pre-workout carbs (~1-2g per kg, 1-2 hours prior) boost high-intensity performance
""",

    'micronutrient_guide': """
### 🌱 **Micronutrient Considerations for Plant-Forward Diets**

**Common shortfalls to monitor:**
- **Vitamin B₁₂**: Critical for vegetarians/vegans (supplement recommended)
- **Iron**: Especially important for vegetarians (combine with vitamin C for better absorption)
- **Zinc**: Important for immune function and testosterone production
- **Calcium**: Ensure adequate intake from fortified foods or supplements
- **Iodine**: Often low in plant-based diets (consider iodized salt)
- **Long-chain Omega-3s (EPA/DHA)**: 1-3g daily if fish intake is low

**Fiber targets:** 
- Aim for 14g fiber per 1000 kcal (≈ 25-38g daily)
- Gradually increase to avoid GI distress; adequate water intake is crucial
- Focus on variety: fruits, vegetables, whole grains, legumes
""",

    'food_quality': """
### 🌱 **Food Quality Within Your Macros**

**The 80/20 principle:**
- 80% of calories from nutrient-dense whole foods
- 20% flexibility for treats and social situations
- This approach is sustainable and prevents restrictive eating patterns

**Prioritize these nutrient-dense options:**
- **Proteins**: Lean meats, fish, eggs, legumes, Greek yogurt, tofu, tempeh
- **Carbs**: Fruits, vegetables, whole grains, potatoes, oats, quinoa
- **Fats**: Nuts, seeds, olive oil, avocados, fatty fish, nut butters

**Quality over quantity principle:**
- Iso-caloric diets with higher whole-food content improve satiety and compliance
- When bulking, "clean surplus" (nutrient-dense foods) limits excess fat gain
- Processed foods can fit within your macros but shouldn't dominate your intake
""",

    'periodization_tips': """
### 🔄 **Long-term Periodization Strategy**

**Fat-loss phases:** 
- Duration: 8-12 weeks maximum
- Follow with 2-week maintenance breaks to restore hormones and adherence
- Monitor for signs of metabolic adaptation

**Muscle-gain phases:** 
- Duration: 12-24 weeks for optimal results
- Consider a mini-cut if body fat increases >5 percentage points above baseline
- Focus on progressive overload in resistance training

**Maintenance phases:**
- Critical for habit reinforcement and metabolic recovery
- Practice flexible tracking (5 days/week instead of 7)
- Regular check-ins: weigh weekly, measure monthly

**Seasonal adjustments:**
- Expect and plan for holiday/vacation fluctuations
- Build in buffer periods around major life events
- Think in years, not weeks - sustainable progress takes time
"""
}

# -----------------------------------------------------------------------------
# Cell 8: Initialize Application
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
# Cell 9: Application Title and Enhanced Input Interface
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
    
    - **Weight Loss:** -20% from TDEE (sustainable fat loss while preserving muscle mass)
    - **Weight Maintenance:** 0% adjustment (maintain current weight and body composition)
    - **Weight Gain:** +10% over TDEE (lean muscle gain while minimizing fat accumulation)
    
    ### **Protein-First Macronutrient Strategy**
    
    **Why Protein Comes First:**
    - Essential for muscle protein synthesis and preservation
    - Highest thermic effect of food (burns ~30% of calories during digestion)
    - Superior satiety compared to carbs and fats
    
    **Goal-Specific Protein Targets:**
    - **Weight Loss:** 1.8g/kg (preserves muscle during caloric deficit)
    - **Maintenance:** 1.6g/kg (maintains muscle mass and metabolic health)
    - **Weight Gain:** 2.0g/kg (supports muscle protein synthesis during surplus)
    
    ### **Fat & Carbohydrate Balance**
    
    After protein needs are met, remaining calories are distributed between fats and carbohydrates:
    - **Fats:** 25-30% of total calories (hormone production, nutrient absorption)
    - **Carbohydrates:** Remaining calories (energy for training and daily activities)
    
    This approach ensures adequate intake of all macronutrients while prioritizing the most metabolically important one: protein.
    """)

# Sidebar Input Interface
st.sidebar.header("Personal Information & Goals 📋")

# Collect user inputs using unified configuration
user_inputs = {}
for field, config in CONFIG['form_fields'].items():
    if not config.get('advanced'):
        user_inputs[field] = create_unified_input(field, config)

# Advanced Settings Section
with st.sidebar.expander("⚙️ Advanced Settings (Optional)", expanded=False):
    st.markdown("*Leave blank to use evidence-based defaults for your goal*")
    for field, config in CONFIG['form_fields'].items():
        if config.get('advanced'):
            value = create_unified_input(field, config, st)
            if config.get('convert'):
                value = config['convert'](value)
            user_inputs[field] = value

# Process inputs and calculate targets
final_values = get_final_values(user_inputs)
targets = calculate_personalized_targets(**final_values)

# Enhanced Hydration Tracking in Sidebar
hydration_needs = calculate_hydration_needs(final_values['weight_kg'], final_values['activity_level'])
st.sidebar.info(f"💧 **Daily Fluid Target:** {hydration_needs} ml ({hydration_needs/250:.1f} cups)")

# Activity Level Descriptions
activity_descriptions = {
    'sedentary': "Little to no exercise, desk job",
    'lightly_active': "Light exercise 1-3 days/week",
    'moderately_active': "Moderate exercise 3-5 days/week",
    'very_active': "Heavy exercise 6-7 days/week",
    'extremely_active': "Very heavy exercise, physical job"
}

current_activity_desc = activity_descriptions.get(final_values['activity_level'], "")
if current_activity_desc:
    st.sidebar.caption(f"*{current_activity_desc}*")

# -----------------------------------------------------------------------------
# Cell 10: Results Display and Educational Guidance
# -----------------------------------------------------------------------------

st.header("Your Personalized Daily Nutrition Targets 🎯")

# Primary metrics display
goal_labels = {
    'weight_loss': 'Weight Loss',
    'weight_maintenance': 'Weight Maintenance', 
    'weight_gain': 'Weight Gain'
}

metrics_data = [
    ("Daily Calories", f"{targets['total_calories']} kcal"),
    ("Protein", f"{targets['protein_g']} g ({targets['protein_percent']:.0f}%)"),
    ("Carbohydrates", f"{targets['carb_g']} g ({targets['carb_percent']:.0f}%)"),
    ("Fat", f"{targets['fat_g']} g ({targets['fat_percent']:.0f}%)")
]

display_metrics_grid(metrics_data)

# Enhanced Results Summary
st.subheader("Detailed Breakdown 📊")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### **Energy Calculations**")
    st.write(f"**Basal Metabolic Rate (BMR):** {targets['bmr']} kcal/day")
    st.write(f"**Total Daily Energy Expenditure (TDEE):** {targets['tdee']} kcal/day")
    st.write(f"**Goal:** {goal_labels.get(targets['goal'], targets['goal'].title())}")
    
    adjustment_sign = "+" if targets['caloric_adjustment'] >= 0 else ""
    st.write(f"**Caloric Adjustment:** {adjustment_sign}{targets['caloric_adjustment']} kcal/day")

with col2:
    st.markdown("### **Expected Progress**")
    if targets['estimated_weekly_change'] > 0:
        st.write(f"**Est. Weekly Weight Change:** +{targets['estimated_weekly_change']:.2f} kg/week")
        st.caption("*Estimated weight gain assuming consistent adherence*")
    elif targets['estimated_weekly_change'] < 0:
        st.write(f"**Est. Weekly Weight Change:** {targets['estimated_weekly_change']:.2f} kg/week")
        st.caption("*Estimated weight loss assuming consistent adherence*")
    else:
        st.write(f"**Est. Weekly Weight Change:** {targets['estimated_weekly_change']:.2f} kg/week")
        st.caption("*Maintenance calories for weight stability*")
    
    st.write(f"**Protein per kg body weight:** {final_values['protein_per_kg']:.1f} g/kg")
    st.write(f"**Fat percentage of calories:** {final_values['fat_percentage']*100:.0f}%")

# Enhanced Educational Content Sections
st.header("Evidence-Based Guidance for Success 📚")

# Create tabs for different educational topics
tab1, tab2, tab3, tab4 = st.tabs([
    "🏃‍♂️ Progress Tracking", 
    "😴 Lifestyle Factors", 
    "🔄 Plateau Solutions", 
    "🧬 Advanced Concepts"
])

with tab1:
    st.markdown(EDUCATIONAL_CONTENT['monitoring_tips'])

with tab2:
    st.markdown(EDUCATIONAL_CONTENT['lifestyle_factors'])

with tab3:
    st.markdown(EDUCATIONAL_CONTENT['plateau_strategies'])

with tab4:
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🧬 **Metabolic Adaptation**", expanded=False):
            st.markdown(EDUCATIONAL_CONTENT['metabolic_adaptation'])
        
        with st.expander("⏰ **Protein Timing**", expanded=False):
            st.markdown(EDUCATIONAL_CONTENT['protein_timing'])
    
    with col2:
        with st.expander("🌱 **Micronutrients**", expanded=False):
            st.markdown(EDUCATIONAL_CONTENT['micronutrient_guide'])
        
        with st.expander("🔄 **Periodization**", expanded=False):
            st.markdown(EDUCATIONAL_CONTENT['periodization_tips'])

# Additional expandable sections
with st.expander("🌱 **Food Quality Guidelines**", expanded=False):
    st.markdown(EDUCATIONAL_CONTENT['food_quality'])

# -----------------------------------------------------------------------------
# Cell 11: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Your Foods 🥗")

# Calculate current totals
totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

# Progress tracking section
if any(totals.values()):
    recommendations = create_progress_tracking(totals, targets)
    
    if recommendations:
        st.subheader("Recommendations to Meet Your Targets 💡")
        for rec in recommendations:
            st.write(rec)

# Food selection interface with enhanced categorization
st.subheader("Food Categories")

# Sort categories to prioritize protein sources
category_priority = {
    'PRIMARY PROTEIN SOURCES': 1,
    'PRIMARY CARBOHYDRATE SOURCES': 2, 
    'PRIMARY FAT SOURCES': 3,
    'PRIMARY MICRONUTRIENT SOURCES': 4
}

sorted_categories = sorted(foods.keys(), key=lambda x: category_priority.get(x, 5))

for category in sorted_categories:
    if not foods[category]:
        continue
        
    with st.expander(f"**{category}** ({len(foods[category])} items)", expanded=True):
        # Sort foods within category by emoji priority, then by primary nutrient
        sorted_foods = sorted(
            foods[category], 
            key=lambda x: (
                CONFIG['emoji_order'].get(x.get('emoji', ''), 4),
                -x.get(CONFIG['nutrient_map'].get(category, {}).get('sort_by', 'calories'), 0)
            )
        )
        render_food_grid(sorted_foods, category, columns=2)

# -----------------------------------------------------------------------------
# Cell 12: Daily Summary and Selected Foods Display
# -----------------------------------------------------------------------------

if selected_foods:
    st.header("Today's Food Selections 📝")
    
    # Create summary table
    summary_data = []
    for item in selected_foods:
        food = item['food']
        servings = item['servings']
        summary_data.append({
            'Food': food['name'],
            'Servings': f"{servings:.1f}",
            'Calories': f"{food['calories'] * servings:.0f}",
            'Protein (g)': f"{food['protein'] * servings:.1f}",
            'Carbs (g)': f"{food['carbs'] * servings:.1f}",
            'Fat (g)': f"{food['fat'] * servings:.1f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Daily totals summary
    st.subheader("Daily Totals Summary")
    
    totals_metrics = [
        ("Total Calories", f"{totals['calories']:.0f} kcal", f"{totals['calories'] - targets['total_calories']:+.0f}"),
        ("Total Protein", f"{totals['protein']:.1f} g", f"{totals['protein'] - targets['protein_g']:+.1f}"),
        ("Total Carbs", f"{totals['carbs']:.1f} g", f"{totals['carbs'] - targets['carb_g']:+.1f}"),
        ("Total Fat", f"{totals['fat']:.1f} g", f"{totals['fat'] - targets['fat_g']:+.1f}")
    ]
    
    display_metrics_grid(totals_metrics)
    
    # Clear selections button
    if st.button("🗑️ Clear All Selections", type="secondary"):
        st.session_state.food_selections = {}
        st.rerun()

# -----------------------------------------------------------------------------
# Cell 13: Footer and Additional Resources
# -----------------------------------------------------------------------------

st.markdown("---")

# Footer with additional tips and disclaimers
st.markdown("""
### 🔬 **Scientific Disclaimer**

This calculator provides evidence-based estimates based on established nutritional science. Individual results may vary due to genetics, medical conditions, medications, and other factors. For personalized nutrition advice, especially if you have medical conditions, consult with a registered dietitian or healthcare provider.

### 📖 **Key References**
- Mifflin-St Jeor Equation: *American Journal of Clinical Nutrition* (1990)
- Protein Requirements: *Journal of the International Society of Sports Nutrition* (2017)
- Activity Multipliers: *Institute of Medicine Dietary Reference Intakes* (2005)

### 💡 **Pro Tips for Success**
- **Consistency over perfection**: Aim for 80-90% adherence rather than 100% perfection
- **Weekly averages matter**: Focus on weekly weight trends, not daily fluctuations  
- **Listen to your body**: Adjust based on energy levels, performance, and wellbeing
- **Be patient**: Sustainable changes take time - think in months, not days

*Built with evidence-based nutrition science and designed for sustainable, long-term success.* 🌟
""")

# Version and update information
st.caption("Version 2.0 - Enhanced with comprehensive educational content and evidence-based guidance")
