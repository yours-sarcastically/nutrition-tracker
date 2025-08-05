import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional

# -----------------------------------------------------------------------------
# Cell 1: Configuration and Constants
# -----------------------------------------------------------------------------

CONFIG = {
    'form_fields': {
        'age': {
            'type': 'number_input',
            'label': 'Age',
            'min_value': 15,
            'max_value': 80,
            'value': 25,
            'step': 1,
            'help': 'Your current age in years',
            'required': True
        },
        'gender': {
            'type': 'selectbox',
            'label': 'Biological Sex',
            'options': ['Male', 'Female'],
            'index': 0,
            'help': 'Biological sex affects metabolic calculations',
            'required': True
        },
        'weight_kg': {
            'type': 'number_input',
            'label': 'Current Weight (kg)',
            'min_value': 40.0,
            'max_value': 200.0,
            'value': 70.0,
            'step': 0.1,
            'help': 'Your current body weight in kilograms',
            'required': True
        },
        'height_cm': {
            'type': 'number_input',
            'label': 'Height (cm)',
            'min_value': 140.0,
            'max_value': 220.0,
            'value': 175.0,
            'step': 0.5,
            'help': 'Your height in centimeters',
            'required': True
        },
        'activity_level': {
            'type': 'selectbox',
            'label': 'Activity Level',
            'options': [
                ('Sedentary (little/no exercise)', 1.2),
                ('Lightly active (light exercise 1-3 days/week)', 1.375),
                ('Moderately active (moderate exercise 3-5 days/week)', 1.55),
                ('Very active (hard exercise 6-7 days/week)', 1.725),
                ('Extremely active (very hard exercise, physical job)', 1.9)
            ],
            'index': 2,
            'help': 'Your typical weekly activity level',
            'convert': lambda x: x[1] if isinstance(x, tuple) else x,
            'required': True
        },
        'goal': {
            'type': 'selectbox',
            'label': 'Primary Goal',
            'options': [
                ('Weight Loss', 'weight_loss'),
                ('Weight Maintenance', 'weight_maintenance'),
                ('Weight Gain', 'weight_gain')
            ],
            'index': 0,
            'help': 'Your primary body composition goal',
            'convert': lambda x: x[1] if isinstance(x, tuple) else x,
            'required': True
        },
        'protein_multiplier': {
            'type': 'slider',
            'label': 'Protein Multiplier (g per kg body weight)',
            'min_value': 1.2,
            'max_value': 2.5,
            'value': 1.8,
            'step': 0.1,
            'help': 'Higher protein supports muscle retention and satiety',
            'advanced': True
        },
        'fat_percentage': {
            'type': 'slider',
            'label': 'Fat Percentage of Total Calories',
            'min_value': 20,
            'max_value': 35,
            'value': 25,
            'step': 1,
            'help': 'Minimum 20% for hormonal health',
            'advanced': True
        },
        'caloric_adjustment': {
            'type': 'slider',
            'label': 'Custom Caloric Adjustment (%)',
            'min_value': -30,
            'max_value': 20,
            'value': 0,
            'step': 1,
            'help': 'Override default goal-based adjustments',
            'advanced': True
        }
    },
    'goal_defaults': {
        'weight_loss': {'protein_multiplier': 1.8, 'fat_percentage': 25, 'caloric_adjustment': -20},
        'weight_maintenance': {'protein_multiplier': 1.6, 'fat_percentage': 30, 'caloric_adjustment': 0},
        'weight_gain': {'protein_multiplier': 2.0, 'fat_percentage': 25, 'caloric_adjustment': 10}
    },
    'nutrient_configs': {
        'calories': {'label': 'Calories', 'unit': 'kcal', 'target_key': 'total_calories'},
        'protein': {'label': 'Protein', 'unit': 'g', 'target_key': 'protein_g'},
        'carbs': {'label': 'Carbohydrates', 'unit': 'g', 'target_key': 'carb_g'},
        'fat': {'label': 'Fat', 'unit': 'g', 'target_key': 'fat_g'}
    },
    'emoji_order': {
        '🥇': 1,  # Top performer
        '🔥': 2,  # High calorie
        '💪': 3,  # Protein power
        '🍚': 3,  # Carb champion
        '🥑': 3,  # Healthy fats
        '': 4     # No emoji
    }
}

# -----------------------------------------------------------------------------
# Cell 2: Core Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(weight_kg: float, height_cm: float, age: int, gender: str) -> float:
    """Calculate BMR using Mifflin-St Jeor equation."""
    if gender.lower() == 'male':
        return 10 * weight_kg + 6.25 * height_cm - 5 * age + 5
    else:
        return 10 * weight_kg + 6.25 * height_cm - 5 * age - 161

def calculate_tdee(bmr: float, activity_level: float) -> float:
    """Calculate Total Daily Energy Expenditure."""
    return bmr * activity_level

def calculate_personalized_targets(
    age: int, gender: str, weight_kg: float, height_cm: float,
    activity_level: float, goal: str, protein_multiplier: float = None,
    fat_percentage: int = None, caloric_adjustment: int = None
) -> Dict[str, Any]:
    """Calculate comprehensive personalized nutrition targets."""
    
    # Use goal defaults if advanced settings not provided
    defaults = CONFIG['goal_defaults'][goal]
    protein_multiplier = protein_multiplier if protein_multiplier is not None else defaults['protein_multiplier']
    fat_percentage = fat_percentage if fat_percentage is not None else defaults['fat_percentage']
    caloric_adjustment = caloric_adjustment if caloric_adjustment is not None else defaults['caloric_adjustment']
    
    # Calculate base metabolic values
    bmr = calculate_bmr(weight_kg, height_cm, age, gender)
    tdee = calculate_tdee(bmr, activity_level)
    
    # Apply caloric adjustment
    adjustment_calories = tdee * (caloric_adjustment / 100)
    total_calories = int(tdee + adjustment_calories)
    
    # Calculate macronutrients
    protein_g = round(weight_kg * protein_multiplier, 1)
    protein_calories = protein_g * 4
    
    fat_calories = int(total_calories * (fat_percentage / 100))
    fat_g = round(fat_calories / 9, 1)
    
    remaining_calories = total_calories - protein_calories - fat_calories
    carb_g = round(remaining_calories / 4, 1)
    carb_calories = int(carb_g * 4)
    
    # Calculate percentages
    protein_percent = (protein_calories / total_calories) * 100
    carb_percent = (carb_calories / total_calories) * 100
    fat_percent = (fat_calories / total_calories) * 100
    
    # Estimate weekly weight change (rough approximation)
    weekly_caloric_difference = adjustment_calories * 7
    estimated_weekly_change = weekly_caloric_difference / 7700  # 7700 kcal ≈ 1 kg fat
    
    return {
        'bmr': int(bmr),
        'tdee': int(tdee),
        'total_calories': total_calories,
        'caloric_adjustment': int(adjustment_calories),
        'estimated_weekly_change': estimated_weekly_change,
        'protein_g': protein_g,
        'protein_calories': int(protein_calories),
        'protein_percent': protein_percent,
        'carb_g': carb_g,
        'carb_calories': carb_calories,
        'carb_percent': carb_percent,
        'fat_g': fat_g,
        'fat_calories': fat_calories,
        'fat_percent': fat_percent,
        'goal': goal
    }

def calculate_hydration_needs(weight_kg: float, activity_level: float) -> int:
    """Calculate daily hydration needs based on weight and activity."""
    base_water = weight_kg * 35  # 35ml per kg base requirement
    
    # Activity adjustment
    activity_multipliers = {1.2: 1.0, 1.375: 1.1, 1.55: 1.2, 1.725: 1.3, 1.9: 1.4}
    activity_mult = activity_multipliers.get(activity_level, 1.2)
    
    return int(base_water * activity_mult)

# -----------------------------------------------------------------------------
# Cell 3: Session State and Input Management
# -----------------------------------------------------------------------------

def initialize_session_state():
    """Initialize session state for food selections."""
    if 'food_selections' not in st.session_state:
        st.session_state.food_selections = {}

def create_unified_input(field_name: str, field_config: Dict[str, Any], container=st) -> Any:
    """Create input widgets based on field configuration."""
    input_type = field_config['type']
    
    if input_type == 'number_input':
        return container.number_input(
            field_config['label'],
            min_value=field_config.get('min_value'),
            max_value=field_config.get('max_value'),
            value=field_config.get('value'),
            step=field_config.get('step', 1),
            help=field_config.get('help')
        )
    elif input_type == 'selectbox':
        return container.selectbox(
            field_config['label'],
            field_config['options'],
            index=field_config.get('index', 0),
            help=field_config.get('help')
        )
    elif input_type == 'slider':
        return container.slider(
            field_config['label'],
            min_value=field_config.get('min_value'),
            max_value=field_config.get('max_value'),
            value=field_config.get('value'),
            step=field_config.get('step', 1),
            help=field_config.get('help')
        )

def get_final_values(all_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Process and validate final input values."""
    final_values = {}
    
    for field_name, value in all_inputs.items():
        field_config = CONFIG['form_fields'][field_name]
        
        # Apply conversion if specified
        if 'convert' in field_config:
            value = field_config['convert'](value)
        
        final_values[field_name] = value
    
    return final_values

# -----------------------------------------------------------------------------
# Cell 4: Food Database Management
# -----------------------------------------------------------------------------

def load_food_database(csv_file: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load and organize food database from CSV."""
    try:
        df = pd.read_csv(csv_file)
        
        # Clean and standardize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Create food dictionary structure
        foods = {
            'PRIMARY PROTEIN SOURCES': [],
            'PRIMARY CARBOHYDRATE SOURCES': [],
            'PRIMARY FAT SOURCES': [],
            'PRIMARY MICRONUTRIENT SOURCES': []
        }
        
        # Process each row and categorize
        for _, row in df.iterrows():
            food_item = {
                'name': row['food_item'],
                'calories': float(row['calories_per_100g']),
                'protein': float(row['protein_g']),
                'carbs': float(row['carbs_g']),
                'fat': float(row['fat_g']),
                'serving_size': row.get('typical_serving_size', '100g')
            }
            
            # Categorize based on primary macronutrient
            protein_ratio = food_item['protein'] / max(food_item['calories'], 1) * 100
            fat_ratio = food_item['fat'] * 9 / max(food_item['calories'], 1) * 100
            carb_ratio = food_item['carbs'] * 4 / max(food_item['calories'], 1) * 100
            
            if protein_ratio >= 40:
                foods['PRIMARY PROTEIN SOURCES'].append(food_item)
            elif fat_ratio >= 60:
                foods['PRIMARY FAT SOURCES'].append(food_item)
            elif carb_ratio >= 60:
                foods['PRIMARY CARBOHYDRATE SOURCES'].append(food_item)
            else:
                foods['PRIMARY MICRONUTRIENT SOURCES'].append(food_item)
        
        return foods
        
    except FileNotFoundError:
        # Return sample data if CSV not found
        return create_sample_food_database()

def create_sample_food_database() -> Dict[str, List[Dict[str, Any]]]:
    """Create sample food database for demonstration."""
    return {
        'PRIMARY PROTEIN SOURCES': [
            {'name': 'Chicken Breast (skinless)', 'calories': 165, 'protein': 31.0, 'carbs': 0.0, 'fat': 3.6, 'serving_size': '100g'},
            {'name': 'Greek Yogurt (plain)', 'calories': 59, 'protein': 10.0, 'carbs': 3.6, 'fat': 0.4, 'serving_size': '100g'},
            {'name': 'Lentils (cooked)', 'calories': 116, 'protein': 9.0, 'carbs': 20.0, 'fat': 0.4, 'serving_size': '100g'},
            {'name': 'Tofu (firm)', 'calories': 144, 'protein': 17.3, 'carbs': 2.8, 'fat': 8.7, 'serving_size': '100g'},
            {'name': 'Eggs (whole)', 'calories': 155, 'protein': 13.0, 'carbs': 1.1, 'fat': 11.0, 'serving_size': '100g'},
            {'name': 'Cottage Cheese', 'calories': 98, 'protein': 11.1, 'carbs': 3.4, 'fat': 4.3, 'serving_size': '100g'},
        ],
        'PRIMARY CARBOHYDRATE SOURCES': [
            {'name': 'Brown Rice (cooked)', 'calories': 123, 'protein': 2.6, 'carbs': 23.0, 'fat': 0.9, 'serving_size': '100g'},
            {'name': 'Oats (dry)', 'calories': 389, 'protein': 16.9, 'carbs': 66.3, 'fat': 6.9, 'serving_size': '100g'},
            {'name': 'Sweet Potato (baked)', 'calories': 86, 'protein': 1.6, 'carbs': 20.0, 'fat': 0.1, 'serving_size': '100g'},
            {'name': 'Quinoa (cooked)', 'calories': 120, 'protein': 4.4, 'carbs': 22.0, 'fat': 1.9, 'serving_size': '100g'},
            {'name': 'Banana', 'calories': 89, 'protein': 1.1, 'carbs': 23.0, 'fat': 0.3, 'serving_size': '100g'},
            {'name': 'Whole Wheat Bread', 'calories': 247, 'protein': 13.0, 'carbs': 41.0, 'fat': 4.2, 'serving_size': '100g'},
        ],
        'PRIMARY FAT SOURCES': [
            {'name': 'Avocado', 'calories': 160, 'protein': 2.0, 'carbs': 9.0, 'fat': 15.0, 'serving_size': '100g'},
            {'name': 'Almonds', 'calories': 579, 'protein': 21.2, 'carbs': 22.0, 'fat': 49.9, 'serving_size': '100g'},
            {'name': 'Olive Oil', 'calories': 884, 'protein': 0.0, 'carbs': 0.0, 'fat': 100.0, 'serving_size': '100g'},
            {'name': 'Peanut Butter', 'calories': 588, 'protein': 25.1, 'carbs': 20.0, 'fat': 50.4, 'serving_size': '100g'},
            {'name': 'Walnuts', 'calories': 654, 'protein': 15.2, 'carbs': 14.0, 'fat': 65.2, 'serving_size': '100g'},
            {'name': 'Chia Seeds', 'calories': 486, 'protein': 16.5, 'carbs': 42.0, 'fat': 30.7, 'serving_size': '100g'},
        ],
        'PRIMARY MICRONUTRIENT SOURCES': [
            {'name': 'Spinach (raw)', 'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'serving_size': '100g'},
            {'name': 'Broccoli (cooked)', 'calories': 35, 'protein': 2.4, 'carbs': 7.0, 'fat': 0.4, 'serving_size': '100g'},
            {'name': 'Bell Peppers', 'calories': 31, 'protein': 1.0, 'carbs': 7.0, 'fat': 0.3, 'serving_size': '100g'},
            {'name': 'Carrots', 'calories': 41, 'protein': 0.9, 'carbs': 10.0, 'fat': 0.2, 'serving_size': '100g'},
            {'name': 'Tomatoes', 'calories': 18, 'protein': 0.9, 'carbs': 3.9, 'fat': 0.2, 'serving_size': '100g'},
            {'name': 'Blueberries', 'calories': 57, 'protein': 0.7, 'carbs': 14.0, 'fat': 0.3, 'serving_size': '100g'},
        ]
    }

# -----------------------------------------------------------------------------
# Cell 5: Food Analysis and Ranking Functions
# -----------------------------------------------------------------------------

def assign_food_emojis(foods: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """Assign ranking emojis to foods based on their nutritional profiles."""
    
    for category, food_list in foods.items():
        if not food_list:
            continue
            
        # Sort by calories (descending) for high-calorie identification
        sorted_by_calories = sorted(food_list, key=lambda x: x['calories'], reverse=True)
        
        # Assign emojis based on category and rankings
        for i, food in enumerate(food_list):
            emoji = ""
            
            if category == 'PRIMARY PROTEIN SOURCES':
                # Top 3 protein sources get protein emoji
                protein_sorted = sorted(food_list, key=lambda x: x['protein'], reverse=True)
                if food in protein_sorted[:3]:
                    emoji = "💪"
                # Highest calorie gets fire emoji
                elif food == sorted_by_calories[0]:
                    emoji = "🔥"
                # Top performer: high calories + high protein
                elif food['calories'] >= 150 and food['protein'] >= 15:
                    emoji = "🥇"
                    
            elif category == 'PRIMARY CARBOHYDRATE SOURCES':
                # Top 3 carb sources get carb emoji
                carb_sorted = sorted(food_list, key=lambda x: x['carbs'], reverse=True)
                if food in carb_sorted[:3]:
                    emoji = "🍚"
                # Highest calorie gets fire emoji
                elif food == sorted_by_calories[0]:
                    emoji = "🔥"
                # Top performer: high calories + high carbs
                elif food['calories'] >= 120 and food['carbs'] >= 20:
                    emoji = "🥇"
                    
            elif category == 'PRIMARY FAT SOURCES':
                # Top 3 fat sources get fat emoji
                fat_sorted = sorted(food_list, key=lambda x: x['fat'], reverse=True)
                if food in fat_sorted[:3]:
                    emoji = "🥑"
                # Highest calorie gets fire emoji
                elif food == sorted_by_calories[0]:
                    emoji = "🔥"
                # Top performer: high calories + high fat
                elif food['calories'] >= 400 and food['fat'] >= 30:
                    emoji = "🥇"
            
            food['emoji'] = emoji
    
    return foods

# -----------------------------------------------------------------------------
# Cell 6: Food Selection Interface Functions
# -----------------------------------------------------------------------------

def render_food_grid(food_list: List[Dict[str, Any]], category: str, columns: int = 2):
    """Render food selection grid with enhanced interface."""
    
    # Create columns
    cols = st.columns(columns)
    
    for i, food in enumerate(food_list):
        col = cols[i % columns]
        
        with col:
            # Create unique key for this food item
            key = f"{category}_{food['name']}"
            
            # Get current serving value
            current_servings = st.session_state.food_selections.get(key, 0.0)
            
            # Display food info with emoji
            emoji = food.get('emoji', '')
            st.markdown(f"**{emoji} {food['name']}**")
            
            # Nutritional info
            st.caption(f"Per 100g: {food['calories']} kcal | P: {food['protein']}g | C: {food['carbs']}g | F: {food['fat']}g")
            
            # Serving input
            servings = st.number_input(
                f"Servings (100g each)",
                min_value=0.0,
                max_value=10.0,
                value=current_servings,
                step=0.1,
                key=key,
                help=f"Number of 100g servings of {food['name']}"
            )
            
            # Update session state
            st.session_state.food_selections[key] = servings
            
            # Show calculated nutrition if servings > 0
            if servings > 0:
                total_cal = food['calories'] * servings
                total_protein = food['protein'] * servings
                total_carbs = food['carbs'] * servings
                total_fat = food['fat'] * servings
                
                st.success(f"**Total:** {total_cal:.0f} kcal | P: {total_protein:.1f}g | C: {total_carbs:.1f}g | F: {total_fat:.1f}g")

def calculate_daily_totals(food_selections: Dict[str, float], foods: Dict[str, List[Dict[str, Any]]]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    """Calculate daily nutritional totals from selected foods."""
    
    totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
    selected_foods = []
    
    # Create a lookup dictionary for all foods
    all_foods = {}
    for category, food_list in foods.items():
        for food in food_list:
            key = f"{category}_{food['name']}"
            all_foods[key] = food
    
    # Calculate totals
    for key, servings in food_selections.items():
        if servings > 0 and key in all_foods:
            food = all_foods[key]
            
            # Add to totals
            totals['calories'] += food['calories'] * servings
            totals['protein'] += food['protein'] * servings
            totals['carbs'] += food['carbs'] * servings
            totals['fat'] += food['fat'] * servings
            
            # Add to selected foods list
            selected_foods.append({
                'food': food,
                'servings': servings
            })
    
    return totals, selected_foods

def create_progress_tracking(totals: Dict[str, float], targets: Dict[str, Any]) -> List[str]:
    """Create personalized recommendations based on current progress."""
    
    recommendations = []
    
    # Caloric balance
    caloric_diff = totals['calories'] - targets['total_calories']
    if caloric_diff > 100:
        recommendations.append("🔥 **Calories:** You're significantly over your target. Consider reducing portion sizes or choosing lower-calorie options.")
    elif caloric_diff < -100:
        recommendations.append("📊 **Calories:** You're well below your target. Add more calorie-dense foods to meet your energy needs.")
    
    # Protein adequacy
    protein_diff = totals['protein'] - targets['protein_g']
    if protein_diff < -10:
        recommendations.append("💪 **Protein:** Add more protein-rich foods like Greek yogurt, lean meats, or legumes to support muscle maintenance.")
    elif protein_diff > 20:
        recommendations.append("💪 **Protein:** Excellent protein intake! This supports muscle building and satiety.")
    
    # Macronutrient balance
    if totals['calories'] > 0:
        fat_percent = (totals['fat'] * 9 / totals['calories']) * 100
        if fat_percent < 20:
            recommendations.append("🥑 **Fats:** Consider adding healthy fats like nuts, avocado, or olive oil for hormonal health.")
        elif fat_percent > 40:
            recommendations.append("🥑 **Fats:** High fat intake detected. Balance with more protein and carbohydrates if needed.")
    
    return recommendations

# -----------------------------------------------------------------------------
# Cell 7: Display and UI Functions
# -----------------------------------------------------------------------------

def display_metrics_grid(metrics: List[Tuple[str, str]], columns: int = 4):
    """Display metrics in a responsive grid layout."""
    
    cols = st.columns(columns)
    
    for i, metric in enumerate(metrics):
        col = cols[i % columns]
        
        if len(metric) == 2:
            label, value = metric
            col.metric(label, value)
        elif len(metric) == 3:
            label, value, delta = metric
            col.metric(label, value, delta)

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
# Cell 8: Application Title and Complete Enhanced Input Interface
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

# Complete Enhanced sidebar with ALL components restored
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

# Activity Level Guide (RESTORED)
if st.sidebar.button("ℹ️ Activity Level Guide", help="Click to see detailed activity level descriptions"):
    st.sidebar.markdown("""
    **Activity Level Definitions:**
    
    **Sedentary (1.2x):** Desk job, minimal exercise
    
    **Lightly Active (1.375x):** Light exercise 1-3 days/week
    
    **Moderately Active (1.55x):** Moderate exercise 3-5 days/week
    
    **Very Active (1.725x):** Hard exercise 6-7 days/week
    
    **Extremely Active (1.9x):** Very hard exercise, physical job
    """)

# Add hydration calculator after activity level
if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_needs = calculate_hydration_needs(all_inputs['weight_kg'], all_inputs['activity_level'])
    st.sidebar.info(f"💧 **Daily Fluid Target:** {hydration_needs} ml ({hydration_needs/250:.1f} cups)")

# Emoji-Based Food Ranking System (RESTORED)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🏆 **Food Ranking System**")
st.sidebar.markdown("""
**Emoji Guide:**
- 🥇 **Top Performer:** High calories + high in primary nutrient
- 🔥 **High Calorie:** Most calorie-dense in category
- 💪 **Protein Power:** Top 3 protein sources
- 🍚 **Carb Champion:** Top 3 carbohydrate sources
- 🥑 **Healthy Fats:** Top 3 fat sources

*Foods are ranked within each category to help you make efficient choices for your goals.*
""")

# Render advanced fields in expander
advanced_expander = st.sidebar.expander("Advanced Settings ⚙️")
for field_name, field_config in advanced_fields.items():
    value = create_unified_input(field_name, field_config, container=advanced_expander)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value

# Resistance Training Guidance (RESTORED)
st.sidebar.markdown("---")
st.sidebar.markdown("### 💪 **Resistance Training Guidelines**")
st.sidebar.markdown("""
**For Optimal Results:**
- **Frequency:** 3-4 sessions per week
- **Progressive overload:** Gradually increase weight/reps
- **Compound movements:** Squats, deadlifts, rows, presses
- **Rest between sets:** 2-3 minutes for strength
- **Rep ranges:** 6-12 reps for muscle building

**Why it matters:** Resistance training preserves muscle mass during weight loss and builds lean tissue during weight gain phases.
""")

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

# Enhanced Dynamic Monitoring Tips with Plateau Troubleshooting
st.sidebar.markdown("---")
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

# -----------------------------------------------------------------------------
# Cell 10: Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Daily Food Selection & Tracking 🍽️")

# Create tabs for each food category
tab_names = list(foods.keys())
tabs = st.tabs(tab_names)

for i, (category, food_list) in enumerate(foods.items()):
    with tabs[i]:
        st.subheader(f"{category}")
        
        # Sort foods by emoji priority for better display
        sorted_foods = sorted(food_list, key=lambda x: CONFIG['emoji_order'].get(x.get('emoji', ''), 4))
        
        render_food_grid(sorted_foods, category, columns=2)

# -----------------------------------------------------------------------------
# Cell 11: Progress Tracking and Analysis
# -----------------------------------------------------------------------------

st.header("Daily Progress Tracking & Analysis 📊")

# Calculate daily totals
daily_totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods)

# Display current totals
col1, col2 = st.columns(2)

with col1:
    st.subheader("Current Daily Totals")
    display_metrics_grid([
        ("Calories", f"{daily_totals['calories']:.0f} kcal"),
        ("Protein", f"{daily_totals['protein']:.1f} g"),
        ("Carbohydrates", f"{daily_totals['carbs']:.1f} g"),
        ("Fat", f"{daily_totals['fat']:.1f} g")
    ], columns=2)

with col2:
    st.subheader("Target vs. Current")
    if user_has_entered_info:
        calorie_diff = daily_totals['calories'] - targets['total_calories']
        protein_diff = daily_totals['protein'] - targets['protein_g']
        carb_diff = daily_totals['carbs'] - targets['carb_g']
        fat_diff = daily_totals['fat'] - targets['fat_g']
        
        display_metrics_grid([
            ("Calorie Difference", f"{calorie_diff:+.0f} kcal"),
            ("Protein Difference", f"{protein_diff:+.1f} g"),
            ("Carb Difference", f"{carb_diff:+.1f} g"),
            ("Fat Difference", f"{fat_diff:+.1f} g")
        ], columns=2)

# Progress bars for visual tracking
if user_has_entered_info and daily_totals['calories'] > 0:
    st.subheader("Progress Visualization")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        calorie_progress = min(daily_totals['calories'] / targets['total_calories'], 2.0)
        st.metric("Calories", f"{calorie_progress:.1%}")
        st.progress(min(calorie_progress, 1.0))
    
    with col2:
        protein_progress = min(daily_totals['protein'] / targets['protein_g'], 2.0)
        st.metric("Protein", f"{protein_progress:.1%}")
        st.progress(min(protein_progress, 1.0))
    
    with col3:
        carb_progress = min(daily_totals['carbs'] / targets['carb_g'], 2.0)
        st.metric("Carbs", f"{carb_progress:.1%}")
        st.progress(min(carb_progress, 1.0))
    
    with col4:
        fat_progress = min(daily_totals['fat'] / targets['fat_g'], 2.0)
        st.metric("Fat", f"{fat_progress:.1%}")
        st.progress(min(fat_progress, 1.0))

# Personalized recommendations
if daily_totals['calories'] > 0 and user_has_entered_info:
    recommendations = create_progress_tracking(daily_totals, targets)
    
    if recommendations:
        st.subheader("Personalized Recommendations")
        for rec in recommendations:
            st.info(rec)

# Selected foods summary
if selected_foods:
    st.subheader("Today's Food Selections")
    
    for selection in selected_foods:
        food = selection['food']
        servings = selection['servings']
        
        total_cal = food['calories'] * servings
        total_protein = food['protein'] * servings
        total_carbs = food['carbs'] * servings
        total_fat = food['fat'] * servings
        
        emoji = food.get('emoji', '')
        st.write(f"**{emoji} {food['name']}** ({servings} servings): {total_cal:.0f} kcal | P: {total_protein:.1f}g | C: {total_carbs:.1f}g | F: {total_fat:.1f}g")

# -----------------------------------------------------------------------------
# Cell 12: Educational Resources and Tips
# -----------------------------------------------------------------------------

st.header("Educational Resources & Advanced Tips 📚")

# Meal Planning Strategies
with st.expander("🍽️ **Meal Planning & Prep Strategies**", expanded=False):
    st.markdown("""
    ### **Weekly Meal Planning**
    
    **Sunday Prep Session (2-3 hours):**
    - Cook 2-3 protein sources in bulk (chicken, lentils, tofu)
    - Prepare 2-3 carb sources (rice, quinoa, sweet potatoes)
    - Wash and chop vegetables for easy access
    - Portion snacks into grab-and-go containers
    
    ### **Flexible Meal Templates**
    
    **Breakfast Template:**
    - 1 protein source (eggs, Greek yogurt, protein powder)
    - 1 carb source (oats, fruit, whole grain toast)
    - 1 fat source (nuts, seeds, avocado)
    
    **Lunch/Dinner Template:**
    - 1 palm-sized protein (150-200g)
    - 1 fist-sized carb (150-200g cooked)
    - 2 handfuls of vegetables
    - 1 thumb-sized fat portion
    
    ### **Emergency Backup Plans**
    
    **Quick 10-Minute Meals:**
    - Greek yogurt + berries + nuts
    - Scrambled eggs + spinach + avocado
    - Protein smoothie with banana and peanut butter
    - Canned beans + pre-cooked quinoa + vegetables
    """)

# Supplementation Guidance
with st.expander("💊 **Evidence-Based Supplementation**", expanded=False):
    st.markdown("""
    ### **Priority Supplements (Evidence-Strong)**
    
    **Vitamin D₃:** 1000-2000 IU daily
    - Supports immune function, bone health, mood
    - Especially important in low-sunlight climates
    
    **Omega-3 (EPA/DHA):** 1-2g daily
    - Reduces inflammation, supports heart/brain health
    - Vegetarians: Consider algae-based sources
    
    **Magnesium:** 200-400mg daily
    - Supports muscle function, sleep quality, stress management
    - Magnesium glycinate is well-absorbed
    
    ### **Goal-Specific Supplements**
    
    **For Weight Loss:**
    - **Caffeine:** 100-200mg pre-workout (if tolerated)
    - **Green tea extract:** May provide modest metabolic boost
    
    **For Muscle Building:**
    - **Creatine monohydrate:** 3-5g daily (any time)
    - **Whey/plant protein:** If struggling to meet protein targets
    
    ### **What NOT to Waste Money On**
    - Fat burners and "detox" products
    - Expensive multivitamins (get nutrients from food first)
    - BCAAs (unnecessary if eating adequate protein)
    """)

# Troubleshooting Common Issues
with st.expander("🔧 **Troubleshooting Common Issues**", expanded=False):
    st.markdown("""
    ### **"I'm Always Hungry"**
    
    **Immediate Solutions:**
    - Increase protein at each meal (aim for 25-30g)
    - Add more fiber-rich vegetables
    - Ensure adequate sleep (7-9 hours)
    - Check if caloric deficit is too aggressive (>20%)
    
    **Long-term Strategies:**
    - Include protein and fiber at every meal
    - Eat slowly and mindfully
    - Stay hydrated throughout the day
    - Manage stress levels
    
    ### **"I'm Not Losing Weight"**
    
    **Checklist:**
    1. ✅ Logging accuracy (weigh foods when possible)
    2. ✅ Hidden calories (cooking oils, condiments, drinks)
    3. ✅ Consistent tracking for 2+ weeks
    4. ✅ Adequate sleep and stress management
    5. ✅ Realistic timeline (1-2 lbs/week maximum)
    
    **When to Adjust:**
    - No weight change for 3+ consecutive weeks
    - Reduce calories by 5-10% OR increase activity
    - Consider diet breaks every 8-12 weeks
    
    ### **"I Can't Stick to the Plan"**
    
    **Flexibility Strategies:**
    - Use the 80/20 rule (80% adherence is success)
    - Plan for social events and special occasions
    - Have backup meal options ready
    - Focus on progress, not perfection
    - Consider working with a registered dietitian
    """)

# Social and Lifestyle Integration
with st.expander("🎉 **Social & Lifestyle Integration**", expanded=False):
    st.markdown("""
    ### **Eating Out Strategies**
    
    **Restaurant Navigation:**
    - Check menus online beforehand
    - Ask for dressings/sauces on the side
    - Request grilled instead of fried proteins
    - Fill half your plate with vegetables when possible
    - Don't arrive overly hungry
    
    **Social Event Survival:**
    - Eat a protein-rich snack beforehand
    - Bring a healthy dish to share
    - Focus on socializing, not just food
    - Practice the "one plate rule"
    - Stay hydrated with water between alcoholic drinks
    
    ### **Travel Nutrition**
    
    **Packing Essentials:**
    - Protein powder or bars
    - Nuts and seeds
    - Instant oatmeal packets
    - Dried fruit (portion-controlled)
    
    **Hotel Room Strategies:**
    - Request a mini-fridge
    - Stock up on Greek yogurt, fruits, vegetables
    - Use hotel fitness facilities
    - Walk whenever possible instead of taxis/ubers
    
    ### **Family Integration**
    
    **Making It Work for Everyone:**
    - Prepare base ingredients that can be customized
    - Teach family members about balanced plates
    - Involve kids in meal planning and prep
    - Make healthy swaps gradually
    - Lead by example, don't preach
    """)

# Reset button for food selections
st.markdown("---")
if st.button("🔄 Reset All Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
<p><strong>Personalized Evidence-Based Nutrition Tracker</strong></p>
<p>Built with scientific precision for sustainable results. Remember: consistency over perfection! 💪</p>
</div>
""", unsafe_allow_html=True)
