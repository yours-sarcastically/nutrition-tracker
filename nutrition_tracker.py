#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker Using the Mifflin-St Jeor Equation
# -----------------------------------------------------------------------------

"""
This script implements an interactive nutrition tracking application built with
Streamlit. It is designed to provide personalized nutrition plans for weight
loss, maintenance, or gain, with a focus on vegetarian food sources.

Core Scientific Principles:
The application calculates personalized daily targets for calories and
macronutrients (protein, fat, and carbohydrates) based on user-provided
attributes. The core calculations rely on established scientific formulas:
- Basal Metabolic Rate (BMR): Calculated using the Mifflin-St Jeor equation,
  which is recognized by the Academy of Nutrition and Dietetics as one of the
  most accurate methods for estimating resting energy needs. The formula is:
  BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + s
  where 's' is +5 for males and -161 for females.
- Total Daily Energy Expenditure (TDEE): Estimated by multiplying the BMR by
  an activity factor corresponding to the user's lifestyle. These multipliers
  are based on validated figures from exercise physiology research.
- Goal-Specific Adjustments: Caloric targets are adjusted based on the user's
  selected goal (e.g., a 20% deficit for weight loss or a 10% surplus for
  weight gain). Macronutrient targets are set using an evidence-based,
  protein-first approach to support muscle preservation and growth.

Application Usage and Features:
1.  User Input: Users enter their personal parameters (age, height, weight,
    sex, activity level, and goal) into the sidebar interface. Advanced
    settings allow for manual adjustments to protein and fat targets.
2.  Target Calculation: The script instantly calculates and displays detailed
    daily nutritional targets, including BMR, TDEE, total calories, and a
    breakdown of macronutrients in both grams and percentages.
3.  Food Selection: Users can select from a database of vegetarian foods,
    organized by category. An emoji guide helps identify foods that are high
    in calories or specific macronutrients.
4.  Progress Tracking: As foods are selected, the application provides a
    real-time summary of nutritional intake. Progress bars show how the
    current intake compares to the daily targets.
5.  Personalized Recommendations: The script generates actionable advice,
    including hydration targets, meal timing suggestions, and tips for
    addressing nutritional shortfalls.
6.  Evidence-Based Playbook: An integrated, tabbed section provides detailed
    information on foundational principles, advanced strategies, and the
    underlying nutrition science to educate the user and support long-term
    adherence and success.
"""

# -----------------------------------------------------------------------------
# # Cell 1
# -----------------------------------------------------------------------------
# # ------ Import Required Libraries and Modules ------
# -----------------------------------------------------------------------------

import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# # Cell 2
# -----------------------------------------------------------------------------
# # ------ Page Configuration and Initial Setup ------
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Personalized Nutrition Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# # Cell 3
# -----------------------------------------------------------------------------
# # ------ Unified Configuration Constants ------
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

# ------ Goal-Specific Targets Based on Evidence-Based Guidelines ------
GOAL_TARGETS = {
    'weight_loss': {
        'caloric_adjustment': -0.20,  # Represents a 20% deficit from TDEE
        'protein_per_kg': 1.8,
        'fat_percentage': 0.25
    },
    'weight_maintenance': {
        'caloric_adjustment': 0.0,  # Represents a 0% adjustment from TDEE
        'protein_per_kg': 1.6,
        'fat_percentage': 0.30
    },
    'weight_gain': {
        'caloric_adjustment': 0.10,  # Represents a 10% surplus over TDEE
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25
    }
}

# ------ Unified Configuration for All Application Components ------
CONFIG = {
    'emoji_order': {'🥇': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
    },
    'nutrient_configs': {
        'calories': {
            'unit': 'kcal', 'label': 'Calories',
            'target_key': 'total_calories'
        },
        'protein': {
            'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'
        },
        'carbs': {
            'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'
        },
        'fat': {
            'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'
        }
    },
    'form_fields': {
        'age': {
            'type': 'number', 'label': 'Age (in Years)', 'min': 16, 'max': 80,
            'step': 1, 'placeholder': 'Enter your age', 'required': True
        },
        'height_cm': {
            'type': 'number', 'label': 'Height (in Centimeters)',
            'min': 140, 'max': 220, 'step': 1,
            'placeholder': 'Enter your height', 'required': True
        },
        'weight_kg': {
            'type': 'number', 'label': 'Weight (in Kilograms)', 'min': 40.0,
            'max': 150.0, 'step': 0.5, 'placeholder': 'Enter your weight',
            'required': True
        },
        'sex': {
            'type': 'selectbox', 'label': 'Sex',
            'options': ["Select Sex", "Male", "Female"], 'required': True,
            'placeholder': "Select Sex"
        },
        'activity_level': {
            'type': 'selectbox', 'label': 'Activity Level',
            'options': [
                ("Select Activity Level", None),
                ("Sedentary", "sedentary"),
                ("Lightly Active", "lightly_active"),
                ("Moderately Active", "moderately_active"),
                ("Very Active", "very_active"),
                ("Extremely Active", "extremely_active")
            ], 'required': True, 'placeholder': None
        },
        'goal': {
            'type': 'selectbox', 'label': 'Nutrition Goal',
            'options': [
                ("Select Goal", None),
                ("Weight Loss", "weight_loss"),
                ("Weight Maintenance", "weight_maintenance"),
                ("Weight Gain", "weight_gain")
            ], 'required': True, 'placeholder': None
        },
        'protein_per_kg': {
            'type': 'number',
            'label': 'Protein Intake (Grams Per Kilogram of Body Weight)',
            'min': 1.2, 'max': 3.0, 'step': 0.1,
            'help': 'Set protein intake per kilogram of body weight',
            'advanced': True, 'required': False
        },
        'fat_percentage': {
            'type': 'number',
            'label': 'Fat Intake (Percentage of Total Calories)',
            'min': 15, 'max': 40, 'step': 1,
            'help': 'Set the percentage of total calories from fat',
            'convert': lambda x: x / 100 if x is not None else None,
            'advanced': True, 'required': False
        }
    }
}

# -----------------------------------------------------------------------------
# # Cell 4
# -----------------------------------------------------------------------------
# # ------ Unified Helper Functions ------
# -----------------------------------------------------------------------------


def initialize_session_state():
    """Initialize all session state variables using a unified approach."""
    session_vars = ['food_selections'] + [
        f'user_{field}' for field in CONFIG['form_fields'].keys()
    ]

    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == 'food_selections' else None


def create_unified_input(field_name, field_config, container=st.sidebar):
    """Create input widgets from a unified configuration."""
    session_key = f'user_{field_name}'

    if field_config['type'] == 'number':
        # Dynamically create a placeholder for advanced fields
        if field_config.get('advanced'):
            default_val = DEFAULTS.get(field_name, 0)
            # Handle the percentage display for the fat input field
            display_val = int(
                default_val * 100
            ) if field_name == 'fat_percentage' else default_val
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
            index = next((
                i for i, (_, val) in enumerate(field_config['options'])
                if val == current_value
            ), 0)
            selection = container.selectbox(
                field_config['label'],
                field_config['options'],
                index=index,
                format_func=lambda x: x[0]
            )
            value = selection[1]
        else:
            index = field_config['options'].index(
                current_value
            ) if current_value in field_config['options'] else 0
            value = container.selectbox(
                field_config['label'], field_config['options'], index=index
            )

    st.session_state[session_key] = value
    return value


def get_final_values(user_inputs):
    """Process all user inputs and apply defaults where needed."""
    final_values = {}

    for field, value in user_inputs.items():
        if field == 'sex':
            final_values[field] = value if value != "Select Sex" \
                else DEFAULTS[field]
        elif field in ['activity_level', 'goal']:
            final_values[field] = value if value is not None \
                else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None \
                else DEFAULTS[field]

    # Apply goal-specific defaults for the advanced settings
    if final_values['goal'] in GOAL_TARGETS:
        goal_config = GOAL_TARGETS[final_values['goal']]
        if user_inputs.get('protein_per_kg') is None:
            final_values['protein_per_kg'] = goal_config['protein_per_kg']
        if user_inputs.get('fat_percentage') is None:
            final_values['fat_percentage'] = goal_config['fat_percentage']

    return final_values


def calculate_hydration_needs(weight_kg, activity_level, climate='temperate'):
    """Calculate daily fluid needs based on body weight and activity."""
    base_needs = weight_kg * 35  # Baseline is 35 milliliters per kilogram

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

    total_ml = (
        base_needs + activity_bonus.get(activity_level, 500)
    ) * climate_multiplier.get(climate, 1.0)
    return round(total_ml)


def display_metrics_grid(metrics_data, num_columns=4):
    """Display metrics in a configurable column layout."""
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
    """Find a food that is a good source for the needed nutrient."""
    best_food = None
    highest_nutrient_val = 0

    # Flatten the food list to search across all categories
    all_foods = [item for sublist in foods.values() for item in sublist]

    for food in all_foods:
        # Prioritize foods rich in the specific nutrient
        if food[nutrient] > highest_nutrient_val:
            highest_nutrient_val = food[nutrient]
            best_food = food

    if best_food and highest_nutrient_val > 0:
        # Suggest one serving for simplicity
        suggestion_servings = 1
        return (
            f"Try adding **{suggestion_servings} serving of "
            f"{best_food['name']}** to get approximately "
            f"{best_food[nutrient] * suggestion_servings:.0f}g of {nutrient}."
        )
    return None


def create_progress_tracking(totals, targets, foods):
    """Create unified progress tracking with bars and recommendations."""
    recommendations = []

    st.subheader("Progress Toward Your Daily Nutritional Targets 🎯")

    purpose_map = {
        'calories': 'to reach your energy target',
        'protein': 'for muscle preservation and growth',
        'carbs': 'for energy and performance',
        'fat': 'for hormone production and health'
    }

    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets[config['target_key']]

        percent = min(actual / target * 100, 100) if target > 0 else 0
        st.progress(
            percent / 100,
            text=(
                f"{config['label']}: {percent:.0f}% of your daily target "
                f"({target:.0f} {config['unit']})"
            )
        )

        if actual < target:
            deficit = target - actual
            purpose = purpose_map.get(nutrient, 'for optimal nutrition')
            base_rec = (
                f"• You need **{deficit:.0f} more {config['unit']}** of "
                f"{config['label'].lower()} {purpose}."
            )

            # Add an actionable food suggestion for macronutrients
            if nutrient in ['protein', 'carbs', 'fat']:
                food_suggestion = find_best_food_for_nutrient(
                    nutrient, deficit, foods
                )
                if food_suggestion:
                    base_rec += f" {food_suggestion}"

            recommendations.append(base_rec)

    return recommendations


def calculate_daily_totals(food_selections, foods):
    """Calculate total daily nutrition from all food selections."""
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
    """Generate personalized recommendations based on intake and goals."""
    recommendations = []
    goal = final_values['goal']

    # Add a hydration recommendation
    hydration_ml = calculate_hydration_needs(
        final_values['weight_kg'], final_values['activity_level']
    )
    recommendations.append(
        f"💧 **Daily Hydration Target:** Aim for {hydration_ml} ml "
        f"({hydration_ml/250:.1f} cups). Drinking 500ml of water before "
        f"meals can help to boost satiety."
    )

    # Add goal-specific recommendations
    if goal == 'weight_loss':
        recommendations.extend([
            "🛏️ **Prioritize Sleep:** Aim for 7 to 9 hours nightly, as poor "
            "sleep can reduce fat loss effectiveness by up to 55%.",
            "📊 **Weigh-In Strategy:** Use daily morning weigh-ins but track "
            "the weekly averages to smooth out daily fluctuations.",
            "🥗 **Practice Volume Eating:** Prioritize high-volume and "
            "low-calorie foods like leafy greens to enhance meal satisfaction."
        ])
    elif goal == 'weight_gain':
        recommendations.extend([
            "🥤 **Incorporate Liquid Calories:** Include smoothies, milk, and "
            "juices to increase your overall calorie density.",
            "🥑 **Embrace Healthy Fats:** Add nuts, oils, and avocados, which "
            "are calorie-dense options that make a surplus easier to achieve.",
            "💪 **Focus on Progressive Overload:** Ensure you are getting "
            "stronger in the gym, as a surplus without training leads to fat "
            "gain."
        ])
    else:  # This case covers weight maintenance
        recommendations.extend([
            "⚖️ **Use Flexible Tracking:** Monitor your intake five days per "
            "week instead of seven for more sustainable maintenance.",
            "📅 **Conduct Regular Check-Ins:** Weigh yourself weekly and take "
            "measurements monthly to catch any changes early.",
            "🎯 **Follow the 80/20 Balance:** Aim for 80% nutrient-dense "
            "foods and 20% flexibility for social situations."
        ])

    # Add a protein timing recommendation
    protein_per_meal = targets['protein_g'] / 4
    recommendations.append(
        f"⏰ **Optimize Protein Timing:** Distribute protein across your "
        f"meals (at approximately {protein_per_meal:.0f}g per meal) for "
        f"optimal muscle protein synthesis."
    )

    return recommendations

# -----------------------------------------------------------------------------
# # Cell 5
# -----------------------------------------------------------------------------
# # ------ Nutritional Calculation Functions ------
# -----------------------------------------------------------------------------


def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation."""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculate Total Daily Energy Expenditure based on activity level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Calculate estimated weekly weight change from a caloric adjustment."""
    # This calculation is based on the approximation that 1 kilogram of body
    # fat contains approximately 7700 kilocalories
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(
    age, height_cm, weight_kg, sex='male',
    activity_level='moderately_active', goal='weight_gain',
    protein_per_kg=None, fat_percentage=None
):
    """Calculate personalized daily nutritional targets."""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)

    # Get the goal-specific configuration
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])

    # Apply the goal-specific caloric adjustment
    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment

    # Use provided values or fall back to goal-specific defaults
    protein_per_kg_final = protein_per_kg if protein_per_kg is not None \
        else goal_config['protein_per_kg']
    fat_percentage_final = fat_percentage if fat_percentage is not None \
        else goal_config['fat_percentage']

    protein_g = protein_per_kg_final * weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage_final
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    # Calculate the estimated weekly weight change
    estimated_weekly_change = calculate_estimated_weekly_change(
        caloric_adjustment
    )

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories),
        'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g),
        'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(estimated_weekly_change, 3),
        'goal': goal
    }

    if targets['total_calories'] > 0:
        targets['protein_percent'] = (
            targets['protein_calories'] / targets['total_calories']
        ) * 100
        targets['carb_percent'] = (
            targets['carb_calories'] / targets['total_calories']
        ) * 100
        targets['fat_percent'] = (
            targets['fat_calories'] / targets['total_calories']
        ) * 100
    else:
        targets['protein_percent'] = 0
        targets['carb_percent'] = 0
        targets['fat_percent'] = 0

    return targets

# -----------------------------------------------------------------------------
# # Cell 6
# -----------------------------------------------------------------------------
# # ------ Food Database Processing Functions ------
# -----------------------------------------------------------------------------


@st.cache_data
def load_food_database(file_path):
    """Load the vegetarian food database from a specified CSV file."""
    df = pd.read_csv(file_path)
    # Use the unique categories directly from the CSV file
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
        if not items:
            continue

        # Rank the top three most calorie-dense foods within each category
        sorted_by_calories = sorted(
            items, key=lambda x: x['calories'], reverse=True
        )
        top_foods['calories'][category] = [
            food['name'] for food in sorted_by_calories[:3]
        ]

        # Rank the top three foods by their primary macronutrient
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(
                items, key=lambda x: x[map_info['sort_by']], reverse=True
            )
            top_foods[map_info['key']] = [
                food['name'] for food in sorted_by_nutrient[:3]
            ]

    # Create a set of all foods that are top nutrient performers
    all_top_nutrient_foods = {
        food for key in ['protein', 'carbs', 'fat']
        for food in top_foods[key]
    }

    # Define the emoji mapping for visual identification
    emoji_mapping = {
        'high_cal_nutrient': '🥇', 'high_calorie': '🔥',
        'protein': '💪', 'carbs': '🍚', 'fat': '🥑'
    }

    # Assign emojis based on the rankings
    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(
                category, []
            )

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
    """Render a single food item with its interaction controls."""
    with st.container(border=True):
        st.subheader(f"{food.get('emoji', '')} {food['name']}")
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(
            food['name'], 0.0
        )

        col1, col2 = st.columns([2, 1.2])

        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    button_type = "primary" if current_serving == float(k) \
                        else "secondary"
                    if st.button(
                        f"{k}", key=f"{key}_{k}", type=button_type,
                        help=f"Set to {k} servings", use_container_width=True
                    ):
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

        # Display the nutritional information per serving
        caption_text = (
            f"Per Serving: {food['calories']} kcal | "
            f"{food['protein']}g protein | {food['carbs']}g carbs | "
            f"{food['fat']}g fat"
        )
        st.caption(caption_text)


def render_food_grid(items, category, columns=2):
    """Render food items in a responsive grid layout."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)

# -----------------------------------------------------------------------------
# # Cell 7
# -----------------------------------------------------------------------------
# # ------ Initialize Application ------
# -----------------------------------------------------------------------------

# Initialize the session state at the start of the script
initialize_session_state()

# Load the food database and assign emojis for visual cues
foods = load_food_database('nutrition_results.csv')
foods = assign_food_emojis(foods)

# Apply custom CSS for enhanced styling of UI elements
st.markdown("""
<style>
[data-testid="InputInstructions"] { display: none; }
.stButton>button[kind="primary"] {
    background-color: #ff6b6b;
    color: white;
    border: 1px solid #ff6b6b;
}
.stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
.sidebar .sidebar-content { background-color: #f0f2f6; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# # Cell 8
# -----------------------------------------------------------------------------
# # ------ Application Title and Unified Input Interface ------
# -----------------------------------------------------------------------------

st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
This advanced nutrition tracker uses evidence-based calculations to provide
personalized daily nutrition goals for **weight loss**, **weight maintenance**,
or **weight gain**. The calculator employs the **Mifflin-St Jeor equation**
for Basal Metabolic Rate and follows a **protein-first macronutrient
strategy** as recommended by nutrition science. 🚀
""")

# ------ Sidebar for an Improved User Experience ------
st.sidebar.header("Your Personal Parameters for Daily Target Calculation 📊")

all_inputs = {}

# Separate standard and advanced fields to control their display order
standard_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if not v.get('advanced')
}
advanced_fields = {
    k: v for k, v in CONFIG['form_fields'].items() if v.get('advanced')
}

# Render the standard (primary) input fields first
for field_name, field_config in standard_fields.items():
    value = create_unified_input(field_name, field_config, container=st.sidebar)
    if 'convert' in field_config:
        value = field_config['convert'](value)
    all_inputs[field_name] = value


# Render the advanced fields inside an expander at the bottom
with st.sidebar.expander("Advanced Settings ⚙️"):
    for field_name, field_config in advanced_fields.items():
        value = create_unified_input(
            field_name, field_config, container=st.container()
        )
        if 'convert' in field_config:
            value = field_config['convert'](value)
        all_inputs[field_name] = value

# ------ Activity Level Guide in the Sidebar ------
with st.sidebar.container(border=True):
    st.markdown("""
    **A Guide to Activity Levels:**

    - 🧑‍💻 **Sedentary:** Little to no exercise, with a desk job.

    - 🚶 **Lightly Active:** Light exercise for 1 to 3 days per week.

    - 🏃 **Moderately Active:** Moderate exercise for 3 to 5 days per week.

    - 🏋️ **Very Active:** Heavy exercise for 6 to 7 days per week.

    - 🚴 **Extremely Active:** Very heavy exercise, a physical job, or training
    twice per day.

    *💡 When in doubt, it is best to choose a lower activity level to avoid
    overestimating your calorie needs.*
    """)

# ------ Process Final Values Using the Unified Approach ------
final_values = get_final_values(all_inputs)

# Display the hydration recommendation in the sidebar
if all_inputs.get('weight_kg') and all_inputs.get('activity_level'):
    hydration_ml = calculate_hydration_needs(
        final_values['weight_kg'], final_values['activity_level']
    )
    st.sidebar.info(
        f"💧 **Daily Hydration Target:** {hydration_ml} ml "
        f"({hydration_ml/250:.1f} cups)"
    )

# ------ Dynamically Check for User Input Completeness ------
required_fields = [
    field for field, config in CONFIG['form_fields'].items()
    if config.get('required')
]
user_has_entered_info = all(
    (
        all_inputs.get(field) is not None and
        all_inputs.get(field) !=
        CONFIG['form_fields'][field].get('placeholder')
    )
    for field in required_fields
)

# ------ Calculate the Personalized Targets ------
targets = calculate_personalized_targets(**final_values)

# -----------------------------------------------------------------------------
# # Cell 9
# -----------------------------------------------------------------------------
# # ------ Unified Target Display System ------
# -----------------------------------------------------------------------------

if not user_has_entered_info:
    st.info(
        "👈 Please enter your personal information in the sidebar to view "
        "your daily nutritional targets."
    )
    st.header("Sample Daily Targets for Your Reference 🎯")
    st.caption(
        "These are example targets. Please enter your information in the "
        "sidebar for personalized calculations."
    )
else:
    goal_labels = {
        'weight_loss': 'Weight Loss',
        'weight_maintenance': 'Weight Maintenance',
        'weight_gain': 'Weight Gain'
    }
    goal_label = goal_labels.get(targets['goal'], 'Weight Gain')
    st.header(
        f"Your Personalized Daily Nutritional Targets for {goal_label} 🎯"
    )

# ------ 80/20 Principle Information Box ------
st.info(
    "🎯 **The 80/20 Principle:** Aim for 80% adherence to your targets rather "
    "than perfection. This approach allows for social flexibility and helps "
    "to prevent the all-or-nothing mentality that often leads to diet cycling."
)

# Calculate hydration needs for the metrics grid
hydration_ml = calculate_hydration_needs(
    final_values['weight_kg'], final_values['activity_level']
)

# ------ Unified Metrics Display Configuration ------
metrics_config = [
    {
        'title': 'Your Metabolic Information', 'columns': 5,
        'metrics': [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal per day"),
            (
                "Total Daily Energy Expenditure (TDEE)",
                f"{targets['tdee']} kcal per day"
            ),
            (
                "Daily Caloric Adjustment",
                f"{targets['caloric_adjustment']:+} kcal per day"
            ),
            (
                "Estimated Weekly Weight Change",
                f"{targets['estimated_weekly_change']:+.2f} kg per week"
            ),
            ("", "")  # Blank entry for alignment purposes
        ]
    },
    {
        'title': 'Your Daily Macronutrient and Hydration Targets',
        'columns': 5,
        'metrics': [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            (
                "Protein", f"{targets['protein_g']} g",
                f"{targets['protein_percent']:.0f}%"
            ),
            (
                "Carbohydrates", f"{targets['carb_g']} g",
                f"{targets['carb_percent']:.0f}%"
            ),
            ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}%"),
            (
                "💧 Hydration", f"{hydration_ml} ml",
                f"~{hydration_ml/250:.1f} cups"
            )
        ]
    }
]


# ------ Display All Metric Sections ------
for config in metrics_config:
    st.subheader(config['title'])
    display_metrics_grid(config['metrics'], config['columns'])
    st.divider()


# -----------------------------------------------------------------------------
# # Cell 10
# -----------------------------------------------------------------------------
# # ------ Enhanced Evidence-Based Tips and Context ------
# -----------------------------------------------------------------------------

st.header("📚 Your Evidence-Based Playbook")
tab1, tab2, tab3, tab4 = st.tabs([
    "Foundations", "Advanced Strategies", "Troubleshooting", "Nutrition Science"
])

with tab1:
    st.subheader("🏆 Essential Tips for Your Success")
    st.markdown(r"""
    ### **The Foundation Trio for Success**

    **💧 The Hydration Protocol:**
    - **Target:** Aim for 35ml per kg of body weight daily
    - **Training Bonus:** Add 500 to 750ml per hour of exercise
    - **Fat Loss Hack:** Drinking 500ml of water before meals increases
      satiety by 13%

    **😴 Sleep Quality (The Game-Changer):**
    - **Less than 7 hours of sleep** can reduce fat loss effectiveness by
      up to 55%
    - **Target:** Aim for 7 to 9 hours nightly with consistent sleep and
      wake times
    - **Optimization:** Use a dark, cool room (18-20°C) and avoid screens
      for 1 to 2 hours before bed

    **⚖️ Best Practices for Weigh-Ins:**
    - **Daily:** Weigh in at the same time, such as in the morning,
      post-bathroom, and with minimal clothing
    - **Track:** Focus on weekly averages, not daily fluctuations
    - **Adjust:** Make changes only after two or more weeks of stalled
      progress
    """)

with tab2:
    st.subheader("📊 Advanced Monitoring and Psychology")
    st.markdown(r"""
    ### **Beyond the Scale: Better Indicators of Progress**
    - **Progress Photos:** Take photos in the same lighting, with the same
      poses, and at the same time of day
    - **Body Measurements:** Measure your waist, hips, arms, and thighs
      monthly
    - **Performance Metrics:** Track your strength gains, energy levels,
      and sleep quality

    ### **The Psychology of Sustainable Change**
    **The 80/20 Rule:** Aim for 80% adherence rather than perfection. This
    practice allows for social flexibility and prevents the "all-or-nothing"
    mentality that leads to diet cycling.

    **Progressive Implementation:**
    - **Weeks 1-2:** Focus only on hitting your calorie targets
    - **Weeks 3-4:** Add your protein targets to the focus
    - **Week 5+:** Fine-tune your fat and carbohydrate distribution

    **Biofeedback Awareness:** Monitor your energy levels, sleep quality,
    gym performance, and hunger patterns, not just the scale.
    """)

with tab3:
    st.subheader("🔄 Plateau Prevention and Meal Timing")
    st.markdown(r"""
    ### **A Troubleshooting Flow for Plateaus**
    **For Weight Loss Plateaus:**
    1. Confirm the accuracy of your logging (within ±5% of calories)
    2. Re-validate your activity multiplier to ensure it is accurate
    3. Add 10 to 15 minutes of daily walking before reducing calories
    4. Implement "diet breaks" by spending 1 to 2 weeks at maintenance
       every 6 to 8 weeks

    **For Weight Gain Plateaus:**
    1. Increase your intake of liquid calories (smoothies, milk)
    2. Add more healthy fats to your diet (nuts, oils, avocados)
    3. Ensure you are applying progressive overload in your training
    4. Make gradual increases of 100 to 150 calories when stalled for
       two or more weeks

    ### **Meal Timing and Distribution**
    **For Protein Optimization:**
    - **Distribution:** Aim for 20 to 30g across 3 to 4 meals, which is
      about 0.4 to 0.5g per kg of body weight per meal
    - **Post-Workout:** Consume 20 to 40g within 2 hours of training
    - **Pre-Sleep:** Have 20 to 30g of casein for overnight muscle protein
      synthesis

    **For Performance Timing:**
    - **Pre-Workout:** Consume moderate carbohydrates and protein 1 to 2
      hours prior to exercise
    - **Post-Workout:** Have a combination of protein and carbohydrates
      within 2 hours
    """)

with tab4:
    st.subheader("🔬 The Scientific Foundation and a Deep Dive into Nutrition")
    st.markdown(r"""
    ### **The Energy Foundation: BMR and TDEE**

    **Basal Metabolic Rate (BMR):** This is your body's energy needs at
    complete rest, calculated using the **Mifflin-St Jeor equation**, which
    is the most accurate formula recognized by the Academy of Nutrition and
    Dietetics.

    **Total Daily Energy Expenditure (TDEE):** These are your maintenance
    calories, including all daily activities, calculated by multiplying your
    BMR by scientifically validated activity factors.

    ### **The Satiety Hierarchy (for Better Adherence)**
    1. **Protein** (provides the highest satiety per calorie)
    2. **Fiber-Rich Carbohydrates** (vegetables, fruits, whole grains)
    3. **Healthy Fats** (nuts, avocado, olive oil)
    4. **Processed Foods** (provide the lowest satiety per calorie)

    **Fiber Target:** Aim for 14g per 1,000 kcal (approximately 25 to 38g
    daily). It is best to increase fiber intake gradually to avoid any
    gastrointestinal distress.

    **The Volume Eating Strategy:** Prioritize low-calorie, high-volume
    foods like leafy greens, cucumbers, and berries to create meal
    satisfaction without exceeding your calorie targets.

    ### **Considerations for Micronutrients**
    **Common Shortfalls in Plant-Forward Diets:**
    - **B₁₂, iron, calcium, zinc, iodine, and omega-3 (EPA/DHA)**
    - **Strategy:** Include fortified foods in your diet or consider
      targeted supplementation based on your lab work.
    """)

# -----------------------------------------------------------------------------
# # Cell 11
# -----------------------------------------------------------------------------
# # ------ Personalized Recommendations System ------
# -----------------------------------------------------------------------------

if user_has_entered_info:
    st.header("🎯 Your Personalized Action Plan")

    # Calculate the current totals to generate recommendations
    totals, _ = calculate_daily_totals(
        st.session_state.food_selections, foods
    )
    recommendations = generate_personalized_recommendations(
        totals, targets, final_values
    )

    for rec in recommendations:
        st.info(rec)

# -----------------------------------------------------------------------------
# # Cell 12
# -----------------------------------------------------------------------------
# # ------ Food Selection Interface ------
# -----------------------------------------------------------------------------

st.header("Your Daily Food Selection and Tracking 🥗")
st.markdown(
    "Select the number of servings for each food item below to track your "
    "daily nutrition intake."
)


with st.expander("💡 View the Food Emoji Guide"):
    st.markdown("""
    **A Guide to the Food Emojis:**

    - 🥇 **Gold Medal:** A top performer in both calories and its primary
      nutrient.

    - 🔥 **High Calorie:** Among the most calorie-dense foods in its category.

    - 💪 **High Protein:** A top source of protein.

    - 🍚 **High Carbohydrate:** A top source of carbohydrates.

    - 🥑 **High Fat:** A top source of healthy fats.

    *Foods are ranked within each category to help you make efficient
    choices that align with your goals.*
    """)


# ------ Reset Selection Button ------
if st.button("🔄 Reset All of Your Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.rerun()

# ------ Food Selection with Tabs ------
available_categories = [
    cat for cat, items in sorted(foods.items()) if items
]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    # Sort items by emoji priority first, then by calories
    sorted_items_in_category = sorted(
        items,
        key=lambda x: (
            CONFIG['emoji_order'].get(x.get('emoji', ''), 4), -x['calories']
        )
    )
    with tabs[i]:
        render_food_grid(sorted_items_in_category, category, columns=2)


# -----------------------------------------------------------------------------
# # Cell 13
# -----------------------------------------------------------------------------
# # ------ Daily Summary and Progress Tracking ------
# -----------------------------------------------------------------------------

st.header("Your Daily Nutrition Summary 📊")

# Calculate the current daily totals from selections
totals, selected_foods = calculate_daily_totals(
    st.session_state.food_selections, foods
)

if selected_foods:
    # Display progress tracking with recommendations
    recommendations = create_progress_tracking(totals, targets, foods)

    # Display the daily summary metrics
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Today's Nutrition Intake")
        summary_metrics = [
            ("Calories Consumed", f"{totals['calories']:.0f} kcal"),
            ("Protein Intake", f"{totals['protein']:.0f} g"),
            ("Carbohydrates", f"{totals['carbs']:.0f} g"),
            ("Fat Intake", f"{totals['fat']:.0f} g")
        ]
        display_metrics_grid(summary_metrics, 2)

    with col2:
        st.subheader("Your Macronutrient Split (in Grams)")
        # Create a donut chart for the macronutrient split
        macro_values = [totals['protein'], totals['carbs'], totals['fat']]
        if sum(macro_values) > 0:
            fig = go.Figure(go.Pie(
                labels=['Protein', 'Carbs', 'Fat'],
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
            st.caption("Please select foods to see the macronutrient split.")

    # Display recommendations based on the current intake
    if recommendations:
        st.subheader("Your Personalized Recommendations for Today")
        for rec in recommendations:
            st.info(rec)

    # Display a detailed food breakdown
    with st.expander("📝 View a Detailed Food Breakdown"):
        st.subheader("The Foods You Have Selected Today")
        for item in selected_foods:
            food = item['food']
            servings = item['servings']
            total_cals = food['calories'] * servings
            total_protein = food['protein'] * servings
            total_carbs = food['carbs'] * servings
            total_fat = food['fat'] * servings

            st.write(f"**{food['name']}** - {servings} serving(s)")
            st.write(
                f"  → {total_cals:.0f} kcal | {total_protein:.1f}g protein | "
                f"{total_carbs:.1f}g carbs | {total_fat:.1f}g fat"
            )
else:
    st.info(
        "You have not selected any foods yet. Please choose foods from the "
        "categories above to track your daily intake."
    )

    # Show sample progress bars with zero values
    st.subheader("Progress Toward Your Daily Nutritional Targets 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        target = targets[config['target_key']]
        st.progress(
            0.0,
            text=(
                f"{config['label']}: 0% of your daily target "
                f"({target:.0f} {config['unit']})"
            )
        )

# -----------------------------------------------------------------------------
# # Cell 14
# -----------------------------------------------------------------------------
# # ------ Footer and Additional Resources ------
# -----------------------------------------------------------------------------

st.divider()
st.markdown("""
### **📚 Evidence-Based References and Methodology**

This nutrition tracker is built on peer-reviewed research and established
evidence-based guidelines to ensure its accuracy and effectiveness.

- **BMR Calculation:** The tracker uses the Mifflin-St Jeor equation, which
  is recommended by the Academy of Nutrition and Dietetics.
- **Activity Factors:** The calculations are based on validated Total Daily
  Energy Expenditure (TDEE) multipliers from exercise physiology research.
- **Protein Targets:** The targets align with the position stands of the
  International Society of Sports Nutrition.
- **Caloric Adjustments:** The adjustments use conservative and sustainable
  rates that are based on body composition research.

### **⚠️ Important Disclaimers**

- This tool provides general nutrition guidance that is based on population
  averages and should not be considered medical advice.
- Individual nutritional needs may vary based on genetics, medical
  conditions, and other personal factors.
- Please consult with a qualified healthcare provider or registered
  dietitian before making significant changes to your diet.
- It is important to monitor your biofeedback, including your energy levels,
  performance, and health markers, and adjust as needed.

### **🔬 A Commitment to Continuous Improvement**

This tracker incorporates the latest findings from nutrition science. As
research evolves, the recommendations may be updated to reflect current
best practices and ensure you receive the most reliable guidance.
""")

# -----------------------------------------------------------------------------
# # Cell 15
# -----------------------------------------------------------------------------
# # ------ Session State Management and Closing Message ------
# -----------------------------------------------------------------------------

# Clean up the session state to prevent potential memory issues
if len(st.session_state.food_selections) > 100:  # Arbitrary limit
    # Keep only the non-zero selections to optimize performance
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# Add a fun and encouraging closing message
st.success(
    "You've got all the tools and information you need to crush your goals! "
    "Thanks for using the tracker. Now go make it happen! 💪"
)
