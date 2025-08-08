#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ---------------------------------------------------------------------------
# A Personalized Evidence-Based Nutrition Tracker for Goal-Specific Meal Planning
# ---------------------------------------------------------------------------

"""
This script implements an interactive, evidence-based nutrition tracker using
Streamlit. It is designed to help users achieve personalized nutrition goals,
such as weight loss, maintenance, or gain, with a focus on vegetarian food
sources.

Core Functionality and Scientific Basis:
- Basal Metabolic Rate (BMR) Calculation: The application uses the Mifflin-St
  Jeor equation, which is widely recognized by organizations like the Academy
  of Nutrition and Dietetics for its accuracy.
  - For Males: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
  - For Females: BMR = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

- Total Daily Energy Expenditure (TDEE): The BMR is multiplied by a
  scientifically validated activity factor to estimate the total number of
  calories burned in a day, including physical activity.

- Goal-Specific Caloric Adjustments:
  - Weight Loss: A conservative 20 percent caloric deficit from TDEE.
  - Weight Maintenance: Caloric intake is set equal to TDEE.
  - Weight Gain: A controlled 10 percent caloric surplus over TDEE.

- Macronutrient Strategy: The script follows a protein-first approach,
  consistent with modern nutrition science.
  1. Protein intake is determined based on grams per kilogram of body weight.
  2. Fat intake is set as a percentage of total daily calories.
  3. Carbohydrate intake is calculated from the remaining caloric budget.

Implementation Details:
- The user interface is built with Streamlit, providing interactive widgets
  for user input and data visualization.
- The food database is managed using the Pandas library.
- Progress visualizations are created with Streamlit's native components and
  Plotly for generating detailed charts.

Usage Documentation:
1. Prerequisites: Ensure you have the required Python libraries installed.
   You can install them using pip:
   pip install streamlit pandas plotly

2. Running the Application: Save this script as a Python file (for example,
   `nutrition_app.py`) and run it from your terminal using the following
   command:
   streamlit run nutrition_app.py

3. Interacting with the Application:
   - Use the sidebar to enter your personal details, such as age, height,
     weight, sex, activity level, and primary nutrition goal.
   - Use the 'Calculate' button to validate your inputs and generate your plan.
   - Your personalized daily targets for calories and macronutrients will be
     calculated and displayed.
   - Navigate through the food tabs to select the number of servings for
     each food item you consume.
   - The daily summary section will update in real time to show your
     progress toward your targets.
"""

# ---------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# ---------------------------------------------------------------------------
import json
import math
from io import StringIO
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Cell 2: Page Configuration and Initial Setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Your Personal Nutrition Coach 🍽️",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Cell 3: Unified Configuration Constants
# ---------------------------------------------------------------------------

# ------ Default Parameter Values Based on Published Research ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180, 'height_in': 71,
    'weight_kg': 57.5, 'weight_lbs': 127,
    'sex': "Male",
    'activity_level': "moderately_active",
    'goal': "weight_gain",
    'protein_per_kg': 2.0,
    'fat_percentage': 0.25
}

# ------ Activity Level Multipliers for TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2, 'lightly_active': 1.375, 'moderately_active': 1.55,
    'very_active': 1.725, 'extremely_active': 1.9
}

# ------ Activity Level Descriptions ------
ACTIVITY_DESCRIPTIONS = {
    'sedentary': "Little to no exercise, desk job",
    'lightly_active': "Light exercise one to three days per week",
    'moderately_active': "Moderate exercise three to five days per week",
    'very_active': "Heavy exercise six to seven days per week",
    'extremely_active': "Very heavy exercise, a physical job, or "
                        "two times per day training"
}

# ------ Goal-Specific Targets Based on an Evidence-Based Guide ------
GOAL_TARGETS = {
    'weight_loss': {'caloric_adjustment': -0.20, 'protein_per_kg': 1.8, 'fat_percentage': 0.25},
    'weight_maintenance': {'caloric_adjustment': 0.0, 'protein_per_kg': 1.6, 'fat_percentage': 0.30},
    'weight_gain': {'caloric_adjustment': 0.10, 'protein_per_kg': 2.0, 'fat_percentage': 0.25}
}

# ------ Unified Configuration for All App Components ------
CONFIG = {
    'emoji_map': {
        '🥇': "A nutritional all-star! High in its target nutrient and very calorie-efficient.",
        '🔥': "One of the more calorie-dense options in its group.",
        '💪': "A true protein powerhouse.",
        '🍚': "A carbohydrate champion.",
        '🥑': "A healthy fat hero."
    },
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
    }
}

# ---------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# ---------------------------------------------------------------------------

def initialize_session_state():
    """Initializes all required session state variables if they don't exist."""
    # Suggestion 4: Part of Save/Load state management
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = {}
    if 'food_selections' not in st.session_state:
        st.session_state.food_selections = {}
    if 'calculated' not in st.session_state:
        st.session_state.calculated = False
    if 'units' not in st.session_state:
        st.session_state.units = 'Metric' # Default unit system

def get_progress_bar_color(percentage):
    """
    Suggestion 1: Returns a color for the progress bar based on completion percentage.
    """
    if percentage >= 80:
        return "green"
    elif percentage >= 50:
        return "orange"
    else:
        return "red"

def create_progress_tracking(totals, targets):
    """
    Suggestion 1: Creates color-coded progress bars and recommendations for nutritional targets.
    """
    st.subheader("Your Daily Dashboard 🎯")
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target = targets.get(config['target_key'], 0)
        percent = min(actual / target * 100, 100) if target > 0 else 0
        color = get_progress_bar_color(percent)

        st.markdown(
            f"**{config['label']}: {actual:.0f} / {target:.0f} {config['unit']}**"
        )
        st.progress(int(percent))
        # This is a CSS hack to color the progress bar. Streamlit does not support it natively.
        st.markdown(f"""
            <style>
                .stProgress > div > div > div > div {{
                    background-color: {color};
                }}
            </style>""",
            unsafe_allow_html=True,
        )


def calculate_daily_totals(food_selections, foods_df):
    """Calculates the total daily nutrition from all selected foods."""
    totals = {nutrient: 0 for nutrient in CONFIG['nutrient_configs'].keys()}
    selected_foods_list = []

    if not food_selections:
        return totals, selected_foods_list

    # Create a small DataFrame of selected foods for efficient calculation
    selected_names = list(food_selections.keys())
    servings = pd.Series(food_selections, name="servings")
    
    # Filter the main DataFrame to only include selected foods
    df_selected = foods_df[foods_df['name_with_unit'].isin(selected_names)].set_index('name_with_unit')
    
    # Join with servings and calculate totals
    df_selected = df_selected.join(servings)

    for nutrient in totals.keys():
        totals[nutrient] = (df_selected[nutrient] * df_selected['servings']).sum()

    for food_name, row in df_selected.iterrows():
        selected_foods_list.append({
            'name': food_name,
            'servings': row['servings'],
            'calories': row['calories'] * row['servings'],
            'protein': row['protein'] * row['servings'],
            'carbs': row['carbs'] * row['servings'],
            'fat': row['fat'] * row['servings']
        })

    return totals, selected_foods_list


def display_metrics_grid(metrics_data, num_columns=4):
    """Displays a grid of metrics in a configurable column layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            label, value, *rest = metric_info
            help_text = rest[1] if len(rest) > 1 else None
            st.metric(label=label, value=value, delta=rest[0] if rest else None, help=help_text)

# ---------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# ---------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculates Basal Metabolic Rate using the Mifflin-St Jeor equation."""
    if not all([age, height_cm, weight_kg, sex]):
        return 0
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == 'male' else -161)


def calculate_tdee(bmr, activity_level):
    """Calculates Total Daily Energy Expenditure."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment):
    """Calculates estimated weekly weight change. 1 kg fat ~= 7700 kcal."""
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(user_data):
    """Calculates all personalized daily nutritional targets from a user data dictionary."""
    bmr = calculate_bmr(user_data['age'], user_data['height_cm'], user_data['weight_kg'], user_data['sex'])
    tdee = calculate_tdee(bmr, user_data['activity_level'])

    goal = user_data['goal']
    goal_config = GOAL_TARGETS.get(goal, GOAL_TARGETS['weight_gain'])
    
    # Use user-defined advanced settings or fall back to goal-based defaults
    protein_per_kg = user_data.get('protein_per_kg') or goal_config['protein_per_kg']
    fat_percentage = user_data.get('fat_percentage') or goal_config['fat_percentage']

    caloric_adjustment = tdee * goal_config['caloric_adjustment']
    total_calories = tdee + caloric_adjustment

    protein_g = protein_per_kg * user_data['weight_kg']
    protein_calories = protein_g * 4
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / 9
    
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    targets = {
        'bmr': round(bmr), 'tdee': round(tdee),
        'total_calories': round(total_calories), 'caloric_adjustment': round(caloric_adjustment),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'estimated_weekly_change': round(calculate_estimated_weekly_change(caloric_adjustment), 2),
        'goal': goal
    }

    # Suggestion 8: Handle division by zero gracefully.
    if targets['total_calories'] > 0:
        targets['protein_percent'] = (targets['protein_calories'] / targets['total_calories']) * 100
        targets['carb_percent'] = (targets['carb_calories'] / targets['total_calories']) * 100
        targets['fat_percent'] = (targets['fat_calories'] / targets['total_calories']) * 100
    else:
        targets['protein_percent'] = targets['carb_percent'] = targets['fat_percent'] = 0

    return targets

# ---------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# ---------------------------------------------------------------------------

@st.cache_data # Suggestion 7: Cache the entire processed DataFrame, including emojis.
def get_processed_food_database(file_path):
    """Loads and processes the food database, assigning emojis. This function is cached."""
    df = pd.read_csv(file_path)
    df['name_with_unit'] = df['name'] + " (" + df['serving_unit'] + ")"
    
    # Assign Emojis
    df['emoji'] = ''
    top_foods = {}
    
    for category, group in df.groupby('category'):
        # Get top 3 by primary nutrient
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            top_nutrient_foods = group.nlargest(3, map_info['sort_by'])['name_with_unit']
            emoji_symbol = '💪' if map_info['key'] == 'protein' else '🍚' if map_info['key'] == 'carbs' else '🥑'
            df.loc[df['name_with_unit'].isin(top_nutrient_foods), 'emoji'] = emoji_symbol
            top_foods[category] = top_nutrient_foods

        # Get top 3 by calories
        top_calorie_foods = group.nlargest(3, 'calories')['name_with_unit']
        df.loc[df['name_with_unit'].isin(top_calorie_foods), 'emoji'] += '🔥'
        
    # Handle 'Gold Medal' foods (top in nutrient AND calories)
    df['emoji'] = df['emoji'].str.replace(r'([💪🍚🥑])🔥', '🥇', regex=True)
    df['emoji'] = df['emoji'].str.replace('🔥', '', regex=False) # remove fire if it was not combined
    df.loc[df['emoji'].str.contains('🥇', na=False), 'emoji'] = '🥇'


    # Assign sort order based on emoji
    df['emoji_order'] = df['emoji'].map(CONFIG['emoji_order']).fillna(4)

    return df

# ---------------------------------------------------------------------------
# Cell 7: UI Rendering Functions
# ---------------------------------------------------------------------------

def render_food_item(food, category_key):
    """Renders a single food item with its interaction controls."""
    # Suggestion 2: Added tooltip to emoji using the help parameter.
    emoji_html = f"<span title='{CONFIG['emoji_map'].get(food['emoji'], '')}'>{food['emoji']}</span>" if food['emoji'] else ""
    
    st.markdown(f"<h5>{emoji_html} {food['name_with_unit']}</h5>", unsafe_allow_html=True)
    
    # Suggestion 9: Use unique keys for all widgets to reduce reruns.
    key_base = f"{category_key}_{food['name_with_unit']}"
    current_serving = st.session_state.food_selections.get(food['name_with_unit'], 0.0)

    col1, col2 = st.columns([3, 2])
    with col1:
        # Suggestion 8: Cap max servings to a reasonable number. Here, it's 10.
        new_serving = st.number_input(
            "Servings", min_value=0.0, max_value=20.0,
            value=float(current_serving), step=0.5, key=f"{key_base}_num",
            label_visibility="collapsed"
        )
        if new_serving != current_serving:
            st.session_state.food_selections[food['name_with_unit']] = new_serving
            # Only keep non-zero servings to keep the state clean
            st.session_state.food_selections = {k: v for k, v in st.session_state.food_selections.items() if v > 0}
            st.rerun()

    with col2:
        if st.button("❌", key=f"{key_base}_clear", help="Reset to 0 servings"):
            if food['name_with_unit'] in st.session_state.food_selections:
                del st.session_state.food_selections[food['name_with_unit']]
                st.rerun()
                
    caption_text = (f"Per Serving: {food['calories']:.0f} kcal | "
                    f"{food['protein']:.1f}g P | {food['carbs']:.1f}g C | {food['fat']:.1f}g F")
    st.caption(caption_text)


def render_food_grid(items_df, category, columns=2):
    """Renders a grid of food items for a given category."""
    # Suggestion 5: Filter food items based on the search query.
    search_query = st.session_state.get('food_search', '').lower()
    if search_query:
        items_df = items_df[items_df['name_with_unit'].str.lower().str.contains(search_query)]
        if items_df.empty:
            st.caption("No food found matching your search in this category.")
            return

    items = items_df.to_dict('records')
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    with st.container(border=True):
                         render_food_item(items[i + j], category)

# ---------------------------------------------------------------------------
# Cell 8: Main Application Logic
# ---------------------------------------------------------------------------

def main():
    """Main function to run the Streamlit application."""
    initialize_session_state()
    
    # Suggestion 7: Load data once using the cached function.
    foods_df = get_processed_food_database('nutrition_results.csv')

    # ------ Apply Custom CSS for Enhanced Styling ------
    # Suggestion 10: Improved color contrast for WCAG compliance.
    st.markdown("""
    <style>
    [data-testid="stMetricValue"] { font-size: 1.5rem; }
    h5 { margin-bottom: 0.1rem; }
    /* Improved button color contrast */
    .stButton>button[kind="primary"] { background-color: #C9302C; color: white; border-color: #C9302C; }
    .stButton>button[kind="primary"]:hover { background-color: #A9201D; border-color: #A9201D; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Your Personal Nutrition Coach 🍽️")
    st.markdown("""
    Welcome to your personalized nutrition guide! This isn’t just another calorie counter—it’s a smart tracker built on rock-solid science to help you achieve your goals. Let’s get rolling! 🚀
    """)
    
    # ------------------ SIDEBAR FOR USER INPUTS ------------------
    with st.sidebar:
        st.header("Your Profile 📊")
        
        # Suggestion 11: Add Metric Units Toggle
        unit_system = st.radio(
            "Choose your preferred units:",
            ('Metric (kg/cm)', 'Imperial (lbs/in)'),
            key='unit_toggle',
            horizontal=True
        )
        st.session_state.units = 'Metric' if 'Metric' in unit_system else 'Imperial'

        # Suggestion 6: Validate User Inputs on Submit using a form.
        with st.form(key='user_input_form'):
            st.subheader("Personal Details")
            age = st.number_input('Age (years)', min_value=16, max_value=100, value=DEFAULTS['age'], step=1, key='age')
            
            if st.session_state.units == 'Imperial':
                height = st.number_input('Height (inches)', min_value=50, max_value=90, value=DEFAULTS['height_in'], step=1, key='height_in')
                weight = st.number_input('Weight (lbs)', min_value=80.0, max_value=350.0, value=float(DEFAULTS['weight_lbs']), step=0.5, key='weight_lbs')
            else:
                height = st.number_input('Height (cm)', min_value=140, max_value=220, value=DEFAULTS['height_cm'], step=1, key='height_cm')
                weight = st.number_input('Weight (kg)', min_value=40.0, max_value=150.0, value=float(DEFAULTS['weight_kg']), step=0.5, key='weight_kg')

            sex = st.selectbox('Biological Sex', ["Male", "Female"], index=["Male", "Female"].index(DEFAULTS['sex']), key='sex')

            st.subheader("Activity & Goals")
            activity_level = st.selectbox(
                'Activity Level', options=ACTIVITY_DESCRIPTIONS.keys(),
                format_func=lambda x: f"{x.replace('_', ' ').title()} - {ACTIVITY_DESCRIPTIONS[x]}",
                index=list(ACTIVITY_DESCRIPTIONS.keys()).index(DEFAULTS['activity_level']), key='activity_level'
            )
            goal = st.selectbox(
                'Your Primary Goal', options=GOAL_TARGETS.keys(),
                format_func=lambda x: x.replace('_', ' ').title(),
                index=list(GOAL_TARGETS.keys()).index(DEFAULTS['goal']), key='goal'
            )

            with st.expander("Advanced Settings ⚙️"):
                protein_per_kg = st.number_input(
                    'Protein Goal (g/kg)', min_value=1.2, max_value=3.0,
                    value=DEFAULTS['protein_per_kg'], step=0.1, key='protein_per_kg',
                    help="Define your daily protein target in grams per kilogram of body weight."
                )
                fat_percentage = st.slider(
                    'Fat Intake (% of calories)', min_value=15, max_value=40,
                    value=int(DEFAULTS['fat_percentage'] * 100), step=1, key='fat_percentage_slider',
                    help="Set the percentage of daily calories from fats."
                )

            # The calculation button that triggers validation.
            submitted = st.form_submit_button('Calculate My Plan', type="primary", use_container_width=True)

        if submitted:
            # Validation logic
            if not all([age, height, weight]):
                st.error("Please fill in your Age, Height, and Weight!")
            else:
                # Store validated inputs in session state
                user_inputs = {
                    'age': age, 'sex': sex, 'activity_level': activity_level, 'goal': goal,
                    'protein_per_kg': protein_per_kg, 'fat_percentage': fat_percentage / 100.0
                }
                # Suggestion 11: Convert units if necessary
                if st.session_state.units == 'Imperial':
                    user_inputs['height_cm'] = height * 2.54
                    user_inputs['weight_kg'] = weight * 0.453592
                else:
                    user_inputs['height_cm'] = height
                    user_inputs['weight_kg'] = weight
                
                st.session_state.user_inputs = user_inputs
                st.session_state.calculated = True
                
                # Suggestion 12: Add Motivational Notification
                targets_temp = calculate_personalized_targets(user_inputs)
                change = targets_temp['estimated_weekly_change']
                goal_text = "lose" if change < 0 else "gain"
                st.success(f"Plan created! You're on track to {goal_text} approx. {abs(change):.2f} kg per week.")
                st.balloons()

        st.divider()

        # Suggestion 4: Auto-Save and Restore Selections
        st.subheader("Save & Load Progress")
        
        # Save Button
        if st.session_state.get('calculated'):
            state_to_save = {
                "user_inputs": st.session_state.user_inputs,
                "food_selections": st.session_state.food_selections
            }
            st.download_button(
                label="💾 Save My Progress",
                data=json.dumps(state_to_save, indent=4),
                file_name="my_nutrition_plan.json",
                mime="application/json",
                key='save_button',
                use_container_width=True
            )

        # Load Button
        uploaded_file = st.file_uploader(
            "📂 Load a Saved Plan", type=['json'], key='load_uploader'
        )
        if uploaded_file is not None:
            try:
                loaded_data = json.load(uploaded_file)
                st.session_state.user_inputs = loaded_data['user_inputs']
                st.session_state.food_selections = loaded_data['food_selections']
                st.session_state.calculated = True
                st.success("Plan loaded successfully!")
                st.rerun() # Rerun to reflect loaded state
            except Exception as e:
                st.error(f"Failed to load file. It might be corrupted. Error: {e}")

        st.divider()

        # Suggestion 15: Add a Dynamic Summary to the Sidebar
        if st.session_state.get('calculated'):
            st.subheader("Live Summary")
            targets = calculate_personalized_targets(st.session_state.user_inputs)
            totals, _ = calculate_daily_totals(st.session_state.food_selections, foods_df)
            
            st.metric("Calories", f"{totals['calories']:.0f} / {targets['total_calories']:.0f} kcal")
            st.metric("Protein", f"{totals['protein']:.0f} / {targets['protein_g']:.0f} g")
            st.metric("Carbs", f"{totals['carbs']:.0f} / {targets['carb_g']:.0f} g")
            st.metric("Fat", f"{totals['fat']:.0f} / {targets['fat_g']:.0f} g")


    # ------------------ MAIN CONTENT AREA ------------------
    if not st.session_state.get('calculated'):
        st.info("👈 **Welcome!** Please enter your details in the sidebar and click **'Calculate My Plan'** to get started.")
        # 
    else:
        # User has calculated their plan, show the main dashboard.
        targets = calculate_personalized_targets(st.session_state.user_inputs)
        
        # ------ Unified Target Display System ------
        goal_label = targets['goal'].replace('_', ' ').title()
        st.header(f"Your Custom Nutrition Roadmap for {goal_label} 🎯")
        st.info(
            "🎯 **The 80/20 Rule**: Aim to hit your targets about 80% of the time. Flexibility builds consistency and helps you avoid the dreaded yo-yo diet trap."
        )
        
        # Suggestion 2: Added tooltips to metrics.
        metrics_config = [
            {"title": "Your Daily Nutrition Targets", "columns": 4, "metrics": [
                ("Total Calories", f"{targets['total_calories']} kcal", f"{targets['caloric_adjustment']:+} vs TDEE", "Your total daily calorie goal."),
                ("Protein", f"{targets['protein_g']} g", f"{targets['protein_percent']:.0f}% of calories", "Target for muscle repair and growth."),
                ("Carbohydrates", f"{targets['carb_g']} g", f"{targets['carb_percent']:.0f}% of calories", "Primary energy source for your body and brain."),
                ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f}% of calories", "Essential for hormone production and health.")
            ]},
            {"title": "Metabolic Information", "columns": 4, "metrics": [
                ("Basal Metabolic Rate", f"{targets['bmr']} kcal", "", "Calories your body burns at complete rest (the 'coma diet')."),
                ("Est. Energy Needs (TDEE)", f"{targets['tdee']} kcal", "", "Total calories burned in a day, including all activities."),
                ("Est. Weekly Change", f"{targets['estimated_weekly_change']:+.2f} kg", "", "Predicted weight change per week based on your calorie goal.")
            ]}
        ]
        for config in metrics_config:
            st.subheader(config['title'])
            display_metrics_grid(config['metrics'], config['columns'])
        st.divider()

        # Suggestion 3: Collapse Long Sections by Default
        with st.expander("Your Evidence-Based Game Plan 📚", expanded=False):
            # Content from original Cell 10 tabs is placed here
            st.markdown("Here you'll find scientifically-backed advice on hydration, sleep, progress tracking, and mindset to complement your nutrition plan.")
            st.markdown("---")
            st.subheader("💧 Master Your Hydration Game")
            st.markdown("Aim for about 35 ml per kg of body weight daily, plus an extra 500-750 ml per hour of exercise. Drinking 500 ml of water before meals can also increase fullness.")
            st.subheader("😴 Sleep Like Your Goals Depend on It")
            st.markdown("Lack of sleep (<7 hours) can impair fat loss and muscle gain. Aim for 7-9 hours in a cool, dark, screen-free room.")
            st.subheader("📅 Track Your Progress Intelligently")
            st.markdown("Weigh yourself daily but focus on the weekly average. Take progress photos and body measurements monthly. Notice non-scale victories like energy levels and gym performance.")


        # ------ Daily Intake Tracking ------
        st.header("Track Your Daily Intake 🥗")
        
        # Suggestion 5: Add a Simple Search for Foods
        st.text_input("Search for a food...", key='food_search', placeholder="E.g., Tofu, Oats, Almonds...")
        
        if st.button("🔄 Reset All Food Selections", key='reset_foods', type="secondary"):
            st.session_state.food_selections = {}
            st.rerun()

        # Food selection tabs
        food_categories = foods_df['category'].unique()
        tabs = st.tabs([cat.replace("_", " ").title() for cat in food_categories])
        for i, category in enumerate(food_categories):
            with tabs[i]:
                # Filter by category and sort by emoji, then name
                items_in_category = foods_df[foods_df['category'] == category].sort_values(
                    by=['emoji_order', 'name_with_unit'], ascending=[True, True]
                )
                render_food_grid(items_in_category, category, columns=3)
        
        st.divider()
        
        # ------ Daily Summary and Progress Tracking ------
        st.header("Today’s Scorecard 📊")
        totals, selected_foods = calculate_daily_totals(st.session_state.food_selections, foods_df)

        if not selected_foods:
            st.info("Add some foods from the categories above to see your progress!")
        else:
            # Suggestion 1: Use the improved color-coded progress tracking.
            create_progress_tracking(totals, targets)
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Your Macronutrient Split")
                macro_values = [totals['protein'], totals['carbs'], totals['fat']]
                if sum(macro_values) > 0:
                    fig = go.Figure(go.Pie(
                        labels=['Protein', 'Carbs', 'Fat'],
                        values=macro_values, hole=.4,
                        marker_colors=['#ff6b6b', '#feca57', '#48dbfb'],
                        textinfo='label+percent', insidetextorientation='radial'
                    ))
                    fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=300)
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                with st.container(height=320):
                    st.subheader("Logged Foods")
                    for item in selected_foods:
                        st.markdown(f"**{item['servings']}x {item['name']}**: {item['calories']:.0f} kcal")
            
            # Suggestion 13: Export Daily Summary
            st.divider()
            st.subheader("Export Your Day")
            summary_df = pd.DataFrame(selected_foods)
            totals_df = pd.DataFrame([totals])
            totals_df.index = ['TOTAL']
            export_df = pd.concat([summary_df, totals_df.rename(columns={'name': ''})])
            
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Daily Summary (CSV)",
                data=csv,
                file_name="daily_nutrition_summary.csv",
                mime="text/csv",
                key='export_csv_button'
            )

    # ------------------ FOOTER AND FEEDBACK ------------------
    st.divider()
    # Suggestion 14: Gather User Feedback
    st.header("Help Us Improve! 💬")
    with st.form("feedback_form", clear_on_submit=True):
        feedback = st.text_area("Share your feedback or suggestions here.")
        submitted_feedback = st.form_submit_button("Submit Feedback")
        if submitted_feedback:
            if feedback: # Simple validation
                st.success("Thank you for your feedback! We appreciate you helping us make this tool better.")
                # In a real app, this would be sent to a backend/database.
            else:
                st.warning("Please enter some feedback before submitting.")


if __name__ == "__main__":
    main()
