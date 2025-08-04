# main.py (or your main application file)
"""
Personalized Evidence-Based Nutrition Tracker

This script implements an interactive nutrition tracking application for healthy weight gain
using vegetarian food sources. It calculates personalized daily targets for calories,
protein, fat, and carbohydrates based on user-specific attributes and activity levels.
"""

import streamlit as st
import pandas as pd
import math
from config import config, ActivityLevel, Gender

# -----------------------------------------------------------------------------
# Core Calculation Functions (Updated to use config)
# -----------------------------------------------------------------------------

def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    """Calculates Basal Metabolic Rate using the Mifflin-St Jeor Equation."""
    if sex.lower() == 'male':
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    else:
        bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
    return bmr

def calculate_tdee(bmr, activity_level):
    """Calculates Total Daily Energy Expenditure based on Activity Level."""
    multiplier = config.activity_multipliers.get(activity_level, ActivityLevel.MODERATELY_ACTIVE.multiplier)
    return bmr * multiplier

def calculate_personalized_targets(age, height_cm, weight_kg, sex, activity_level,
                                   caloric_surplus=None, protein_per_kg=None, fat_percentage=None):
    """Calculates Personalized Daily Nutritional Targets."""
    # Use config defaults if not provided
    caloric_surplus = caloric_surplus or config.nutritional_constants.caloric_surplus
    protein_per_kg = protein_per_kg or config.nutritional_constants.protein_per_kg
    fat_percentage = fat_percentage or config.nutritional_constants.fat_percentage
    
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)
    total_calories = tdee + caloric_surplus
    
    protein_g = protein_per_kg * weight_kg
    protein_calories = protein_g * config.nutritional_constants.CALORIES_PER_GRAM_PROTEIN
    
    fat_calories = total_calories * fat_percentage
    fat_g = fat_calories / config.nutritional_constants.CALORIES_PER_GRAM_FAT
    
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / config.nutritional_constants.CALORIES_PER_GRAM_CARB

    return {
        'bmr': round(bmr), 'tdee': round(tdee), 'total_calories': round(total_calories),
        'protein_g': round(protein_g), 'protein_calories': round(protein_calories),
        'fat_g': round(fat_g), 'fat_calories': round(fat_calories),
        'carb_g': round(carb_g), 'carb_calories': round(carb_calories),
        'target_weight_gain_per_week': round(weight_kg * config.nutritional_constants.target_weekly_gain_rate, 2)
    }

@st.cache_data(ttl=config.ui_config.food_data_cache_ttl)
def load_and_process_foods(file_path):
    """Loads and processes the food database, assigning categories and emojis."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Food database file '{file_path}' not found!")
        return {}
    except Exception as e:
        st.error(f"Error loading food database: {str(e)}")
        return {}
    
    categories = df['category'].unique()
    foods = {cat: [] for cat in categories}

    for _, row in df.iterrows():
        foods[row['category']].append({
            'name': f"{row['name']} ({row['serving_unit']})", 
            'calories': row['calories'],
            'protein': row['protein'], 
            'carbs': row['carbs'], 
            'fat': row['fat']
        })

    # Assign emojis based on nutritional hierarchy using config
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}
    
    for category, items in foods.items():
        if not items: 
            continue
        
        sorted_by_calories = sorted(items, key=lambda x: x['calories'], reverse=True)
        top_foods['calories'][category] = [food['name'] for food in sorted_by_calories[:config.food_config.top_foods_count]]
        
        nutrient = config.food_config.nutrient_category_mapping.get(category)
        if nutrient:
            sorted_by_nutrient = sorted(items, key=lambda x: x[nutrient], reverse=True)
            top_foods[nutrient] = [food['name'] for food in sorted_by_nutrient[:config.food_config.top_foods_count]]

    all_top_nutrient_foods = set(top_foods['protein'] + top_foods['carbs'] + top_foods['fat'] + top_foods['micro'])
    food_rank_counts = {
        food_name: sum(1 for nutrient_list in ['protein', 'carbs', 'fat', 'micro'] 
                      if food_name in top_foods[nutrient_list]) 
        for food_name in all_top_nutrient_foods
    }
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    # Assign emojis using config
    for category, items in foods.items():
        for food in items:
            food_name = food['name']
            is_top_nutrient = food_name in all_top_nutrient_foods
            is_high_calorie = food_name in top_foods['calories'].get(category, [])
            
            if food_name in superfoods: 
                food['emoji'] = config.get_emoji_info('superfood')['emoji']
            elif is_high_calorie and is_top_nutrient: 
                food['emoji'] = config.get_emoji_info('nutrient_calorie_dense')['emoji']
            elif is_high_calorie: 
                food['emoji'] = config.get_emoji_info('high_calorie')['emoji']
            elif food_name in top_foods['protein']: 
                food['emoji'] = config.get_emoji_info('top_protein')['emoji']
            elif food_name in top_foods['carbs']: 
                food['emoji'] = config.get_emoji_info('top_carb')['emoji']
            elif food_name in top_foods['fat']: 
                food['emoji'] = config.get_emoji_info('top_fat')['emoji']
            elif food_name in top_foods['micro']: 
                food['emoji'] = config.get_emoji_info('top_micronutrient')['emoji']
            else: 
                food['emoji'] = config.get_emoji_info('default')['emoji']
    
    return foods

# -----------------------------------------------------------------------------
# UI Helper Functions (Updated to use config)
# -----------------------------------------------------------------------------

def setup_sidebar():
    """Handles all user input widgets in the sidebar and returns final values."""
    st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")

    # Personal Info using config defaults
    age = st.sidebar.number_input(
        "Age (Years)", 
        config.nutritional_constants.min_age, 
        config.nutritional_constants.max_age, 
        st.session_state.user_age or config.user_defaults.age, 
        placeholder="Enter your age"
    )
    
    height_cm = st.sidebar.number_input(
        "Height (Centimeters)", 
        config.nutritional_constants.min_height_cm, 
        config.nutritional_constants.max_height_cm, 
        st.session_state.user_height or config.user_defaults.height_cm, 
        placeholder="Enter your height"
    )
    
    weight_kg = st.sidebar.number_input(
        "Weight (kg)", 
        config.nutritional_constants.min_weight_kg, 
        config.nutritional_constants.max_weight_kg, 
        st.session_state.user_weight or config.user_defaults.weight_kg, 
        0.5, 
        placeholder="Enter your weight"
    )

    sex_options = ["Select Sex"] + config.gender_options_for_ui
    sex_index = sex_options.index(st.session_state.user_sex) if st.session_state.user_sex in sex_options else 0
    sex = st.sidebar.selectbox("Sex", sex_options, index=sex_index)

    activity_labels = list(config.activity_options_for_ui.keys())
    activity_values = list(config.activity_options_for_ui.values())
    activity_index = activity_values.index(st.session_state.user_activity) if st.session_state.user_activity in activity_values else 0
    activity_selection = st.sidebar.selectbox("Activity Level", activity_labels, index=activity_index)
    activity_level = config.activity_options_for_ui[activity_selection]

    # Update Session State
    st.session_state.update(
        user_age=age, 
        user_height=height_cm, 
        user_weight=weight_kg, 
        user_sex=sex, 
        user_activity=activity_level
    )

    # Advanced Settings using config
    with st.sidebar.expander("Advanced Settings ⚙️"):
        caloric_surplus = st.number_input(
            "Caloric Surplus (kcal)", 
            config.nutritional_constants.min_caloric_surplus, 
            config.nutritional_constants.max_caloric_surplus, 
            None, 
            config.nutritional_constants.caloric_surplus_step, 
            placeholder=f"Default: {config.nutritional_constants.caloric_surplus}", 
            help="Additional calories above maintenance for weight gain."
        )
        
        protein_per_kg = st.number_input(
            "Protein (g/kg)", 
            config.nutritional_constants.min_protein_per_kg, 
            config.nutritional_constants.max_protein_per_kg, 
            None, 
            config.nutritional_constants.protein_per_kg_step, 
            placeholder=f"Default: {config.nutritional_constants.protein_per_kg}", 
            help="Protein intake per kilogram of body weight."
        )
        
        fat_percentage_input = st.number_input(
            "Fat (% of Calories)", 
            config.nutritional_constants.min_fat_percentage, 
            config.nutritional_constants.max_fat_percentage, 
            None, 
            config.nutritional_constants.fat_percentage_step, 
            placeholder=f"Default: {int(config.nutritional_constants.fat_percentage * 100)}", 
            help="Percentage of total calories from fat."
        )

    user_has_entered_info = all([
        age, height_cm, weight_kg, 
        sex != "Select Sex", 
        activity_level
    ])

    return {
        "age": age or config.user_defaults.age, 
        "height_cm": height_cm or config.user_defaults.height_cm, 
        "weight_kg": weight_kg or config.user_defaults.weight_kg,
        "sex": sex if sex != "Select Sex" else config.user_defaults.gender, 
        "activity_level": activity_level or config.user_defaults.activity_level,
        "caloric_surplus": caloric_surplus or config.nutritional_constants.caloric_surplus, 
        "protein_per_kg": protein_per_kg or config.nutritional_constants.protein_per_kg,
        "fat_percentage": (fat_percentage_input / 100) if fat_percentage_input else config.nutritional_constants.fat_percentage,
        "user_has_entered_info": user_has_entered_info
    }

# Rest of your functions remain the same but can now use config values...

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title=config.ui_config.page_title, 
        page_icon=config.ui_config.page_icon, 
        layout=config.ui_config.layout, 
        initial_sidebar_state=config.ui_config.initial_sidebar_state
    )
    
    # Initialize Session State
    for key in ['user_age', 'user_height', 'user_weight', 'user_sex', 'user_activity']:
        if key not in st.session_state: 
            st.session_state[key] = None
    if 'food_selections' not in st.session_state: 
        st.session_state.food_selections = {}

    st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
    st.markdown("Ready to turbocharge your health game? This awesome tool dishes out daily nutrition goals made just for you and makes tracking meals as easy as pie. Let's get those macros on your team! 🚀")

    # Load Data using config
    foods = load_and_process_foods(config.food_config.database_file)

    # Rest of your main function...
    user_inputs = setup_sidebar()
    targets = calculate_personalized_targets(**{k: v for k, v in user_inputs.items() if k != 'user_has_entered_info'})

    # Continue with the rest of your application logic...

if __name__ == "__main__":
    main()
