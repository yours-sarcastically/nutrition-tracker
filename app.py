# app.py
# Description: Main application file for the Personalized Nutrition Tracker.

import streamlit as st
from config import CONFIG
from core import calculations, data
from ui import sidebar, components

def initialize_session_state():
    """Initialize all required session state variables."""
    if 'food_selections' not in st.session_state:
        st.session_state.food_selections = {}
    
    for field in CONFIG['form_fields'].keys():
        if f'user_{field}' not in st.session_state:
            st.session_state[f'user_{field}'] = None

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Personalized Nutrition Tracker",
        page_icon="🍽️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    [data-testid="InputInstructions"] { display: none; }
    .stButton>button[kind="primary"] { background-color: #ff6b6b; color: white; border: 1px solid #ff6b6b; }
    .stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    </style>
    """, unsafe_allow_html=True)

    initialize_session_state()

    # --- Data Loading ---
    foods = data.load_food_database('data/nutrition_results.csv')
    foods = data.assign_food_emojis(foods)

    # --- Sidebar and User Input ---
    user_profile, advanced_settings, user_has_entered_info = sidebar.render_sidebar()

    # --- Core Calculation ---
    targets = calculations.calculate_personalized_targets(user_profile, advanced_settings)

    # --- Main Page UI ---
    st.title("Personalized Nutrition Tracker 🍽️")
    st.markdown("Ready to turbocharge your health game? This tool dishes out daily nutrition goals and makes tracking meals easy. Let's get those macros on your team! 🚀")

    # --- Display Nutritional Targets ---
    if not user_has_entered_info:
        st.info("👈 Please enter your personal information in the sidebar to view your personalized targets.")
        st.header("Sample Daily Targets for Reference 🎯")
    else:
        st.header("Your Personalized Daily Nutritional Targets 🎯")

    metrics_config = [
        {'title': 'Metabolic Information', 'columns': 4, 'metrics': [
            ("BMR", f"{targets.bmr} kcal"),
            ("TDEE", f"{targets.tdee} kcal"),
            ("Est. Weekly Gain", f"{targets.target_weight_gain_per_week} kg"),
            ("", ""),  # Blank placeholder for alignment
        ]},
        {'title': 'Daily Nutritional Target Breakdown', 'columns': 4, 'metrics': [
            ("Calorie Target", f"{targets.total_calories} kcal"),
            ("Protein Target", f"{targets.protein_g:.0f} g"),
            ("Carb Target", f"{targets.carb_g:.0f} g"),
            ("Fat Target", f"{targets.fat_g:.0f} g"),
        ]},
        {'title': 'Macronutrient Distribution (% of Calories)', 'columns': 4, 'metrics': [
            ("Protein", f"{targets.protein_percent:.1f}%", f"{targets.protein_calories:.0f} kcal"),
            ("Carbohydrates", f"{targets.carb_percent:.1f}%", f"{targets.carb_calories:.0f} kcal"),
            ("Fat", f"{targets.fat_percent:.1f}%", f"{targets.fat_calories:.0f} kcal"),
            ("", ""),  # Blank placeholder for alignment
        ]}
    ]
    for config in metrics_config:
        st.subheader(config['title'])
        components.display_metrics_grid(config['metrics'], config['columns'])
    st.markdown("---")

    # --- Interactive Food Selection ---
    st.header("Select Foods and Log Servings for Today 📝")
    available_categories = [cat for cat, items in foods.items() if items]
    tabs = st.tabs(available_categories)

    for i, category in enumerate(available_categories):
        with tabs[i]:
            sorted_items = sorted(foods[category], key=lambda x: (CONFIG['emoji_order'].get(x.emoji, 4), -x.calories))
            components.render_food_grid(sorted_items, category, 2)
    st.markdown("---")

    # --- Results and Analysis ---
    if st.button("Calculate Daily Intake", type="primary", use_container_width=True):
        totals, selected_foods = calculations.calculate_daily_totals(st.session_state.food_selections, foods)
        components.display_results_summary(totals, selected_foods, targets)

    # --- Clear Selections ---
    if st.button("Clear All Selections", use_container_width=True):
        st.session_state.food_selections.clear()
        st.rerun()

if __name__ == "__main__":
    main()
