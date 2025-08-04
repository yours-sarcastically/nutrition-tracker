"""
UI components for the Personalized Evidence-Based Nutrition Tracker.
Contains all Streamlit interface components as reusable functions.
"""

import streamlit as st
from typing import Dict, List, Tuple
from src.data_models import UserProfile, NutritionTargets, FoodItem, UserInputs
from src.config import config

class UIComponents:
    """Handles all UI component rendering."""
    
    @staticmethod
    def setup_page_config():
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="Personalized Nutrition Tracker", 
            page_icon="🍽️", 
            layout="wide", 
            initial_sidebar_state="expanded"
        )
        st.markdown("""<style>[data-testid="InputInstructions"] {display: none;}</style>""", unsafe_allow_html=True)
    
    @staticmethod
    def render_header():
        """Render the main application header."""
        st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
        st.markdown("Ready to turbocharge your health game? This awesome tool dishes out daily nutrition goals made just for you and makes tracking meals as easy as pie. Let's get those macros on your team! 🚀")
    
    @staticmethod
    def render_user_input_sidebar() -> UserInputs:
        """
        Render the sidebar for user input and return collected data.
        
        Returns:
            UserInputs object containing all user input values
        """
        st.sidebar.header("Personal Parameters for Daily Target Calculation 📊")
        
        # Personal Information Section
        age = st.sidebar.number_input(
            "Age (Years)", 
            16, 80, 
            st.session_state.get('user_age'), 
            placeholder="Enter your age"
        )
        
        height_cm = st.sidebar.number_input(
            "Height (Centimeters)", 
            140, 220, 
            st.session_state.get('user_height'), 
            placeholder="Enter your height"
        )
        
        weight_kg = st.sidebar.number_input(
            "Weight (kg)", 
            40.0, 150.0, 
            st.session_state.get('user_weight'), 
            0.5, 
            placeholder="Enter your weight"
        )
        
        # Sex Selection
        sex_options = ["Select Sex"] + config.GENDER_OPTIONS
        current_sex = st.session_state.get('user_sex')
        sex_index = sex_options.index(current_sex) if current_sex in sex_options else 0
        sex = st.sidebar.selectbox("Sex", sex_options, index=sex_index)
        
        # Activity Level Selection
        activity_labels = list(config.ACTIVITY_OPTIONS.keys())
        activity_values = list(config.ACTIVITY_OPTIONS.values())
        current_activity = st.session_state.get('user_activity')
        activity_index = activity_values.index(current_activity) if current_activity in activity_values else 0
        activity_selection = st.sidebar.selectbox("Activity Level", activity_labels, index=activity_index)
        activity_level = config.ACTIVITY_OPTIONS[activity_selection]
        
        # Update session state
        st.session_state.update({
            'user_age': age,
            'user_height': height_cm,
            'user_weight': weight_kg,
            'user_sex': sex,
            'user_activity': activity_level
        })
        
        # Advanced Settings
        with st.sidebar.expander("Advanced Settings ⚙️"):
            caloric_surplus = st.number_input(
                "Caloric Surplus (kcal)", 
                200, 800, 
                None, 50, 
                placeholder=f"Default: {config.DEFAULT_CALORIC_SURPLUS}",
                help="Additional calories above maintenance for weight gain."
            )
            
            protein_per_kg = st.number_input(
                "Protein (g/kg)", 
                1.2, 3.0, 
                None, 0.1, 
                placeholder=f"Default: {config.DEFAULT_PROTEIN_PER_KG}",
                help="Protein intake per kilogram of body weight."
            )
            
            fat_percentage_input = st.number_input(
                "Fat (% of Calories)", 
                15, 40, 
                None, 1, 
                placeholder=f"Default: {int(config.DEFAULT_FAT_PERCENTAGE * 100)}",
                help="Percentage of total calories from fat."
            )
        
        # Determine if user has entered complete information
        user_has_entered_info = all([
            age, height_cm, weight_kg, 
            sex != "Select Sex", 
            activity_level
        ])
        
        return UserInputs(
            age=age or config.DEFAULT_AGE,
            height_cm=height_cm or config.DEFAULT_HEIGHT_CM,
            weight_kg=weight_kg or config.DEFAULT_WEIGHT_KG,
            sex=sex if sex != "Select Sex" else config.DEFAULT_GENDER,
            activity_level=activity_level or config.DEFAULT_ACTIVITY_LEVEL,
            caloric_surplus=caloric_surplus or config.DEFAULT_CALORIC_SURPLUS,
            protein_per_kg=protein_per_kg or config.DEFAULT_PROTEIN_PER_KG,
            fat_percentage=(fat_percentage_input / 100) if fat_percentage_input else config.DEFAULT_FAT_PERCENTAGE,
            user_has_entered_info=user_has_entered_info
        )
    
    @staticmethod
    def render_nutrition_dashboard(targets: NutritionTargets, user_has_entered_info: bool):
        """
        Render the nutrition targets dashboard.
        
        Args:
            targets: Calculated nutritional targets
            user_has_entered_info: Whether user has provided complete information
        """
        if not user_has_entered_info:
            st.info("👈 Please enter your personal information in the sidebar to view your daily nutritional targets.")
            st.header("Sample Daily Targets for Reference 🎯")
            st.caption("These are example targets. Enter your information in the sidebar for personalized calculations.")
        else:
            st.header("Your Personalized Daily Nutritional Targets for Healthy Weight Gain 🎯")
        
        # Metabolic Information
        col1, col2, col3, _ = st.columns(4)
        col1.metric("Basal Metabolic Rate (BMR)", f"{targets.bmr} kcal/day")
        col2.metric("Total Daily Energy Expenditure (TDEE)", f"{targets.tdee} kcal/day")
        col3.metric("Est. Weekly Weight Gain", f"{targets.target_weight_gain_per_week} kg/week")
        
        # Daily Targets
        st.subheader("Daily Nutritional Target Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Daily Calorie Target", f"{targets.total_calories} kcal")
        col2.metric("Protein Target", f"{targets.protein_g} g")
        col3.metric("Carbohydrate Target", f"{targets.carb_g} g")
        col4.metric("Fat Target", f"{targets.fat_g} g")
        
        # Macronutrient Distribution
        st.subheader("Macronutrient Distribution as a Percent of Daily Calories")
        col1, col2, col3, _ = st.columns(4)
        col1.metric("Protein", f"{targets.get_protein_percentage():.1f}%", f"{targets.protein_calories} kcal")
        col2.metric("Carbohydrates", f"{targets.get_carb_percentage():.1f}%", f"{targets.carb_calories} kcal")
        col3.metric("Fat", f"{targets.get_fat_percentage():.1f}%", f"{targets.fat_calories} kcal")
        st.markdown("---")
    
    @staticmethod
    def render_food_selection_tabs(foods: Dict[str, List[FoodItem]]):
        """
        Render the food selection interface with tabs.
        
        Args:
            foods: Dictionary mapping categories to food items
        """
        st.header("Select Foods and Log Servings for Today 📝")
        st.markdown("Choose foods using the buttons for preset servings or enter a custom serving amount for each item.")
        
        available_categories = sorted([cat for cat, items in foods.items() if items])
        tabs = st.tabs(available_categories)
        
        for i, category in enumerate(available_categories):
            with tabs[i]:
                UIComponents._render_food_category(category, foods[category])
        
        st.markdown("---")
    
    @staticmethod
    def _render_food_category(category: str, food_items: List[FoodItem]):
        """
        Render food items for a specific category.
        
        Args:
            category: Category name
            food_items: List of food items in the category
        """
        # Sort items by emoji ranking and calories
        sorted_items = sorted(
            food_items, 
            key=lambda x: (config.EMOJI_ORDER.get(x.emoji, 4), -x.calories)
        )
        
        # Display items in two-column layout
        for j in range(0, len(sorted_items), 2):
            col1, col2 = st.columns(2)
            if j < len(sorted_items):
                UIComponents._render_food_item(sorted_items[j], category, col1)
            if j + 1 < len(sorted_items):
                UIComponents._render_food_item(sorted_items[j + 1], category, col2)
    
    @staticmethod
    def _render_food_item(food: FoodItem, category: str, col):
        """
        Render a single food item selection interface.
        
        Args:
            food: FoodItem to render
            category: Category the food belongs to
            col: Streamlit column to render in
        """
        with col:
            display_name = food.get_display_name()
            st.subheader(f"{food.emoji} {display_name}")
            key_prefix = f"{category}_{display_name}"
            current_serving = st.session_state.food_selections.get(display_name, 0.0)
            
            # Quick selection buttons
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k-1]:
                    button_type = "primary" if current_serving == float(k) else "secondary"
                    if st.button(f"{k}", key=f"{key_prefix}_{k}", type=button_type, use_container_width=True):
                        st.session_state.food_selections[display_name] = float(k)
                        st.rerun()
            
            # Custom serving input
            custom_serving = st.number_input(
                "Custom Servings:", 
                0.0, 10.0, 
                float(current_serving), 
                0.1, 
                key=f"{key_prefix}_custom"
            )
            
            if custom_serving != current_serving:
                if custom_serving > 0:
                    st.session_state.food_selections[display_name] = custom_serving
                elif display_name in st.session_state.food_selections:
                    del st.session_state.food_selections[display_name]
                st.rerun()
            
            # Nutritional information
            st.caption(f"Per Serving: {food.calories} kcal | {food.protein}g protein | {food.carbs}g carbs | {food.fat}g fat")
    
    @staticmethod
    def render_calculation_results(foods: Dict[str, List[FoodItem]], targets: NutritionTargets):
        """
        Calculate and display daily intake results.
        
        Args:
            foods: Dictionary of all food items
            targets: User's nutritional targets
        """
        # Calculate totals from selections
        total_calories, total_protein, total_carbs, total_fat = 0, 0, 0, 0
        selected_foods = []
        
        for category, items in foods.items():
            for food in items:
                display_name = food.get_display_name()
                servings = st.session_state.food_selections.get(display_name, 0)
                if servings > 0:
                    nutrition = food.calculate_nutrition_for_servings(servings)
                    total_calories += nutrition['calories']
                    total_protein += nutrition['protein']
