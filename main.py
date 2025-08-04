"""
Main application file for the Personalized Evidence-Based Nutrition Tracker.
Orchestrates all components and manages the application flow.
"""

import streamlit as st
from src.data_models import UserProfile
from src.nutrition_calculator import NutritionCalculator
from src.food_database import FoodDatabase
from src.ui_components import UIComponents
from src.config import config

class NutritionTrackerApp:
    """Main application class that orchestrates all components."""
    
    def __init__(self):
        """Initialize the application with required components."""
        self.ui = UIComponents()
        self.calculator = NutritionCalculator()
        self.food_db = FoodDatabase()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state with default values."""
        default_values = {
            'user_age': config.DEFAULT_AGE,
            'user_height': config.DEFAULT_HEIGHT_CM,
            'user_weight': config.DEFAULT_WEIGHT_KG,
            'user_sex': config.DEFAULT_GENDER,
            'user_activity': config.DEFAULT_ACTIVITY_LEVEL,
            'food_selections': {}
        }
        
        for key, value in default_values.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def run(self):
        """Run the main application."""
        # Configure page
        self.ui.setup_page_config()
        
        # Render main header
        self.ui.render_header()
        
        # Get user inputs from sidebar
        user_inputs = self.ui.render_user_input_sidebar()
        
        # Render sidebar guides
        self.ui.render_activity_guide()
        self.ui.render_emoji_legend()
        
        # Create user profile
        try:
            user_profile = UserProfile(
                age=user_inputs.age,
                height_cm=user_inputs.height_cm,
                weight_kg=user_inputs.weight_kg,
                sex=user_inputs.sex,
                activity_level=user_inputs.activity_level,
                caloric_surplus=user_inputs.caloric_surplus,
                protein_per_kg=user_inputs.protein_per_kg,
                fat_percentage=user_inputs.fat_percentage
            )
        except ValueError as e:
            st.error(f"Invalid input: {str(e)}")
            return
        
        # Calculate nutritional targets
        targets = self.calculator.calculate_targets(user_profile)
        
        # Render nutrition dashboard
        self.ui.render_nutrition_dashboard(targets, user_inputs.user_has_entered_info)
        
        # Load and display food selection interface
        foods = self.food_db.load_foods()
        if foods:
            self.ui.render_food_selection_tabs(foods)
            
            # Calculate and display results
            self.ui.render_calculation_results(foods, targets)
        else:
            st.error("Unable to load food database. Please check the file path and try again.")
        
        # Render footer
        self.ui.render_footer()

def main():
    """Main entry point for the application."""
    app = NutritionTrackerApp()
    app.run()

if __name__ == "__main__":
    main()
