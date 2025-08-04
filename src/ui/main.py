"""Main UI controller for the nutrition tracker application."""

import streamlit as st
import logging
from typing import Dict, Any
from ..models import UserProfile, DailyIntake, SelectedFood
from ..calculator import NutritionCalculator
from ..database import FoodDatabase
from ..session_manager import SessionManager
from ..config import NutritionConfig
from .components import UIComponents

logger = logging.getLogger(__name__)

class NutritionTrackerUI:
    """Main UI controller for the nutrition tracker application."""
    
    def __init__(self):
        """Initialize the nutrition tracker UI."""
        self.config = NutritionConfig()
        self.session_manager = SessionManager(self.config)
        self.calculator = NutritionCalculator(self.config)
        self.ui_components = UIComponents(self.config, self.session_manager)
        
        # Initialize database
        try:
            self.food_database = FoodDatabase(config=self.config)
            logger.info("Food database loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load food database: {e}")
            st.error(f"Failed to load food database: {e}")
            st.stop()
    
    def run(self) -> None:
        """Run the main application."""
        try:
            # Setup page configuration
            self.ui_components.setup_page_config()
            
            # Initialize session state
            self.session_manager.initialize_session()
            
            # Render header
            self.ui_components.render_header()
            
            # Setup sidebar and get user inputs
            user_inputs = self.ui_components.setup_sidebar()
            
            # Calculate targets
            targets = self._calculate_targets(user_inputs)
            
            # Display dashboard
            self.ui_components.display_dashboard(targets, user_inputs['user_has_entered_info'])
            
            # Create food logging interface
            foods_by_category = self.food_database.get_foods_by_category()
            self.ui_components.create_food_log_ui(foods_by_category)
            
            # Handle action buttons
            self._handle_action_buttons(targets)
            
        except Exception as e:
            logger.error(f"Application error: {e}")
            st.error(f"An error occurred: {e}")
    
    def _calculate_targets(self, user_inputs: Dict[str, Any]):
        """Calculate nutrition targets based on user inputs."""
        try:
            # Create user profile
            if user_inputs['user_has_entered_info']:
                profile = UserProfile(
                    age=user_inputs['age'],
                    height_cm=user_inputs['height_cm'],
                    weight_kg=user_inputs['weight_kg'],
                    gender=user_inputs['sex'],
                    activity_level=user_inputs['activity_level']
                )
            else:
                # Use default profile for demo
                profile = UserProfile(
                    age=self.config.DEFAULT_USER['age'],
                    height_cm=self.config.DEFAULT_USER['height_cm'],
                    weight_kg=self.config.DEFAULT_USER['weight_kg'],
                    gender=self.config.DEFAULT_USER['gender'],
                    activity_level=self.config.DEFAULT_USER['activity_level']
                )
            
            # Calculate targets
            targets = self.calculator.calculate_targets(
                profile=profile,
                caloric_surplus=user_inputs['caloric_surplus'],
                protein_per_kg=user_inputs['protein_per_kg'],
                fat_percentage=user_inputs['fat_percentage']
            )
            
            # Validate targets
            if not self.calculator.validate_targets(targets):
                st.warning("Calculated targets may be unusual. Please verify your inputs.")
            
            return targets
            
        except Exception as e:
            logger.error(f"Error calculating targets: {e}")
            st.error(f"Error calculating targets: {e}")
            # Return default targets
            return self.calculator.calculate_targets(
                UserProfile(
                    age=self.config.DEFAULT_USER['age'],
                    height_cm=self.config.DEFAULT_USER['height_cm'],
                    weight_kg=self.config.DEFAULT_USER['weight_kg'],
                    gender=self.config.DEFAULT_USER['gender'],
                    activity_level=self.config.DEFAULT_USER['activity_level']
                )
            )
    
    def _handle_action_buttons(self, targets) -> None:
        """Handle calculate and clear button actions."""
        calculate_clicked, clear_clicked = self.ui_components.render_action_buttons()
        
        if clear_clicked:
            self.session_manager.clear_food_selections()
            st.rerun()
        
        if calculate_clicked:
            self._perform_calculation(targets)
    
    def _perform_calculation(self, targets) -> None:
        """Perform daily intake calculation and display results."""
        try:
            # Get food selections
            food_selections = self.session_manager.get_food_selections()
            
            # Create daily intake object
            daily_intake = DailyIntake()
            
            # Process selected foods
            foods_by_category = self.food_database.get_foods_by_category()
            
            for category, foods in foods_by_category.items():
                for food in foods:
                    servings = food_selections.get(food.name, 0)
                    if servings > 0:
                        daily_intake.add_food(food, servings)
            
            # Store results in session
            self.session_manager.store_calculation_results(targets, daily_intake)
            
            # Display results
            self.ui_components.display_calculation_results(daily_intake, targets)
            
            logger.info(f"Calculation completed: {len(daily_intake.selected_foods)} foods selected")
            
        except Exception as e:
            logger.error(f"Error during calculation: {e}")
            st.error(f"Error during calculation: {e}")
