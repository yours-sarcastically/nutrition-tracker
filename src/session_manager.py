"""Streamlit session state management."""

import streamlit as st
import logging
from typing import Dict, Any, Optional
from .models import UserProfile, DailyIntake
from .config import NutritionConfig

logger = logging.getLogger(__name__)

class SessionManager:
    """Manages Streamlit session state for the nutrition tracker."""
    
    def __init__(self, config: NutritionConfig = None):
        """Initialize session manager with configuration."""
        self.config = config or NutritionConfig()
    
    def initialize_session(self) -> None:
        """Initialize session state with default values."""
        logger.debug("Initializing session state")
        
        defaults = {
            # User profile data
            'user_age': None,
            'user_height': None,
            'user_weight': None,
            'user_sex': None,
            'user_activity': None,
            
            # Food selections
            'food_selections': {},
            'daily_intake': DailyIntake(),
            
            # Calculation results
            'last_targets': None,
            'last_calculation_time': None,
            
            # UI state
            'show_advanced_settings': False,
            'selected_tab': 0
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    def get_user_profile(self) -> Optional[UserProfile]:
        """
        Get user profile from session state.
        
        Returns:
            UserProfile if all required data is present, None otherwise
        """
        try:
            if all([
                st.session_state.get('user_age'),
                st.session_state.get('user_height'),
                st.session_state.get('user_weight'),
                st.session_state.get('user_sex'),
                st.session_state.get('user_activity')
            ]):
                return UserProfile(
                    age=st.session_state.user_age,
                    height_cm=st.session_state.user_height,
                    weight_kg=st.session_state.user_weight,
                    gender=st.session_state.user_sex,
                    activity_level=st.session_state.user_activity
                )
        except Exception as e:
            logger.warning(f"Error creating user profile: {e}")
        
        return None
    
    def update_user_data(self, **kwargs) -> None:
        """
        Update user data in session state.
        
        Args:
            **kwargs: User data fields to update
        """
        for key, value in kwargs.items():
            if f'user_{key}' in st.session_state:
                st.session_state[f'user_{key}'] = value
    
    def get_food_selections(self) -> Dict[str, float]:
        """Get current food selections from session state."""
        return st.session_state.get('food_selections', {})
    
    def update_food_selection(self, food_name: str, servings: float) -> None:
        """
        Update food selection in session state.
        
        Args:
            food_name: Name of the food item
            servings: Number of servings selected
        """
        if 'food_selections' not in st.session_state:
            st.session_state.food_selections = {}
        
        if servings > 0:
            st.session_state.food_selections[food_name] = servings
        elif food_name in st.session_state.food_selections:
            del st.session_state.food_selections[food_name]
    
    def clear_food_selections(self) -> None:
        """Clear all food selections."""
        st.session_state.food_selections = {}
        if 'daily_intake' in st.session_state:
            st.session_state.daily_intake.clear()
    
    def has_user_entered_info(self) -> bool:
        """Check if user has entered all required information."""
        profile = self.get_user_profile()
        return profile is not None
    
    def get_user_input_defaults(self) -> Dict[str, Any]:
        """Get default values for user input fields."""
        return {
            'age': st.session_state.get('user_age') or self.config.DEFAULT_USER['age'],
            'height': st.session_state.get('user_height') or self.config.DEFAULT_USER['height_cm'],
            'weight': st.session_state.get('user_weight') or self.config.DEFAULT_USER['weight_kg'],
            'sex': st.session_state.get('user_sex') or self.config.DEFAULT_USER['gender'],
            'activity': st.session_state.get('user_activity') or self.config.DEFAULT_USER['activity_level']
        }
    
    def store_calculation_results(self, targets, daily_intake) -> None:
        """
        Store calculation results in session state.
        
        Args:
            targets: Calculated nutrition targets
            daily_intake: Daily intake summary
        """
        import datetime
        
        st.session_state.last_targets = targets
        st.session_state.daily_intake = daily_intake
        st.session_state.last_calculation_time = datetime.datetime.now()
    
    def reset_session(self) -> None:
        """Reset all session state data."""
        logger.info("Resetting session state")
        for key in list(st.session_state.keys()):
            if key.startswith(('user_', 'food_', 'daily_', 'last_')):
                del st.session_state[key]
        self.initialize_session()      
