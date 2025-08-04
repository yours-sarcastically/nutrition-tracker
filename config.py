"""
Configuration management for the Personalized Evidence-Based Nutrition Tracker.

This module centralizes all configuration parameters, constants, and settings
used throughout the application. It provides a clean separation between
configuration and business logic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
from enum import Enum

class ActivityLevel(Enum):
    """Enumeration for activity levels with their multipliers."""
    SEDENTARY = ("sedentary", 1.2, "Little to no exercise")
    LIGHTLY_ACTIVE = ("lightly_active", 1.375, "Light exercise 1-3 days/week")
    MODERATELY_ACTIVE = ("moderately_active", 1.55, "Moderate exercise 3-5 days/week")
    VERY_ACTIVE = ("very_active", 1.725, "Hard exercise 6-7 days/week")
    EXTREMELY_ACTIVE = ("extremely_active", 1.9, "Very hard exercise/physical job")
    
    def __init__(self, key: str, multiplier: float, description: str):
        self.key = key
        self.multiplier = multiplier
        self.description = description

class Gender(Enum):
    """Enumeration for gender options."""
    MALE = ("Male", "male")
    FEMALE = ("Female", "female")
    
    def __init__(self, display_name: str, key: str):
        self.display_name = display_name
        self.key = key

@dataclass
class UserDefaults:
    """Default user attributes for the application."""
    age: int = 26
    height_cm: int = 180
    weight_kg: float = 57.5
    gender: str = "Male"
    activity_level: str = "moderately_active"

@dataclass
class NutritionalConstants:
    """Constants used in nutritional calculations."""
    # Caloric values per gram
    CALORIES_PER_GRAM_PROTEIN: int = 4
    CALORIES_PER_GRAM_CARB: int = 4
    CALORIES_PER_GRAM_FAT: int = 9
    
    # Default nutritional targets
    caloric_surplus: int = 400
    protein_per_kg: float = 2.0
    fat_percentage: float = 0.25
    target_weekly_gain_rate: float = 0.0025  # 0.25% of body weight per week
    
    # Validation ranges
    min_age: int = 16
    max_age: int = 80
    min_height_cm: int = 140
    max_height_cm: int = 220
    min_weight_kg: float = 40.0
    max_weight_kg: float = 150.0
    
    # Advanced settings ranges
    min_caloric_surplus: int = 200
    max_caloric_surplus: int = 800
    caloric_surplus_step: int = 50
    
    min_protein_per_kg: float = 1.2
    max_protein_per_kg: float = 3.0
    protein_per_kg_step: float = 0.1
    
    min_fat_percentage: int = 15
    max_fat_percentage: int = 40
    fat_percentage_step: int = 1

@dataclass
class UIConfiguration:
    """UI-related configuration settings."""
    # Page configuration
    page_title: str = "Personalized Nutrition Tracker"
    page_icon: str = "🍽️"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    
    # Food display settings
    foods_per_row: int = 2
    max_custom_servings: float = 10.0
    custom_servings_step: float = 0.1
    
    # Quick selection buttons
    quick_serving_options: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    
    # Cache settings
    food_data_cache_ttl: int = 3600  # 1 hour in seconds

@dataclass
class FoodConfiguration:
    """Configuration for food database and categorization."""
    # Database file path
    database_file: str = "nutrition_database_final.csv"
    
    # Food ranking configuration
    top_foods_count: int = 3  # Number of top foods to highlight per category
    
    # Category mappings for nutrient analysis
    nutrient_category_mapping: Dict[str, str] = field(default_factory=lambda: {
        'PRIMARY PROTEIN SOURCES': 'protein',
        'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
        'PRIMARY FAT SOURCES': 'fat',
        'PRIMARY MICRONUTRIENT SOURCES': 'protein'  # Treated as protein for ranking
    })
    
    # Emoji configuration for food ranking
    emoji_config: Dict[str, Dict] = field(default_factory=lambda: {
        'superfood': {'emoji': '🥇', 'description': 'High in multiple nutrients', 'priority': 0},
        'nutrient_calorie_dense': {'emoji': '💥', 'description': 'High in both nutrients and calories', 'priority': 1},
        'high_calorie': {'emoji': '🔥', 'description': 'Energy-dense', 'priority': 2},
        'top_protein': {'emoji': '💪', 'description': 'Top protein source', 'priority': 3},
        'top_carb': {'emoji': '🍚', 'description': 'Top carbohydrate source', 'priority': 3},
        'top_fat': {'emoji': '🥑', 'description': 'Top fat source', 'priority': 3},
        'top_micronutrient': {'emoji': '🥦', 'description': 'Top micronutrient source', 'priority': 3},
        'default': {'emoji': '', 'description': 'Standard food item', 'priority': 4}
    })

@dataclass
class ApplicationConfig:
    """Main configuration class that combines all configuration sections."""
    user_defaults: UserDefaults = field(default_factory=UserDefaults)
    nutritional_constants: NutritionalConstants = field(default_factory=NutritionalConstants)
    ui_config: UIConfiguration = field(default_factory=UIConfiguration)
    food_config: FoodConfiguration = field(default_factory=FoodConfiguration)
    
    # Environment-specific settings
    debug_mode: bool = field(default_factory=lambda: os.getenv('DEBUG_MODE', 'False').lower() == 'true')
    log_level: str = field(default_factory=lambda: os.getenv('LOG_LEVEL', 'INFO'))
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Override database file path from environment if available
        env_db_path = os.getenv('FOOD_DATABASE_PATH')
        if env_db_path:
            self.food_config.database_file = env_db_path
    
    @property
    def activity_multipliers(self) -> Dict[str, float]:
        """Get activity level multipliers as a dictionary."""
        return {level.key: level.multiplier for level in ActivityLevel}
    
    @property
    def activity_options_for_ui(self) -> Dict[str, Optional[str]]:
        """Get activity options formatted for UI selectbox."""
        options = {"Select Activity Level": None}
        for level in ActivityLevel:
            display_name = level.name.replace('_', ' ').title()
            options[display_name] = level.key
        return options
    
    @property
    def gender_options_for_ui(self) -> List[str]:
        """Get gender options formatted for UI."""
        return [gender.display_name for gender in Gender]
    
    def get_activity_description(self, activity_key: str) -> str:
        """Get description for an activity level."""
        for level in ActivityLevel:
            if level.key == activity_key:
                return level.description
        return "Unknown activity level"
    
    def get_emoji_info(self, emoji_type: str) -> Dict:
        """Get emoji configuration for a given type."""
        return self.food_config.emoji_config.get(emoji_type, self.food_config.emoji_config['default'])
    
    def validate_user_input(self, age: int, height_cm: int, weight_kg: float, 
                           gender: str, activity_level: str) -> List[str]:
        """Validate user inputs and return list of error messages."""
        errors = []
        
        if not (self.nutritional_constants.min_age <= age <= self.nutritional_constants.max_age):
            errors.append(f"Age must be between {self.nutritional_constants.min_age} and {self.nutritional_constants.max_age} years")
        
        if not (self.nutritional_constants.min_height_cm <= height_cm <= self.nutritional_constants.max_height_cm):
            errors.append(f"Height must be between {self.nutritional_constants.min_height_cm} and {self.nutritional_constants.max_height_cm} cm")
        
        if not (self.nutritional_constants.min_weight_kg <= weight_kg <= self.nutritional_constants.max_weight_kg):
            errors.append(f"Weight must be between {self.nutritional_constants.min_weight_kg} and {self.nutritional_constants.max_weight_kg} kg")
        
        valid_genders = [g.display_name for g in Gender]
        if gender not in valid_genders:
            errors.append(f"Please select a valid gender: {', '.join(valid_genders)}")
        
        valid_activities = [level.key for level in ActivityLevel]
        if activity_level not in valid_activities:
            errors.append("Please select a valid activity level")
        
        return errors

# Create global configuration instance
config = ApplicationConfig()

# Convenience functions for backward compatibility
def get_default_age() -> int:
    return config.user_defaults.age

def get_default_height() -> int:
    return config.user_defaults.height_cm

def get_default_weight() -> float:
    return config.user_defaults.weight_kg

def get_default_gender() -> str:
    return config.user_defaults.gender

def get_default_activity_level() -> str:
    return config.user_defaults.activity_level

def get_activity_multiplier(activity_level: str) -> float:
    return config.activity_multipliers.get(activity_level, ActivityLevel.MODERATELY_ACTIVE.multiplier)

def get_caloric_surplus() -> int:
    return config.nutritional_constants.caloric_surplus

def get_protein_per_kg() -> float:
    return config.nutritional_constants.protein_per_kg

def get_fat_percentage() -> float:
    return config.nutritional_constants.fat_percentage

def get_target_weekly_gain_rate() -> float:
    return config.nutritional_constants.target_weekly_gain_rate

def get_database_file_path() -> str:
    return config.food_config.database_file

# Export commonly used constants
DEFAULT_AGE = config.user_defaults.age
DEFAULT_HEIGHT_CM = config.user_defaults.height_cm
DEFAULT_WEIGHT_KG = config.user_defaults.weight_kg
DEFAULT_GENDER = config.user_defaults.gender
DEFAULT_ACTIVITY_LEVEL = config.user_defaults.activity_level
DEFAULT_CALORIC_SURPLUS = config.nutritional_constants.caloric_surplus
DEFAULT_PROTEIN_PER_KG = config.nutritional_constants.protein_per_kg
DEFAULT_FAT_PERCENTAGE = config.nutritional_constants.fat_percentage
TARGET_WEEKLY_GAIN_RATE = config.nutritional_constants.target_weekly_gain_rate
GENDER_OPTIONS = config.gender_options_for_ui
ACTIVITY_OPTIONS = config.activity_options_for_ui
