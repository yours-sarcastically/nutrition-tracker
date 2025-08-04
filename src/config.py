"""Configuration settings for the nutrition tracker application."""

import os
from pathlib import Path
from typing import Dict, Any

class NutritionConfig:
    """Central configuration for all nutrition calculations and UI settings."""
    
    # File paths
    DATA_DIR = Path("data")
    NUTRITION_DB_FILE = DATA_DIR / "nutrition_database_final.csv"
    
    # Environment settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    # Default user attributes
    DEFAULT_USER = {
        'age': 26,
        'height_cm': 180,
        'weight_kg': 57.5,
        'gender': 'Male',
        'activity_level': 'moderately_active'
    }
    
    # Nutritional calculation constants
    NUTRITION_CONSTANTS = {
        'caloric_surplus': 400,
        'protein_per_kg': 2.0,
        'fat_percentage': 0.25,
        'target_weekly_gain_rate': 0.0025,  # 0.25% of body weight per week
        'calories_per_gram_protein': 4,
        'calories_per_gram_carb': 4,
        'calories_per_gram_fat': 9
    }
    
    # Activity level multipliers for TDEE calculation
    ACTIVITY_MULTIPLIERS = {
        'sedentary': 1.2,
        'lightly_active': 1.375,
        'moderately_active': 1.55,
        'very_active': 1.725,
        'extremely_active': 1.9
    }
    
    # UI configuration
    UI_CONFIG = {
        'gender_options': ["Male", "Female"],
        'activity_options': {
            "Select Activity Level": None,
            "Sedentary": "sedentary",
            "Lightly Active": "lightly_active",
            "Moderately Active": "moderately_active",
            "Very Active": "very_active",
            "Extremely Active": "extremely_active"
        },
        'age_range': (16, 80),
        'height_range': (140, 220),
        'weight_range': (40.0, 150.0),
        'caloric_surplus_range': (200, 800),
        'protein_range': (1.2, 3.0),
        'fat_percentage_range': (15, 40)
    }
    
    # Food ranking emoji configuration
    EMOJI_CONFIG = {
        'superfood': '🥇',
        'nutrient_calorie_dense': '💥',
        'high_calorie': '🔥',
        'top_protein': '💪',
        'top_carb': '🍚',
        'top_fat': '🥑',
        'top_micronutrient': '🥦',
        'default': ''
    }
    
    # Emoji priority order for sorting
    EMOJI_ORDER = {
        '🥇': 0, '💥': 1, '🔥': 2, '💪': 3, 
        '🍚': 3, '🥑': 3, '🥦': 3, '': 4
    }
    
    # Nutrient category mapping
    NUTRIENT_CATEGORY_MAP = {
        'PRIMARY PROTEIN SOURCES': 'protein',
        'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
        'PRIMARY FAT SOURCES': 'fat',
        'PRIMARY MICRONUTRIENT SOURCES': 'protein'
    }

    @classmethod
    def get_activity_multiplier(cls, activity_level: str) -> float:
        """Get activity multiplier with fallback to moderate activity."""
        return cls.ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    
    @classmethod
    def validate_file_paths(cls) -> bool:
        """Validate that required data files exist."""
        return cls.NUTRITION_DB_FILE.exists()
