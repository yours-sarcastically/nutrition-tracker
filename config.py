from dataclasses import dataclass
from typing import Dict, List

@dataclass
class NutritionConfig:
    """Configuration class for all nutrition calculation constants and defaults."""
    
    # User defaults
    DEFAULT_AGE: int = 26
    DEFAULT_HEIGHT_CM: int = 180
    DEFAULT_WEIGHT_KG: float = 57.5
    DEFAULT_GENDER: str = "Male"
    DEFAULT_ACTIVITY_LEVEL: str = "moderately_active"
    
    # Calculation constants
    DEFAULT_CALORIC_SURPLUS: int = 400
    DEFAULT_PROTEIN_PER_KG: float = 2.0
    DEFAULT_FAT_PERCENTAGE: float = 0.25
    TARGET_WEEKLY_GAIN_RATE: float = 0.0025  # 0.25% of body weight per week
    
    # Calories per gram
    PROTEIN_CALORIES_PER_GRAM: int = 4
    CARB_CALORIES_PER_GRAM: int = 4
    FAT_CALORIES_PER_GRAM: int = 9
    
    # UI Options
    GENDER_OPTIONS: List[str] = None
    ACTIVITY_OPTIONS: Dict[str, str] = None
    ACTIVITY_MULTIPLIERS: Dict[str, float] = None
    
    # File paths
    FOOD_DATABASE_PATH: str = "nutrition_database_final.csv"
    
    # Emoji mappings
    EMOJI_ORDER: Dict[str, int] = None
    NUTRIENT_CATEGORY_MAP: Dict[str, str] = None
    
    def __post_init__(self):
        if self.GENDER_OPTIONS is None:
            self.GENDER_OPTIONS = ["Male", "Female"]
        
        if self.ACTIVITY_OPTIONS is None:
            self.ACTIVITY_OPTIONS = {
                "Select Activity Level": None,
                "Sedentary": "sedentary",
                "Lightly Active": "lightly_active",
                "Moderately Active": "moderately_active",
                "Very Active": "very_active",
                "Extremely Active": "extremely_active"
            }
        
        if self.ACTIVITY_MULTIPLIERS is None:
            self.ACTIVITY_MULTIPLIERS = {
                'sedentary': 1.2,
                'lightly_active': 1.375,
                'moderately_active': 1.55,
                'very_active': 1.725,
                'extremely_active': 1.9
            }
        
        if self.EMOJI_ORDER is None:
            self.EMOJI_ORDER = {'🥇': 0, '💥': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '🥦': 3, '': 4}
        
        if self.NUTRIENT_CATEGORY_MAP is None:
            self.NUTRIENT_CATEGORY_MAP = {
                'PRIMARY PROTEIN SOURCES': 'protein',
                'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
                'PRIMARY FAT SOURCES': 'fat',
                'PRIMARY MICRONUTRIENT SOURCES': 'protein'
            }

# Global config instance
config = NutritionConfig()
