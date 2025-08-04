"""
Configuration file for the Personalized Evidence-Based Nutrition Tracker.
Contains all constants, default values, and configuration settings.
"""

class Config:
    """Configuration class containing all application settings."""
    
    # File Paths
    FOOD_DATABASE_PATH = "nutrition_database_final.csv"
    
    # Default User Values
    DEFAULT_AGE = 25
    DEFAULT_HEIGHT_CM = 175
    DEFAULT_WEIGHT_KG = 70.0
    DEFAULT_GENDER = "Male"
    DEFAULT_ACTIVITY_LEVEL = "moderately_active"
    DEFAULT_CALORIC_SURPLUS = 400
    DEFAULT_PROTEIN_PER_KG = 2.0
    DEFAULT_FAT_PERCENTAGE = 0.25  # 25% of total calories
    
    # Gender Options
    GENDER_OPTIONS = ["Male", "Female"]
    
    # Activity Level Options and Mappings
    ACTIVITY_OPTIONS = {
        "Sedentary (little to no exercise)": "sedentary",
        "Lightly Active (light exercise 1-3 days/week)": "lightly_active",
        "Moderately Active (moderate exercise 3-5 days/week)": "moderately_active",
        "Very Active (hard exercise 6-7 days/week)": "very_active",
        "Extremely Active (very hard exercise, physical job)": "extremely_active"
    }
    
    # TDEE Activity Multipliers (based on research)
    ACTIVITY_MULTIPLIERS = {
        "sedentary": 1.2,
        "lightly_active": 1.375,
        "moderately_active": 1.55,
        "very_active": 1.725,
        "extremely_active": 1.9
    }
    
    # Nutritional Constants
    CALORIES_PER_GRAM_PROTEIN = 4
    CALORIES_PER_GRAM_CARB = 4
    CALORIES_PER_GRAM_FAT = 9
    
    # Weight Gain Parameters
    TARGET_WEEKLY_GAIN_RATE = 0.005  # 0.5% of body weight per week
    CALORIES_PER_KG_WEIGHT_GAIN = 7700  # Approximate calories needed to gain 1kg
    
    # Food Category to Nutrient Mapping
    NUTRIENT_CATEGORY_MAP = {
        "PRIMARY PROTEIN SOURCES": "protein",
        "SECONDARY PROTEIN SOURCES": "protein",
        "COMPLEX CARBOHYDRATES": "carbs",
        "SIMPLE CARBOHYDRATES": "carbs",
        "HEALTHY FATS": "fat",
        "NUTS AND SEEDS": "fat",
        "DAIRY AND ALTERNATIVES": "protein",
        "VEGETABLES": "micro",
        "FRUITS": "carbs",
        "BEVERAGES": "carbs",
        "SUPPLEMENTS": "protein"
    }
    
    # Emoji Rankings (lower number = higher priority)
    EMOJI_ORDER = {
        '🥇': 0,  # Superfood
        '💥': 1,  # Nutrient & Calorie Dense
        '🔥': 2,  # High-Calorie
        '💪': 2,  # Top Protein
        '🍚': 2,  # Top Carb
        '🥑': 2,  # Top Fat
        '🥦': 3,  # Top Micronutrient
        '': 4     # No special ranking
    }
    
    # UI Configuration
    PAGE_TITLE = "Personalized Nutrition Tracker"
    PAGE_ICON = "🍽️"
    LAYOUT = "wide"
    SIDEBAR_STATE = "expanded"
    
    # Validation Ranges
    MIN_AGE = 16
    MAX_AGE = 80
    MIN_HEIGHT_CM = 140
    MAX_HEIGHT_CM = 220
    MIN_WEIGHT_KG = 40.0
    MAX_WEIGHT_KG = 150.0
    MIN_CALORIC_SURPLUS = 200
    MAX_CALORIC_SURPLUS = 800
    MIN_PROTEIN_PER_KG = 1.2
    MAX_PROTEIN_PER_KG = 3.0
    MIN_FAT_PERCENTAGE = 0.15  # 15%
    MAX_FAT_PERCENTAGE = 0.40  # 40%
    
    # Progress Bar Thresholds
    LOW_INTAKE_THRESHOLD = 80    # Below 80% of target
    HIGH_INTAKE_THRESHOLD = 120  # Above 120% of target
    CARB_LOW_THRESHOLD = 70      # Below 70% for carbs
    FAT_LOW_THRESHOLD = 70       # Below 70% for fats
    
    # Recommendation Messages
    RECOMMENDATIONS = {
        "low_calories": "🔥 You're below your calorie target. Consider adding more calorie-dense foods.",
        "high_calories": "⚠️ You're significantly over your calorie target. Consider reducing portion sizes.",
        "low_protein": "💪 Your protein intake is low. Add more protein-rich foods like lean meats, eggs, or protein powder.",
        "low_carbs": "🍚 Your carbohydrate intake is low. Add more complex carbs like oats, rice, or fruits.",
        "low_fats": "🥑 Your fat intake is low. Include healthy fats like nuts, avocado, or olive oil.",
        "balanced": "🎉 Great job! Your nutrition is well-balanced and on track with your targets!"
    }
    
    # Food Selection Configuration
    MAX_CUSTOM_SERVINGS = 10.0
    SERVING_INCREMENT = 0.1
    QUICK_SELECT_BUTTONS = [1, 2, 3, 4, 5]
    
    # Display Configuration
    DECIMAL_PLACES_WEIGHT = 1
    DECIMAL_PLACES_PERCENTAGE = 1
    DECIMAL_PLACES_CALORIES = 0
    
    # Cache Configuration
    CACHE_TTL = 3600  # 1 hour in seconds
    
    # Error Messages
    ERROR_MESSAGES = {
        "invalid_age": f"Age must be between {MIN_AGE} and {MAX_AGE} years",
        "invalid_height": f"Height must be between {MIN_HEIGHT_CM} and {MAX_HEIGHT_CM} cm",
        "invalid_weight": f"Weight must be between {MIN_WEIGHT_KG} and {MAX_WEIGHT_KG} kg",
        "invalid_sex": "Sex must be 'Male' or 'Female'",
        "file_not_found": "Food database file not found. Please check the file path.",
        "empty_database": "Food database is empty or corrupted.",
        "loading_error": "Error loading food database. Please try again."
    }
    
    # Success Messages
    SUCCESS_MESSAGES = {
        "profile_created": "✅ User profile created successfully!",
        "targets_calculated": "✅ Nutritional targets calculated!",
        "food_added": "✅ Food added to daily intake!",
        "food_removed": "✅ Food removed from daily intake!"
    }
    
    # Info Messages
    INFO_MESSAGES = {
        "enter_info": "👈 Please enter your personal information in the sidebar to view your daily nutritional targets.",
        "no_foods_selected": "No foods have been selected for today. 🍽️",
        "sample_targets": "These are example targets. Enter your information in the sidebar for personalized calculations."
    }

# Create a global config instance
config = Config()

# Export commonly used constants for convenience
DEFAULT_VALUES = {
    'age': config.DEFAULT_AGE,
    'height_cm': config.DEFAULT_HEIGHT_CM,
    'weight_kg': config.DEFAULT_WEIGHT_KG,
    'sex': config.DEFAULT_GENDER,
    'activity_level': config.DEFAULT_ACTIVITY_LEVEL,
    'caloric_surplus': config.DEFAULT_CALORIC_SURPLUS,
    'protein_per_kg': config.DEFAULT_PROTEIN_PER_KG,
    'fat_percentage': config.DEFAULT_FAT_PERCENTAGE
}

# Validation functions
def validate_age(age: int) -> bool:
    """Validate age input."""
    return config.MIN_AGE <= age <= config.MAX_AGE

def validate_height(height_cm: int) -> bool:
    """Validate height input."""
    return config.MIN_HEIGHT_CM <= height_cm <= config.MAX_HEIGHT_CM

def validate_weight(weight_kg: float) -> bool:
    """Validate weight input."""
    return config.MIN_WEIGHT_KG <= weight_kg <= config.MAX_WEIGHT_KG

def validate_sex(sex: str) -> bool:
    """Validate sex input."""
    return sex in config.GENDER_OPTIONS

def validate_caloric_surplus(surplus: int) -> bool:
    """Validate caloric surplus input."""
    return config.MIN_CALORIC_SURPLUS <= surplus <= config.MAX_CALORIC_SURPLUS

def validate_protein_per_kg(protein: float) -> bool:
    """Validate protein per kg input."""
    return config.MIN_PROTEIN_PER_KG <= protein <= config.MAX_PROTEIN_PER_KG

def validate_fat_percentage(fat_pct: float) -> bool:
    """Validate fat percentage input."""
    return config.MIN_FAT_PERCENTAGE <= fat_pct <= config.MAX_FAT_PERCENTAGE

# Helper functions for common operations
def get_activity_multiplier(activity_level: str) -> float:
    """Get TDEE multiplier for given activity level."""
    return config.ACTIVITY_MULTIPLIERS.get(activity_level, config.ACTIVITY_MULTIPLIERS["moderately_active"])

def get_emoji_priority(emoji: str) -> int:
    """Get priority ranking for emoji (lower = higher priority)."""
    return config.EMOJI_ORDER.get(emoji, 4)

def format_calories(calories: float) -> str:
    """Format calories for display."""
    return f"{calories:.{config.DECIMAL_PLACES_CALORIES}f}"

def format_weight(weight: float) -> str:
    """Format weight for display."""
    return f"{weight:.{config.DECIMAL_PLACES_WEIGHT}f}"

def format_percentage(percentage: float) -> str:
    """Format percentage for display."""
    return f"{percentage:.{config.DECIMAL_PLACES_PERCENTAGE}f}%"
