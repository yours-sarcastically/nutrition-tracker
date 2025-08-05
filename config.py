# config.py
# Description: Centralized configuration for the Nutrition Tracker application.

# ------ Default Parameter Values Based on Published Research ------
DEFAULTS = {
    'age': 26,
    'height_cm': 180,
    'weight_kg': 57.5,
    'sex': "Male",
    'activity_level': "moderately_active",
    'caloric_surplus': 400,
    'protein_per_kg': 2.0,
    'fat_percentage': 0.25
}

# ------ Activity Level Multipliers for TDEE Calculation ------
ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

# ------ Unified Configuration for All App Components ------
CONFIG = {
    'emoji_order': {'🥇': 0, '💥': 1, '🔥': 2, '💪': 3, '🍚': 3, '🥑': 3, '🥦': 3, '': 4},
    'nutrient_map': {
        'PRIMARY PROTEIN SOURCES': {'sort_by': 'protein', 'key': 'protein'},
        'PRIMARY CARBOHYDRATE SOURCES': {'sort_by': 'carbs', 'key': 'carbs'},
        'PRIMARY FAT SOURCES': {'sort_by': 'fat', 'key': 'fat'},
        'PRIMARY MICRONUTRIENT SOURCES': {'sort_by': 'protein', 'key': 'micro'}
    },
    'nutrient_configs': {
        'calories': {'unit': 'kcal', 'label': 'Calories', 'target_key': 'total_calories'},
        'protein': {'unit': 'g', 'label': 'Protein', 'target_key': 'protein_g'},
        'carbs': {'unit': 'g', 'label': 'Carbohydrates', 'target_key': 'carb_g'},
        'fat': {'unit': 'g', 'label': 'Fat', 'target_key': 'fat_g'}
    },
    'form_fields': {
        'age': {
            'type': 'number', 
            'label': 'Age (Years)', 
            'min': 16, 
            'max': 80, 
            'step': 1, 
            'placeholder': 'Enter your age', 
            'required': True
        },
        'height_cm': {
            'type': 'number', 
            'label': 'Height (Centimeters)', 
            'min': 140, 
            'max': 220, 
            'step': 1, 
            'placeholder': 'Enter your height', 
            'required': True
        },
        'weight_kg': {
            'type': 'number', 
            'label': 'Weight (kg)', 
            'min': 40.0, 
            'max': 150.0, 
            'step': 0.5, 
            'placeholder': 'Enter your weight', 
            'required': True
        },
        'sex': {
            'type': 'selectbox', 
            'label': 'Sex', 
            'options': ["Select Sex", "Male", "Female"], 
            'required': True, 
            'placeholder': "Select Sex"
        },
        'activity_level': {
            'type': 'selectbox', 
            'label': 'Activity Level', 
            'options': [
                ("Select Activity Level", None),
                ("Sedentary", "sedentary"),
                ("Lightly Active", "lightly_active"),
                ("Moderately Active", "moderately_active"),
                ("Very Active", "very_active"),
                ("Extremely Active", "extremely_active")
            ], 
            'required': True, 
            'placeholder': None
        },
        'caloric_surplus': {
            'type': 'number', 
            'label': 'Caloric Surplus (kcal Per Day)', 
            'min': 200, 
            'max': 800, 
            'step': 50, 
            'help': 'Additional calories above maintenance for weight gain', 
            'advanced': True, 
            'required': False
        },
        'protein_per_kg': {
            'type': 'number', 
            'label': 'Protein (g Per Kilogram Body Weight)', 
            'min': 1.2, 
            'max': 3.0, 
            'step': 0.1, 
            'help': 'Protein intake per kilogram of body weight', 
            'advanced': True, 
            'required': False
        },
        'fat_percentage': {
            'type': 'number', 
            'label': 'Fat (Percent of Total Calories)', 
            'min': 15, 
            'max': 40, 
            'step': 1, 
            'help': 'Percentage of total calories from fat', 
            'convert': lambda x: x / 100 if x is not None else None, 
            'advanced': True, 
            'required': False
        }
    }
}
