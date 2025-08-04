"""
Business logic for nutrition calculations.
Contains all mathematical formulas and nutritional computation logic.
"""

from src.data_models import UserProfile, NutritionTargets
from src.config import config

class NutritionCalculator:
    """Handles all nutrition-related calculations."""
    
    def __init__(self):
        """Initialize the nutrition calculator with configuration."""
        self.activity_multipliers = config.ACTIVITY_MULTIPLIERS
        self.target_weekly_gain_rate = config.TARGET_WEEKLY_GAIN_RATE
    
    def calculate_bmr(self, age: int, height_cm: int, weight_kg: float, sex: str) -> float:
        """
        Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation.
        
        Args:
            age: Age in years
            height_cm: Height in centimeters
            weight_kg: Weight in kilograms
            sex: 'Male' or 'Female'
            
        Returns:
            BMR in calories per day
        """
        if sex.lower() == 'male':
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
        else:
            bmr = (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161
        return bmr
    
    def calculate_tdee(self, bmr: float, activity_level: str) -> float:
        """
        Calculate Total Daily Energy Expenditure based on Activity Level.
        
        Args:
            bmr: Basal Metabolic Rate
            activity_level: Activity level key
            
        Returns:
            TDEE in calories per day
        """
        multiplier = self.activity_multipliers.get(activity_level, 1.55)
        return bmr * multiplier
    
    def calculate_targets(self, user_profile: UserProfile) -> NutritionTargets:
        """
        Calculate personalized daily nutritional targets.
        
        Args:
            user_profile: User's personal information and preferences
            
        Returns:
            Complete nutritional targets for the user
        """
        # Calculate BMR and TDEE
        bmr = self.calculate_bmr(
            user_profile.age, 
            user_profile.height_cm, 
            user_profile.weight_kg, 
            user_profile.sex
        )
        tdee = self.calculate_tdee(bmr, user_profile.activity_level)
        
        # Calculate total calories with surplus
        total_calories = tdee + user_profile.caloric_surplus
        
        # Calculate protein targets
        protein_g = user_profile.protein_per_kg * user_profile.weight_kg
        protein_calories = protein_g * 4
        
        # Calculate fat targets
        fat_calories = total_calories * user_profile.fat_percentage
        fat_g = fat_calories / 9
        
        # Calculate carbohydrate targets (remainder)
        carb_calories = total_calories - protein_calories - fat_calories
        carb_g = carb_calories / 4
        
        # Calculate target weight gain
        target_weight_gain_per_week = user_profile.weight_kg * self.target_weekly_gain_rate
        
        return NutritionTargets(
            bmr=round(bmr),
            tdee=round(tdee),
            total_calories=round(total_calories),
            protein_g=round(protein_g),
            protein_calories=round(protein_calories),
            fat_g=round(fat_g),
            fat_calories=round(fat_calories),
            carb_g=round(carb_g),
            carb_calories=round(carb_calories),
            target_weight_gain_per_week=round(target_weight_gain_per_week, 2)
        )
