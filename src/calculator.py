"""Nutrition calculation engine."""

import logging
from typing import Dict, Any
from .models import UserProfile, NutritionTargets
from .config import NutritionConfig

logger = logging.getLogger(__name__)

class NutritionCalculator:
    """Handles all nutrition-related calculations."""
    
    def __init__(self, config: NutritionConfig = None):
        """Initialize calculator with configuration."""
        self.config = config or NutritionConfig()
    
    def calculate_bmr(self, profile: UserProfile) -> float:
        """
        Calculate Basal Metabolic Rate using the Mifflin-St Jeor Equation.
        
        Args:
            profile: User profile containing age, height, weight, and gender
            
        Returns:
            BMR in calories per day
        """
        logger.debug(f"Calculating BMR for {profile.gender}, age {profile.age}")
        
        if profile.gender.lower() == 'male':
            bmr = (10 * profile.weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age) + 5
        else:
            bmr = (10 * profile.weight_kg) + (6.25 * profile.height_cm) - (5 * profile.age) - 161
        
        logger.debug(f"BMR calculated: {bmr}")
        return bmr
    
    def calculate_tdee(self, bmr: float, activity_level: str) -> float:
        """
        Calculate Total Daily Energy Expenditure.
        
        Args:
            bmr: Basal Metabolic Rate
            activity_level: Activity level string
            
        Returns:
            TDEE in calories per day
        """
        multiplier = self.config.get_activity_multiplier(activity_level)
        tdee = bmr * multiplier
        
        logger.debug(f"TDEE calculated: {tdee} (BMR: {bmr}, multiplier: {multiplier})")
        return tdee
    
    def calculate_targets(
        self, 
        profile: UserProfile,
        caloric_surplus: int = None,
        protein_per_kg: float = None,
        fat_percentage: float = None
    ) -> NutritionTargets:
        """
        Calculate personalized daily nutritional targets.
        
        Args:
            profile: User profile
            caloric_surplus: Additional calories for weight gain
            protein_per_kg: Protein grams per kg body weight
            fat_percentage: Fat as percentage of total calories
            
        Returns:
            Complete nutrition targets
        """
        logger.info(f"Calculating targets for user: {profile.age}y, {profile.weight_kg}kg, {profile.gender}")
        
        # Use defaults if not provided
        caloric_surplus = caloric_surplus or self.config.NUTRITION_CONSTANTS['caloric_surplus']
        protein_per_kg = protein_per_kg or self.config.NUTRITION_CONSTANTS['protein_per_kg']
        fat_percentage = fat_percentage or self.config.NUTRITION_CONSTANTS['fat_percentage']
        
        # Calculate basic metabolic values
        bmr = self.calculate_bmr(profile)
        tdee = self.calculate_tdee(bmr, profile.activity_level)
        total_calories = tdee + caloric_surplus
        
        # Calculate macronutrients
        protein_g = protein_per_kg * profile.weight_kg
        protein_calories = protein_g * self.config.NUTRITION_CONSTANTS['calories_per_gram_protein']
        
        fat_calories = total_calories * fat_percentage
        fat_g = fat_calories / self.config.NUTRITION_CONSTANTS['calories_per_gram_fat']
        
        carb_calories = total_calories - protein_calories - fat_calories
        carb_g = carb_calories / self.config.NUTRITION_CONSTANTS['calories_per_gram_carb']
        
        # Calculate target weight gain
        target_weight_gain = profile.weight_kg * self.config.NUTRITION_CONSTANTS['target_weekly_gain_rate']
        
        targets = NutritionTargets(
            bmr=round(bmr),
            tdee=round(tdee),
            total_calories=round(total_calories),
            protein_g=round(protein_g),
            protein_calories=round(protein_calories),
            fat_g=round(fat_g),
            fat_calories=round(fat_calories),
            carb_g=round(carb_g),
            carb_calories=round(carb_calories),
            target_weight_gain_per_week=round(target_weight_gain, 2)
        )
        
        logger.info(f"Targets calculated: {total_calories} kcal, {protein_g}g protein")
        return targets
    
    def validate_targets(self, targets: NutritionTargets) -> bool:
        """
        Validate that calculated targets are reasonable.
        
        Args:
            targets: Calculated nutrition targets
            
        Returns:
            True if targets are valid
        """
        # Basic sanity checks
        if targets.total_calories < 1200 or targets.total_calories > 5000:
            logger.warning(f"Unusual calorie target: {targets.total_calories}")
            return False
        
        if targets.protein_g < 50 or targets.protein_g > 300:
            logger.warning(f"Unusual protein target: {targets.protein_g}")
            return False
        
        # Check macronutrient distribution
        percentages = targets.get_macronutrient_percentages()
        if not (10 <= percentages['protein'] <= 35):
            logger.warning(f"Protein percentage out of range: {percentages['protein']:.1f}%")
            return False
        
        if not (15 <= percentages['fat'] <= 40):
            logger.warning(f"Fat percentage out of range: {percentages['fat']:.1f}%")
            return False
        
        return True
