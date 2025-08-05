# core/calculations.py
# Description: Core business logic for nutritional calculations.

from typing import Dict, List, Tuple
from .models import UserProfile, AdvancedSettings, NutritionalTargets, FoodItem
from config import ACTIVITY_MULTIPLIERS, CONFIG

def calculate_bmr(user: UserProfile) -> float:
    """Calculate Basal Metabolic Rate Using the Mifflin-St Jeor Equation."""
    base_calc = (10 * user.weight_kg) + (6.25 * user.height_cm) - (5 * user.age)
    return base_calc + (5 if user.sex.lower() == 'male' else -161)

def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Calculate Total Daily Energy Expenditure Based on Activity Level."""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier

def calculate_personalized_targets(user: UserProfile, settings: AdvancedSettings) -> NutritionalTargets:
    """Calculate Personalized Daily Nutritional Targets."""
    bmr = calculate_bmr(user)
    tdee = calculate_tdee(bmr, user.activity_level)
    total_calories = tdee + settings.caloric_surplus
    
    protein_g = settings.protein_per_kg * user.weight_kg
    protein_calories = protein_g * 4
    fat_calories = total_calories * settings.fat_percentage
    fat_g = fat_calories / 9
    carb_calories = total_calories - protein_calories - fat_calories
    carb_g = carb_calories / 4

    targets = NutritionalTargets(
        bmr=round(bmr),
        tdee=round(tdee),
        total_calories=round(total_calories),
        protein_g=round(protein_g),
        protein_calories=round(protein_calories),
        fat_g=round(fat_g),
        fat_calories=round(fat_calories),
        carb_g=round(carb_g),
        carb_calories=round(carb_calories),
        target_weight_gain_per_week=round(user.weight_kg * 0.0025, 2)
    )
    return targets

def calculate_daily_totals(
    food_selections: Dict[str, float], 
    foods: Dict[str, List[FoodItem]]
) -> Tuple[Dict[str, float], List[Dict]]:
    """Calculate total daily nutrition from food selections."""
    totals = {nutrient: 0.0 for nutrient in CONFIG['nutrient_configs'].keys()}
    selected_foods_details = []
    
    all_foods_flat = {food.name: food for sublist in foods.values() for food in sublist}

    for food_name, servings in food_selections.items():
        if servings > 0 and food_name in all_foods_flat:
            food_item = all_foods_flat[food_name]
            totals['calories'] += food_item.calories * servings
            totals['protein'] += food_item.protein * servings
            totals['carbs'] += food_item.carbs * servings
            totals['fat'] += food_item.fat * servings
            selected_foods_details.append({'food': food_item, 'servings': servings})
    
    return totals, selected_foods_details
