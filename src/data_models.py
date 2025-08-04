"""
Data classes for the Personalized Evidence-Based Nutrition Tracker.
Defines the structure for all data objects used throughout the application.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class UserProfile:
    """Represents a user's personal information and preferences."""
    age: int
    height_cm: int
    weight_kg: float
    sex: str
    activity_level: str
    caloric_surplus: int = 400
    protein_per_kg: float = 2.0
    fat_percentage: float = 0.25
    
    def __post_init__(self):
        """Validate user profile data after initialization."""
        if not (16 <= self.age <= 80):
            raise ValueError("Age must be between 16 and 80 years")
        if not (140 <= self.height_cm <= 220):
            raise ValueError("Height must be between 140 and 220 cm")
        if not (40.0 <= self.weight_kg <= 150.0):
            raise ValueError("Weight must be between 40 and 150 kg")
        if self.sex not in ["Male", "Female"]:
            raise ValueError("Sex must be 'Male' or 'Female'")

@dataclass
class NutritionTargets:
    """Represents calculated daily nutritional targets for a user."""
    bmr: int
    tdee: int
    total_calories: int
    protein_g: int
    protein_calories: int
    fat_g: int
    fat_calories: int
    carb_g: int
    carb_calories: int
    target_weight_gain_per_week: float
    
    def get_protein_percentage(self) -> float:
        """Calculate protein as percentage of total calories."""
        return (self.protein_calories / self.total_calories) * 100
    
    def get_carb_percentage(self) -> float:
        """Calculate carbohydrates as percentage of total calories."""
        return (self.carb_calories / self.total_calories) * 100
    
    def get_fat_percentage(self) -> float:
        """Calculate fat as percentage of total calories."""
        return (self.fat_calories / self.total_calories) * 100

@dataclass
class FoodItem:
    """Represents a single food item with its nutritional information."""
    name: str
    calories: int
    protein: float
    carbs: float
    fat: float
    category: str
    serving_unit: str = ""
    emoji: str = ""
    
    def get_display_name(self) -> str:
        """Get the formatted display name including serving unit."""
        if self.serving_unit:
            return f"{self.name} ({self.serving_unit})"
        return self.name
    
    def calculate_nutrition_for_servings(self, servings: float) -> Dict[str, float]:
        """Calculate total nutrition for given number of servings."""
        return {
            'calories': self.calories * servings,
            'protein': self.protein * servings,
            'carbs': self.carbs * servings,
            'fat': self.fat * servings
        }

@dataclass
class FoodSelection:
    """Represents a user's selection of a food item with serving amount."""
    food_item: FoodItem
    servings: float
    
    def get_total_nutrition(self) -> Dict[str, float]:
        """Get total nutritional values for this selection."""
        return self.food_item.calculate_nutrition_for_servings(self.servings)

@dataclass
class DailyIntake:
    """Represents a user's total daily food intake."""
    selections: List[FoodSelection]
    
    def get_totals(self) -> Dict[str, float]:
        """Calculate total nutritional intake for the day."""
        totals = {'calories': 0, 'protein': 0, 'carbs': 0, 'fat': 0}
        for selection in self.selections:
            nutrition = selection.get_total_nutrition()
            for key in totals:
                totals[key] += nutrition[key]
        return totals
    
    def get_progress_vs_targets(self, targets: NutritionTargets) -> Dict[str, float]:
        """Calculate progress percentages toward daily targets."""
        totals = self.get_totals()
        return {
            'calories': (totals['calories'] / targets.total_calories) * 100,
            'protein': (totals['protein'] / targets.protein_g) * 100,
            'carbs': (totals['carbs'] / targets.carb_g) * 100,
            'fat': (totals['fat'] / targets.fat_g) * 100
        }

@dataclass
class UserInputs:
    """Represents all user inputs from the sidebar."""
    age: int
    height_cm: int
    weight_kg: float
    sex: str
    activity_level: str
    caloric_surplus: int
    protein_per_kg: float
    fat_percentage: float
    user_has_entered_info: bool
