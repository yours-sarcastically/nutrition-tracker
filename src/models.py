"""Data models for the nutrition tracker application."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

class Gender(Enum):
    """Gender enumeration for BMR calculations."""
    MALE = "Male"
    FEMALE = "Female"

class ActivityLevel(Enum):
    """Activity level enumeration."""
    SEDENTARY = "sedentary"
    LIGHTLY_ACTIVE = "lightly_active"
    MODERATELY_ACTIVE = "moderately_active"
    VERY_ACTIVE = "very_active"
    EXTREMELY_ACTIVE = "extremely_active"

@dataclass
class UserProfile:
    """User profile data for nutrition calculations."""
    age: int
    height_cm: int
    weight_kg: float
    gender: str
    activity_level: str
    
    def __post_init__(self):
        """Validate user profile data."""
        if not (16 <= self.age <= 80):
            raise ValueError("Age must be between 16 and 80")
        if not (140 <= self.height_cm <= 220):
            raise ValueError("Height must be between 140 and 220 cm")
        if not (40.0 <= self.weight_kg <= 150.0):
            raise ValueError("Weight must be between 40 and 150 kg")
        if self.gender not in ["Male", "Female"]:
            raise ValueError("Gender must be 'Male' or 'Female'")

@dataclass
class NutritionTargets:
    """Daily nutrition targets calculated for a user."""
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
    
    def get_macronutrient_percentages(self) -> Dict[str, float]:
        """Calculate macronutrient distribution as percentages."""
        return {
            'protein': (self.protein_calories / self.total_calories) * 100,
            'carbs': (self.carb_calories / self.total_calories) * 100,
            'fat': (self.fat_calories / self.total_calories) * 100
        }

@dataclass
class FoodItem:
    """Individual food item with nutritional information."""
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    serving_unit: str = ""
    category: str = ""
    emoji: str = ""
    
    def __post_init__(self):
        """Format the display name with serving unit."""
        if self.serving_unit and not self.name.endswith(f"({self.serving_unit})"):
            self.name = f"{self.name} ({self.serving_unit})"
    
    def calculate_nutrition(self, servings: float) -> 'NutritionSummary':
        """Calculate nutrition for given number of servings."""
        return NutritionSummary(
            calories=self.calories * servings,
            protein=self.protein * servings,
            carbs=self.carbs * servings,
            fat=self.fat * servings
        )

@dataclass
class NutritionSummary:
    """Summary of nutritional intake."""
    calories: float = 0.0
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    
    def __add__(self, other: 'NutritionSummary') -> 'NutritionSummary':
        """Add two nutrition summaries together."""
        return NutritionSummary(
            calories=self.calories + other.calories,
            protein=self.protein + other.protein,
            carbs=self.carbs + other.carbs,
            fat=self.fat + other.fat
        )

@dataclass
class SelectedFood:
    """A food item with selected serving amount."""
    food: FoodItem
    servings: float
    
    @property
    def nutrition(self) -> NutritionSummary:
        """Get nutrition summary for selected servings."""
        return self.food.calculate_nutrition(self.servings)

@dataclass
class DailyIntake:
    """Complete daily food intake summary."""
    selected_foods: List[SelectedFood] = field(default_factory=list)
    
    @property
    def total_nutrition(self) -> NutritionSummary:
        """Calculate total nutrition from all selected foods."""
        total = NutritionSummary()
        for selected_food in self.selected_foods:
            total += selected_food.nutrition
        return total
    
    def add_food(self, food: FoodItem, servings: float) -> None:
        """Add a food item to daily intake."""
        if servings > 0:
            self.selected_foods.append(SelectedFood(food, servings))
    
    def clear(self) -> None:
        """Clear all selected foods."""
        self.selected_foods.clear()

@dataclass
class NutritionProgress:
    """Progress tracking against targets."""
    targets: NutritionTargets
    actual: NutritionSummary
    
    @property
    def calorie_progress(self) -> float:
        """Calorie progress as percentage."""
        return (self.actual.calories / self.targets.total_calories) * 100
    
    @property
    def protein_progress(self) -> float:
        """Protein progress as percentage."""
        return (self.actual.protein / self.targets.protein_g) * 100
    
    @property
    def carb_progress(self) -> float:
        """Carbohydrate progress as percentage."""
        return (self.actual.carbs / self.targets.carb_g) * 100
    
    @property
    def fat_progress(self) -> float:
        """Fat progress as percentage."""
        return (self.actual.fat / self.targets.fat_g) * 100
    
    def get_all_progress(self) -> Dict[str, float]:
        """Get all progress metrics."""
        return {
            'calories': self.calorie_progress,
            'protein': self.protein_progress,
            'carbs': self.carb_progress,
            'fat': self.fat_progress
        }
