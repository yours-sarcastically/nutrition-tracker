from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class UserProfile:
    """Data class representing a user's profile and preferences."""
    age: int
    height_cm: int
    weight_kg: float
    sex: str
    activity_level: str
    caloric_surplus: Optional[int] = None
    protein_per_kg: Optional[float] = None
    fat_percentage: Optional[float] = None
    
    def __post_init__(self):
        """Set defaults if not provided."""
        from config import config
        if self.caloric_surplus is None:
            self.caloric_surplus = config.DEFAULT_CALORIC_SURPLUS
        if self.protein_per_kg is None:
            self.protein_per_kg = config.DEFAULT_PROTEIN_PER_KG
        if self.fat_percentage is None:
            self.fat_percentage = config.DEFAULT_FAT_PERCENTAGE

@dataclass
class NutritionTargets:
    """Data class representing calculated nutrition targets."""
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
        """Calculate macronutrient percentages of total calories."""
        return {
            'protein': (self.protein_calories / self.total_calories) * 100,
            'carbs': (self.carb_calories / self.total_calories) * 100,
            'fat': (self.fat_calories / self.total_calories) * 100
        }

@dataclass
class FoodItem:
    """Data class representing a food item."""
    name: str
    calories: int
    protein: float
    carbs: float
    fat: float
    category: str
    serving_unit: str
    emoji: str = ""
    
    @property
    def display_name(self) -> str:
        """Get formatted display name with serving unit."""
        return f"{self.name} ({self.serving_unit})"
    
    @property
    def nutrition_summary(self) -> str:
        """Get formatted nutrition summary."""
        return f"Per Serving: {self.calories} kcal | {self.protein}g protein | {self.carbs}g carbs | {self.fat}g fat"

@dataclass
class FoodSelection:
    """Data class representing a selected food with servings."""
    food_item: FoodItem
    servings: float
    
    @property
    def total_calories(self) -> float:
        return self.food_item.calories * self.servings
    
    @property
    def total_protein(self) -> float:
        return self.food_item.protein * self.servings
    
    @property
    def total_carbs(self) -> float:
        return self.food_item.carbs * self.servings
    
    @property
    def total_fat(self) -> float:
        return self.food_item.fat * self.servings

@dataclass
class DailyIntake:
    """Data class representing total daily nutritional intake."""
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    selected_foods: List[FoodSelection]
    
    def get_progress_vs_targets(self, targets: NutritionTargets) -> Dict[str, float]:
        """Calculate progress percentages against targets."""
        return {
            'calories': (self.total_calories / targets.total_calories) * 100,
            'protein': (self.total_protein / targets.protein_g) * 100,
            'carbs': (self.total_carbs / targets.carb_g) * 100,
            'fat': (self.total_fat / targets.fat_g) * 100
        }
    
    def get_remaining_targets(self, targets: NutritionTargets) -> Dict[str, float]:
        """Calculate remaining amounts to reach targets."""
        return {
            'calories': max(0, targets.total_calories - self.total_calories),
            'protein': max(0, targets.protein_g - self.total_protein),
            'carbs': max(0, targets.carb_g - self.total_carbs),
            'fat': max(0, targets.fat_g - self.total_fat)
        }
