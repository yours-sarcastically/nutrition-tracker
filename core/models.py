# core/models.py
# Description: Data models for the application using dataclasses.

from dataclasses import dataclass, field

@dataclass
class FoodItem:
    """Represents a single food item from the database."""
    name: str
    calories: float
    protein: float
    carbs: float
    fat: float
    emoji: str = ""

@dataclass
class UserProfile:
    """Represents the user's core physical attributes."""
    age: int
    height_cm: float
    weight_kg: float
    sex: str
    activity_level: str

@dataclass
class AdvancedSettings:
    """Represents the user's advanced nutritional settings."""
    caloric_surplus: int
    protein_per_kg: float
    fat_percentage: float

@dataclass
class NutritionalTargets:
    """Represents all calculated nutritional targets for the user."""
    bmr: int
    tdee: int
    total_calories: int
    protein_g: float
    protein_calories: float
    fat_g: float
    fat_calories: float
    carb_g: float
    carb_calories: float
    target_weight_gain_per_week: float
    protein_percent: float = field(init=False)
    carb_percent: float = field(init=False)
    fat_percent: float = field(init=False)

    def __post_init__(self):
        """Calculate percentage-based fields after initialization."""
        if self.total_calories > 0:
            self.protein_percent = (self.protein_calories / self.total_calories) * 100
            self.carb_percent = (self.carb_calories / self.total_calories) * 100
            self.fat_percent = (self.fat_calories / self.total_calories) * 100
        else:
            self.protein_percent = self.carb_percent = self.fat_percent = 0
