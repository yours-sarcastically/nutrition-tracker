"""Food database management and processing."""

import pandas as pd
import logging
from typing import Dict, List, Set
from pathlib import Path
from .models import FoodItem
from .config import NutritionConfig

logger = logging.getLogger(__name__)

class FoodDatabase:
    """Manages food data loading, processing, and categorization."""
    
    def __init__(self, file_path: Path = None, config: NutritionConfig = None):
        """
        Initialize food database.
        
        Args:
            file_path: Path to nutrition database CSV file
            config: Configuration object
        """
        self.config = config or NutritionConfig()
        self.file_path = file_path or self.config.NUTRITION_DB_FILE
        self.foods: Dict[str, List[FoodItem]] = {}
        self._load_and_process_foods()
    
    def _load_and_process_foods(self) -> None:
        """Load and process foods from CSV file."""
        try:
            logger.info(f"Loading food database from {self.file_path}")
            df = pd.read_csv(self.file_path)
            
            # Validate required columns
            required_columns = ['name', 'category', 'calories', 'protein', 'carbs', 'fat', 'serving_unit']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Initialize category dictionary
            categories = df['category'].unique()
            self.foods = {cat: [] for cat in categories}
            
            # Process each food item
            for _, row in df.iterrows():
                try:
                    food_item = FoodItem(
                        name=row['name'],
                        calories=float(row['calories']),
                        protein=float(row['protein']),
                        carbs=float(row['carbs']),
                        fat=float(row['fat']),
                        serving_unit=str(row['serving_unit']),
                        category=row['category']
                    )
                    self.foods[row['category']].append(food_item)
                except Exception as e:
                    logger.warning(f"Error processing food item {row.get('name', 'unknown')}: {e}")
                    continue
            
            # Assign emojis based on nutritional ranking
            self._assign_food_emojis()
            
            logger.info(f"Successfully loaded {sum(len(items) for items in self.foods.values())} food items")
            
        except FileNotFoundError:
            logger.error(f"Food database file not found: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading food database: {e}")
            raise
    
    def _assign_food_emojis(self) -> None:
        """Assign emojis to foods based on nutritional ranking."""
        logger.debug("Assigning food emojis based on nutritional ranking")
        
        # Find top foods for each nutrient
        top_foods = {
            'protein': [],
            'carbs': [],
            'fat': [],
            'micro': [],
            'calories': {}
        }
        
        # Get top foods by nutrient for each category
        for category, items in self.foods.items():
            if not items:
                continue
            
            # Top calorie foods by category
            sorted_by_calories = sorted(items, key=lambda x: x.calories, reverse=True)
            top_foods['calories'][category] = [food.name for food in sorted_by_calories[:3]]
            
            # Top nutrient foods based on category mapping
            nutrient = self.config.NUTRIENT_CATEGORY_MAP.get(category)
            if nutrient and hasattr(items[0], nutrient):
                sorted_by_nutrient = sorted(items, key=lambda x: getattr(x, nutrient), reverse=True)
                top_foods[nutrient].extend([food.name for food in sorted_by_nutrient[:3]])
        
        # Identify superfoods (high in multiple nutrients)
        all_top_nutrient_foods = set(
            top_foods['protein'] + top_foods['carbs'] + 
            top_foods['fat'] + top_foods['micro']
        )
        
        food_rank_counts = {}
        for food_name in all_top_nutrient_foods:
            count = sum(1 for nutrient_list in ['protein', 'carbs', 'fat', 'micro'] 
                       if food_name in top_foods[nutrient_list])
            food_rank_counts[food_name] = count
        
        superfoods = {name for name, count in food_rank_counts.items() if count > 1}
        
        # Assign emojis
        for category, items in self.foods.items():
            for food in items:
                food.emoji = self._determine_food_emoji(
                    food, category, superfoods, top_foods, all_top_nutrient_foods
                )
    
    def _determine_food_emoji(
        self, 
        food: FoodItem, 
        category: str, 
        superfoods: Set[str],
        top_foods: Dict[str, List[str]], 
        all_top_nutrient_foods: Set[str]
    ) -> str:
        """Determine appropriate emoji for a food item."""
        food_name = food.name
        is_top_nutrient = food_name in all_top_nutrient_foods
        is_high_calorie = food_name in top_foods['calories'].get(category, [])
        
        if food_name in superfoods:
            return self.config.EMOJI_CONFIG['superfood']
        elif is_high_calorie and is_top_nutrient:
            return self.config.EMOJI_CONFIG['nutrient_calorie_dense']
        elif is_high_calorie:
            return self.config.EMOJI_CONFIG['high_calorie']
        elif food_name in top_foods['protein']:
            return self.config.EMOJI_CONFIG['top_protein']
        elif food_name in top_foods['carbs']:
            return self.config.EMOJI_CONFIG['top_carb']
        elif food_name in top_foods['fat']:
            return self.config.EMOJI_CONFIG['top_fat']
        elif food_name in top_foods['micro']:
            return self.config.EMOJI_CONFIG['top_micronutrient']
        else:
            return self.config.EMOJI_CONFIG['default']
    
    def get_foods_by_category(self) -> Dict[str, List[FoodItem]]:
        """Get all foods organized by category."""
        return self.foods
    
    def get_sorted_foods_by_category(self, category: str) -> List[FoodItem]:
        """
        Get foods in a category sorted by emoji priority and calories.
        
        Args:
            category: Food category name
            
        Returns:
            Sorted list of food items
        """
        if category not in self.foods:
            return []
        
        return sorted(
            self.foods[category],
            key=lambda x: (self.config.EMOJI_ORDER.get(x.emoji, 4), -x.calories)
        )
    
    def get_available_categories(self) -> List[str]:
        """Get list of available food categories with items."""
        return sorted([cat for cat, items in self.foods.items() if items])
    
    def search_foods(self, query: str) -> List[FoodItem]:
        """
        Search for foods by name across all categories.
        
        Args:
            query: Search query string
            
        Returns:
            List of matching food items
        """
        query_lower = query.lower()
        results = []
        
        for category, items in self.foods.items():
            for food in items:
                if query_lower in food.name.lower():
                    results.append(food)
        
        return results
    
    def get_food_by_name(self, name: str) -> FoodItem:
        """
        Get a specific food item by name.
        
        Args:
            name: Food name to search for
            
        Returns:
            Food item if found
            
        Raises:
            ValueError: If food not found
        """
        for category, items in self.foods.items():
            for food in items:
                if food.name == name:
                    return food
        
        raise ValueError(f"Food not found: {name}")
