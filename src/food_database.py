"""
Food database management and processing.
Handles loading, categorizing, and ranking food items.
"""

import pandas as pd
import streamlit as st
from typing import Dict, List
from src.data_models import FoodItem
from src.config import config

class FoodDatabase:
    """Handles food data loading and processing."""
    
    def __init__(self, file_path: str = None):
        """Initialize the food database."""
        self.file_path = file_path or config.FOOD_DATABASE_PATH
        self.foods = {}
        self.nutrient_map = config.NUTRIENT_CATEGORY_MAP
        self.emoji_order = config.EMOJI_ORDER
    
    @st.cache_data
    def load_foods(_self) -> Dict[str, List[FoodItem]]:
        """
        Load and process the food database, assigning categories and emojis.
        
        Returns:
            Dictionary mapping categories to lists of FoodItem objects
        """
        try:
            df = pd.read_csv(_self.file_path)
            if df.empty:
                st.error("Food database is empty")
                return {}
        except FileNotFoundError:
            st.error(f"Food database file '{_self.file_path}' not found")
            return {}
        except Exception as e:
            st.error(f"Error loading food database: {str(e)}")
            return {}
        
        # Initialize categories
        categories = df['category'].unique()
        foods = {cat: [] for cat in categories}
        
        # Process each food item
        for _, row in df.iterrows():
            food_item = FoodItem(
                name=row['name'],
                calories=row['calories'],
                protein=row['protein'],
                carbs=row['carbs'],
                fat=row['fat'],
                category=row['category'],
                serving_unit=row.get('serving_unit', '')
            )
            foods[row['category']].append(food_item)
        
        # Assign emojis based on nutritional ranking
        _self._assign_emojis(foods)
        
        return foods
    
    def _assign_emojis(self, foods: Dict[str, List[FoodItem]]) -> None:
        """Assign emojis to food items based on nutritional ranking."""
        # Find top foods by nutrient type
        top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}
        
        for category, items in foods.items():
            if not items:
                continue
            
            # Top foods by calories
            sorted_by_calories = sorted(items, key=lambda x: x.calories, reverse=True)
            top_foods['calories'][category] = [food.get_display_name() for food in sorted_by_calories[:3]]
            
            # Top foods by specific nutrients
            nutrient = self.nutrient_map.get(category)
            if nutrient:
                nutrient_attr = nutrient if nutrient != 'micro' else 'protein'  # Use protein for micronutrients
                sorted_by_nutrient = sorted(items, key=lambda x: getattr(x, nutrient_attr), reverse=True)
                top_foods[nutrient] = [food.get_display_name() for food in sorted_by_nutrient[:3]]
        
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
        
        # Assign emojis to each food item
        for category, items in foods.items():
            for food in items:
                food_name = food.get_display_name()
                is_top_nutrient = food_name in all_top_nutrient_foods
                is_high_calorie = food_name in top_foods['calories'].get(category, [])
                
                if food_name in superfoods:
                    food.emoji = '🥇'
                elif is_high_calorie and is_top_nutrient:
                    food.emoji = '💥'
                elif is_high_calorie:
                    food.emoji = '🔥'
                elif food_name in top_foods['protein']:
                    food.emoji = '💪'
                elif food_name in top_foods['carbs']:
                    food.emoji = '🍚'
                elif food_name in top_foods['fat']:
                    food.emoji = '🥑'
                elif food_name in top_foods['micro']:
                    food.emoji = '🥦'
                else:
                    food.emoji = ''
    
    def get_sorted_foods_by_category(self, category: str, foods: Dict[str, List[FoodItem]]) -> List[FoodItem]:
        """
        Get foods in a category sorted by emoji ranking and calories.
        
        Args:
            category: Food category name
            foods: Dictionary of all foods
            
        Returns:
            Sorted list of FoodItem objects
        """
        if category not in foods:
            return []
        
        return sorted(
            foods[category], 
            key=lambda x: (self.emoji_order.get(x.emoji, 4), -x.calories)
        )
