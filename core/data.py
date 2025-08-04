# core/data.py
# Description: Functions for loading and processing the food database.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG

@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """Load the Vegetarian Food Database From a CSV File into FoodItem objects."""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in CONFIG['nutrient_map'].keys()}

    for _, row in df.iterrows():
        category = row['category']
        if category in foods:
            foods[category].append(FoodItem(
                name=f"{row['name']} ({row['serving_unit']})",
                calories=row['calories'],
                protein=row['protein'],
                carbs=row['carbs'],
                fat=row['fat']
            ))
    return foods

def assign_food_emojis(foods: Dict[str, List[FoodItem]]) -> Dict[str, List[FoodItem]]:
    """Assign emojis to foods using unified ranking system."""
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}
    # FIX: Create a single set to hold all high-calorie foods for easy lookup.
    all_high_calorie_foods = set()
    
    # First, identify all top-ranking foods across all categories.
    for category, items in foods.items():
        if not items: continue
        
        # Identify top 3 calorie-dense foods within the category
        sorted_by_calories = sorted(items, key=lambda x: x.calories, reverse=True)
        top_calorie_foods_in_cat = [food.name for food in sorted_by_calories[:3]]
        
        # FIX: Add these foods to our unified set of high-calorie foods.
        for food_name in top_calorie_foods_in_cat:
            all_high_calorie_foods.add(food_name)
        
        # Identify top 3 nutrient-dense foods within the category
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(items, key=lambda x: getattr(x, map_info['sort_by']), reverse=True)
            top_foods[map_info['key']] = [food.name for food in sorted_by_nutrient[:3]]

    # Create a set of all foods that are top in any nutrient category
    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat', 'micro'] for food in top_foods[key]}
    
    # Identify "superfoods" (top in more than one nutrient category)
    food_rank_counts = {name: sum(1 for key in ['protein', 'carbs', 'fat', 'micro'] if name in top_foods[key]) for name in all_top_nutrient_foods}
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    emoji_mapping = {'superfoods': '🥇', 'high_cal_nutrient': '💥', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑', 'micro': '🥦'}
    
    # Now, iterate through all foods and assign the correct emoji based on our findings.
    for items in foods.values():
        for food in items:
            is_top_nutrient = food.name in all_top_nutrient_foods
            # FIX: Check against the unified set of high-calorie foods.
            is_high_calorie = food.name in all_high_calorie_foods

            if food.name in superfoods:
                food.emoji = emoji_mapping['superfoods']
            elif is_high_calorie and is_top_nutrient:
                food.emoji = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie:
                food.emoji = emoji_mapping['high_calorie']
            elif food.name in top_foods['protein']:
                food.emoji = emoji_mapping['protein']
            elif food.name in top_foods['carbs']:
                food.emoji = emoji_mapping['carbs']
            elif food.name in top_foods['fat']:
                food.emoji = emoji_mapping['fat']
            elif food.name in top_foods['micro']:
                food.emoji = emoji_mapping['micro']
            else:
                food.emoji = ''
                
    return foods
