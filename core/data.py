# core/data.py
# Description: Functions for loading and processing the food database.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG

@st.cache_data
def get_processed_food_data(file_path: str) -> Dict[str, List[FoodItem]]:
    """
    Loads the food database from a CSV, processes it into FoodItem objects,
    and assigns ranking emojis in a single, cacheable operation.
    """
    # --- Part 1: Load the database from CSV ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{file_path}'. Please ensure the file exists and the path is correct.")
        return {}

    foods: Dict[str, List[FoodItem]] = {cat: [] for cat in CONFIG['nutrient_map'].keys()}
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

    # --- Part 2: Assign Emojis (The logic from the previous function) ---
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}
    
    for category, items in foods.items():
        if not items: continue
        
        sorted_by_calories = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods['calories'][category] = [food.name for food in sorted_by_calories[:3]]
        
        map_info = CONFIG['nutrient_map'].get(category)
        if map_info:
            sorted_by_nutrient = sorted(items, key=lambda x: getattr(x, map_info['sort_by']), reverse=True)
            top_foods[map_info['key']] = [food.name for food in sorted_by_nutrient[:3]]

    all_top_foods = {food for key in ['protein', 'carbs', 'fat', 'micro'] for food in top_foods[key]}
    food_rank_counts = {name: sum(1 for key in ['protein', 'carbs', 'fat', 'micro'] if name in top_foods[key]) for name in all_top_foods}
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    emoji_mapping = {'superfoods': '🥇', 'high_cal_nutrient': '💥', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑', 'micro': '🥦'}
    
    for category, items in foods.items():
        for food in items:
            is_top_nutrient = food.name in all_top_foods
            is_high_calorie = food.name in top_foods['calories'].get(category, [])

            if food.name in superfoods: food.emoji = emoji_mapping['superfoods']
            elif is_high_calorie and is_top_nutrient: food.emoji = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: food.emoji = emoji_mapping['high_calorie']
            elif food.name in top_foods['protein']: food.emoji = emoji_mapping['protein']
            elif food.name in top_foods['carbs']: food.emoji = emoji_mapping['carbs']
            elif food.name in top_foods['fat']: food.emoji = emoji_mapping['fat']
            elif food.name in top_foods['micro']: food.emoji = emoji_mapping['micro']
            else: food.emoji = ''
            
    return foods
