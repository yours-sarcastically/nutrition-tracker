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
    
    # Collect all foods across categories for global ranking
    all_foods = []
    for category, items in foods.items():
        for food in items:
            all_foods.append((food, category))
    
    if not all_foods:
        return foods
    
    # Global rankings across all foods
    all_foods_sorted_calories = sorted(all_foods, key=lambda x: x[0].calories, reverse=True)
    all_foods_sorted_protein = sorted(all_foods, key=lambda x: x[0].protein, reverse=True)
    all_foods_sorted_carbs = sorted(all_foods, key=lambda x: x[0].carbs, reverse=True)
    all_foods_sorted_fat = sorted(all_foods, key=lambda x: x[0].fat, reverse=True)
    
    # Get top foods for each category (top 20% or minimum 3)
    top_count = max(3, len(all_foods) // 5)
    
    top_calorie_foods = {food.name for food, _ in all_foods_sorted_calories[:top_count]}
    top_protein_foods = {food.name for food, _ in all_foods_sorted_protein[:top_count]}
    top_carb_foods = {food.name for food, _ in all_foods_sorted_carbs[:top_count]}
    top_fat_foods = {food.name for food, _ in all_foods_sorted_fat[:top_count]}
    
    # Identify superfoods (foods that appear in multiple top categories)
    all_nutrient_categories = [top_protein_foods, top_carb_foods, top_fat_foods]
    superfoods = set()
    
    for food_name in top_protein_foods | top_carb_foods | top_fat_foods:
        category_count = sum(1 for category_set in all_nutrient_categories if food_name in category_set)
        if category_count >= 2:  # Appears in 2+ nutrient categories
            superfoods.add(food_name)
    
    # Assign emojis based on priority
    for category, items in foods.items():
        for food in items:
            food_name = food.name
            
            # Priority 1: Superfoods (top in multiple nutrients)
            if food_name in superfoods:
                food.emoji = '🥇'
            
            # Priority 2: High-calorie AND nutrient-dense
            elif (food_name in top_calorie_foods and 
                  food_name in (top_protein_foods | top_carb_foods | top_fat_foods)):
                food.emoji = '💥'
            
            # Priority 3: High-calorie only
            elif food_name in top_calorie_foods:
                food.emoji = '🔥'
            
            # Priority 4: Top in specific nutrients
            elif food_name in top_protein_foods:
                food.emoji = '💪'
            elif food_name in top_carb_foods:
                food.emoji = '🍚'
            elif food_name in top_fat_foods:
                food.emoji = '🥑'
            
            # Priority 5: Category-specific micronutrient sources
            elif category == 'PRIMARY MICRONUTRIENT SOURCES':
                food.emoji = '🥦'
            
            # No emoji for other foods
            else:
                food.emoji = ''
    
    return foods

