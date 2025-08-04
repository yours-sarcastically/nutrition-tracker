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
    and assigns ranking emojis in a single, cacheable operation using corrected logic.
    """
    # --- Part 1: Load the database from CSV ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{file_path}'. Please ensure the file exists and the path is correct.")
        return {}

    foods: Dict[str, List[FoodItem]] = {cat: [] for cat in CONFIG['nutrient_map'].keys()}
    all_foods_flat: List[FoodItem] = []

    for _, row in df.iterrows():
        category = row['category']
        food_item = FoodItem(
            name=f"{row['name']} ({row['serving_unit']})",
            calories=row['calories'],
            protein=row['protein'],
            carbs=row['carbs'],
            fat=row['fat']
        )
        if category in foods:
            foods[category].append(food_item)
        all_foods_flat.append(food_item)

    # --- Part 2: Assign Emojis (with Final Corrected Logic) ---
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'calories': {}}

    if all_foods_flat:
        top_foods['protein'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.protein, reverse=True)[:3]]
        top_foods['carbs'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.carbs, reverse=True)[:3]]
        top_foods['fat'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.fat, reverse=True)[:3]]

    for category, items in foods.items():
        top_foods['calories'][category] = [f.name for f in sorted(items, key=lambda x: x.calories, reverse=True)[:3]]

    # FINAL FIX: Superfood calculation now only considers the 3 distinct macronutrient lists.
    macronutrient_keys = ['protein', 'carbs', 'fat']
    all_top_macronutrient_foods = {food for key in macronutrient_keys for food in top_foods[key]}
    food_rank_counts = {name: sum(1 for key in macronutrient_keys if name in top_foods[key]) for name in all_top_macronutrient_foods}
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    # The 'micro' key is still needed for basic emoji assignment, but it's not part of the superfood logic.
    # It is populated by the top protein sources from the "Micronutrient Sources" category specifically.
    micro_items = foods.get('PRIMARY MICRONUTRIENT SOURCES', [])
    top_foods['micro'] = [f.name for f in sorted(micro_items, key=lambda x: x.protein, reverse=True)[:3]]

    emoji_mapping = {'superfoods': '🥇', 'high_cal_nutrient': '💥', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑', 'micro': '🥦'}

    for category, items in foods.items():
        for food in items:
            is_top_nutrient = food.name in all_top_macronutrient_foods
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
