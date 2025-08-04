# core/data.py
# Description: Functions for loading and processing the food database.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG


@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """Load the Vegetarian Food Database from a CSV file into FoodItem objects."""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in CONFIG['nutrient_map'].keys()}

    for _, row in df.iterrows():
        category = row['category']
        if category in foods:
            foods[category].append(
                FoodItem(
                    name=f"{row['name']} ({row['serving_unit']})",
                    calories=row['calories'],
                    protein=row['protein'],
                    carbs=row['carbs'],
                    fat=row['fat'],
                )
            )
    return foods


def assign_food_emojis(foods: Dict[str, List[FoodItem]]) -> Dict[str, List[FoodItem]]:
    """
    Assign an emoji to every food item based on its rank in:
        • Protein, Carbs, Fat, Micronutrients  (💪 / 🍚 / 🥑 / 🥦)
        • Calories (🔥)
        • Both nutrient-top & high-calorie (💥)
        • Appearing in ≥ 2 nutrient leaderboards (🥇)
    """
    # ------------------------------------------------------------------
    # 1. Build leaderboards
    # ------------------------------------------------------------------
    top_foods = {
        "protein": [],
        "carbs": [],
        "fat": [],
        "micro": [],
        "calories": {},  # per-category top-3 lists
    }

    for category, items in foods.items():
        if not items:
            continue

        # Top-3 by calories (store per category – later collapsed)
        sorted_by_cal = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods["calories"][category] = [food.name for food in sorted_by_cal[:3]]

        # Top-3 by the category’s key nutrient
        map_info = CONFIG["nutrient_map"].get(category)
        if map_info:
            nutrient_key = map_info["key"]          # protein / carbs / fat / micro
            sort_attr = map_info["sort_by"]         # attribute on FoodItem
            sorted_by_nutrient = sorted(
                items, key=lambda x: getattr(x, sort_attr), reverse=True
            )
            top_foods[nutrient_key] = [
                food.name for food in sorted_by_nutrient[:3]
            ]

    # ------------------------------------------------------------------
    # 2. Identify superfoods (appear in ≥2 nutrient leaderboards)
    # ------------------------------------------------------------------
    all_top_nutrient_foods = {
        food
        for key in ["protein", "carbs", "fat", "micro"]
        for food in top_foods[key]
    }
    nutrient_rank_count = {
        name: sum(
            1
            for key in ["protein", "carbs", "fat", "micro"]
            if name in top_foods[key]
        )
        for name in all_top_nutrient_foods
    }
    superfoods = {name for name, count in nutrient_rank_count.items() if count > 1}

    # ------------------------------------------------------------------
    # 3. Emoji mapping
    # ------------------------------------------------------------------
    emoji_mapping = {
        "superfoods": "🥇",
        "high_cal_nutrient": "💥",
        "high_calorie": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
        "micro": "🥦",
    }

    # Helper: flatten all high-calorie lists into one set
    high_calorie_set = {
        food_name
        for cal_list in top_foods["calories"].values()
        for food_name in cal_list
    }

    # ------------------------------------------------------------------
    # 4. Walk through every food and assign its emoji
    # ------------------------------------------------------------------
    for items in foods.values():
        for food in items:
            is_top_nutrient = food.name in all_top_nutrient_foods
            is_high_calorie = food.name in high_calorie_set

            if food.name in superfoods:
                food.emoji = emoji_mapping["superfoods"]        # 🥇
            elif is_high_calorie and is_top_nutrient:
                food.emoji = emoji_mapping["high_cal_nutrient"] # 💥
            elif is_high_calorie:
                food.emoji = emoji_mapping["high_calorie"]      # 🔥
            elif food.name in top_foods["protein"]:
                food.emoji = emoji_mapping["protein"]           # 💪
            elif food.name in top_foods["carbs"]:
                food.emoji = emoji_mapping["carbs"]             # 🍚
            elif food.name in top_foods["fat"]:
                food.emoji = emoji_mapping["fat"]               # 🥑
            elif food.name in top_foods["micro"]:
                food.emoji = emoji_mapping["micro"]             # 🥦
            else:
                food.emoji = ""  # fallback – no emoji

    return foods
