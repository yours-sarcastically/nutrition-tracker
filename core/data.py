# core/data.py
# Description: Load and process the food database and attach ranking emojis.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG


@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """Load the Vegetarian Food Database from a CSV file into FoodItem objects."""
    df = pd.read_csv(file_path)
    foods: Dict[str, List[FoodItem]] = {cat: [] for cat in CONFIG["nutrient_map"].keys()}

    for _, row in df.iterrows():
        category = row["category"]
        if category in foods:
            foods[category].append(
                FoodItem(
                    name=f"{row['name']} ({row['serving_unit']})",
                    calories=row["calories"],
                    protein=row["protein"],
                    carbs=row["carbs"],
                    fat=row["fat"],
                )
            )
    return foods


def assign_food_emojis(foods: Dict[str, List[FoodItem]]) -> Dict[str, List[FoodItem]]:
    """
    Assign an emoji to every food item based on its rank in:

        • Protein, Carbs, Fat (💪 / 🍚 / 🥑)
        • Calories (🔥)
        • Appearing in BOTH a top-nutrient leaderboard and the high-calorie list (🥇)
    """
    # ------------------------------------------------------------------
    # 1. Build leaderboards
    # ------------------------------------------------------------------
    top_foods = {
        "protein": [],
        "carbs": [],
        "fat": [],
        "calories": {},  # per-category top-3 lists
    }

    for category, items in foods.items():
        if not items:
            continue

        # Top-3 by calories (store per category – later flattened)
        sorted_by_cal = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods["calories"][category] = [food.name for food in sorted_by_cal[:3]]

        # Determine the nutrient that defines this category
        map_info = CONFIG["nutrient_map"].get(category)
        if not map_info:
            continue

        nutrient_key = map_info["key"]          # one of: protein / carbs / fat / micro
        if nutrient_key == "micro":
            # Micronutrient leaderboard removed
            continue

        sort_attr = map_info["sort_by"]         # attribute on FoodItem
        sorted_by_nutrient = sorted(
            items, key=lambda x: getattr(x, sort_attr), reverse=True
        )
        top_foods[nutrient_key] = [
            food.name for food in sorted_by_nutrient[:3]
        ]

    # ------------------------------------------------------------------
    # 2. Utility sets
    # ------------------------------------------------------------------
    nutrient_keys = ["protein", "carbs", "fat"]

    all_top_nutrient_foods = {
        food_name for key in nutrient_keys for food_name in top_foods[key]
    }

    high_calorie_set = {
        food_name
        for cal_list in top_foods["calories"].values()
        for food_name in cal_list
    }

    # ------------------------------------------------------------------
    # 3. Emoji mapping
    # ------------------------------------------------------------------
    emoji_mapping = {
        "combo": "🥇",        # high-calorie AND top nutrient
        "high_calorie": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
    }

    # ------------------------------------------------------------------
    # 4. Walk through every food and assign its emoji
    # ------------------------------------------------------------------
    for items in foods.values():
        for food in items:
            is_high_calorie = food.name in high_calorie_set
            is_top_nutrient = food.name in all_top_nutrient_foods

            if is_high_calorie and is_top_nutrient:
                food.emoji = emoji_mapping["combo"]            # 🥇
            elif is_high_calorie:
                food.emoji = emoji_mapping["high_calorie"]     # 🔥
            elif food.name in top_foods["protein"]:
                food.emoji = emoji_mapping["protein"]          # 💪
            elif food.name in top_foods["carbs"]:
                food.emoji = emoji_mapping["carbs"]            # 🍚
            elif food.name in top_foods["fat"]:
                food.emoji = emoji_mapping["fat"]              # 🥑
            else:
                food.emoji = ""                                # no emoji

    return foods
