# core/data.py
# Description: Load the food database and assign nutrient / calorie emojis.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG


@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """Load the Vegetarian Food Database from a CSV file into FoodItem objects."""
    df = pd.read_csv(file_path)
    foods = {cat: [] for cat in CONFIG["nutrient_map"].keys()}

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
    Assign emojis to each food item.

        • Protein leaderboard  – 💪
        • Carbs leaderboard    – 🍚
        • Fat leaderboard      – 🥑
        • High-calorie (top-3 per category)                 – 🔥
        • High-calorie 𝘢𝘯𝘥 on any nutrient leaderboard      – 🥇
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

        # Top-3 by calories (per category)
        sorted_by_cal = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods["calories"][category] = [food.name for food in sorted_by_cal[:3]]

        # Top-3 by the category’s key macronutrient
        map_info = CONFIG["nutrient_map"].get(category)
        if map_info:
            nutrient_key = map_info["key"]          # protein / carbs / fat
            sort_attr = map_info["sort_by"]         # attribute on FoodItem
            if nutrient_key in {"protein", "carbs", "fat"}:
                sorted_by_nutrient = sorted(
                    items, key=lambda x: getattr(x, sort_attr), reverse=True
                )
                top_foods[nutrient_key] = [
                    food.name for food in sorted_by_nutrient[:3]
                ]

    # ------------------------------------------------------------------
    # 2. Convenience sets for quick lookup
    # ------------------------------------------------------------------
    nutrient_leaderboard_set = {
        food for key in ["protein", "carbs", "fat"] for food in top_foods[key]
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
        "combo": "🥇",        # high-calorie + nutrient-leaderboard
        "high_calorie": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
    }

    # ------------------------------------------------------------------
    # 4. Assign emojis
    # ------------------------------------------------------------------
    for items in foods.values():
        for food in items:
            is_top_nutrient = food.name in nutrient_leaderboard_set
            is_high_calorie = food.name in high_calorie_set

            if is_high_calorie and is_top_nutrient:
                food.emoji = emoji_mapping["combo"]          # 🥇
            elif is_high_calorie:
                food.emoji = emoji_mapping["high_calorie"]   # 🔥
            elif food.name in top_foods["protein"]:
                food.emoji = emoji_mapping["protein"]        # 💪
            elif food.name in top_foods["carbs"]:
                food.emoji = emoji_mapping["carbs"]          # 🍚
            elif food.name in top_foods["fat"]:
                food.emoji = emoji_mapping["fat"]            # 🥑
            else:
                food.emoji = ""  # no badge

    return foods
