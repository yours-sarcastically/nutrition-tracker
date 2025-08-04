# core/data.py
# Description: Load the food database and assign ranking emojis.

from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG

# ---------------------------------------------------------------------
# Tunable constant – how many foods should receive the 🥇 emoji?
SUPERFOODS_TOP_N = 10
# ---------------------------------------------------------------------


@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """
    Load the Vegetarian Food Database from a CSV file and return
    {category: [FoodItem, …]}.
    """
    df = pd.read_csv(file_path)
    foods: Dict[str, List[FoodItem]] = {cat: [] for cat in CONFIG["nutrient_map"].keys()}

    for _, row in df.iterrows():
        category = row["category"]
        if category not in foods:
            continue
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


def _calculate_nutrient_density(food: FoodItem) -> float:
    """
    (protein*4 + carbs*4 + fat*9) / calories  →  kcal‐weighted nutrient density.
    Returns 0 when calories are absent or zero.
    """
    if not food.calories:
        return 0.0
    kcal_from_macros = food.protein * 4 + food.carbs * 4 + food.fat * 9
    return kcal_from_macros / food.calories


def assign_food_emojis(foods: Dict[str, List[FoodItem]]) -> Dict[str, List[FoodItem]]:
    """
    Mutates the FoodItem objects in `foods`, adding an `.emoji` attribute.

    Emoji legend
    ------------
    💪  Top-3 protein food in its category
    🍚  Top-3 carbs   food in its category
    🥑  Top-3 fat     food in its category
    🥦  Top-3 micro   food in its category  (mapped via CONFIG)
    🔥  Top-3 calories food in its category
    💥  Appears in *both* a nutrient leaderboard (💪/🍚/🥑/🥦) **and** 🔥
    🥇  One of the `SUPERFOODS_TOP_N` most nutrient-dense foods globally
    """
    # ------------------------------------------------------------------
    # 1. Build leaderboards (exactly as before)
    # ------------------------------------------------------------------
    top_foods = {
        "protein": [],
        "carbs": [],
        "fat": [],
        "micro": [],
        "calories": {},  # category → [food names]
    }

    for category, items in foods.items():
        if not items:
            continue

        # High-calorie (🔥) – per category
        sorted_by_cal = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods["calories"][category] = [f.name for f in sorted_by_cal[:3]]

        # Category’s key nutrient leaderboard (💪/🍚/🥑/🥦)
        map_info = CONFIG["nutrient_map"].get(category)
        if map_info:
            board_key = map_info["key"]      # protein / carbs / fat / micro
            sort_attr = map_info["sort_by"]  # the FoodItem attribute
            sorted_by_nutrient = sorted(
                items, key=lambda x: getattr(x, sort_attr), reverse=True
            )
            top_foods[board_key].extend(f.name for f in sorted_by_nutrient[:3])

    # ------------------------------------------------------------------
    # 2. Compute global Superfoods by nutrient-density score  (🥇)
    # ------------------------------------------------------------------
    all_items: List[FoodItem] = [
        food for category_items in foods.values() for food in category_items
    ]
    for food in all_items:
        food._nutrient_density = _calculate_nutrient_density(food)

    superfoods = {
        f.name
        for f in sorted(all_items, key=lambda x: x._nutrient_density, reverse=True)[
            : SUPERFOODS_TOP_N
        ]
    }

    # ------------------------------------------------------------------
    # 3. Emoji mapping dictionary
    # ------------------------------------------------------------------
    emoji = {
        "super": "🥇",
        "high_cal_nutrient": "💥",
        "high_cal": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
        "micro": "🥦",
    }

    # Flatten high-calorie lists into one set for quick membership tests
    high_calorie_set = {
        food_name
        for cal_list in top_foods["calories"].values()
        for food_name in cal_list
    }

    # ------------------------------------------------------------------
    # 4. Walk through every food and assign an emoji
    # ------------------------------------------------------------------
    for category_items in foods.values():
        for food in category_items:
            is_high_cal = food.name in high_calorie_set
            is_top_nutrient = any(
                food.name in top_foods[key] for key in ("protein", "carbs", "fat", "micro")
            )

            if food.name in superfoods:
                food.emoji = emoji["super"]                       # 🥇
            elif is_high_cal and is_top_nutrient:
                food.emoji = emoji["high_cal_nutrient"]           # 💥
            elif is_high_cal:
                food.emoji = emoji["high_cal"]                    # 🔥
            elif food.name in top_foods["protein"]:
                food.emoji = emoji["protein"]                     # 💪
            elif food.name in top_foods["carbs"]:
                food.emoji = emoji["carbs"]                       # 🍚
            elif food.name in top_foods["fat"]:
                food.emoji = emoji["fat"]                         # 🥑
            elif food.name in top_foods["micro"]:
                food.emoji = emoji["micro"]                       # 🥦
            else:
                food.emoji = ""                                   # no badge

            # Clean up helper attribute
            if hasattr(food, "_nutrient_density"):
                delattr(food, "_nutrient_density")

    return foods
