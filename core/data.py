# core/data.py
# Description: Loading the food database and assigning nutrient / calorie emojis.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------
@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """
    Load the Vegetarian Food Database from a CSV file and return a dict:
        {category_name: [FoodItem, …], …}
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


# ---------------------------------------------------------------------------
# Emoji assignment
# ---------------------------------------------------------------------------
def assign_food_emojis(foods: Dict[str, List[FoodItem]]) -> Dict[str, List[FoodItem]]:
    """
    Annotate every FoodItem with an emoji that reflects its nutritional standing.

    Rules (priority order):
    1. 🥇  Superfood               – appears in ≥ 2 of the four nutrient leaderboards
    2. 💥  High-calorie nutrient   – 🔥 + in any nutrient leaderboard
    3. 🔥  High-calorie            – top-3 calories in its category
    4. 💪 / 🍚 / 🥑 / 🥦             – top-3 in protein / carbs / fat / micro respectively
    """
    # ----------------------------------------------------------------------
    # 1. Build leaderboards
    # ----------------------------------------------------------------------
    top_foods = {
        "protein": set(),   # filled across ALL categories
        "carbs":   set(),
        "fat":     set(),
        "micro":   set(),
        "calories": {},     # per-category list
    }

    for category, items in foods.items():
        if not items:
            continue

        # --- top-3 by calories (per category) -----------------------------
        sorted_by_cal = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods["calories"][category] = [f.name for f in sorted_by_cal[:3]]

        # --- top-3 by the category's key nutrient ------------------------
        map_info = CONFIG["nutrient_map"].get(category)
        if not map_info:
            continue

        nutrient_key = map_info["key"]       # protein / carbs / fat / micro
        sort_attr    = map_info["sort_by"]   # FoodItem attribute

        sorted_by_nutrient = sorted(
            items, key=lambda x: getattr(x, sort_attr), reverse=True
        )
        top_foods[nutrient_key].update(
            f.name for f in sorted_by_nutrient[:3]
        )

    # Convert nutrient sets back to lists for eventual downstream use
    for k in ("protein", "carbs", "fat", "micro"):
        top_foods[k] = list(top_foods[k])

    # ----------------------------------------------------------------------
    # 2. Identify superfoods (≥ 2 nutrient leaderboards)
    # ----------------------------------------------------------------------
    all_nutrient_leaders = {
        name
        for k in ("protein", "carbs", "fat", "micro")
        for name in top_foods[k]
    }

    leaderboard_count = {
        name: sum(name in top_foods[k] for k in ("protein", "carbs", "fat", "micro"))
        for name in all_nutrient_leaders
    }
    superfoods = {name for name, cnt in leaderboard_count.items() if cnt >= 2}

    # ----------------------------------------------------------------------
    # 3. Flatten high-calorie winners
    # ----------------------------------------------------------------------
    high_calorie_set = {
        food_name
        for cal_list in top_foods["calories"].values()
        for food_name in cal_list
    }

    # ----------------------------------------------------------------------
    # 4. Emoji mapping
    # ----------------------------------------------------------------------
    EMOJI = {
        "super": "🥇",
        "high_cal_nutrient": "💥",
        "high_cal": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
        "micro": "🥦",
    }

    # ----------------------------------------------------------------------
    # 5. Walk through every FoodItem and assign its emoji
    # ----------------------------------------------------------------------
    for items in foods.values():
        for food in items:
            in_nutrient_lb = food.name in all_nutrient_leaders
            high_cal       = food.name in high_calorie_set

            if food.name in superfoods:
                food.emoji = EMOJI["super"]                      # 🥇
            elif high_cal and in_nutrient_lb:
                food.emoji = EMOJI["high_cal_nutrient"]          # 💥
            elif high_cal:
                food.emoji = EMOJI["high_cal"]                   # 🔥
            elif food.name in top_foods["protein"]:
                food.emoji = EMOJI["protein"]                    # 💪
            elif food.name in top_foods["carbs"]:
                food.emoji = EMOJI["carbs"]                      # 🍚
            elif food.name in top_foods["fat"]:
                food.emoji = EMOJI["fat"]                        # 🥑
            elif food.name in top_foods["micro"]:
                food.emoji = EMOJI["micro"]                      # 🥦
            else:
                food.emoji = ""

    return foods
