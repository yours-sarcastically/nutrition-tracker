# core/data.py
# Description: Load the database and attach nutrient / calorie emojis to every food.

import pandas as pd
import streamlit as st
from typing import Dict, List
from .models import FoodItem
from config import CONFIG


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load the CSV into FoodItem objects
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[FoodItem]]:
    """Return {category: [FoodItem, …]} from the CSV file."""
    df = pd.read_csv(file_path)
    foods: Dict[str, List[FoodItem]] = {cat: [] for cat in CONFIG["nutrient_map"].keys()}

    for _, row in df.iterrows():
        cat = row["category"]
        if cat not in foods:
            continue
        foods[cat].append(
            FoodItem(
                name=f"{row['name']} ({row['serving_unit']})",
                calories=row["calories"],
                protein=row["protein"],
                carbs=row["carbs"],
                fat=row["fat"],
            )
        )
    return foods


# ─────────────────────────────────────────────────────────────────────────────
# 2. Emoji assignment
# ─────────────────────────────────────────────────────────────────────────────
def assign_food_emojis(foods: Dict[str, List[FoodItem]]) -> Dict[str, List[FoodItem]]:
    """
    Attach an emoji to every FoodItem.

    Priority:
      🥇  Superfood              – appears in ≥ 2 of the four nutrient leaderboards
      💥  High-calorie nutrient  – 🔥 AND in any nutrient leaderboard
      🔥  High-calorie           – top-3 calories *within its own category*
      💪  Protein top-3 (global)
      🍚  Carbs   top-3 (global)
      🥑  Fat     top-3 (global)
      🥦  Micro   top-3 (global)
    """
    # ------------------------------------------------------------------
    # 2-A. Collect a flat list of every FoodItem for global sorting
    # ------------------------------------------------------------------
    all_items: List[FoodItem] = [f for lst in foods.values() for f in lst]

    # Mapping nutrient key → FoodItem attribute
    nutrient_attr = {
        "protein": "protein",
        "carbs":   "carbs",
        "fat":     "fat",
        "micro":   "micro_score",   # depends on your model—adjust if needed
    }

    # ------------------------------------------------------------------
    # 2-B. Build GLOBAL top-3 leaderboards for protein / carbs / fat / micro
    # ------------------------------------------------------------------
    TOP_N = 3
    top_foods = {
        "protein": [
            itm.name
            for itm in sorted(all_items, key=lambda x: x.protein, reverse=True)[:TOP_N]
        ],
        "carbs": [
            itm.name
            for itm in sorted(all_items, key=lambda x: x.carbs, reverse=True)[:TOP_N]
        ],
        "fat": [
            itm.name
            for itm in sorted(all_items, key=lambda x: x.fat, reverse=True)[:TOP_N]
        ],
        "micro": [
            itm.name
            for itm in sorted(all_items, key=lambda x: getattr(x, "micro", 0), reverse=True)[:TOP_N]
        ],
        "calories": {},   # per-category high-cal lists filled next
    }

    # ------------------------------------------------------------------
    # 2-C. Top-3 by calories *per category*
    # ------------------------------------------------------------------
    for cat, items in foods.items():
        sorted_by_cal = sorted(items, key=lambda x: x.calories, reverse=True)
        top_foods["calories"][cat] = [f.name for f in sorted_by_cal[:TOP_N]]

    # ------------------------------------------------------------------
    # 2-D. Derive helper sets
    # ------------------------------------------------------------------
    all_nutrient_leaders = {
        name for k in ("protein", "carbs", "fat", "micro") for name in top_foods[k]
    }

    # food → how many nutrient leaderboards it is in
    leaderboard_count = {
        name: sum(name in top_foods[k] for k in ("protein", "carbs", "fat", "micro"))
        for name in all_nutrient_leaders
    }
    superfoods = {n for n, cnt in leaderboard_count.items() if cnt >= 2}

    high_calorie_set = {
        n for lst in top_foods["calories"].values() for n in lst
    }

    # ------------------------------------------------------------------
    # 2-E. Emoji lookup table
    # ------------------------------------------------------------------
    EMOJI = {
        "super": "🥇",
        "high_cal_nutrient": "💥",
        "high_cal": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
        "micro": "🥦",
    }

    # ------------------------------------------------------------------
    # 2-F. Walk through every FoodItem and assign its emoji
    # ------------------------------------------------------------------
    for item in all_items:
        in_nutrient = item.name in all_nutrient_leaders
        high_cal    = item.name in high_calorie_set

        if item.name in superfoods:
            item.emoji = EMOJI["super"]
        elif high_cal and in_nutrient:
            item.emoji = EMOJI["high_cal_nutrient"]
        elif high_cal:
            item.emoji = EMOJI["high_cal"]
        elif item.name in top_foods["protein"]:
            item.emoji = EMOJI["protein"]
        elif item.name in top_foods["carbs"]:
            item.emoji = EMOJI["carbs"]
        elif item.name in top_foods["fat"]:
            item.emoji = EMOJI["fat"]
        elif item.name in top_foods["micro"]:
            item.emoji = EMOJI["micro"]
        else:
            item.emoji = ""

    return foods
