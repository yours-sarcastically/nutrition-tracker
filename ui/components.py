# ui/components.py
# Description: Reusable Streamlit UI components.

import streamlit as st
import pandas as pd
from typing import Dict, List, Tuple
from config import CONFIG
from core.models import FoodItem, NutritionalTargets

def display_metrics_grid(metrics_data: List[Tuple], num_columns: int = 4):
    """Display metrics in a configurable column layout."""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                st.metric(label=metric_info[0], value=metric_info[1])
            elif len(metric_info) == 3:
                st.metric(label=metric_info[0], value=metric_info[1], delta=metric_info[2])

def create_progress_tracking(totals: Dict, targets: NutritionalTargets) -> List[str]:
    """Create unified progress tracking with bars and recommendations."""
    recommendations = []
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    
    purpose_map = {
        'calories': 'to reach your weight gain target', 'protein': 'for muscle building',
        'carbs': 'for energy and performance', 'fat': 'for hormone production'
    }
    
    for nutrient, config in CONFIG['nutrient_configs'].items():
        actual = totals[nutrient]
        target_val = getattr(targets, config['target_key'])
        
        percent = min(actual / target_val * 100, 100) if target_val > 0 else 0
        st.progress(percent / 100, text=f"{config['label']}: {percent:.0f}% of daily target ({target_val:.0f} {config['unit']})")
        
        if actual < target_val:
            deficit = target_val - actual
            purpose = purpose_map.get(nutrient, 'for optimal nutrition')
            recommendations.append(f"• You need {deficit:.0f} more {config['unit']} of {config['label'].lower()} {purpose}.")
    
    return recommendations

def render_food_item(food: FoodItem, category: str):
    """Render a single food item with interaction controls."""
    st.subheader(f"{food.emoji} {food.name}")
    key = f"{category}_{food.name}"
    current_serving = st.session_state.food_selections.get(food.name, 0.0)
    
    button_cols = st.columns(5)
    for k in range(1, 6):
        with button_cols[k - 1]:
            button_type = "primary" if current_serving == float(k) else "secondary"
            if st.button(f"{k}", key=f"{key}_{k}", type=button_type, help=f"Set to {k} servings"):
                st.session_state.food_selections[food.name] = float(k)
                st.rerun()
    
    custom_serving = st.number_input(
        "Custom Number of Servings:", min_value=0.0, max_value=10.0,
        value=float(current_serving), step=0.1, key=f"{key}_custom"
    )
    
    if custom_serving != current_serving:
        if custom_serving > 0:
            st.session_state.food_selections[food.name] = custom_serving
        elif food.name in st.session_state.food_selections:
            del st.session_state.food_selections[food.name]
        st.rerun()
    
    st.caption(f"Per Serving: {food.calories:.0f} kcal | {food.protein:.1f} g protein | {food.carbs:.1f} g carbs | {food.fat:.1f} g fat")

def render_food_grid(items: List[FoodItem], category: str, columns: int = 2):
    """Render food items in a grid layout."""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)

def display_results_summary(totals: Dict, selected_foods: List[Dict], targets: NutritionalTargets):
    """Display the entire summary section after calculation."""
    st.header("Summary of Daily Nutritional Intake 📊")

    if selected_foods:
        st.subheader("Foods Logged for Today 🥣")
        cols = st.columns(3)
        for i, item in enumerate(selected_foods):
            with cols[i % 3]:
                st.write(f"• {item['food'].emoji} {item['food'].name} × {item['servings']:.1f}")
    else:
        st.info("No foods have been selected for today. 🍽️")

    st.subheader("Total Nutritional Intake for the Day 📈")
    intake_metrics = []
    for nutrient, config in CONFIG['nutrient_configs'].items():
        label = f"Total {config['label']} Consumed"
        value_format = "{:.0f}" if nutrient == 'calories' else "{:.1f}"
        value_str = f"{value_format.format(totals[nutrient])} {config['unit']}"
        intake_metrics.append((label, value_str))
    display_metrics_grid(intake_metrics, 4)

    recommendations = create_progress_tracking(totals, targets)

    st.subheader("Personalized Recommendations 💡")
    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("All daily nutritional targets have been met. Keep up the good work! 🎉")

    st.subheader("Daily Caloric Balance ⚖️")
    cal_balance = totals['calories'] - targets.tdee
    if cal_balance > 0:
        st.info(f"📈 You are consuming {cal_balance:.0f} kcal above maintenance, supporting weight gain.")
    else:
        st.warning(f"📉 You are consuming {abs(cal_balance):.0f} kcal below maintenance.")

    if selected_foods:
        st.subheader("Detailed Food Log 📋")
        food_log_data = [{
            'Food Item Name': f"{item['food'].emoji} {item['food'].name}", 'Servings': item['servings'],
            'Calories': item['food'].calories * item['servings'], 'Protein (g)': item['food'].protein * item['servings'],
            'Carbs (g)': item['food'].carbs * item['servings'], 'Fat (g)': item['food'].fat * item['servings']
        } for item in selected_foods]
        df_log = pd.DataFrame(food_log_data)
        st.dataframe(df_log.style.format({
            'Servings': '{:.1f}', 'Calories': '{:.0f}', 'Protein (g)': '{:.1f}',
            'Carbs (g)': '{:.1f}', 'Fat (g)': '{:.1f}'
        }), use_container_width=True)
    st.markdown("---")
