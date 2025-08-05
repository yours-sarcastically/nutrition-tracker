# -----------------------------------------------------------------------------
# Personalized Evidence-Based Vegetarian Nutrition Tracker Using Streamlit
# -----------------------------------------------------------------------------

"""
Overview
    This script builds an interactive Streamlit application that calculates
    personalized vegetarian nutrition targets and tracks daily intake.  Basal
    Metabolic Rate is estimated with the Mifflin-St Jeor equation, then
    multiplied by an activity factor to determine Total Daily Energy
    Expenditure.  Goal-specific caloric adjustments are applied for weight
    loss, maintenance, or gain.  Protein, fat, and carbohydrate targets are
    calculated with a protein-first approach that follows evidence-based
    sports-nutrition guidelines.

Implementation Steps
    1.  Collect user data through a unified sidebar form.
    2.  Compute BMR, TDEE, and macronutrient targets.
    3.  Load a vegetarian food database and assign emoji-based rankings.
    4.  Let users log food servings; aggregate intake in real time.
    5.  Display progress bars, donut charts, and metric grids.
    6.  Provide individualized recommendations and evidence summaries.
    7.  Offer a reset button and lightweight session-state management.

Command-Line Usage
    streamlit run nutrition_tracker.py

Available Sidebar Fields
    •   Age (Years)                       – Integer between 16 and 80  
    •   Height (Centimeters)              – Integer between 140 and 220  
    •   Weight (kg)                       – Float between 40 and 150  
    •   Sex                               – Male or Female  
    •   Activity Level                    – Sedentary … Extremely Active  
    •   Nutrition Goal                    – Weight Loss | Maintenance | Gain  
    •   Protein (g Per Kilogram) ⚙        – Advanced; 1.2 – 3.0  
    •   Fat (Percent Of Total Calories) ⚙ – Advanced; 15 – 40 %

The script is divided into notebook-style sections that can be pasted into a
Jupyter Notebook.  Each cell is clearly marked for easy navigation.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries And Modules
# -----------------------------------------------------------------------------

import math
from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ------ Purpose Of Imports ------
# math            – Lightweight arithmetic helpers
# typing          – Type hints for clarity
# pandas          – CSV loading and DataFrame utilities
# plotly.graph_objects – Interactive charts (donut)
# streamlit       – Web-app framework


# -----------------------------------------------------------------------------
# Cell 2: Page Configuration And Initial Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Personalized Nutrition Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------ User Experience Note ------
# Wide layout offers more horizontal space for metric grids and charts


# -----------------------------------------------------------------------------
# Cell 3: Unified Configuration Constants
# -----------------------------------------------------------------------------

# ------ Default Parameter Values Based On Published Research ------
DEFAULTS: Dict[str, Any] = {
    "age": 26,
    "height_cm": 180,
    "weight_kg": 57.5,
    "sex": "Male",
    "activity_level": "moderately_active",
    "goal": "weight_gain",
    "protein_per_kg": 2.0,
    "fat_percentage": 0.25,
}

# ------ Activity Level Multipliers For TDEE Calculation ------
ACTIVITY_MULTIPLIERS: Dict[str, float] = {
    "sedentary": 1.2,
    "lightly_active": 1.375,
    "moderately_active": 1.55,
    "very_active": 1.725,
    "extremely_active": 1.9,
}

# ------ Activity Level Descriptions ------
ACTIVITY_DESCRIPTIONS: Dict[str, str] = {
    "sedentary": "Little to no exercise, desk job",
    "lightly_active": "Light exercise one to three days per week",
    "moderately_active": "Moderate exercise three to five days per week",
    "very_active": "Heavy exercise six to seven days per week",
    "extremely_active": (
        "Very heavy exercise, physical job, or training twice per day"
    ),
}

# ------ Goal-Specific Targets Based On Evidence-Based Guide ------
GOAL_TARGETS: Dict[str, Dict[str, float]] = {
    "weight_loss": {
        "caloric_adjustment": -0.20,  # −20 percent below TDEE
        "protein_per_kg": 1.8,
        "fat_percentage": 0.25,
    },
    "weight_maintenance": {
        "caloric_adjustment": 0.0,  # maintenance
        "protein_per_kg": 1.6,
        "fat_percentage": 0.30,
    },
    "weight_gain": {
        "caloric_adjustment": 0.10,  # +10 percent above TDEE
        "protein_per_kg": 2.0,
        "fat_percentage": 0.25,
    },
}

# ------ Unified Configuration For All App Components ------
CONFIG: Dict[str, Any] = {
    "emoji_order": {"🥇": 1, "🔥": 2, "💪": 3, "🍚": 3, "🥑": 3, "": 4},
    "nutrient_map": {
        "PRIMARY PROTEIN SOURCES": {"sort_by": "protein", "key": "protein"},
        "PRIMARY CARBOHYDRATE SOURCES": {"sort_by": "carbs", "key": "carbs"},
        "PRIMARY FAT SOURCES": {"sort_by": "fat", "key": "fat"},
    },
    "nutrient_configs": {
        "calories": {
            "unit": "kcal",
            "label": "Calories",
            "target_key": "total_calories",
        },
        "protein": {
            "unit": "g",
            "label": "Protein",
            "target_key": "protein_g",
        },
        "carbs": {
            "unit": "g",
            "label": "Carbohydrates",
            "target_key": "carb_g",
        },
        "fat": {"unit": "g", "label": "Fat", "target_key": "fat_g"},
    },
    "form_fields": {
        "age": {
            "type": "number",
            "label": "Age (Years)",
            "min": 16,
            "max": 80,
            "step": 1,
            "placeholder": "Enter your age",
            "required": True,
        },
        "height_cm": {
            "type": "number",
            "label": "Height (Centimeters)",
            "min": 140,
            "max": 220,
            "step": 1,
            "placeholder": "Enter your height",
            "required": True,
        },
        "weight_kg": {
            "type": "number",
            "label": "Weight (kg)",
            "min": 40.0,
            "max": 150.0,
            "step": 0.5,
            "placeholder": "Enter your weight",
            "required": True,
        },
        "sex": {
            "type": "selectbox",
            "label": "Sex",
            "options": ["Select Sex", "Male", "Female"],
            "required": True,
            "placeholder": "Select Sex",
        },
        "activity_level": {
            "type": "selectbox",
            "label": "Activity Level",
            "options": [
                ("Select Activity Level", None),
                ("Sedentary", "sedentary"),
                ("Lightly Active", "lightly_active"),
                ("Moderately Active", "moderately_active"),
                ("Very Active", "very_active"),
                ("Extremely Active", "extremely_active"),
            ],
            "required": True,
            "placeholder": None,
        },
        "goal": {
            "type": "selectbox",
            "label": "Nutrition Goal",
            "options": [
                ("Select Goal", None),
                ("Weight Loss", "weight_loss"),
                ("Weight Maintenance", "weight_maintenance"),
                ("Weight Gain", "weight_gain"),
            ],
            "required": True,
            "placeholder": None,
        },
        "protein_per_kg": {
            "type": "number",
            "label": "Protein (g Per Kilogram Body Weight)",
            "min": 1.2,
            "max": 3.0,
            "step": 0.1,
            "help": "Protein intake per kilogram of body weight",
            "advanced": True,
            "required": False,
        },
        "fat_percentage": {
            "type": "number",
            "label": "Fat (Percent Of Total Calories)",
            "min": 15,
            "max": 40,
            "step": 1,
            "help": "Percentage of total calories from fat",
            "convert": lambda x: x / 100 if x is not None else None,
            "advanced": True,
            "required": False,
        },
    },
}

# -----------------------------------------------------------------------------
# Cell 4: Unified Helper Functions
# -----------------------------------------------------------------------------

def initialize_session_state() -> None:
    """Create all session-state keys if they do not exist"""
    session_vars: List[str] = ["food_selections"] + [
        f"user_{field}" for field in CONFIG["form_fields"]
    ]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = {} if var == "food_selections" else None


def create_unified_input(
    field_name: str, field_config: Dict[str, Any], container=st.sidebar
) -> Any:
    """Render a sidebar widget and sync its value to session state"""
    session_key: str = f"user_{field_name}"

    if field_config["type"] == "number":
        placeholder: str | None
        if field_config.get("advanced"):
            default_val = DEFAULTS.get(field_name, 0)
            display_val = (
                int(default_val * 100) if field_name == "fat_percentage" else default_val
            )
            placeholder = f"Default: {display_val}"
        else:
            placeholder = field_config.get("placeholder")

        value = container.number_input(
            field_config["label"],
            min_value=field_config["min"],
            max_value=field_config["max"],
            value=st.session_state[session_key],
            step=field_config["step"],
            placeholder=placeholder,
            help=field_config.get("help"),
        )

    elif field_config["type"] == "selectbox":
        current_value = st.session_state[session_key]
        if field_name in {"activity_level", "goal"}:
            index = next(
                (
                    i
                    for i, (_, val) in enumerate(field_config["options"])
                    if val == current_value
                ),
                0,
            )
            selection = container.selectbox(
                field_config["label"],
                field_config["options"],
                index=index,
                format_func=lambda x: x[0],
            )
            value = selection[1]
        else:
            index = (
                field_config["options"].index(current_value)
                if current_value in field_config["options"]
                else 0
            )
            value = container.selectbox(
                field_config["label"], field_config["options"], index=index
            )
    else:
        raise ValueError(f"Unsupported widget type: {field_config['type']}")

    st.session_state[session_key] = value
    return value


def get_final_values(user_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge user inputs with defaults and goal-specific overrides"""
    final_values: Dict[str, Any] = {}

    for field, value in user_inputs.items():
        if field == "sex":
            final_values[field] = value if value != "Select Sex" else DEFAULTS[field]
        elif field in {"activity_level", "goal"}:
            final_values[field] = value if value is not None else DEFAULTS[field]
        else:
            final_values[field] = value if value is not None else DEFAULTS[field]

    # Goal-level fallback for advanced settings
    if final_values["goal"] in GOAL_TARGETS:
        goal_cfg = GOAL_TARGETS[final_values["goal"]]
        if user_inputs.get("protein_per_kg") is None:
            final_values["protein_per_kg"] = goal_cfg["protein_per_kg"]
        if user_inputs.get("fat_percentage") is None:
            final_values["fat_percentage"] = goal_cfg["fat_percentage"]

    return final_values


def calculate_hydration_needs(
    weight_kg: float, activity_level: str, climate: str = "temperate"
) -> int:
    """Estimate daily fluid needs in milliliters"""
    base_needs = weight_kg * 35  # 35 ml per kg baseline

    activity_bonus = {
        "sedentary": 0,
        "lightly_active": 300,
        "moderately_active": 500,
        "very_active": 700,
        "extremely_active": 1000,
    }

    climate_multiplier = {
        "cold": 0.9,
        "temperate": 1.0,
        "hot": 1.2,
        "very_hot": 1.4,
    }

    total_ml = (
        base_needs + activity_bonus.get(activity_level, 500)
    ) * climate_multiplier.get(climate, 1.0)
    return round(total_ml)


def display_metrics_grid(
    metrics_data: List[Tuple[str, str, str | None]], num_columns: int = 4
) -> None:
    """Show a metric grid with a custom column count"""
    columns = st.columns(num_columns)
    for i, metric_info in enumerate(metrics_data):
        with columns[i % num_columns]:
            if len(metric_info) == 2:
                label, value = metric_info
                st.metric(label, value)
            elif len(metric_info) == 3:
                label, value, delta = metric_info
                st.metric(label, value, delta)


# -----------------------------------------------------------------------------
# Cell 5: Nutritional Calculation Functions
# -----------------------------------------------------------------------------

def calculate_bmr(age: int, height_cm: int, weight_kg: float, sex: str = "male") -> float:
    """Return Basal Metabolic Rate using the Mifflin-St Jeor equation"""
    base_calc = (10 * weight_kg) + (6.25 * height_cm) - (5 * age)
    return base_calc + (5 if sex.lower() == "male" else -161)


def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Return Total Daily Energy Expenditure"""
    multiplier = ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)
    return bmr * multiplier


def calculate_estimated_weekly_change(daily_caloric_adjustment: float) -> float:
    """Convert a daily caloric adjustment to an estimated weekly weight change"""
    # 1 kg fat ≈ 7700 kcal
    return (daily_caloric_adjustment * 7) / 7700


def calculate_personalized_targets(
    age: int,
    height_cm: int,
    weight_kg: float,
    sex: str = "male",
    activity_level: str = "moderately_active",
    goal: str = "weight_gain",
    protein_per_kg: float | None = None,
    fat_percentage: float | None = None,
) -> Dict[str, Any]:
    """Compute daily calorie and macronutrient goals"""
    bmr = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity_level)

    goal_cfg = GOAL_TARGETS.get(goal, GOAL_TARGETS["weight_gain"])
    caloric_adjustment = tdee * goal_cfg["caloric_adjustment"]
    total_calories = tdee + caloric_adjustment

    protein_per_kg = protein_per_kg or goal_cfg["protein_per_kg"]
    fat_percentage = fat_percentage or goal_cfg["fat_percentage"]

    protein_g = protein_per_kg * weight_kg
    protein_cal = protein_g * 4
    fat_cal = total_calories * fat_percentage
    fat_g = fat_cal / 9
    carb_cal = total_calories - protein_cal - fat_cal
    carb_g = carb_cal / 4

    weekly_change = calculate_estimated_weekly_change(caloric_adjustment)

    targets = {
        "bmr": round(bmr),
        "tdee": round(tdee),
        "total_calories": round(total_calories),
        "caloric_adjustment": round(caloric_adjustment),
        "protein_g": round(protein_g),
        "protein_calories": round(protein_cal),
        "fat_g": round(fat_g),
        "fat_calories": round(fat_cal),
        "carb_g": round(carb_g),
        "carb_calories": round(carb_cal),
        "estimated_weekly_change": round(weekly_change, 3),
        "goal": goal,
    }

    if targets["total_calories"] > 0:
        targets["protein_percent"] = (
            targets["protein_calories"] / targets["total_calories"] * 100
        )
        targets["carb_percent"] = (
            targets["carb_calories"] / targets["total_calories"] * 100
        )
        targets["fat_percent"] = (
            targets["fat_calories"] / targets["total_calories"] * 100
        )
    else:
        targets["protein_percent"] = targets["carb_percent"] = targets["fat_percent"] = 0

    return targets

# -----------------------------------------------------------------------------
# Cell 6: Food Database Processing Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_food_database(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Load a vegetarian food database from a CSV file"""
    df = pd.read_csv(file_path)
    foods: Dict[str, List[Dict[str, Any]]] = {
        cat: [] for cat in df["category"].unique()
    }

    for _, row in df.iterrows():
        category = row["category"]
        if category in foods:
            foods[category].append(
                {
                    "name": f"{row['name']} ({row['serving_unit']})",
                    "calories": row["calories"],
                    "protein": row["protein"],
                    "carbs": row["carbs"],
                    "fat": row["fat"],
                }
            )
    return foods


def assign_food_emojis(
    foods: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, List[Dict[str, Any]]]:
    """Add an emoji label to each food based on nutrient ranking"""
    top_foods: Dict[str, Any] = {"protein": [], "carbs": [], "fat": [], "calories": {}}

    # Identify category leaders
    for category, items in foods.items():
        if not items:
            continue

        # Top three by calories
        sorted_cals = sorted(items, key=lambda x: x["calories"], reverse=True)
        top_foods["calories"][category] = [f["name"] for f in sorted_cals[:3]]

        # Top three by primary nutrient
        map_info = CONFIG["nutrient_map"].get(category)
        if map_info:
            key = map_info["sort_by"]
            sorted_items = sorted(items, key=lambda x: x[key], reverse=True)
            top_foods[map_info["key"]] = [f["name"] for f in sorted_items[:3]]

    all_top_nutrient = {
        food for n_key in ["protein", "carbs", "fat"] for food in top_foods[n_key]
    }

    emoji_map = {
        "high_cal_nutrient": "🥇",
        "high_calorie": "🔥",
        "protein": "💪",
        "carbs": "🍚",
        "fat": "🥑",
    }

    for category, items in foods.items():
        for food in items:
            fname = food["name"]
            top_nutrient = fname in all_top_nutrient
            high_cal = fname in top_foods["calories"].get(category, [])

            if high_cal and top_nutrient:
                food["emoji"] = emoji_map["high_cal_nutrient"]
            elif high_cal:
                food["emoji"] = emoji_map["high_calorie"]
            elif fname in top_foods["protein"]:
                food["emoji"] = emoji_map["protein"]
            elif fname in top_foods["carbs"]:
                food["emoji"] = emoji_map["carbs"]
            elif fname in top_foods["fat"]:
                food["emoji"] = emoji_map["fat"]
            else:
                food["emoji"] = ""
    return foods


def render_food_item(food: Dict[str, Any], category: str) -> None:
    """Render an interactive card for a single food"""
    with st.container(border=True):
        st.subheader(f"{food.get('emoji', '')} {food['name']}")
        key = f"{category}_{food['name']}"
        current_serving = st.session_state.food_selections.get(food["name"], 0.0)

        col1, col2 = st.columns([2, 1.2])

        # ------ Quick-Select Buttons ------
        with col1:
            button_cols = st.columns(5)
            for k in range(1, 6):
                with button_cols[k - 1]:
                    btn_type = (
                        "primary" if current_serving == float(k) else "secondary"
                    )
                    if st.button(
                        f"{k}",
                        key=f"{key}_{k}",
                        type=btn_type,
                        help=f"Set to {k} servings",
                        use_container_width=True,
                    ):
                        st.session_state.food_selections[food["name"]] = float(k)
                        st.rerun()

        # ------ Custom Input ------
        with col2:
            custom_serving = st.number_input(
                "Custom",
                min_value=0.0,
                max_value=10.0,
                value=float(current_serving),
                step=0.1,
                key=f"{key}_custom",
                label_visibility="collapsed",
            )

        # Sync custom value
        if custom_serving != current_serving:
            if custom_serving > 0:
                st.session_state.food_selections[food["name"]] = custom_serving
            elif food["name"] in st.session_state.food_selections:
                del st.session_state.food_selections[food["name"]]
            st.rerun()

        caption = (
            f"Per Serving: {food['calories']} kcal | {food['protein']} g protein | "
            f"{food['carbs']} g carbs | {food['fat']} g fat"
        )
        st.caption(caption)


def render_food_grid(
    items: List[Dict[str, Any]], category: str, columns: int = 2
) -> None:
    """Display foods in a multi-column grid"""
    for i in range(0, len(items), columns):
        cols = st.columns(columns)
        for j in range(columns):
            if i + j < len(items):
                with cols[j]:
                    render_food_item(items[i + j], category)

# -----------------------------------------------------------------------------
# Cell 7: Initialize Application
# -----------------------------------------------------------------------------

initialize_session_state()
foods_db = load_food_database("nutrition_results.csv")
foods_db = assign_food_emojis(foods_db)

# ------ Minimal CSS Tweaks ------
st.markdown(
    """
    <style>
    [data-testid="InputInstructions"] { display: none; }
    .stButton>button[kind="primary"] {
        background-color: #ff6b6b;
        color: white;
        border: 1px solid #ff6b6b;
    }
    .stButton>button[kind="secondary"] { border: 1px solid #ff6b6b; }
    .sidebar .sidebar-content { background-color: #f0f2f6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Cell 8: Application Title And Unified Input Interface
# -----------------------------------------------------------------------------

st.title("Personalized Evidence-Based Nutrition Tracker 🍽️")
st.markdown(
    """
    This application estimates daily calorie and macronutrient targets for
    vegetarian diets.  Calculations rely on the Mifflin–St Jeor equation and a
    protein-first strategy recommended in sports-nutrition research. 🚀
    """
)

# ------ Sidebar Input Collection ------
st.sidebar.header("Personal Parameters For Daily Target Calculation 📊")

user_inputs: Dict[str, Any] = {}

# Split fields into standard and advanced groups
standard_fields = {k: v for k, v in CONFIG["form_fields"].items() if not v.get("advanced")}
advanced_fields = {k: v for k, v in CONFIG["form_fields"].items() if v.get("advanced")}

# Standard entries
for name, cfg in standard_fields.items():
    val = create_unified_input(name, cfg, container=st.sidebar)
    if "convert" in cfg:
        val = cfg["convert"](val)
    user_inputs[name] = val

# Advanced entries in an expander
with st.sidebar.expander("Advanced Settings ⚙️"):
    for name, cfg in advanced_fields.items():
        val = create_unified_input(name, cfg, container=st)
        if "convert" in cfg:
            val = cfg["convert"](val)
        user_inputs[name] = val

# Activity level guide
with st.sidebar.container(border=True):
    st.markdown(
        """
        **Activity Level Guide**

        • Sedentary – Little to no exercise, desk job  
        • Lightly Active – Light exercise one to three days per week  
        • Moderately Active – Moderate exercise three to five days per week  
        • Very Active – Heavy exercise six to seven days per week  
        • Extremely Active – Physical job or two training sessions per day

        💡 When uncertain, select a lower level to avoid overestimation
        """
    )

# -----------------------------------------------------------------------------
# Cell 9: Unified Target Display System
# -----------------------------------------------------------------------------

final_vals = get_final_values(user_inputs)

# Hydration info in sidebar
if final_vals.get("weight_kg") and final_vals.get("activity_level"):
    h2o_ml = calculate_hydration_needs(
        final_vals["weight_kg"], final_vals["activity_level"]
    )
    st.sidebar.info(
        f"💧 Daily Hydration Target: {h2o_ml} ml "
        f"({h2o_ml / 250:.1f} cups)"
    )

required_ok = all(
    user_inputs.get(f) not in {None, "Select Sex"} for f, conf in CONFIG["form_fields"].items() if conf.get("required")
)

targets = calculate_personalized_targets(**final_vals)

if not required_ok:
    st.info(
        "👈 Enter your information in the sidebar to generate personalized "
        "targets"
    )
    st.header("Sample Daily Targets For Reference 🎯")
else:
    goal_names = {
        "weight_loss": "Weight Loss",
        "weight_maintenance": "Weight Maintenance",
        "weight_gain": "Weight Gain",
    }
    st.header(
        f"Your Personalized Daily Nutritional Targets For "
        f"{goal_names.get(targets['goal'], 'Weight Gain')} 🎯"
    )

st.info(
    "🎯 80–20 Principle: Aim for eighty percent adherence rather than perfect "
    "compliance to support social flexibility"
)

# Hydration for grid
hydration_ml = calculate_hydration_needs(
    final_vals["weight_kg"], final_vals["activity_level"]
)

metric_sections = [
    {
        "title": "Metabolic Information",
        "columns": 5,
        "metrics": [
            ("Basal Metabolic Rate (BMR)", f"{targets['bmr']} kcal per day"),
            ("Total Daily Energy Expenditure (TDEE)", f"{targets['tdee']} kcal per day"),
            ("Daily Caloric Adjustment", f"{targets['caloric_adjustment']:+} kcal"),
            (
                "Estimated Weekly Weight Change",
                f"{targets['estimated_weekly_change']:+.2f} kg",
            ),
            ("", ""),
        ],
    },
    {
        "title": "Daily Macronutrient And Hydration Targets",
        "columns": 5,
        "metrics": [
            ("Total Calories", f"{targets['total_calories']} kcal"),
            (
                "Protein",
                f"{targets['protein_g']} g",
                f"{targets['protein_percent']:.0f} percent",
            ),
            (
                "Carbohydrates",
                f"{targets['carb_g']} g",
                f"{targets['carb_percent']:.0f} percent",
            ),
            ("Fat", f"{targets['fat_g']} g", f"{targets['fat_percent']:.0f} percent"),
            (
                "💧 Hydration",
                f"{hydration_ml} ml",
                f"≈{hydration_ml / 250:.1f} cups",
            ),
        ],
    },
]

for sec in metric_sections:
    st.subheader(sec["title"])
    display_metrics_grid(sec["metrics"], sec["columns"])
    st.divider()

# -----------------------------------------------------------------------------
# Cell 10: Enhanced Evidence-Based Tips And Context
# -----------------------------------------------------------------------------

st.header("📚 Evidence-Based Playbook")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Foundations", "Advanced Strategies", "Troubleshooting", "Nutrition Science"]
)

# ------ Foundations ------
with tab1:
    st.subheader("🏆 Essential Tips For Success")
    st.markdown(
        """
        ### The Foundation Trio

        **💧 Hydration Protocol**  
        • Target – 35 ml per kg body weight daily  
        • Training Bonus – 500 to 750 ml per hour of exercise  
        • Appetite Aid – 500 ml water before meals increases satiety

        **😴 Sleep Quality**  
        • Less than seven hours reduces fat-loss effectiveness  
        • Target – Seven to nine hours nightly with set sleep–wake times  
        • Environment – Dark, cool room at eighteen to twenty °C

        **⚖️ Weigh-In Best Practices**  
        • Daily – Morning, post-bathroom, minimal clothing  
        • Track – Weekly averages instead of daily fluctuations  
        • Adjust – Only after two or more stalled weeks
        """
    )

# ------ Advanced Strategies ------
with tab2:
    st.subheader("📊 Advanced Monitoring And Psychology")
    st.markdown(
        """
        ### Beyond The Scale

        • Progress photos – Consistent lighting and timing  
        • Body measurements – Waist, hips, arms, thighs monthly  
        • Performance metrics – Strength, energy, sleep

        ### Psychology Of Sustainable Change

        **Progressive Implementation**  
        • Weeks 1–2 – Hit calorie targets only  
        • Weeks 3–4 – Add protein goals  
        • Week 5 + – Refine fat and carbohydrate distribution
        """
    )

# ------ Troubleshooting ------
with tab3:
    st.subheader("🔄 Plateau Prevention And Meal Timing")
    st.markdown(
        """
        ### Plateau Flow – Weight Loss

        1. Confirm logging accuracy within five percent  
        2. Re-check activity multiplier  
        3. Add ten to fifteen minutes of daily walking  
        4. Insert diet breaks – One to two weeks at maintenance

        ### Meal Timing

        **Protein** – Twenty to thirty grams across three to four meals  
        **Post-Workout** – Twenty to forty grams protein within two hours
        """
    )

# ------ Nutrition Science ------
with tab4:
    st.subheader("🔬 Scientific Foundation And Nutrition Deep Dive")
    st.markdown(
        """
        ### Energy Primer

        • **BMR** – Resting energy via Mifflin–St Jeor equation  
        • **TDEE** – BMR multiplied by activity factor

        ### Satiety Hierarchy

        1. Protein  
        2. Fiber-rich carbohydrates  
        3. Healthy fats  
        4. Refined foods
        """
    )

# -----------------------------------------------------------------------------
# Cell 11: Personalized Recommendations System
# -----------------------------------------------------------------------------

if required_ok:
    st.header("🎯 Your Personalized Action Plan")
    totals_now, _ = calculate_daily_totals(st.session_state.food_selections, foods_db)
    todays_recs = generate_personalized_recommendations(
        totals_now, targets, final_vals
    )
    for rec in todays_recs:
        st.info(rec)

# -----------------------------------------------------------------------------
# Cell 12: Daily Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Daily Food Selection And Tracking 🥗")
st.markdown(
    "Select the number of servings for each food item to monitor daily intake"
)

with st.expander("💡 Food Emoji Guide"):
    st.markdown(
        """
        **Food Emoji Guide**

        • 🥇 Gold Medal – Leading food for calories and its primary nutrient  
        • 🔥 High Calorie – Among the most calorie-dense foods in category  
        • 💪 High Protein – Top protein source  
        • 🍚 High Carbohydrate – Top carbohydrate source  
        • 🥑 High Fat – Top fat source
        """
    )

if st.button("🔄 Reset All Food Selections", type="secondary"):
    st.session_state.food_selections = {}
    st.rerun()

categories = [cat for cat, items in sorted(foods_db.items()) if items]
tabs = st.tabs(categories)

for idx, cat in enumerate(categories):
    cat_items = foods_db[cat]
    cat_sorted = sorted(
        cat_items,
        key=lambda x: (CONFIG["emoji_order"].get(x.get("emoji", ""), 4), -x["calories"]),
    )
    with tabs[idx]:
        render_food_grid(cat_sorted, cat, columns=2)

# -----------------------------------------------------------------------------
# Cell 13: Daily Summary And Progress Tracking
# -----------------------------------------------------------------------------

st.header("Daily Nutrition Summary 📊")
totals_today, chosen_foods = calculate_daily_totals(
    st.session_state.food_selections, foods_db
)

if chosen_foods:
    progress_notes = create_progress_tracking(totals_today, targets, foods_db)

    col_left, col_right = st.columns(2)

    # ------ Summary Metrics ------
    with col_left:
        st.subheader("Today's Nutrition Intake")
        day_metrics = [
            ("Calories Consumed", f"{totals_today['calories']:.0f} kcal"),
            ("Protein Intake", f"{totals_today['protein']:.0f} g"),
            ("Carbohydrates", f"{totals_today['carbs']:.0f} g"),
            ("Fat Intake", f"{totals_today['fat']:.0f} g"),
        ]
        display_metrics_grid(day_metrics, 2)

    # ------ Donut Chart ------
    with col_right:
        st.subheader("Macronutrient Split (Grams)")
        macro_vals = [
            totals_today["protein"],
            totals_today["carbs"],
            totals_today["fat"],
        ]
        if sum(macro_vals) > 0:
            fig = go.Figure(
                go.Pie(
                    labels=["Protein", "Carbohydrates", "Fat"],
                    values=macro_vals,
                    hole=0.4,
                    marker_colors=["#ff6b6b", "#feca57", "#48dbfb"],
                    textinfo="label+percent",
                    insidetextorientation="radial",
                )
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                height=250,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Select foods to view the macronutrient split")

    # ------ Additional Recommendations ------
    if progress_notes:
        st.subheader("Personalized Recommendations For Today")
        for note in progress_notes:
            st.info(note)

    # ------ Detailed Food Breakdown ------
    with st.expander("📝 Detailed Food Breakdown"):
        st.subheader("Foods Selected Today")
        for entry in chosen_foods:
            food_item = entry["food"]
            serve = entry["servings"]
            cal = food_item["calories"] * serve
            pro = food_item["protein"] * serve
            carb = food_item["carbs"] * serve
            fat = food_item["fat"] * serve
            st.write(
                f"**{food_item['name']}** – {serve} serving(s)  "
                f"→ {cal:.0f} kcal | {pro:.1f} g protein | "
                f"{carb:.1f} g carbs | {fat:.1f} g fat"
            )
else:
    st.info(
        "No foods selected yet.  Pick items from the tabs above to start "
        "tracking"
    )
    st.subheader("Progress Toward Daily Nutritional Targets 🎯")
    for nut, cfg in CONFIG["nutrient_configs"].items():
        targ = targets[cfg["target_key"]]
        st.progress(
            0.0, text=f"{cfg['label']}: 0 percent of daily target ({targ:.0f} {cfg['unit']})"
        )

# -----------------------------------------------------------------------------
# Cell 14: Footer And Additional Resources
# -----------------------------------------------------------------------------

st.divider()
st.markdown(
    """
    ### 📚 Evidence-Based References And Methodology

    • **BMR** – Mifflin-St Jeor equation (Academy of Nutrition and Dietetics)  
    • **Activity Factors** – Exercise-physiology research  
    • **Protein Targets** – International Society of Sports Nutrition  
    • **Caloric Adjustments** – Body-composition literature

    ### ⚠️ Important Disclaimers

    • Guidance is general and may not suit every individual  
    • Consult a qualified professional before major dietary changes  
    • Monitor health markers and adjust as required

    ### 🔬 Continuous Improvement

    This tracker reflects current research and will evolve as new data emerge
    """
)

# -----------------------------------------------------------------------------
# Cell 15: Session State Management And Performance
# -----------------------------------------------------------------------------

# ------ Compaction For Large Food Lists ------
if len(st.session_state.food_selections) > 100:
    st.session_state.food_selections = {
        k: v for k, v in st.session_state.food_selections.items() if v > 0
    }

# ------ Friendly Sign-Off ------
st.success(
    "🙏 Thanks for fueling your day with science-backed choices.  Until next "
    "time, keep your plate colorful and your goals in sight 🌈"
)
