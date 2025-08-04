# -----------------------------------------------------------------------------
# Personalized Evidence-Based Nutrition Tracker (refactored)
# -----------------------------------------------------------------------------

"""
Interactive nutrition tracking application for healthy weight gain using
vegetarian food sources.  Calculates personalised calorie & macro targets and
lets the user log foods from a CSV database.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports & page config
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Personalised Nutrition Tracker",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults, constants
# ─────────────────────────────────────────────────────────────────────────────
DEFAULTS = dict(
    AGE=26,
    HEIGHT_CM=180,
    WEIGHT_KG=57.5,
    GENDER="Male",
    ACTIVITY="moderately_active",
    SURPLUS=400,
    PROT_PER_KG=2.0,
    FAT_FRAC=0.25      # 25 %
)

ACTIVITY_MULTIPLIERS = {
    'sedentary': 1.2,
    'lightly_active': 1.375,
    'moderately_active': 1.55,
    'very_active': 1.725,
    'extremely_active': 1.9
}

EMOJI_ORDER = {'🥇': 0, '💥': 1, '🔥': 2, '💪': 3, '🍚': 3,
               '🥑': 3, '🥦': 3, '': 4}

NUTRIENT_MAP = {
    'PRIMARY PROTEIN SOURCES': 'protein',
    'PRIMARY CARBOHYDRATE SOURCES': 'carbs',
    'PRIMARY FAT SOURCES': 'fat',
    'PRIMARY MICRONUTRIENT SOURCES': 'micro'
}

SESSION_KEYS = ("user_age", "user_height", "user_weight",
                "user_sex", "user_activity", "food_selections")

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────
def initialise_state():
    """Ensure all required session-state keys exist."""
    for key in SESSION_KEYS:
        st.session_state.setdefault(key, None if key != "food_selections" else {})

def show_metrics(metrics: list[tuple[str, str]], cols=4):
    """Display Streamlit metrics from list of (label, value[, delta])."""
    columns = st.columns(cols)
    for (label, value, *delta), col in zip(metrics, columns):
        if delta:
            col.metric(label, value, delta[0])
        else:
            col.metric(label, value)

def show_progress(bars: list[tuple[str, float, float, str]]):
    """
    bars: list of (label, current, goal, unit)
    Produces a Streamlit progress bar for each.
    """
    for label, current, goal, unit in bars:
        pct = min(current / goal, 1.0) if goal else 0
        st.progress(
            pct,
            text=f"{label}: {pct*100:.0f}% of daily target ({goal:.0f} {unit})"
        )

def render_preset_buttons(current_serving, key_prefix, max_buttons=5):
    """Generate 1…max_buttons serving buttons and return new value or None."""
    btn_cols = st.columns(max_buttons)
    for k in range(1, max_buttons + 1):
        with btn_cols[k - 1]:
            button_type = "primary" if current_serving == float(k) else "secondary"
            if st.button(f"{k} Servings", key=f"{key_prefix}_{k}", type=button_type):
                return float(k)          # new selection
    return None                        # unchanged

# ─────────────────────────────────────────────────────────────────────────────
# Metabolic calculations
# ─────────────────────────────────────────────────────────────────────────────
def calculate_bmr(age, height_cm, weight_kg, sex='male'):
    if sex.lower() == 'male':
        return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) + 5
    return (10 * weight_kg) + (6.25 * height_cm) - (5 * age) - 161

def calculate_tdee(bmr, activity_level):
    return bmr * ACTIVITY_MULTIPLIERS.get(activity_level, 1.55)

def calculate_targets(age, height_cm, weight_kg, sex,
                      activity, surplus, prot_per_kg, fat_frac):
    bmr  = calculate_bmr(age, height_cm, weight_kg, sex)
    tdee = calculate_tdee(bmr, activity)
    total_cal = tdee + surplus

    protein_g       = prot_per_kg * weight_kg
    protein_cal     = protein_g * 4
    fat_cal         = total_cal * fat_frac
    fat_g           = fat_cal / 9
    carb_cal        = total_cal - protein_cal - fat_cal
    carb_g          = carb_cal / 4
    weekly_gain_kg  = round(weight_kg * 0.0025, 2)

    return dict(
        bmr=round(bmr), tdee=round(tdee), total_calories=round(total_cal),
        protein_g=round(protein_g), protein_calories=round(protein_cal),
        fat_g=round(fat_g), fat_calories=round(fat_cal),
        carb_g=round(carb_g), carb_calories=round(carb_cal),
        weekly_gain=weekly_gain_kg
    )

# ─────────────────────────────────────────────────────────────────────────────
# Food database & emoji assignment
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_food_database(path="nutrition_results.csv"):
    df = pd.read_csv(path)
    foods = {c: [] for c in NUTRIENT_MAP.keys()}
    for _, row in df.iterrows():
        food = dict(
            name=f"{row['name']} ({row['serving_unit']})",
            calories=row['calories'],
            protein=row['protein'],
            carbs=row['carbs'],
            fat=row['fat']
        )
        foods.get(row["category"], []).append(food)
    return foods

def assign_food_emojis(foods: dict):
    """Tag foods with emoji based on nutrient & calorie ranking."""
    top_foods = {nut: [] for nut in NUTRIENT_MAP.values()}
    top_cals  = {}

    # top-3 per nutrient + per calorie
    for category, items in foods.items():
        if not items: continue
        sorted_by_cal = sorted(items, key=lambda x: x['calories'], reverse=True)[:3]
        top_cals[category] = {f['name'] for f in sorted_by_cal}

        nutrient = NUTRIENT_MAP[category]
        sorted_by_nut = sorted(items, key=lambda x: x[nutrient], reverse=True)[:3]
        top_foods[nutrient] = {f['name'] for f in sorted_by_nut}

    # foods that rank in ≥2 nutrient categories → superfood
    all_tops = set().union(*top_foods.values())
    rank_counts = {name: sum(name in s for s in top_foods.values())
                   for name in all_tops}
    superfoods = {n for n, c in rank_counts.items() if c > 1}

    # apply emoji
    for category, items in foods.items():
        for food in items:
            name = food['name']
            food['emoji'] = ''
            if name in superfoods:
                food['emoji'] = '🥇'
            elif name in top_cals.get(category, set()) and name in all_tops:
                food['emoji'] = '💥'
            elif name in top_cals.get(category, set()):
                food['emoji'] = '🔥'
            elif name in top_foods['protein']:
                food['emoji'] = '💪'
            elif name in top_foods['carbs']:
                food['emoji'] = '🍚'
            elif name in top_foods['fat']:
                food['emoji'] = '🥑'
            elif name in top_foods['micro']:
                food['emoji'] = '🥦'
    return foods

# ─────────────────────────────────────────────────────────────────────────────
# Food-item UI
# ─────────────────────────────────────────────────────────────────────────────
def render_food_item(food: dict, category: str):
    st.subheader(f"{food.get('emoji','')} {food['name']}")
    key_prefix = f"{category}_{food['name']}"
    current = st.session_state.food_selections.get(food['name'], 0.0)

    # preset buttons
    new_val = render_preset_buttons(current, key_prefix)
    if new_val is not None:
        st.session_state.food_selections[food['name']] = new_val
        st.rerun()

    # custom input
    custom = st.number_input(
        "Custom # Servings", 0.0, 10.0, value=current, step=0.1,
        key=f"{key_prefix}_custom"
    )
    if custom != current:
        if custom > 0:
            st.session_state.food_selections[food['name']] = custom
        else:
            st.session_state.food_selections.pop(food['name'], None)
        st.rerun()

    st.caption(f"Per Serving: {food['calories']} kcal | "
               f"{food['protein']} g protein | {food['carbs']} g carbs | "
               f"{food['fat']} g fat")

# ─────────────────────────────────────────────────────────────────────────────
# Initialise state & load foods
# ─────────────────────────────────────────────────────────────────────────────
initialise_state()
foods = assign_food_emojis(load_food_database())

# ─────────────────────────────────────────────────────────────────────────────
# ────────────────  Sidebar – personal parameters  ───────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.header("Personal Parameters 📊")

age = st.sidebar.number_input(
    "Age (years)", 16, 80, value=st.session_state.user_age,
    placeholder="Enter age"
)
height = st.sidebar.number_input(
    "Height (cm)", 140, 220, value=st.session_state.user_height,
    placeholder="Enter height"
)
weight = st.sidebar.number_input(
    "Weight (kg)", 40.0, 150.0, value=st.session_state.user_weight,
    step=0.5, placeholder="Enter weight"
)

sex = st.sidebar.selectbox(
    "Sex", ["Select Sex", "Male", "Female"],
    index=["Select Sex", "Male", "Female"]
    .index(st.session_state.user_sex or "Select Sex")
)

activity_options = [
    ("Select Activity Level", None),
    ("Sedentary", "sedentary"), ("Lightly Active", "lightly_active"),
    ("Moderately Active", "moderately_active"), ("Very Active", "very_active"),
    ("Extremely Active", "extremely_active")
]
activity = st.sidebar.selectbox(
    "Activity Level", activity_options,
    index=next((i for i, (_, val) in enumerate(activity_options)
                if val == st.session_state.user_activity), 0),
    format_func=lambda x: x[0]
)[1]

# store back
st.session_state.update(user_age=age, user_height=height,
                        user_weight=weight, user_sex=sex,
                        user_activity=activity)

with st.sidebar.expander("Advanced Settings ⚙️"):
    surplus = st.number_input(
        "Caloric Surplus (kcal)", 200, 800, step=50, value=None,
        placeholder=f"Default: {DEFAULTS['SURPLUS']}"
    )
    prot_per_kg = st.number_input(
        "Protein (g/kg BW)", 1.2, 3.0, step=0.1, value=None,
        placeholder=f"Default: {DEFAULTS['PROT_PER_KG']}"
    )
    fat_pct = st.number_input(
        "Fat (% calories)", 15, 40, step=1, value=None,
        placeholder=f"Default: {int(DEFAULTS['FAT_FRAC']*100)}"
    )

# fill missing with defaults
params = dict(
    age     = age   or DEFAULTS['AGE'],
    height  = height or DEFAULTS['HEIGHT_CM'],
    weight  = weight or DEFAULTS['WEIGHT_KG'],
    sex     = sex   if sex != "Select Sex" else DEFAULTS['GENDER'],
    activity= activity or DEFAULTS['ACTIVITY'],
    surplus = surplus or DEFAULTS['SURPLUS'],
    protkg  = prot_per_kg or DEFAULTS['PROT_PER_KG'],
    fatfrac = (fat_pct/100 if fat_pct is not None else DEFAULTS['FAT_FRAC'])
)

required_entered = (age and height and weight and sex != "Select Sex" and activity)

# ─────────────────────────────────────────────────────────────────────────────
# ────────────────  Calculate targets  ───────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
targets = calculate_targets(
    params['age'], params['height'], params['weight'],
    params['sex'].lower(), params['activity'], params['surplus'],
    params['protkg'], params['fatfrac']
)

# ─────────────────────────────────────────────────────────────────────────────
# ────────────────  Main page  ───────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
st.title("Personalised Evidence-Based Nutrition Tracker 🍽️")
st.markdown("""
Ready to turbo-charge your health? Input your details, pick your foods and let
the app do the maths.  Lean gains, here we come! 🚀
""")

# Targets section
if not required_entered:
    st.info("👈 Enter your personal info in the sidebar for personalised targets.")
    st.header("Sample Daily Targets 🎯")
else:
    st.header("Your Personal Daily Targets 🎯")

show_metrics([
    ("BMR", f"{targets['bmr']} kcal"),
    ("TDEE", f"{targets['tdee']} kcal"),
    ("Weekly Gain (est.)", f"{targets['weekly_gain']} kg")
])

show_metrics([
    ("Calories",     f"{targets['total_calories']} kcal"),
    ("Protein",      f"{targets['protein_g']} g"),
    ("Carbohydrates",f"{targets['carb_g']} g"),
    ("Fat",          f"{targets['fat_g']} g")
])

# Macro % of calories
prot_pct = targets['protein_calories'] / targets['total_calories'] * 100
carb_pct = targets['carb_calories']  / targets['total_calories'] * 100
fat_pct  = targets['fat_calories']   / targets['total_calories'] * 100
show_metrics([
    ("Protein %",      f"{prot_pct:.1f} %", f"+{targets['protein_calories']} kcal"),
    ("Carbohydrate %", f"{carb_pct:.1f} %", f"+{targets['carb_calories']} kcal"),
    ("Fat %",          f"{fat_pct:.1f} %",  f"+{targets['fat_calories']} kcal")
])

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Food selection interface
# ─────────────────────────────────────────────────────────────────────────────
st.header("Select Foods & Log Servings 📝")
st.markdown("Use the preset buttons or enter a custom amount.")

tabs = st.tabs([c for c, items in foods.items() if items])
for tab, category in zip(tabs, foods):
    with tab:
        sorted_items = sorted(foods[category],
                              key=lambda x: (EMOJI_ORDER.get(x['emoji'], 4),
                                             -x['calories']))
        for i in range(0, len(sorted_items), 2):
            row_cols = st.columns(2)
            for food, col in zip(sorted_items[i:i+2], row_cols):
                with col: render_food_item(food, category)

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Calculation & results
# ─────────────────────────────────────────────────────────────────────────────
if st.button("Calculate Daily Intake", type="primary", use_container_width=True):
    totals = dict(cal=0, prot=0, carb=0, fat=0)
    selected = []

    for items in foods.values():
        for food in items:
            servings = st.session_state.food_selections.get(food['name'], 0.0)
            if servings:
                selected.append((food, servings))
                totals['cal']  += food['calories'] * servings
                totals['prot'] += food['protein']  * servings
                totals['carb'] += food['carbs']    * servings
                totals['fat']  += food['fat']      * servings

    st.header("Daily Intake Summary 📊")

    if selected:
        st.subheader("Foods Logged 🥣")
        cols = st.columns(3)
        for (food, servings), col in zip(selected, cols * ((len(selected)+2)//3)):
            with col:
                st.write(f"• {food['emoji']} {food['name']} × {servings:.1f}")
    else:
        st.info("No foods selected.")

    show_metrics([
        ("Calories Consumed", f"{totals['cal']:.0f} kcal"),
        ("Protein Consumed",  f"{totals['prot']:.1f} g"),
        ("Carbs Consumed",    f"{totals['carb']:.1f} g"),
        ("Fat Consumed",      f"{totals['fat']:.1f} g")
    ])

    st.subheader("Progress Toward Targets 🎯")
    show_progress([
        ("Calories", totals['cal'],  targets['total_calories'], "kcal"),
        ("Protein",  totals['prot'], targets['protein_g'], "g"),
        ("Carbs",    totals['carb'], targets['carb_g'], "g"),
        ("Fat",      totals['fat'],  targets['fat_g'], "g")
    ])

    # recommendations
    recs, fields = [], dict(cal=("cal","kcal"), prot=("protein","g"),
                            carb=("carbs","g"), fat=("fat","g"))
    for key,(tkey,unit) in fields.items():
        diff = targets[f"{tkey if key!='cal' else 'total_calories'}"] - totals[key]
        if diff > 0:
            recs.append(f"• {diff:.0f} more {unit} of {tkey} needed.")

    if recs:
        st.subheader("Recommendations 💡")
        st.write("\n".join(recs))
    else:
        st.success("All daily targets met! 🎉")

    # caloric balance
    st.subheader("Caloric Balance ⚖️")
    balance = totals['cal'] - targets['tdee']
    if balance >= 0:
        st.info(f"📈 Surplus of {balance:.0f} kcal vs maintenance.")
    else:
        st.warning(f"📉 Deficit of {abs(balance):.0f} kcal vs maintenance.")

    # detailed log
    if selected:
        st.subheader("Detailed Food Log 📋")
        df_log = pd.DataFrame([{
            "Food": f"{f['emoji']} {f['name']}",
            "Servings": s,
            "Calories": f['calories']*s,
            "Protein (g)": f['protein']*s,
            "Carbs (g)":   f['carbs']*s,
            "Fat (g)":     f['fat']*s
        } for f, s in selected])
        st.dataframe(df_log.style.format({
            "Calories":"{:.0f}", "Protein (g)":"{:.1f}",
            "Carbs (g)":"{:.1f}", "Fat (g)":"{:.1f}"}), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Clear selections
# ─────────────────────────────────────────────────────────────────────────────
if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar info blocks (unchanged text)
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Activity Level Guide 🏃‍♂️")
st.sidebar.markdown("""
- Sedentary: Little/no exercise  
- Lightly Active: 1–3 days/wk  
- Moderately Active: 3–5 days/wk  
- Very Active: 6–7 days/wk  
- Extremely Active: hard training & physical job  
""")

st.sidebar.markdown("### Emoji Guide 💡")
st.sidebar.markdown("""
🥇 Superfood 💥 Nutrient + Calorie dense 🔥 High-calorie  
💪 Top protein 🍚 Top carb 🥑 Top fat 🥦 Top micro
""")

st.sidebar.markdown("### Methodology 📖")
st.sidebar.markdown("""
- BMR: Mifflin-St Jeor  
- Protein: 2 g/kg BW (default)  
- Fat: 25 % of kcal (default)  
- Carbs: remaining kcal  
- Weight-gain target: 0.25 % BW/week  
""")
