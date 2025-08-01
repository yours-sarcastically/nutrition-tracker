# -----------------------------------------------------------------------------
# Streamlit Vegetarian Nutrition Tracker for Healthy Weight Gain
# -----------------------------------------------------------------------------

"""
This Streamlit application offers a tool for tracking daily nutritional intake 
using vegetarian food sources. It calculates total caloric and macronutrient  
consumption based on user-selected foods and serving sizes, then compares the 
results to established daily targets for healthy weight gain.

The application includes a categorized food database organized by nutritional
focus, such as primary protein sources, carbohydrate sources, fat sources,
and micronutrient sources. Users can select foods using quick-select buttons
or custom serving inputs, with real-time calculation of nutritional totals.

Daily Targets:
- Calories: 2800–2900 kcal for healthy weight gain
- Protein: 110–120g for muscle building and recovery
- Carbohydrates: 410–430g for energy and performance
- Fat: 75–85g for hormone production and absorption
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Modules
# -----------------------------------------------------------------------------

import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# Cell 2: Page Configuration and Initial Setup
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Nutrition Tracker",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Cell 3: Daily Nutritional Targets and Food Database Configuration
# -----------------------------------------------------------------------------

# ------ Daily Nutritional Targets for Weight Gain ------

# Define daily targets for weight gain with minimum and maximum ranges
daily_targets = {
    'calories': {'min': 2800, 'max': 2900},
    'protein': {'min': 110, 'max': 120},
    'carbs': {'min': 410, 'max': 430},
    'fat': {'min': 75, 'max': 85}
}

# ------ Load and Process Food Database from CSV ------

# Function to load and categorize food data from the CSV file according to the specified scheme
@st.cache_data
def load_food_database(file_path):
    # Define the custom classification and order
    classification_scheme = {
        'PRIMARY PROTEIN SOURCES': [
            'Eggs', 'Greek Yogurt', 'Protein Powder', 'Lentils', 'Chickpeas',
            'Cottage Cheese', 'Kidney Beans', 'Milk', 'Cheese', 'Hummus', 'Tortellini'
        ],
        'PRIMARY FAT SOURCES': [
            'Olive Oil', 'Almonds', 'Chia Seeds', 'Avocado', 'Sunflower Seeds',
            'Mixed Nuts', 'Peanut Butter', 'Tahini', 'Trail Mix', 'Heavy Cream'
        ],
        'CARBOHYDRATE SOURCES': [
            'Oats', 'Potato', 'Mixed Vegetables', 'Green Peas', 'Bread',
            'Corn', 'Banana', 'Couscous', 'Rice', 'Pasta',
            'Spinach Tortellini', 'Pizza'
        ],
        'PRIMARY MICRONUTRIENT SOURCES': [
            'Spinach', 'Broccoli', 'Berries', 'Tomatoes', 'Carrots',
            'Cauliflower', 'Green Beans', 'Mushrooms', 'Orange Juice',
            'Apple Juice', 'Fruit Juice'
        ]
    }

    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)

    # Create a mapping from Food Name to Category and Sort Order
    food_map = {}
    for category, food_list in classification_scheme.items():
        for i, food_name in enumerate(food_list):
            food_map[food_name] = {'category': category, 'order': i}

    # Apply the mapping to the DataFrame
    df['Category'] = df['Food Name'].map(lambda x: food_map.get(x, {}).get('category'))
    df['SortOrder'] = df['Food Name'].map(lambda x: food_map.get(x, {}).get('order'))

    # Drop foods that are not in the classification scheme and sort
    df.dropna(subset=['Category'], inplace=True)
    df.sort_values(by=['Category', 'SortOrder'], inplace=True)

    # Create the food dictionary in the required format
    foods = {
        'PRIMARY PROTEIN SOURCES': [],
        'PRIMARY FAT SOURCES': [],
        'CARBOHYDRATE SOURCES': [],
        'PRIMARY MICRONUTRIENT SOURCES': []
    }

    # Populate the foods dictionary from the sorted DataFrame
    for _, row in df.iterrows():
        category = row['Category']
        food_item = {
            'name': f"{row['Food Name']} ({row['Serving Size']})",
            'calories': row['Calories (kcal)'],
            'protein': row['Protein (g)'],
            'carbs': row['Carbohydrates (g)'],
            'fat': row['Fat (g)']
        }
        if category in foods:
            foods[category].append(food_item)

    return foods


# Load the food database from the CSV file
try:
    foods = load_food_database('nutrition_results.csv')
except FileNotFoundError:
    st.error("The 'nutrition_results.csv' file was not found. Please make sure it's in the same directory as the script. 📄")
    st.stop()


# -----------------------------------------------------------------------------
# Cell 4: Session State Initialization and Custom Styling
# -----------------------------------------------------------------------------

# ------ Initialize Session State for Food Selections ------

if 'food_selections' not in st.session_state:
    st.session_state.food_selections = {}

# ------ Custom CSS for Enhanced Button Styling ------

st.markdown("""
<style>
/* Active button styling */
.active-button {
    background-color: #ff6b6b !important;
    color: white !important;
    border: 2px solid #ff6b6b !important;
}

/* Custom button container */
.button-container {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Cell 5: Application Title and Introduction Display
# -----------------------------------------------------------------------------

st.title("Nutrition Tracker 🥗")
st.markdown("""
Welcome to your interactive nutrition tracking application! This tool helps
you plan and monitor your daily food intake for healthy weight gain using
vegetarian food sources.
""")

# -----------------------------------------------------------------------------
# Cell 6: Daily Nutritional Targets Display Section
# -----------------------------------------------------------------------------

st.header("Daily Nutritional Targets 🎯")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Calories", f"{daily_targets['calories']['min']}-"
                        f"{daily_targets['calories']['max']} kcal")
with col2:
    st.metric("Protein", f"{daily_targets['protein']['min']}-"
                       f"{daily_targets['protein']['max']}g")
with col3:
    st.metric("Carbohydrates", f"{daily_targets['carbs']['min']}-"
                             f"{daily_targets['carbs']['max']}g")
with col4:
    st.metric("Fat", f"{daily_targets['fat']['min']}-"
                   f"{daily_targets['fat']['max']}g")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 7: Interactive Food Selection Interface
# -----------------------------------------------------------------------------

st.header("Select Your Foods 📝")
st.markdown("Choose foods using the buttons (0-5 servings) or enter custom "
            "serving amounts.")

# ------ Create Category Tabs for Food Organization ------

# Create tabs for different food categories, filtering out empty ones
available_categories = [cat for cat, items in foods.items() if items]
tabs = st.tabs(available_categories)

for i, category in enumerate(available_categories):
    items = foods[category]
    with tabs[i]:
        # Create columns for better layout
        for j in range(0, len(items), 2):
            col1, col2 = st.columns(2)

            # ------ First Item in Row ------
            if j < len(items):
                with col1:
                    food = items[j]
                    st.subheader(food['name'])

                    key = f"{category}_{food['name']}"
                    current_serving = st.session_state.food_selections.get(food['name'], 0.0)

                    button_cols = st.columns(5)
                    for k in range(1, 6):
                        with button_cols[k-1]:
                            button_type = "primary" if current_serving == float(k) else "secondary"
                            if st.button(f"{k}×", key=f"{key}_{k}", type=button_type):
                                st.session_state.food_selections[food['name']] = float(k)
                                st.rerun()

                    custom_serving = st.number_input(
                        "Custom servings:",
                        min_value=0.0, max_value=10.0,
                        value=float(current_serving), step=0.1,
                        key=f"{key}_custom"
                    )

                    if custom_serving != current_serving:
                        if custom_serving > 0:
                            st.session_state.food_selections[food['name']] = custom_serving
                        elif food['name'] in st.session_state.food_selections:
                            del st.session_state.food_selections[food['name']]
                        st.rerun()

                    st.caption(f"Per Serving: {food['calories']} kcal | "
                               f"{food['protein']}g Protein | "
                               f"{food['carbs']}g Carbs | "
                               f"{food['fat']}g Fat")

            # ------ Second Item in Row ------
            if j + 1 < len(items):
                with col2:
                    food = items[j + 1]
                    st.subheader(food['name'])

                    key = f"{category}_{food['name']}"
                    current_serving = st.session_state.food_selections.get(food['name'], 0.0)

                    button_cols = st.columns(5)
                    for k in range(1, 6):
                        with button_cols[k-1]:
                            button_type = "primary" if current_serving == float(k) else "secondary"
                            if st.button(f"{k}×", key=f"{key}_{k}", type=button_type):
                                st.session_state.food_selections[food['name']] = float(k)
                                st.rerun()

                    custom_serving = st.number_input(
                        "Custom servings:",
                        min_value=0.0, max_value=10.0,
                        value=float(current_serving), step=0.1,
                        key=f"{key}_custom"
                    )

                    if custom_serving != current_serving:
                        if custom_serving > 0:
                            st.session_state.food_selections[food['name']] = custom_serving
                        elif food['name'] in st.session_state.food_selections:
                            del st.session_state.food_selections[food['name']]
                        st.rerun()

                    st.caption(f"Per Serving: {food['calories']} kcal | "
                               f"{food['protein']}g Protein | "
                               f"{food['carbs']}g Carbs | "
                               f"{food['fat']}g Fat")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 8: Calculation Button and Nutritional Results Display
# -----------------------------------------------------------------------------

if st.button("Calculate Daily Intake", type="primary", use_container_width=True):

    total_calories, total_protein, total_carbs, total_fat = 0, 0, 0, 0
    selected_foods = []

    for category, items in foods.items():
        for food in items:
            servings = st.session_state.food_selections.get(food['name'], 0)
            if servings > 0:
                total_calories += food['calories'] * servings
                total_protein += food['protein'] * servings
                total_carbs += food['carbs'] * servings
                total_fat += food['fat'] * servings
                selected_foods.append({'food': food, 'servings': servings})

    st.header("Daily Intake Summary 📊")

    if selected_foods:
        st.subheader("Foods Consumed Today: 🍽️")
        cols = st.columns(3)
        for i, item in enumerate(selected_foods):
            with cols[i % 3]:
                st.write(f"• {item['food']['name']} x {item['servings']:.1f}")
    else:
        st.info("No foods selected. 🥗")

    st.subheader("Total Nutritional Intake: 📈")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calories", f"{total_calories:.0f} kcal")
    with col2:
        st.metric("Protein", f"{total_protein:.1f}g")
    with col3:
        st.metric("Carbohydrates", f"{total_carbs:.1f}g")
    with col4:
        st.metric("Fat", f"{total_fat:.1f}g")

    st.subheader("Daily Target Achievement: 🎯")

    cal_percent = (min(total_calories / daily_targets['calories']['min'] * 100, 100) if daily_targets['calories']['min'] > 0 else 0)
    st.progress(cal_percent / 100, text=f"Calories: {cal_percent:.0f}% of minimum target")

    prot_percent = (min(total_protein / daily_targets['protein']['min'] * 100, 100) if daily_targets['protein']['min'] > 0 else 0)
    st.progress(prot_percent / 100, text=f"Protein: {prot_percent:.0f}% of minimum target")

    carb_percent = (min(total_carbs / daily_targets['carbs']['min'] * 100, 100) if daily_targets['carbs']['min'] > 0 else 0)
    st.progress(carb_percent / 100, text=f"Carbohydrates: {carb_percent:.0f}% of minimum target")

    fat_percent = (min(total_fat / daily_targets['fat']['min'] * 100, 100) if daily_targets['fat']['min'] > 0 else 0)
    st.progress(fat_percent / 100, text=f"Fat: {fat_percent:.0f}% of minimum target")

    st.subheader("Personalized Recommendations: 💡")
    recommendations = []
    if total_calories < daily_targets['calories']['min']:
        recommendations.append(f"• You need {daily_targets['calories']['min'] - total_calories:.0f} more calories.")
    if total_protein < daily_targets['protein']['min']:
        recommendations.append(f"• You need {daily_targets['protein']['min'] - total_protein:.0f}g more protein.")
    if total_carbs < daily_targets['carbs']['min']:
        recommendations.append(f"• You need {daily_targets['carbs']['min'] - total_carbs:.0f}g more carbohydrates.")
    if total_fat < daily_targets['fat']['min']:
        recommendations.append(f"• You need {daily_targets['fat']['min'] - total_fat:.0f}g more healthy fats.")

    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("Outstanding work! You've met all your minimum daily nutritional targets! 🎉")

    if selected_foods:
        st.subheader("Detailed Food Log 📋")
        food_log = [
            {
                'Food': item['food']['name'],
                'Servings': f"{item['servings']:.1f}",
                'Calories': item['food']['calories'] * item['servings'],
                'Protein (g)': item['food']['protein'] * item['servings'],
                'Carbs (g)': item['food']['carbs'] * item['servings'],
                'Fat (g)': item['food']['fat'] * item['servings']
            }
            for item in selected_foods
        ]
        df_log = pd.DataFrame(food_log)
        st.dataframe(df_log.style.format({
            'Calories': '{:.0f}',
            'Protein (g)': '{:.1f}',
            'Carbs (g)': '{:.1f}',
            'Fat (g)': '{:.1f}'
        }), use_container_width=True)

    st.markdown("---")
    st.markdown("Thanks for using the Nutrition Tracker! Keep up the great work! 💪")
    print("Daily nutritional intake calculated successfully! 🎯")

# -----------------------------------------------------------------------------
# Cell 9: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("Clear All Selections", use_container_width=True):
    st.session_state.food_selections.clear()
    st.rerun()
    print("All food selections have been cleared successfully! ✨")
