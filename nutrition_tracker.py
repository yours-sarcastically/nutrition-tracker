# -----------------------------------------------------------------------------
# Streamlit Vegetarian Nutrition Tracker for Healthy Weight Gain
# -----------------------------------------------------------------------------

"""
Interactive nutrition tracking application for vegetarian weight gain planning.

This Streamlit application provides a comprehensive tool for tracking daily
nutritional intake using vegetarian food sources. The application calculates
total caloric and macronutrient consumption based on user-selected foods and
serving sizes, then compares results against established daily targets for
healthy weight gain.

The application features a categorized food database organized by nutritional
focus including primary protein sources, carbohydrate sources, fat sources,
and micronutrient sources. Users can select foods through quick-select buttons
or custom serving inputs, with real-time calculation of nutritional totals.

Key Features:
- Daily nutritional targets with minimum and maximum ranges for calories,
  protein, carbohydrates, and fat
- Categorized vegetarian food database with detailed nutritional information
- Interactive food selection interface with quick-select buttons and custom
  serving inputs
- Real-time calculation and display of total nutritional intake
- Progress tracking against daily targets with visual progress bars
- Personalized recommendations for meeting nutritional goals
- Detailed food log with tabular display of selected items

Usage:
1. Run the Streamlit application using 'streamlit run nutrition_tracker.py'
2. View daily nutritional targets displayed at the top of the interface
3. Navigate through food category tabs to select desired foods
4. Use quick-select buttons (1x-5x) or custom serving inputs for portions
5. Click 'Calculate Daily Intake' to view comprehensive nutritional summary
6. Review progress bars and personalized recommendations
7. Use 'Clear All Selections' to reset all food choices

Daily Targets:
- Calories: 2800-2900 kcal for healthy weight gain
- Protein: 110-120g for muscle building and recovery
- Carbohydrates: 410-430g for energy and performance
- Fat: 75-85g for hormone production and absorption

The application maintains session state to preserve food selections across
user interactions and provides comprehensive feedback on nutritional adequacy
through progress tracking and targeted recommendations.
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

# Function to load and categorize food data from the CSV file
@st.cache_data
def load_food_database(file_path):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Define the exact categorization mapping based on the provided specification
    category_mapping = {
        'Eggs': 'PRIMARY PROTEIN SOURCES',
        'Greek Yogurt': 'PRIMARY PROTEIN SOURCES',
        'Protein Powder': 'PRIMARY PROTEIN SOURCES',
        'Lentils': 'PRIMARY PROTEIN SOURCES',
        'Chickpeas': 'PRIMARY PROTEIN SOURCES',
        'Cottage Cheese': 'PRIMARY PROTEIN SOURCES',
        'Kidney Beans': 'PRIMARY PROTEIN SOURCES',
        'Milk': 'PRIMARY PROTEIN SOURCES',
        'Mozzarella Cheese': 'PRIMARY PROTEIN SOURCES',
        'Hummus': 'PRIMARY PROTEIN SOURCES',
        'Cheese Tortellini': 'PRIMARY PROTEIN SOURCES',
        'Olive Oil': 'PRIMARY FAT SOURCES',
        'Almonds': 'PRIMARY FAT SOURCES',
        'Chia Seeds': 'PRIMARY FAT SOURCES',
        'Avocados': 'PRIMARY FAT SOURCES',
        'Sunflower Seeds': 'PRIMARY FAT SOURCES',
        'Mixed Nuts': 'PRIMARY FAT SOURCES',
        'Peanut Butter': 'PRIMARY FAT SOURCES',
        'Tahini': 'PRIMARY FAT SOURCES',
        'Trail Mix': 'PRIMARY FAT SOURCES',
        'Heavy Cream': 'PRIMARY FAT SOURCES',
        'Oats': 'CARBOHYDRATE SOURCES',
        'Potatoes': 'CARBOHYDRATE SOURCES',
        'Mixed Vegetables': 'CARBOHYDRATE SOURCES',
        'Green Peas': 'CARBOHYDRATE SOURCES',
        'Multigrain Bread': 'CARBOHYDRATE SOURCES',
        'Corn': 'CARBOHYDRATE SOURCES',
        'Bananas': 'CARBOHYDRATE SOURCES',
        'Couscous': 'CARBOHYDRATE SOURCES',
        'White Rice': 'CARBOHYDRATE SOURCES',
        'Pasta': 'CARBOHYDRATE SOURCES',
        'Spinach Tortellini': 'CARBOHYDRATE SOURCES',
        'Spinach': 'PRIMARY MICRONUTRIENT SOURCES',
        'Broccoli': 'PRIMARY MICRONUTRIENT SOURCES',
        'Berries': 'PRIMARY MICRONUTRIENT SOURCES',
        'Tomatoes': 'PRIMARY MICRONUTRIENT SOURCES',
        'Carrots': 'PRIMARY MICRONUTRIENT SOURCES',
        'Mushrooms': 'PRIMARY MICRONUTRIENT SOURCES',
        'Orange Juice': 'PRIMARY MICRONUTRIENT SOURCES',
        'Apple Juice': 'PRIMARY MICRONUTRIENT SOURCES',
        'Fruit Juice': 'PRIMARY MICRONUTRIENT SOURCES'
    }

    # Create the food dictionary in the required format with exact ordering
    foods = {
        'PRIMARY PROTEIN SOURCES': [],
        'PRIMARY FAT SOURCES': [],
        'CARBOHYDRATE SOURCES': [],
        'PRIMARY MICRONUTRIENT SOURCES': []
    }

    # Define the exact order for each category as specified
    protein_order = ['Eggs', 'Greek Yogurt', 'Protein Powder', 'Lentils', 'Chickpeas', 'Cottage Cheese', 'Kidney Beans', 'Milk', 'Mozzarella Cheese', 'Hummus', 'Cheese Tortellini']
    fat_order = ['Olive Oil', 'Almonds', 'Chia Seeds', 'Avocados', 'Sunflower Seeds', 'Mixed Nuts', 'Peanut Butter', 'Tahini', 'Trail Mix', 'Heavy Cream']
    carb_order = ['Oats', 'Potatoes', 'Mixed Vegetables', 'Green Peas', 'Multigrain Bread', 'Corn', 'Bananas', 'Couscous', 'White Rice', 'Pasta', 'Spinach Tortellini']
    micro_order = ['Spinach', 'Broccoli', 'Berries', 'Tomatoes', 'Carrots', 'Mushrooms', 'Orange Juice', 'Apple Juice', 'Fruit Juice']

    # Process foods in the specified order
    for food_name in protein_order + fat_order + carb_order + micro_order:
        # Find the matching row in the DataFrame
        matching_row = df[df['Food Name'] == food_name]
        if not matching_row.empty:
            row = matching_row.iloc[0]
            category = category_mapping.get(food_name, 'PRIMARY MICRONUTRIENT SOURCES')
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
