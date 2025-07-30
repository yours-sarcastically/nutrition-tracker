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

# ------ Comprehensive Vegetarian Food Database ------

# Food database organized by nutritional categories for optimal meal planning
foods = {
    'PRIMARY PROTEIN SOURCES': [
        {'name': 'Greek Yogurt (1 Cup)', 'calories': 130, 'protein': 23,
         'carbs': 9, 'fat': 0},
        {'name': 'Cottage Cheese (1 Cup)', 'calories': 220, 'protein': 25,
         'carbs': 6, 'fat': 9},
        {'name': 'Whey Protein (1 Tablespoon)', 'calories': 28, 'protein': 5,
         'carbs': 1, 'fat': 0.5},
        {'name': 'Chickpeas (1 Cup Cooked)', 'calories': 270, 'protein': 14,
         'carbs': 45, 'fat': 4},
        {'name': 'Black Beans (1 Cup Cooked)', 'calories': 240, 'protein': 15,
         'carbs': 41, 'fat': 1},
        {'name': 'Tofu (200 Grams)', 'calories': 152, 'protein': 17,
         'carbs': 3, 'fat': 9},
        {'name': 'Lentils (1 Cup Cooked)', 'calories': 230, 'protein': 18,
         'carbs': 40, 'fat': 1},
        {'name': 'Beans (1 Cup Cooked)', 'calories': 225, 'protein': 15,
         'carbs': 40, 'fat': 1},
        {'name': 'Mozzarella Cheese (100g)', 'calories': 280, 'protein': 22,
         'carbs': 3, 'fat': 20},
        {'name': 'Hummus (1 Cup)', 'calories': 410, 'protein': 20,
         'carbs': 36, 'fat': 24},
        {'name': 'Nuts (1 Cup)', 'calories': 815, 'protein': 30,
         'carbs': 28, 'fat': 70},
        {'name': 'Eggs (1 Medium)', 'calories': 63, 'protein': 5.5,
         'carbs': 0.5, 'fat': 4.5},
        {'name': 'Cheese (1 Slice)', 'calories': 115, 'protein': 7,
         'carbs': 0.5, 'fat': 9},
        {'name': 'Peas (1 Cup Cooked)', 'calories': 135, 'protein': 9,
         'carbs': 25, 'fat': 0.5},
    ],
    'PRIMARY CARBOHYDRATE SOURCES': [
        {'name': 'Oats (1 Cup Cooked)', 'calories': 165, 'protein': 6,
         'carbs': 28, 'fat': 3.5},
        {'name': 'Rice (1 Cup Cooked)', 'calories': 205, 'protein': 4,
         'carbs': 45, 'fat': 0.5},
        {'name': 'Pasta (1 Cup Cooked)', 'calories': 220, 'protein': 8,
         'carbs': 43, 'fat': 1},
        {'name': 'Whole Wheat Bread (2 Slices)', 'calories': 160, 'protein': 8,
         'carbs': 30, 'fat': 2},
        {'name': 'Potatoes (1 Medium)', 'calories': 160, 'protein': 4,
         'carbs': 37, 'fat': 0},
        {'name': 'Bananas (1 Medium)', 'calories': 105, 'protein': 1,
         'carbs': 27, 'fat': 0.5},
        {'name': 'Corn (1 Cup Cooked)', 'calories': 175, 'protein': 5,
         'carbs': 41, 'fat': 2},
        {'name': 'Muesli (1 Cup)', 'calories': 340, 'protein': 8,
         'carbs': 66, 'fat': 5},
        {'name': 'Couscous (1 Cup Cooked)', 'calories': 175, 'protein': 6,
         'carbs': 36, 'fat': 0.5},
        {'name': 'Dates (5 Pieces)', 'calories': 115, 'protein': 1,
         'carbs': 30, 'fat': 0},
        {'name': 'Dried Fruits (1 Cup)', 'calories': 480, 'protein': 4,
         'carbs': 127, 'fat': 1},
        {'name': 'Trail Mix (1 Cup)', 'calories': 700, 'protein': 21,
         'carbs': 67, 'fat': 44},
    ],
    'PRIMARY FAT SOURCES': [
        {'name': 'Peanut Butter (1 Tablespoon)', 'calories': 95, 'protein': 4,
         'carbs': 3, 'fat': 8},
        {'name': 'Olive Oil (1 Tablespoon)', 'calories': 120, 'protein': 0,
         'carbs': 0, 'fat': 14},
        {'name': 'Avocados (1 Medium)', 'calories': 320, 'protein': 4,
         'carbs': 17, 'fat': 29},
        {'name': 'Coconut Milk - Full Fat (1 Cup)', 'calories': 552,
         'protein': 5.5, 'carbs': 13, 'fat': 57},
        {'name': 'Whole Milk (1 Cup)', 'calories': 150, 'protein': 8,
         'carbs': 12, 'fat': 8},
        {'name': 'Chia Seeds (1 Tablespoon)', 'calories': 60, 'protein': 2,
         'carbs': 5, 'fat': 4},
        {'name': 'Cream Cheese (1 Tablespoon)', 'calories': 50, 'protein': 1,
         'carbs': 1, 'fat': 5},
    ],
    'PRIMARY MICRONUTRIENT SOURCES': [
        {'name': 'Spinach (1 Cup Raw)', 'calories': 7, 'protein': 1,
         'carbs': 1, 'fat': 0},
        {'name': 'Broccoli (1 Cup Cooked)', 'calories': 55, 'protein': 4,
         'carbs': 11, 'fat': 0.5},
        {'name': 'Berries (1 Cup)', 'calories': 85, 'protein': 1,
         'carbs': 21, 'fat': 0.5},
        {'name': 'Carrots (1 Cup Cooked)', 'calories': 55, 'protein': 1,
         'carbs': 13, 'fat': 0},
        {'name': 'Mushrooms (1 Cup Cooked)', 'calories': 45, 'protein': 3,
         'carbs': 8, 'fat': 0.5},
        {'name': 'Apples (1 Medium)', 'calories': 95, 'protein': 0.5,
         'carbs': 25, 'fat': 0},
        {'name': 'Tomato Puree (1 Cup)', 'calories': 98, 'protein': 4,
         'carbs': 22, 'fat': 0.5},
        {'name': 'Oranges (1 Medium)', 'calories': 60, 'protein': 1,
         'carbs': 15, 'fat': 0},
        {'name': 'Cauliflower (1 Cup Cooked)', 'calories': 30, 'protein': 2,
         'carbs': 5, 'fat': 0.5},
    ]
}

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

st.title("🥗 Nutrition Tracker")
st.markdown("""
Welcome to your interactive nutrition tracking application! This tool helps
you plan and monitor your daily food intake for healthy weight gain using
vegetarian food sources.
""")

# -----------------------------------------------------------------------------
# Cell 6: Daily Nutritional Targets Display Section
# -----------------------------------------------------------------------------

st.header("🎯 Daily Nutritional Targets")

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

st.header("📝 Select Your Foods")
st.markdown("Choose foods using the buttons (0-5 servings) or enter custom "
            "serving amounts.")

# ------ Create Category Tabs for Food Organization ------

# Create tabs for different food categories
tabs = st.tabs(list(foods.keys()))

for i, (category, items) in enumerate(foods.items()):
    with tabs[i]:
        # Create columns for better layout
        for j in range(0, len(items), 2):
            col1, col2 = st.columns(2)

            # ------ First Item in Row ------
            if j < len(items):
                with col1:
                    food = items[j]
                    st.subheader(food['name'])

                    # Create unique key for this food item
                    key = f"{category}_{food['name']}"

                    # Get current serving value
                    current_serving = st.session_state.food_selections.get(
                        food['name'], 0.0)

                    # Quick select buttons with active state styling
                    button_col1, button_col2, button_col3, button_col4, \
                        button_col5 = st.columns(5)

                    with button_col1:
                        button_type = ("primary" if current_serving == 1.0
                                       else "secondary")
                        if st.button("1×", key=f"{key}_1", type=button_type):
                            st.session_state.food_selections[food['name']] = 1.0
                            st.rerun()

                    with button_col2:
                        button_type = ("primary" if current_serving == 2.0
                                       else "secondary")
                        if st.button("2×", key=f"{key}_2", type=button_type):
                            st.session_state.food_selections[food['name']] = 2.0
                            st.rerun()

                    with button_col3:
                        button_type = ("primary" if current_serving == 3.0
                                       else "secondary")
                        if st.button("3×", key=f"{key}_3", type=button_type):
                            st.session_state.food_selections[food['name']] = 3.0
                            st.rerun()

                    with button_col4:
                        button_type = ("primary" if current_serving == 4.0
                                       else "secondary")
                        if st.button("4×", key=f"{key}_4", type=button_type):
                            st.session_state.food_selections[food['name']] = 4.0
                            st.rerun()

                    with button_col5:
                        button_type = ("primary" if current_serving == 5.0
                                       else "secondary")
                        if st.button("5×", key=f"{key}_5", type=button_type):
                            st.session_state.food_selections[food['name']] = 5.0
                            st.rerun()

                    # Custom serving input - ensure value is always float
                    custom_serving = st.number_input(
                        "Custom servings:",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(current_serving),
                        step=0.1,
                        key=f"{key}_custom"
                    )

                    if custom_serving > 0:
                        st.session_state.food_selections[
                            food['name']] = custom_serving
                    elif (food['name'] in st.session_state.food_selections
                          and custom_serving == 0):
                        del st.session_state.food_selections[food['name']]

                    # Display nutritional info
                    st.caption(f"Per Serving: {food['calories']} kcal | "
                               f"{food['protein']} g Protein | "
                               f"{food['carbs']} g Carbohydrates | "
                               f"{food['fat']} g Fat")

            # ------ Second Item in Row ------
            if j + 1 < len(items):
                with col2:
                    food = items[j + 1]
                    st.subheader(food['name'])

                    # Create unique key for this food item
                    key = f"{category}_{food['name']}"

                    # Get current serving value
                    current_serving = st.session_state.food_selections.get(
                        food['name'], 0.0)

                    # Quick select buttons with active state styling
                    button_col1, button_col2, button_col3, button_col4, \
                        button_col5 = st.columns(5)

                    with button_col1:
                        button_type = ("primary" if current_serving == 1.0
                                       else "secondary")
                        if st.button("1×", key=f"{key}_1", type=button_type):
                            st.session_state.food_selections[food['name']] = 1.0
                            st.rerun()

                    with button_col2:
                        button_type = ("primary" if current_serving == 2.0
                                       else "secondary")
                        if st.button("2×", key=f"{key}_2", type=button_type):
                            st.session_state.food_selections[food['name']] = 2.0
                            st.rerun()

                    with button_col3:
                        button_type = ("primary" if current_serving == 3.0
                                       else "secondary")
                        if st.button("3×", key=f"{key}_3", type=button_type):
                            st.session_state.food_selections[food['name']] = 3.0
                            st.rerun()

                    with button_col4:
                        button_type = ("primary" if current_serving == 4.0
                                       else "secondary")
                        if st.button("4×", key=f"{key}_4", type=button_type):
                            st.session_state.food_selections[food['name']] = 4.0
                            st.rerun()

                    with button_col5:
                        button_type = ("primary" if current_serving == 5.0
                                       else "secondary")
                        if st.button("5×", key=f"{key}_5", type=button_type):
                            st.session_state.food_selections[food['name']] = 5.0
                            st.rerun()

                    # Custom serving input - ensure value is always float
                    custom_serving = st.number_input(
                        "Custom servings:",
                        min_value=0.0,
                        max_value=10.0,
                        value=float(current_serving),
                        step=0.1,
                        key=f"{key}_custom"
                    )

                    if custom_serving > 0:
                        st.session_state.food_selections[
                            food['name']] = custom_serving
                    elif (food['name'] in st.session_state.food_selections
                          and custom_serving == 0):
                        del st.session_state.food_selections[food['name']]

                    # Display nutritional info
                    st.caption(f"Per Serving: {food['calories']} kcal | "
                               f"{food['protein']} g Protein | "
                               f"{food['carbs']} g Carbohydrates | "
                               f"{food['fat']} g Fat")

st.markdown("---")

# -----------------------------------------------------------------------------
# Cell 8: Calculation Button and Nutritional Results Display
# -----------------------------------------------------------------------------

if st.button("📊 Calculate Daily Intake", type="primary",
             use_container_width=True):

    # ------ Initialize Nutritional Totals ------

    # Initialize totals
    total_calories = 0
    total_protein = 0
    total_carbs = 0
    total_fat = 0
    selected_foods = []

    # ------ Calculate Total Nutritional Values ------

    # Calculate totals
    for category, items in foods.items():
        for food in items:
            if food['name'] in st.session_state.food_selections:
                servings = st.session_state.food_selections[food['name']]
                if servings > 0:
                    total_calories += food['calories'] * servings
                    total_protein += food['protein'] * servings
                    total_carbs += food['carbs'] * servings
                    total_fat += food['fat'] * servings
                    selected_foods.append(f"{food['name']} x {servings:.1f}")

    # ------ Display Comprehensive Results ------

    # Display results
    st.header("📊 Daily Intake Summary")

    # Selected foods
    if selected_foods:
        st.subheader("🍽️ Foods Consumed Today:")
        # Create columns for better display
        cols = st.columns(3)
        for i, food in enumerate(selected_foods):
            with cols[i % 3]:
                st.write(f"• {food}")
    else:
        st.info("No foods selected")

    # ------ Nutritional Totals Display ------

    # Nutritional totals
    st.subheader("📈 Total Nutritional Intake:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calories", f"{total_calories:.0f} kcal")
    with col2:
        st.metric("Protein", f"{total_protein:.1f}g")
    with col3:
        st.metric("Carbohydrates", f"{total_carbs:.1f}g")
    with col4:
        st.metric("Fat", f"{total_fat:.1f}g")

    # ------ Progress Bars for Target Achievement ------

    # Progress bars
    st.subheader("🎯 Daily Target Achievement:")

    # Calories progress
    cal_percent = (min((total_calories / daily_targets['calories']['min'])
                       * 100, 100) if daily_targets['calories']['min'] > 0
                   else 0)
    st.progress(cal_percent / 100,
              text=f"Calories: {cal_percent:.0f}% of minimum target")

    # Protein progress
    prot_percent = (min((total_protein / daily_targets['protein']['min'])
                        * 100, 100) if daily_targets['protein']['min'] > 0
                    else 0)
    st.progress(prot_percent / 100,
              text=f"Protein: {prot_percent:.0f}% of minimum target")

    # Carbs progress
    carb_percent = (min((total_carbs / daily_targets['carbs']['min'])
                        * 100, 100) if daily_targets['carbs']['min'] > 0
                    else 0)
    st.progress(carb_percent / 100,
              text=f"Carbohydrates: {carb_percent:.0f}% of minimum target")

    # Fat progress
    fat_percent = (min((total_fat / daily_targets['fat']['min'])
                       * 100, 100) if daily_targets['fat']['min'] > 0
                   else 0)
    st.progress(fat_percent / 100,
              text=f"Fat: {fat_percent:.0f}% of minimum target")

    # ------ Personalized Recommendations ------

    # Recommendations
    st.subheader("💡 Personalized Recommendations:")
    recommendations = []

    if total_calories < daily_targets['calories']['min']:
        deficit = daily_targets['calories']['min'] - total_calories
        recommendations.append(
            f"• You need {deficit:.0f} more calories to reach your minimum "
            f"daily target")

    if total_protein < daily_targets['protein']['min']:
        deficit = daily_targets['protein']['min'] - total_protein
        recommendations.append(
            f"• You need {deficit:.0f}g more protein for muscle building "
            f"and recovery")

    if total_carbs < daily_targets['carbs']['min']:
        deficit = daily_targets['carbs']['min'] - total_carbs
        recommendations.append(
            f"• You need {deficit:.0f}g more carbohydrates for energy "
            f"and performance")

    if total_fat < daily_targets['fat']['min']:
        deficit = daily_targets['fat']['min'] - total_fat
        recommendations.append(
            f"• You need {deficit:.0f}g more healthy fats for hormone "
            f"production")

    if recommendations:
        for rec in recommendations:
            st.write(rec)
    else:
        st.success("🎉 Outstanding work! You have successfully met all your "
                   "minimum daily nutritional targets!")

    # ------ Detailed Food Log Table ------

    # Display selected foods as a table
    if selected_foods:
        st.subheader("📋 Detailed Food Log")
        food_log = []
        for category, items in foods.items():
            for food in items:
                if food['name'] in st.session_state.food_selections:
                    servings = st.session_state.food_selections[food['name']]
                    if servings > 0:
                        food_log.append({
                            'Food': food['name'],
                            'Servings': servings,
                            'Calories': food['calories'] * servings,
                            'Protein (g)': food['protein'] * servings,
                            'Carbs (g)': food['carbs'] * servings,
                            'Fat (g)': food['fat'] * servings
                        })

        df = pd.DataFrame(food_log)
        st.dataframe(df, use_container_width=True)

    # ------ Thank You Message ------

    # Footer message - only appears after calculation
    st.markdown("---")
    st.markdown("""
    🌟 Thanks for using the Nutrition Tracker!
    Keep fueling your body with wholesome plant-based goodness, and remember -
    every healthy choice you make is a step toward your goals. You've got this! 💪
    """)
    print("📊 Daily nutritional intake calculated successfully! 🎯")

# -----------------------------------------------------------------------------
# Cell 9: Clear Selections Button and Application Reset
# -----------------------------------------------------------------------------

if st.button("🔄 Clear All Selections", use_container_width=True):
    st.session_state.food_selections = {}
    st.rerun()
    print("🔄 All food selections have been cleared successfully! ✨")
