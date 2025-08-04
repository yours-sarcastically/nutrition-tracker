# core/data.py

@st.cache_data
def get_processed_food_data(file_path: str) -> Dict[str, List[FoodItem]]:
    """
    Loads the food database from a CSV, processes it into FoodItem objects,
    and assigns ranking emojis in a single, cacheable operation using corrected logic.
    """
    # --- Part 1: Load the database from CSV ---
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: The data file was not found at '{file_path}'. Please ensure the file exists and the path is correct.")
        return {}

    foods: Dict[str, List[FoodItem]] = {cat: [] for cat in CONFIG['nutrient_map'].keys()}
    all_foods_flat: List[FoodItem] = [] # A single list of all food items

    for _, row in df.iterrows():
        category = row['category']
        food_item = FoodItem(
            name=f"{row['name']} ({row['serving_unit']})",
            calories=row['calories'],
            protein=row['protein'],
            carbs=row['carbs'],
            fat=row['fat']
        )
        if category in foods:
            foods[category].append(food_item)
        all_foods_flat.append(food_item)

    # --- Part 2: Assign Emojis (with Corrected Global Ranking Logic) ---
    top_foods = {'protein': [], 'carbs': [], 'fat': [], 'micro': [], 'calories': {}}

    # CORRECTED LOGIC: Rank across ALL foods, not within categories
    if all_foods_flat:
        top_foods['protein'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.protein, reverse=True)[:3]]
        top_foods['carbs'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.carbs, reverse=True)[:3]]
        top_foods['fat'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.fat, reverse=True)[:3]]
        # 'micro' is still sorted by protein as per config, but across all foods
        top_foods['micro'] = [f.name for f in sorted(all_foods_flat, key=lambda x: x.protein, reverse=True)[:3]]

    # High-calorie check remains category-specific as it's for context within the tab
    for category, items in foods.items():
        top_foods['calories'][category] = [f.name for f in sorted(items, key=lambda x: x.calories, reverse=True)[:3]]

    all_top_nutrient_foods = {food for key in ['protein', 'carbs', 'fat', 'micro'] for food in top_foods[key]}
    food_rank_counts = {name: sum(1 for key in ['protein', 'carbs', 'fat', 'micro'] if name in top_foods[key]) for name in all_top_nutrient_foods}
    superfoods = {name for name, count in food_rank_counts.items() if count > 1}

    emoji_mapping = {'superfoods': '🥇', 'high_cal_nutrient': '💥', 'high_calorie': '🔥', 'protein': '💪', 'carbs': '🍚', 'fat': '🥑', 'micro': '🥦'}

    for category, items in foods.items():
        for food in items:
            is_top_nutrient = food.name in all_top_nutrient_foods
            is_high_calorie = food.name in top_foods['calories'].get(category, [])

            if food.name in superfoods: food.emoji = emoji_mapping['superfoods']
            elif is_high_calorie and is_top_nutrient: food.emoji = emoji_mapping['high_cal_nutrient']
            elif is_high_calorie: food.emoji = emoji_mapping['high_calorie']
            elif food.name in top_foods['protein']: food.emoji = emoji_mapping['protein']
            elif food.name in top_foods['carbs']: food.emoji = emoji_mapping['carbs']
            elif food.name in top_foods['fat']: food.emoji = emoji_mapping['fat']
            elif food.name in top_foods['micro']: food.emoji = emoji_mapping['micro']
            else: food.emoji = ''

    return foods
