"""
Food Nutrition Analysis and Data Export Tool Using USDA FoodData Central API

This script retrieves nutrition information for food items from the USDA Survey
(FNDDS) database using the FoodData Central API. The script automatically
determines appropriate serving sizes based on food categories and descriptions,
then calculates nutrition values for those realistic portions. All results are
displayed in the console and saved to a CSV file for further analysis.

The script requires a valid USDA FoodData Central API key. Users should obtain 
their own key for production use from: https://fdc.nal.usda.gov/api-guide.html
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Dependencies
# -----------------------------------------------------------------------------

import requests
import time
import re
import csv
from typing import Dict, List, Optional, Any

# -----------------------------------------------------------------------------
# Cell 2: Food Priority Configuration
# -----------------------------------------------------------------------------

# ------ Unified Food Priority Configuration Dictionary ------
# A single, unified dictionary to manage all serving size priorities
# The logic will check 'SPECIFIC' first, then 'CATEGORIES'
FOOD_PRIORITIES = {
    # === HIGH PRIORITY: Specific Food Overrides (for exceptions) ===
    'SPECIFIC': {
        # Keyword from description : [Prioritized list of units]
        'potato, nfs': ['potato'],
        'nutritional powder mix': ['scoop', 'packet'],
        'high protein': ['scoop', 'packet'],
        'mozzarella': ['cup'],
        'cream cheese': ['tablespoon'],
        'hummus': ['tablespoon'],
        'peanut butter': ['tablespoon'],
        'almond butter': ['tablespoon'],
        'tahini': ['tablespoon'],
        'avocado': ['fruit'],
        'banana': ['banana'],
        'tortellini': ['cup'],
    },

    # === LOWER PRIORITY: General Category Rules ===
    'CATEGORIES': {
        # Dairy & Dairy Products
        'yogurt': ['cup', 'container', 'oz'],
        'milk': ['cup', 'oz', 'container'],
        'cheese': ['slice', 'stick', 'cup', 'curd', 'inch'],
        'cream': ['tablespoon', 'container', 'cup'],
        'cottage': ['cup'],

        # Protein Foods
        'beans': ['cup'],
        'lentils': ['cup'],
        'chickpeas': ['cup', 'pea'],
        'edamame': ['cup', 'pod'],
        'almonds': ['oz', 'cup', 'nut', 'package'],
        'mixed nuts': ['oz', 'cup', 'package'],
        'seeds': ['oz', 'tablespoon', 'cup', 'package'],
        'egg': ['egg', 'cup'],

        # Grains
        'oats': ['cup'],
        'rice': ['cup'],
        'pasta': ['cup', 'oz'],
        'bread': ['slice', 'inch'],
        'bagel': ['bagel', 'large', 'regular', 'small', 'miniature'],
        'quinoa': ['cup'],
        'corn': ['cup', 'ear'],
        'couscous': ['cup', 'oz'],

        # Fruits
        'apple': ['medium', 'large', 'small', 'cup', 'slice', 'package'],
        'orange': ['fruit', 'medium', 'large', 'small', 'cup', 'section',
                   'slice'],
        'berries': ['cup', 'berry'],
        'raisins': ['cup', 'box', 'raisin', 'oz'],
        'date': ['date', 'cup'],

        # Vegetables
        'spinach': ['cup', 'leaf'],
        'broccoli': ['cup', 'floweret', 'piece'],
        'cauliflower': ['cup', 'floweret', 'piece'],
        'carrots': ['cup', 'carrot', 'slice', 'stick'],
        'mushrooms': ['cup', 'whole', 'slice', 'piece'],
        'tomatoes': ['cup', 'whole', 'tomato', 'slice', 'cherry', 'grape',
                     'plum'],
        'potato': ['potato', 'cup'],
        'sweet potato': ['medium', 'large', 'small', 'cup', 'oz'],
        'brussels sprouts': ['sprout', 'cup'],
        'peas': ['cup'],
        'green beans': ['cup', 'bean', 'piece'],

        # Snacks & Mixtures
        'trail mix': ['cup', 'package'],

        # Mixed Dishes
        'pizza': ['small', 'personal', 'slice', 'piece'],

        # Fats, Oils & Sweets
        'oil': ['tablespoon', 'cup'],

        # Beverages
        'juice': ['cup', 'oz', 'box', 'container', 'pouch'],
        'drink': ['cup', 'bottle', 'can', 'oz'],
        'shake': ['cup', 'bottle', 'can', 'oz'],
    }
}

# -----------------------------------------------------------------------------
# Cell 3: Food Name Cleaning Function
# -----------------------------------------------------------------------------


def clean_food_name(raw_name: str, is_branded: bool = False) -> str:
    """
    Cleans up food description strings for better readability.
    
    This function removes unnecessary descriptors, standardizes formatting,
    and applies food-specific simplifications to make food names more
    user-friendly while maintaining their essential identity.
    
    Args:
        raw_name: The original food description from USDA database
        is_branded: Whether the food item is a branded product
        
    Returns:
        A cleaned and standardized food name string
    """
    # ------ Handle Branded Food Items ------
    if is_branded:
        cleaned_name = raw_name.split(',')[0].strip()
        return cleaned_name.title()

    cleaned_name = raw_name.lower()
    
    # ------ Remove Common USDA Database Descriptors ------
    junk_patterns = [
        r',?\s*ns as to.*', r',?\s*nfs.*', r',?\s*from canned.*',
        r',?\s*from frozen.*', r',?\s*cooked.*', r',?\s*raw.*',
        r',?\s*plain.*', r',?\s*100%.*', r',?\s*regular.*',
        r',?\s*unsweetened.*', r',?\s*ready-to-drink.*',
        r',?\s*no added fat.*', r',?\s*no sauce.*', r',?\s*unsalted.*',
        r',?\s*with oil.*', r',?\s*bottled or in a carton.*',
        r',?\s*thin crust.*', r',?\s*creamed.*',
        r',?\s*large or small curd.*', r',?\s*part skim.*',
        r',?\s*reduced fat.*', r',?\s*\(\d+%\).*', r',?\s*whole.*',
        r',?\s*high protein.*', r',?\s*drained.*', r',?\s*in water.*'
    ]
    for pattern in junk_patterns:
        cleaned_name = re.sub(pattern, '', cleaned_name)

    # ------ Apply Food-Specific Simplifications ------
    food_simplifications = {
        r'^egg,?\s*': 'eggs',
        r'^nutritional powder mix.*': 'protein powder',
        r'.*protein.*powder.*': 'protein powder',
        r'^yogurt,?\s*greek,?\s*nonfat milk': 'greek yogurt',
        r'^milk,?\s*.*': 'milk',
        r'^cheese,?\s*cottage': 'cottage cheese',
        r'^cheese,?\s*mozzarella': 'mozzarella cheese',
        r'^cream,?\s*heavy': 'heavy cream',
        r'^mixed nuts,?\s*with peanuts': 'mixed nuts',
        r'^peanut butter': 'peanut butter',
        r'^almond butter': 'almond butter',
        r'^tahini': 'tahini',
        r'^hummus': 'hummus',
        r'^bread,?\s*multigrain': 'multigrain bread',
        r'^pasta': 'pasta',
        r'^rice,?\s*white': 'white rice',
        r'^oats': 'oats',
        r'^couscous': 'couscous',
        r'^spinach': 'spinach',
        r'^broccoli': 'broccoli',
        r'^mushrooms': 'mushrooms',
        r'^cauliflower': 'cauliflower',
        r'^carrots': 'carrots',
        r'^tomatoes': 'tomatoes',
        r'^green beans': 'green beans',
        r'^green peas': 'green peas',
        r'^corn': 'corn',
        r'^potato': 'potatoes',
        r'^classic mixed vegetables': 'mixed vegetables',
        r'^banana': 'bananas',
        r'^avocado': 'avocados',
        r'^berries': 'berries',
        r'^orange juice': 'orange juice',
        r'^apple juice': 'apple juice',
        r'^fruit juice': 'fruit juice',
        r'^tortellini,?\s*cheese-filled': 'cheese tortellini',
        r'^tortellini,?\s*spinach-filled': 'spinach tortellini',
        r'^trail mix with nuts and fruit': 'trail mix',
        r'^pizza,?\s*cheese with vegetables.*frozen': 'pizza',
        r'^sunflower seeds': 'sunflower seeds',
        r'^almonds': 'almonds',
        r'^chia seeds': 'chia seeds',
        r'^olive oil': 'olive oil',
        r'^lentils': 'lentils',
        r'^chickpeas': 'chickpeas',
        r'^kidney beans': 'kidney beans',
    }
    
    for pattern, replacement in food_simplifications.items():
        if re.match(pattern, cleaned_name):
            cleaned_name = replacement
            break

    # ------ Final Cleanup for Non-Matched Items ------
    if not any(re.match(pattern, raw_name.lower())
               for pattern in food_simplifications.keys()):
        parts = [part.strip() for part in cleaned_name.split(',')
                 if part.strip()]
        if parts:
            cleaned_name = parts[0]

    cleaned_name = re.sub(r',.*', '', cleaned_name)
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name).strip()
    
    if not cleaned_name:
        first_word = (raw_name.split()[0] if raw_name.split()
                      else "Unknown Food")
        cleaned_name = first_word.lower()
    
    return cleaned_name.title()

# -----------------------------------------------------------------------------
# Cell 4: USDA Nutrition API Class Definition
# -----------------------------------------------------------------------------


class USDANutritionAPI:
    """
    A class to retrieve nutrition data from USDA FoodData Central API.
    
    This class handles API communication, food searching, portion size
    determination, and nutrition calculation using intelligent default
    serving sizes based on food categories and descriptions.
    """

    def __init__(self, api_key: str):
        """
        Initializes the API client with authentication credentials.
        
        Args:
            api_key: Valid USDA FoodData Central API key
        """
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()

    def _find_exact_match_id(self, query: str) -> Optional[int]:
        """
        Searches FNDDS database for an exact match and returns its FDC ID.
        
        This method searches the Survey (FNDDS) database specifically for
        foods that match the query exactly, ensuring we get standardized
        survey foods rather than branded products.
        
        Args:
            query: Food description to search for
            
        Returns:
            FDC ID of the matching food item, or None if no match found
        """
        url = f"{self.base_url}/foods/search"
        params = {
            'api_key': self.api_key,
            'query': query,
            'dataType': ["Survey (FNDDS)"],
            'pageSize': 5
        }
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            for food in response.json().get('foods', []):
                if food.get('description', '').lower() == query.lower():
                    return food.get('fdcId')
                    
        except requests.exceptions.RequestException as e:
            print(f"Error searching for '{query}': {e} ❌")
        
        return None

    def _get_food_details(self, fdc_id: int) -> Dict:
        """
        Fetches complete food details for a given FDC ID.
        
        This method retrieves full nutrition and portion information
        for a specific food item from the USDA database.
        
        Args:
            fdc_id: Food Data Central ID number
            
        Returns:
            Dictionary containing complete food details
        """
        url = f"{self.base_url}/food/{fdc_id}"
        params = {'api_key': self.api_key, 'format': 'full'}
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error getting details for FDC ID {fdc_id}: {e} ❌")
        
        return {}

    def _get_available_measures(self, food_details: Dict) -> List[str]:
        """
        Extracts all available household measures from food portion data.
        
        This method processes the foodPortions data to create a list of
        available serving size options with their gram weights.
        
        Args:
            food_details: Complete food details from API
            
        Returns:
            List of formatted measure descriptions with gram weights
        """
        portions = food_details.get('foodPortions', [])
        measures = []
        
        for portion in portions:
            desc = portion.get('portionDescription')
            grams = portion.get('gramWeight')
            
            if (desc and grams and desc != 'N/A' and
                    "Quantity not specified" not in desc):
                measures.append(f"{desc} ({grams:.1f}g)")
        
        return sorted(list(set(measures)))

    def _get_default_measure(self, food_category: Optional[str],
                             available_measures: List[str],
                             food_description: str = "") -> str:
        """
        Determines the most logical default measure using hierarchical priority.
        
        This method implements the core logic for intelligent serving size
        selection using the unified FOOD_PRIORITIES configuration. It checks
        specific food overrides first, then general category rules, and
        finally falls back to common units.
        
        Args:
            food_category: WWEIA food category description
            available_measures: List of available portion measures
            food_description: Original food description for matching
            
        Returns:
            The most appropriate unit name for the default serving
        """
        # ------ Filter Out Guideline Amounts ------
        filtered_measures = [m for m in available_measures
                             if 'guideline amount' not in m.lower()]
        if not filtered_measures:
            filtered_measures = available_measures

        food_desc_lower = food_description.lower()

        # ------ Step 1: Check High-Priority Specific Food Overrides ------
        for keyword, priority_units in FOOD_PRIORITIES['SPECIFIC'].items():
            if keyword in food_desc_lower:
                for unit in priority_units:
                    for measure in filtered_measures:
                        if unit in measure.lower():
                            return unit

        # ------ Step 2: Check Lower-Priority General Category Rules ------
        if food_category:
            category_lower = food_category.lower()
            for keyword, priority_units in FOOD_PRIORITIES['CATEGORIES'].items():
                if keyword in category_lower:
                    for unit in priority_units:
                        for measure in filtered_measures:
                            if unit in measure.lower():
                                return unit

        # ------ Step 3: Apply Generic Fallback Units ------
        fallback_units = ['medium', 'large', 'small', 'cup', 'container',
                          'piece', 'slice', 'tablespoon', 'oz']
        for unit in fallback_units:
            for measure in filtered_measures:
                if unit in measure.lower():
                    return unit

        # ------ Final Fallback ------
        return (filtered_measures[0].split('(')[0].strip()
                if filtered_measures else "100g")

    def _find_portion_grams(self, food_details: Dict,
                            unit_to_find: str) -> Optional[float]:
        """
        Finds the gram weight for a specified unit from food portions data.
        
        This method searches through the foodPortions data to find the
        gram weight corresponding to a specific serving unit, with special
        handling for common conversions like fluid ounces to cups.
        
        Args:
            food_details: Complete food details from API
            unit_to_find: The serving unit to find gram weight for
            
        Returns:
            Gram weight for the specified unit, or None if not found
        """
        if unit_to_find == "100g":
            return 100.0

        unit_lower = unit_to_find.lower()

        # ------ Special Handling for Cup Conversions ------
        if unit_lower == 'cup':
            # First try to find a direct cup measure
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if 'cup' in desc and 'guideline' not in desc:
                    return portion.get('gramWeight')
            
            # If no direct cup, try to convert from fluid ounces
            base_grams_per_fl_oz = None
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if '1 fl oz' in desc and 'nfs' in desc:
                    base_grams_per_fl_oz = portion.get('gramWeight')
                    break
            
            if base_grams_per_fl_oz:
                return base_grams_per_fl_oz * 8.0

        # ------ General Search for Portion Description ------
        best_match_grams = None
        for portion in food_details.get('foodPortions', []):
            desc = portion.get('portionDescription', '').lower()
            
            if 'guideline' in desc:
                continue

            # Prefer exact matches
            if (f"1 {unit_lower}" in desc or
                    (unit_lower == 'potato' and 'any size' in desc)):
                return portion.get('gramWeight')

            # Keep first partial match as fallback
            if unit_lower in desc and best_match_grams is None:
                best_match_grams = portion.get('gramWeight')

        return best_match_grams

    def _validate_serving_size(self, food_name: str, grams: float,
                               calories: float):
        """
        Validates serving sizes and provides warnings for unrealistic values.
        
        This method checks if the calculated serving sizes are reasonable
        and alerts users to potentially problematic values that might
        indicate data issues or unusual portion sizes.
        
        Args:
            food_name: Name of the food item being validated
            grams: Calculated gram weight of the serving
            calories: Calculated calories for the serving
        """
        warnings = []
        
        if grams > 500 and 'pizza' not in food_name.lower():
            warnings.append(f"very large serving size ({grams:.1f}g)")
        
        if calories > 800 and 'pizza' not in food_name.lower():
            warnings.append(f"high calorie serving ({calories:.0f} kcal)")
        
        # Avoid flagging tablespoon servings for milk
        if ('milk' in food_name.lower() and grams < 200 and grams > 10):
            warnings.append(f"milk serving seems light for a cup "
                            f"({grams:.1f}g)")

        if warnings:
            print(f"  > WARNING for {food_name}: {'; '.join(warnings)} ⚠️")

    def display_nutrition_for_food(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Processes a food item, displays its nutrition, and returns the data.
        
        This is the main method that orchestrates the entire process of
        finding a food, determining its serving size, calculating nutrition,
        and displaying results to the user.
        
        Args:
            query: Food description to search for and analyze
            
        Returns:
            Dictionary containing all calculated nutrition data, or None if
            the food could not be found or processed
        """
        # ------ Find Food in Database ------
        fdc_id = self._find_exact_match_id(query)
        if not fdc_id:
            return None

        # ------ Get Complete Food Details ------
        food_details = self._get_food_details(fdc_id)
        if not food_details:
            return None

        # ------ Determine Serving Size ------
        wweia_category = (food_details.get('wweiaFoodCategory', {})
                          .get('wweiaFoodCategoryDescription'))
        available_measures = self._get_available_measures(food_details)
        default_unit = self._get_default_measure(wweia_category,
                                                 available_measures, query)
        grams_per_portion = self._find_portion_grams(food_details,
                                                     default_unit)

        # ------ Extract Key Nutrients ------
        nutrient_map = {
            1008: 'Calories',
            1003: 'Protein',
            1004: 'Fat',
            1005: 'Carbohydrates'
        }
        nutrients_100g = {key: 0.0 for key in nutrient_map.values()}
        
        for n in food_details.get('foodNutrients', []):
            if n.get('nutrient', {}).get('id') in nutrient_map:
                key = nutrient_map[n['nutrient']['id']]
                nutrients_100g[key] = n.get('amount', 0.0)

        # ------ Prepare Result Data Structure ------
        cleaned_description = clean_food_name(food_details.get('description',
                                                               'N/A'))
        result_data = {
            "Food Name": cleaned_description,
            "FDC ID": fdc_id,
            "Serving Unit": f"1 {default_unit.capitalize()}",
            "Serving Weight in Grams": "N/A",
            "Calories per Serving": "N/A",
            "Protein in Grams": "N/A",
            "Fat in Grams": "N/A",
            "Carbohydrates in Grams": "N/A"
        }

        # ------ Display Basic Information ------
        print(f"Food Information: {cleaned_description} "
              f"(FDC ID: {fdc_id})")

        # ------ Calculate and Display Nutrition ------
        if grams_per_portion:
            scale = grams_per_portion / 100.0
            result_data.update({
                "Serving Weight in Grams": f"{grams_per_portion:.1f}",
                "Calories per Serving": f"{nutrients_100g['Calories'] * scale:.0f}",
                "Protein in Grams": f"{nutrients_100g['Protein'] * scale:.1f}",
                "Fat in Grams": f"{nutrients_100g['Fat'] * scale:.1f}",
                "Carbohydrates in Grams": f"{nutrients_100g['Carbohydrates'] * scale:.1f}"
            })

            self._validate_serving_size(cleaned_description,
                                        grams_per_portion,
                                        float(result_data['Calories per Serving']))

            print(f"Serving Size: {result_data['Serving Unit']} "
                  f"({result_data['Serving Weight in Grams']}g)")
            print(f"  - Nutrition per Serving: "
                  f"{result_data['Calories per Serving']} kcal, "
                  f"{result_data['Protein in Grams']}g Protein, "
                  f"{result_data['Fat in Grams']}g Fat, "
                  f"{result_data['Carbohydrates in Grams']}g Carbs")
        else:
            print("No serving size data available ❌")

        # ------ Display Available Measures ------
        if available_measures:
            print(f"Available Serving Options: "
                  f"{', '.join(available_measures)}")
        print()
        
        return result_data

# -----------------------------------------------------------------------------
# Cell 5: CSV Export Functionality
# -----------------------------------------------------------------------------


def export_to_csv(nutrition_data: List[Dict[str, Any]],
                  filename: str = "usda_nutrition_data.csv"):
    """
    Exports nutrition data to a CSV file with proper formatting.
    
    This function takes the collected nutrition data and exports it to
    a CSV file with descriptive column headers and proper data formatting.
    Only valid data entries are included in the export.
    
    Args:
        nutrition_data: List of nutrition data dictionaries
        filename: Output CSV filename
    """
    # ------ Filter Valid Data Entries ------
    valid_data = [item for item in nutrition_data if item is not None]
    
    if not valid_data:
        print("No valid nutrition data available for export ❌")
        return

    # ------ Define CSV Column Structure ------
    fieldnames = [
        'Food Name', 'FDC ID', 'Serving Unit', 'Serving Weight in Grams',
        'Calories per Serving', 'Protein in Grams', 'Fat in Grams',
        'Carbohydrates in Grams'
    ]
    
    # ------ Write Data to CSV File ------
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_data)
        
        print(f"Successfully exported nutrition data for {len(valid_data)} "
              f"food items to {filename} ✅")
        
    except IOError as e:
        print(f"Error occurred while exporting data to CSV file: {e} ❌")

# -----------------------------------------------------------------------------
# Cell 6: Main Function and Food List Processing
# -----------------------------------------------------------------------------


def main():
    """
    Main function to run the nutrition analysis for all foods.
    
    This function initializes the API client, processes the predefined
    food list, collects nutrition data, and exports results to CSV.
    It includes progress tracking and summary reporting.
    """
    # ------ Initialize API Client ------
    # NOTE: Using the public DEMO_KEY. For production use, get your own key
    api_key = "PodqZM9xrI5ByN5sS8zlEMf2haudDydBMCzt3U4N"
    nutrition_api = USDANutritionAPI(api_key)

    # ------ Define Food List for Analysis ------
    food_list = [
        "Egg, Whole, Raw",
        "Yogurt, Greek, Nonfat Milk, Plain",
        "Nutritional Powder Mix, High Protein, Nfs",
        "Milk, Reduced Fat (2%)",
        "Cheese, Cottage, Creamed, Large Or Small Curd",
        "Cheese, Mozzarella, Part Skim",
        "Lentils, From Canned",
        "Chickpeas, From Canned, No Added Fat",
        "Kidney Beans, From Canned, No Added Fat",
        "Hummus, Plain",
        "Tortellini, Cheese-Filled, No Sauce",
        "Olive Oil",
        "Peanut Butter",
        "Almonds, Unsalted",
        "Mixed Nuts, With Peanuts, Unsalted",
        "Avocado, Raw",
        "Sunflower Seeds, Plain, Unsalted",
        "Chia Seeds",
        "Tahini",
        "Cream, Heavy",
        "Oats, Raw",
        "Potato, Nfs",
        "Rice, White, Cooked, No Added Fat",
        "Bread, Multigrain",
        "Pasta, Cooked",
        "Banana, Raw",
        "Couscous, Plain, Cooked",
        "Corn, Frozen, Cooked, No Added Fat",
        "Green Peas, Frozen, Cooked, No Added Fat",
        "Classic Mixed Vegetables, Frozen, Cooked, No Added Fat",
        "Spinach, Frozen, Cooked, No Added Fat",
        "Broccoli, Frozen, Cooked With Oil",
        "Berries, Frozen",
        "Carrots, Frozen, Cooked, No Added Fat",
        "Tomatoes, Canned, Cooked",
        "Mushrooms, Raw",
        "Cauliflower, Frozen, Cooked, No Added Fat",
        "Green Beans, Frozen, Cooked, No Added Fat",
        "Orange Juice, 100%, Canned, Bottled Or In A Carton",
        "Apple Juice, 100%",
        "Trail Mix With Nuts And Fruit",
        "Tortellini, Spinach-Filled, No Sauce",
        "Pizza, Cheese, With Vegetables, From Frozen, Thin Crust",
        "Fruit Juice, Nfs"
    ]

    # ------ Display Program Header ------
    print("USDA Food Nutrition Analysis Tool 🍎")
    print("=" * 60)
    print()

    # ------ Process Each Food Item ------
    nutrition_results = []
    for food_item in food_list:
        print(f"Processing Food Item: {food_item}")
        result = nutrition_api.display_nutrition_for_food(food_item)
        nutrition_results.append(result)
        time.sleep(1)  # Rate limiting for API requests

    # ------ Export Results and Display Summary ------
    print("=" * 60)
    export_to_csv(nutrition_results, "nutrition_results.csv")
    print("=" * 60)

    print("\nNutrition analysis process completed successfully! 🎉")
    print("Key design improvements implemented in this version:")
    print("  - Unified configuration system for easy food priority updates")
    print("  - Clear hierarchy where specific food rules override general categories")
    print("  - Robust data cleaning and serving size validation system")
    print("  - Comprehensive nutrition data export with descriptive headers")
    print("\nThanks for using our nutrition analysis tool! 🥗")
    print("May your meals be as balanced as your macros! 💪")

# -----------------------------------------------------------------------------
# Cell 7: Script Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
