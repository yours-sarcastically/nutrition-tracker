# -----------------------------------------------------------------------------
# USDA Food Nutrition Analysis and Data Export Tool
# -----------------------------------------------------------------------------

"""
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
from fractions import Fraction
from typing import Dict, List, Optional, Any

# -----------------------------------------------------------------------------
# Cell 2: Food Category Unit Preferences Configuration
# -----------------------------------------------------------------------------

# ------ Category Unit Mapping Configuration ------

# Category keywords to a prioritized list of preferred units
# This is now matched against the official 'wweiaFoodCategoryDescription' from the API
CATEGORY_UNITS = {
    # Dairy & Dairy Products
    'yogurt': ['cup', 'container', 'oz'],
    'milk': ['cup', 'oz', 'container'],
    'cheese': ['slice', 'stick', 'cup', 'curd', 'inch'],
    'cream': ['tablespoon', 'container', 'cup'],  # Prioritize tablespoon for cream cheese
    'cottage': ['cup'],  # Specific for cottage cheese
    
    # Protein Foods
    'beans': ['cup'],
    'lentils': ['cup'],
    'chickpeas': ['cup', 'pea'],
    'edamame': ['cup', 'pod'],
    'hummus': ['tablespoon', 'container'],  # Prioritize tablespoon
    'almonds': ['oz', 'cup', 'nut', 'package'],
    'mixed nuts': ['oz', 'cup', 'package'],
    'seeds': ['oz', 'tablespoon', 'cup', 'package'],
    'peanut butter': ['tablespoon', 'serving'],  # Prioritize tablespoon
    'almond butter': ['tablespoon'],
    'tahini': ['tablespoon'],  # Prioritize tablespoon
    'egg': ['egg', 'cup'],
    
    # Grains
    'oats': ['cup'],
    'rice': ['cup'],
    'pasta': ['cup', 'oz'],  # Prioritize cup for pasta dishes
    'tortellini': ['cup'],   # Specific for tortellini
    'bread': ['slice', 'inch'],
    'bagel': ['bagel', 'large', 'regular', 'small', 'miniature'],
    'quinoa': ['cup'],
    'corn': ['cup', 'ear'],
    'couscous': ['cup', 'oz'],
    
    # Fruits
    'apple': ['medium', 'large', 'small', 'cup', 'slice', 'package'],
    'banana': ['banana', 'cup', 'slice', 'inch'],
    'orange': ['fruit', 'medium', 'large', 'small', 'cup', 'section', 'slice'],  # Prioritize fruit
    'berries': ['cup', 'berry'],
    'raisins': ['cup', 'box', 'raisin', 'oz'],  # Prioritize cup
    'date': ['date', 'cup'],
    'avocado': ['fruit', 'medium', 'large', 'small', 'cup', 'slice'],  # Prioritize fruit
    
    # Vegetables
    'spinach': ['cup', 'leaf'],
    'broccoli': ['cup', 'floweret', 'piece'],
    'cauliflower': ['cup', 'floweret', 'piece'],  # Prioritize cup
    'carrots': ['cup', 'carrot', 'slice', 'stick'],
    'mushrooms': ['cup', 'whole', 'slice', 'piece'],  # Prioritize cup
    'tomatoes': ['cup', 'whole', 'tomato', 'slice', 'cherry', 'grape', 'plum'],  # Prioritize cup for canned
    'potato': ['potato', 'cup'],
    'sweet potato': ['medium', 'large', 'small', 'cup', 'oz'],  # Prioritize medium
    'brussels sprouts': ['sprout', 'cup'],
    'peas': ['cup'],
    'green beans': ['cup', 'bean', 'piece'],
    
    # Snacks & Mixtures
    'trail mix': ['cup', 'package'],
    
    # Fats, Oils & Sweets
    'oil': ['tablespoon', 'cup'],
    
    # Beverages
    'juice': ['cup', 'oz', 'box', 'container', 'pouch'],  # Prioritize cup for juice
    'drink': ['cup', 'bottle', 'can', 'oz'],
    'shake': ['cup', 'bottle', 'can', 'oz'],
}

# -----------------------------------------------------------------------------
# Cell 3: Food Name Cleaning Function
# -----------------------------------------------------------------------------

def clean_food_name(raw_name: str, is_branded: bool = False) -> str:
    """
    Cleans up food description strings for better readability.
    
    Args:
        raw_name: The original food description from the database
        is_branded: Whether the food is a branded product
        
    Returns:
        A cleaned and formatted food name
    """
    if is_branded:
        # For branded items, take the primary name before a comma
        cleaned_name = raw_name.split(',')[0].strip()
        return cleaned_name.title()
    else:
        # For generic foods, apply comprehensive cleaning logic
        
        # Step 1: Convert to lowercase for processing
        cleaned_name = raw_name.lower()
        
        # Step 2: Remove common USDA descriptors and preparation methods
        junk_patterns = [
            r',?\s*ns as to.*',
            r',?\s*nfs.*',
            r',?\s*from canned.*',
            r',?\s*from frozen.*',
            r',?\s*cooked.*',
            r',?\s*raw.*',
            r',?\s*plain.*',
            r',?\s*100%.*',
            r',?\s*regular.*',
            r',?\s*unsweetened.*',
            r',?\s*ready-to-drink.*',
            r',?\s*no added fat.*',
            r',?\s*no sauce.*',
            r',?\s*unsalted.*',
            r',?\s*with oil.*',
            r',?\s*bottled or in a carton.*',
            r',?\s*thin crust.*',
            r',?\s*creamed.*',
            r',?\s*large or small curd.*',
            r',?\s*part skim.*',
            r',?\s*reduced fat.*',
            r',?\s*\(\d+%\).*',  # Remove percentage indicators like (2%)
            r',?\s*whole.*',
            r',?\s*high protein.*',
        ]
        
        # Remove all the junk patterns
        for pattern in junk_patterns:
            cleaned_name = re.sub(pattern, '', cleaned_name)
        
        # Step 3: Handle specific food type simplifications
        food_simplifications = {
            # Eggs
            r'^egg,?\s*': 'eggs',
            
            # Protein powders
            r'^nutritional powder mix.*': 'protein powder',
            r'.*protein.*powder.*': 'protein powder',
            
            # Dairy products
            r'^yogurt,?\s*greek,?\s*nonfat milk': 'greek yogurt',
            r'^milk,?\s*.*': 'milk',
            r'^cheese,?\s*cottage': 'cottage cheese',
            r'^cheese,?\s*mozzarella': 'mozzarella cheese',
            r'^cream,?\s*heavy': 'heavy cream',
            
            # Proteins
            r'^mixed nuts,?\s*with peanuts': 'mixed nuts',
            r'^peanut butter': 'peanut butter',
            r'^almond butter': 'almond butter',
            r'^tahini': 'tahini',
            r'^hummus': 'hummus',
            
            # Grains & Starches
            r'^bread,?\s*multigrain': 'multigrain bread',
            r'^pasta': 'pasta',
            r'^rice,?\s*white': 'white rice',
            r'^oats': 'oats',
            r'^couscous': 'couscous',
            
            # Vegetables - Updated to plural
            r'^spinach': 'spinach',  # Already plural-like
            r'^broccoli': 'broccoli',  # Already plural-like
            r'^mushrooms': 'mushrooms',  # Already plural
            r'^cauliflower': 'cauliflower',  # Already plural-like
            r'^carrots': 'carrots',  # Already plural
            r'^tomatoes': 'tomatoes',  # Already plural
            r'^green beans': 'green beans',  # Already plural
            r'^green peas': 'green peas',  # Already plural
            r'^corn': 'corn',  # Mass noun, stays singular
            r'^potato': 'potatoes',  # Changed to plural
            r'^classic mixed vegetables': 'mixed vegetables',
            
            # Fruits - Updated to plural where appropriate
            r'^banana': 'bananas',  # Changed to plural
            r'^avocado': 'avocados',  # Changed to plural
            r'^berries': 'berries',  # Already plural
            
            # Beverages (stay singular as they are mass nouns)
            r'^orange juice': 'orange juice',
            r'^apple juice': 'apple juice',
            r'^fruit juice': 'fruit juice',
            
            # Specialty items
            r'^tortellini,?\s*cheese-filled': 'cheese tortellini',
            r'^tortellini,?\s*spinach-filled': 'spinach tortellini',
            r'^trail mix with nuts and fruit': 'trail mix',
            r'^pizza,?\s*cheese with vegetables.*frozen': 'pizza',
            
            # Seeds and nuts
            r'^sunflower seeds': 'sunflower seeds',  # Already plural
            r'^almonds': 'almonds',  # Already plural
            r'^chia seeds': 'chia seeds',  # Already plural
            
            # Oils
            r'^olive oil': 'olive oil',
            
            # Legumes
            r'^lentils': 'lentils',  # Already plural
            r'^chickpeas': 'chickpeas',  # Already plural
            r'^kidney beans': 'kidney beans',  # Already plural
        }
        
        # Apply food-specific simplifications
        for pattern, replacement in food_simplifications.items():
            if re.match(pattern, cleaned_name):
                cleaned_name = replacement
                break
        
        # Step 4: If no specific simplification matched, do general cleanup
        if not any(re.match(pattern, raw_name.lower()) for pattern in food_simplifications.keys()):
            # Split by comma, strip whitespace, remove empty parts
            parts = [part.strip() for part in cleaned_name.split(',') if part.strip()]
            
            # For multi-part names, take the first meaningful part
            if parts:
                cleaned_name = parts[0]
        
        # Step 5: Final cleanup - remove any remaining commas and extra spaces
        cleaned_name = re.sub(r',.*', '', cleaned_name)  # Remove everything after first comma
        cleaned_name = re.sub(r'\s+', ' ', cleaned_name)  # Normalize whitespace
        cleaned_name = cleaned_name.strip()
        
        # Step 6: Handle edge case where name becomes empty
        if not cleaned_name:
            # Fall back to the first word of the original name
            first_word = raw_name.split()[0] if raw_name.split() else "Unknown Food"
            cleaned_name = first_word.lower()
        
        # Step 7: Convert to title case
        return cleaned_name.title()

# -----------------------------------------------------------------------------
# Cell 4: USDA Nutrition API Class Definition
# -----------------------------------------------------------------------------

class USDANutritionAPI:
    """
    A class to get nutrition for single foods from the USDA Survey (FNDDS) database,
    using a smart default serving size for calculations based on the food's official category.
    """

    def __init__(self, api_key: str):
        """
        Initialize the API client with the provided API key.
        
        Args:
            api_key: Valid USDA FoodData Central API key
        """
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()

    # ------ Food Search and Identification Methods ------

    def _find_exact_match_id(self, query: str) -> Optional[int]:
        """
        Searches only FNDDS for an exact match and returns its FDC ID.
        
        Args:
            query: The food name to search for
            
        Returns:
            The FDC ID if an exact match is found, None otherwise
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
            results = response.json()
            for food in results.get('foods', []):
                if food.get('description', '').lower() == query.lower():
                    return food.get('fdcId')
        except requests.exceptions.RequestException as e:
            print(f"Error searching for '{query}': {e} ❌")
        return None

    def _get_food_details(self, fdc_id: int) -> Dict:
        """
        Fetches the full details for a given FDC ID.
        
        Args:
            fdc_id: The Food Data Central ID
            
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

    # ------ Serving Size and Measurement Methods ------

    def _get_available_measures(self, food_details: Dict) -> List[str]:
        """
        Extracts all available household measures from the food's portion data.
        
        Args:
            food_details: Complete food details from the API
            
        Returns:
            List of available measures with gram weights
        """
        portions = food_details.get('foodPortions', [])
        if not portions:
            return []  # Return empty list if no portions
        
        measures = []
        for portion in portions:
            desc = portion.get('portionDescription')
            grams = portion.get('gramWeight')
            if desc and grams and desc != 'N/A' and "Quantity not specified" not in desc:
                # Special formatting for fluid ounces to show both volume and weight
                if 'fl oz' in desc:
                    measures.append(f"{desc} ({grams:.1f}g)")
                else:
                    measures.append(f"{desc} ({grams:.1f}g)")
        
        return sorted(list(set(measures)))  # Use set to remove duplicates

    def _filter_guideline_amounts(self, available_measures: List[str]) -> List[str]:
        """
        Filters out guideline amounts which are portion control suggestions, not typical servings.
        
        Args:
            available_measures: List of all available measures
            
        Returns:
            List of measures with guideline amounts removed
        """
        return [measure for measure in available_measures 
                if 'guideline amount' not in measure.lower()]

    def _get_default_measure(self, food_category: Optional[str], available_measures: List[str], food_description: str = "") -> str:
        """
        Determines the most logical default measure based on the food's WWEIA category and description.
        
        Args:
            food_category: The WWEIA food category description
            available_measures: List of available household measures
            food_description: The food description for additional context
            
        Returns:
            The best default measure unit as a string
        """
        # Filter out guideline amounts first
        filtered_measures = self._filter_guideline_amounts(available_measures)
        if not filtered_measures:
            filtered_measures = available_measures  # Fallback if all are guideline amounts

        food_desc_lower = food_description.lower()

        # Specific food-based fixes based on the serving size data
        
        # Fix for potatoes - prioritize "any size" over baby/new potato
        if 'potato' in food_desc_lower and 'nfs' in food_desc_lower:
            for measure in filtered_measures:
                if 'any size' in measure.lower():
                    return 'potato'
        
        # Fix for protein powder - prioritize scoop over packet
        if 'nutritional powder mix' in food_desc_lower or 'high protein' in food_desc_lower:
            # Look for scoop first, prefer "NFS" version for consistency
            for measure in filtered_measures:
                if 'scoop' in measure.lower() and 'nfs' in measure.lower():
                    return 'scoop'
            for measure in filtered_measures:
                if 'scoop' in measure.lower():
                    return 'scoop'

        # Fix for mozzarella cheese - prioritize shredded cup
        if 'mozzarella' in food_desc_lower:
            for measure in filtered_measures:
                if 'cup' in measure.lower() and 'shredded' in measure.lower():
                    return 'cup'
            # Fallback to any cup measure
            for measure in filtered_measures:
                if 'cup' in measure.lower() and 'nfs' in measure.lower():
                    return 'cup'

        # Standard specific food overrides
        if 'cream cheese' in food_desc_lower:
            for measure in filtered_measures:
                if 'tablespoon' in measure.lower():
                    return 'tablespoon'

        if 'hummus' in food_desc_lower:
            for measure in filtered_measures:
                if 'tablespoon' in measure.lower():
                    return 'tablespoon'

        if 'peanut butter' in food_desc_lower or 'almond butter' in food_desc_lower:
            for measure in filtered_measures:
                if 'tablespoon' in measure.lower():
                    return 'tablespoon'

        if 'tahini' in food_desc_lower:
            for measure in filtered_measures:
                if 'tablespoon' in measure.lower():
                    return 'tablespoon'

        if 'avocado' in food_desc_lower:
            for measure in filtered_measures:
                if 'fruit' in measure.lower():
                    return 'fruit'

        if 'banana' in food_desc_lower:
            for measure in filtered_measures:
                if 'banana' in measure.lower():
                    return 'banana'

        # Fix for tortellini - prioritize cup over piece
        if 'tortellini' in food_desc_lower:
            for measure in filtered_measures:
                if 'cup' in measure.lower():
                    return 'cup'

        # Category-based selection using filtered measures
        if food_category:
            category_lower = food_category.lower()

            # Special handling for juice - look for direct cup measures first
            if 'juice' in category_lower or 'drink' in category_lower:
                # Check if a cup measure exists directly
                for measure in filtered_measures:
                    if 'cup' in measure.lower():
                        return 'cup'
                # If no cup, we'll handle conversion later
                for measure in filtered_measures:
                    if 'fl oz' in measure.lower() and 'nfs' in measure.lower():
                        return 'cup'  # Signal that we want cup conversion

            # Apply category-based matching
            for category_keyword, units in CATEGORY_UNITS.items():
                if category_keyword in category_lower:
                    # Go through units in priority order
                    for unit in units:
                        for measure in filtered_measures:
                            measure_lower = measure.lower()
                            # More precise matching to avoid false positives
                            if unit == 'fruit' and 'fruit' in measure_lower:
                                return 'fruit'
                            elif unit == 'medium' and 'medium' in measure_lower:
                                return 'medium'
                            elif unit == 'cup' and 'cup' in measure_lower:
                                # For cheese, prefer shredded if available
                                if 'cheese' in category_lower and 'shredded' in measure_lower:
                                    return 'cup'
                                elif 'cheese' not in category_lower:
                                    return 'cup'
                            elif unit == 'tablespoon' and ('tablespoon' in measure_lower or 'tbsp' in measure_lower):
                                return 'tablespoon'
                            elif unit in measure_lower:
                                return unit

        # Fallback for items without a matching category
        preferred_units = [
            'medium', 'large', 'small', 'cup', 'container', 'piece', 'slice',
            'tablespoon', 'tbsp', 'oz', 'ounce', 'tsp', 'teaspoon'
        ]

        for unit in preferred_units:
            for measure in filtered_measures:
                if unit in measure.lower():
                    return unit

        if filtered_measures:
            return filtered_measures[0].split('(')[0].strip()

        return "100g"

    def _find_portion_grams(self, food_details: Dict, unit_to_find: str) -> Optional[float]:
        """
        Finds the gram weight for a given unit from the food's portions.
        
        Args:
            food_details: Complete food details from the API
            unit_to_find: The unit to find the gram weight for
            
        Returns:
            The gram weight for the specified unit, or None if not found
        """
        if unit_to_find == "100g":
            return 100.0
            
        unit_to_find = unit_to_find.lower()
        
        # Handle juice cup conversion using actual USDA ratios
        if unit_to_find == 'cup':
            # First check if a direct cup measure exists
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if 'cup' in desc and 'guideline' not in desc:
                    return portion.get('gramWeight')
            
            # If no direct cup, convert from fl oz using USDA data
            # Look for "1 fl oz (NFS)" measure for most accurate conversion
            base_grams_per_fl_oz = None
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if '1 fl oz' in desc and 'nfs' in desc:
                    base_grams_per_fl_oz = portion.get('gramWeight')
                    break
            
            if base_grams_per_fl_oz:
                # 1 cup = 8 fl oz, using actual USDA gram conversion
                return base_grams_per_fl_oz * 8.0
        
        # Specific fixes based on serving size data
        
        # Fix for potatoes - look for "any size" specifically
        if unit_to_find == 'potato':
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if 'any size' in desc:
                    return portion.get('gramWeight')
        
        # Fix for protein powder - look for scoop, prefer NFS
        if unit_to_find == 'scoop':
            # First try to find "scoop, NFS"
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if 'scoop' in desc and 'nfs' in desc:
                    return portion.get('gramWeight')
            # Fallback to any scoop
            for portion in food_details.get('foodPortions', []):
                desc = portion.get('portionDescription', '').lower()
                if 'scoop' in desc:
                    return portion.get('gramWeight')
        
        # Standard matching with exact word boundaries
        exact_match_pattern = re.compile(r'\b' + re.escape(unit_to_find) + r'\b')
        
        best_match = None
        
        for portion in food_details.get('foodPortions', []):
            desc = portion.get('portionDescription', '').lower()
            # Skip guideline amounts
            if 'guideline amount' in desc:
                continue
            # Prefer exact word matches first
            if exact_match_pattern.search(desc):
                return portion.get('gramWeight')
            # Fallback to substring match if no exact match is found yet
            if unit_to_find in desc and not best_match:
                best_match = portion.get('gramWeight')
        
        return best_match

    def _validate_serving_size(self, food_name: str, selected_measure: str, grams: float, calories: float):
        """
        Validates serving sizes and provides warnings for unrealistic values.
        
        Args:
            food_name: Name of the food
            selected_measure: The selected serving measure
            grams: Gram weight of the serving
            calories: Calories in the serving
        """
        warnings = []
        
        # Check for very large servings (except pizza which is intentionally large)
        if grams > 500 and 'pizza' not in food_name.lower():
            warnings.append(f"Very large serving size ({grams:.1f}g)")
        
        # Check for very high calorie servings
        if calories > 800 and 'pizza' not in food_name.lower():
            warnings.append(f"High calorie serving ({calories:.0f} kcal)")
        
        # Check if guideline amount was used (shouldn't happen with fixes)
        if 'guideline amount' in selected_measure.lower():
            warnings.append("Using guideline amount, not typical serving")
        
        # Check for suspiciously light servings for certain foods
        if 'milk' in food_name.lower() and 'cup' in selected_measure and grams < 200:
            warnings.append(f"Milk serving seems too light ({grams:.1f}g for cup)")
        
        if warnings:
            print(f"{food_name}: {'; '.join(warnings)} ⚠️")

    # ------ Main Processing Method ------

    def display_nutrition_for_food(self, query: str) -> Optional[Dict[str, Any]]:
        """
        Processes a food item, prints its nutrition, and returns the data as a dictionary.
        
        Args:
            query: The food name to search for and analyze
            
        Returns:
            Dictionary containing nutrition data, or None if food not found
        """
        fdc_id = self._find_exact_match_id(query)
        if not fdc_id:
            return None

        food_details = self._get_food_details(fdc_id)
        if not food_details:
            return None
        
        wweia_category_obj = food_details.get('wweiaFoodCategory', {})
        food_category = wweia_category_obj.get('wweiaFoodCategoryDescription')
        
        available_measures = self._get_available_measures(food_details)
        # Pass food description to help with unit selection
        default_unit = self._get_default_measure(food_category, available_measures, query)
        grams_per_portion = self._find_portion_grams(food_details, default_unit)

        # Data extraction
        nutrients_100g = {'Calories': 0.0, 'Protein': 0.0, 'Fat': 0.0, 'Carbohydrates': 0.0}
        nutrient_map = {1008: 'Calories', 1003: 'Protein', 1004: 'Fat', 1005: 'Carbohydrates'}
        
        for n in food_details.get('foodNutrients', []):
            nutrient_id = n.get('nutrient', {}).get('id')
            if nutrient_id in nutrient_map:
                key = nutrient_map[nutrient_id]
                nutrients_100g[key] = n.get('amount', 0.0)

        description = food_details.get('description', 'N/A')
        cleaned_description = clean_food_name(description, is_branded=False)
        
        # Prepare results for printing and returning
        result_data = {
            "name": cleaned_description,
            "fdcId": fdc_id,
            "serving_unit": f"1 {default_unit}",
            "serving_grams": None,
            "calories": None,
            "protein": None,
            "fat": None,
            "carbs": None
        }

        # Console output and final calculations
        print(f"Basic Info: {cleaned_description}")
        print(f"FDC ID: {fdc_id}")

        if grams_per_portion:
            scale = grams_per_portion / 100.0
            result_data['serving_grams'] = f"{grams_per_portion:.1f}"
            result_data['calories'] = f"{nutrients_100g['Calories'] * scale:.0f}"
            result_data['protein'] = f"{nutrients_100g['Protein'] * scale:.1f}"
            result_data['fat'] = f"{nutrients_100g['Fat'] * scale:.1f}"
            result_data['carbs'] = f"{nutrients_100g['Carbohydrates'] * scale:.1f}"

            # Validate serving size
            self._validate_serving_size(
                cleaned_description, 
                result_data['serving_unit'], 
                grams_per_portion, 
                float(result_data['calories'])
            )

            print(f"Serving Size: {result_data['serving_unit']} ({result_data['serving_grams']}g)")
            print("Key Nutrition Facts:")
            print(f"  - Calories: {result_data['calories']} kcal")
            print(f"  - Protein: {result_data['protein']} g")
            print(f"  - Fat: {result_data['fat']} g")
            print(f"  - Carbohydrates: {result_data['carbs']} g")
        else:
            print("No serving size data available ❌")
            result_data['serving_grams'] = "N/A"
            result_data['calories'] = "N/A"
            result_data['protein'] = "N/A"
            result_data['fat'] = "N/A"
            result_data['carbs'] = "N/A"

        if available_measures:
            print(f"Available Measures: {', '.join(available_measures)}")
        
        print()  # Blank line for readability
        return result_data

# -----------------------------------------------------------------------------
# Cell 5: CSV Export Functionality
# -----------------------------------------------------------------------------

def export_to_csv(nutrition_data: List[Dict[str, Any]], filename: str = "usda_nutrition_data.csv"):
    """
    Exports nutrition data to a CSV file.
    
    Args:
        nutrition_data: List of nutrition dictionaries
        filename: Output CSV filename
    """
    if not nutrition_data:
        print("No data to export ❌")
        return

    # Filter out None entries
    valid_data = [item for item in nutrition_data if item is not None]
    
    if not valid_data:
        print("No valid data to export ❌")
        return

    fieldnames = ['name', 'fdcId', 'serving_unit', 'serving_grams', 'calories', 'protein', 'fat', 'carbs']
    
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(valid_data)
        
        print(f"Successfully exported {len(valid_data)} food items to {filename} ✅")
    except Exception as e:
        print(f"Error exporting to CSV: {e} ❌")

# -----------------------------------------------------------------------------
# Cell 6: Main Function and Food List Processing
# -----------------------------------------------------------------------------

def main():
    """
    Main function to run the nutrition analysis for all foods.
    """
    # ------ API Configuration ------
    
    # For production use, obtain your own key from: https://fdc.nal.usda.gov/api-guide.html
    api_key = ""
    nutrition_api = USDANutritionAPI(api_key)

    # ------ Food Items List ------
    
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

    # ------ Process Each Food Item ------
    
    print("USDA Food Nutrition Analysis Tool 🍎")
    print("=" * 60)
    print()

    nutrition_results = []
    
    for food_item in food_list:
        print(f"Processing: {food_item}")
        result = nutrition_api.display_nutrition_for_food(food_item)
        nutrition_results.append(result)
        time.sleep(1)  # Be respectful to the API

    # ------ Export Results ------
    
    print("=" * 60)
    print("Export Summary 📊")
    print("=" * 60)
    
    export_to_csv(nutrition_results, "nutrition_results.csv")
    
    # ------ Summary Statistics ------
    
    successful_items = [item for item in nutrition_results if item is not None]
    failed_items = len(nutrition_results) - len(successful_items)
    
    print(f"Successfully processed: {len(successful_items)} foods ✅")
    if failed_items > 0:
        print(f"Failed to process: {failed_items} foods ❌")
    
    print()
    print("Analysis complete! Check the CSV file for detailed nutrition data 🎉")
    print("Key fixes applied 💡")
    print("   - Corrected milk serving size (244g vs 61g)")
    print("   - Fixed potato serving size (170g vs 60g)")
    print("   - Improved juice conversions using actual USDA ratios")
    print("   - Filtered out guideline amounts")
    print("   - Added serving size validation warnings")

# -----------------------------------------------------------------------------
# Cell 7: Script Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
