# -----------------------------------------------------------------------------
# USDA Survey Food Database Serving Size Finder
# -----------------------------------------------------------------------------

"""
The script utilizes the USDA FoodData Central API to access the Survey (FNDDS)
database, which contains food consumption data from dietary surveys. For each
food item, the script searches for an exact match by comparing food
descriptions in a case-insensitive manner. When a match is found, it retrieves
the complete food details including available food portions.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Dependencies
# -----------------------------------------------------------------------------

import requests
import time
from typing import Dict, List, Optional

# -----------------------------------------------------------------------------
# Cell 2: USDA Serving Finder Class Definition
# -----------------------------------------------------------------------------


class USDAServingFinder:
    """
    A class to find available servings for single foods from the USDA Survey
    (FNDDS) database using an exact match search.
    """

    def __init__(self, api_key: str):
        """
        Initialize the USDAServingFinder with API credentials and session.

        Args:
            api_key (str): Valid USDA FoodData Central API key
        """
        self.api_key = api_key
        self.base_url = "https://api.nal.usda.gov/fdc/v1"
        self.session = requests.Session()

    # ------ Private Method for Exact Match Food Search ------

    def _find_exact_match_id(self, query: str) -> Optional[int]:
        """
        Search FNDDS for an exact match and return its FDC ID.

        Args:
            query (str): Food description to search for

        Returns:
            Optional[int]: FDC ID if exact match found, None otherwise
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
                # Ensure the description is an exact match (case-insensitive)
                if food.get('description', '').lower() == query.lower():
                    return food.get('fdcId')
        except requests.exceptions.RequestException as e:
            print(f"Error searching for '{query}': {e} ❌")
        return None

    # ------ Private Method for Food Detail Retrieval ------

    def _get_food_details(self, fdc_id: int) -> Dict:
        """
        Fetch the complete details for a given FDC ID.

        Args:
            fdc_id (int): Food Data Central identification number

        Returns:
            Dict: Complete food details including portions, or empty dict if error
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

    # ------ Public Method for Displaying Food Servings ------

    def display_servings_for_food(self, query: str):
        """
        Find a food by exact match and print its available servings.

        Args:
            query (str): Food description to search for and display servings
        """
        print(f"Searching for Food Item: '{query}'")
        fdc_id = self._find_exact_match_id(query)

        if not fdc_id:
            print("  No exact match found in the database")
            print()
            return

        food_details = self._get_food_details(fdc_id)
        if not food_details:
            print("  Could not retrieve food details from the database")
            print()
            return

        portions = food_details.get('foodPortions', [])

        if not portions:
            print("  No household servings listed in the database")
            print()
            return

        print(f"  Match Found Successfully (FDC ID: {fdc_id}) ✅")
        print("  Available Household Serving Sizes:")

        measures = []
        for portion in portions:
            desc = portion.get('portionDescription')
            grams = portion.get('gramWeight')
            if (desc and grams and desc != 'N/A' and
                    "Quantity not specified" not in desc):
                measures.append(f"{desc} ({grams:.1f}g)")

        if not measures:
            print("    No valid household servings found in the database")
        else:
            for measure in sorted(measures):
                print(f"    {measure}")

        print()  # Add a blank line for readability

# -----------------------------------------------------------------------------
# Cell 3: Main Function and Food List Processing
# -----------------------------------------------------------------------------


def main():
    """
    Main function to run the USDA food serving searches.
    """
    # ------ API Configuration and Initialization ------

    # NOTE: Using the public DEMO_KEY from api.nal.usda.gov
    api_key = ""
    finder = USDAServingFinder(api_key)

    # ------ Food Items List for Processing ------

    food_list = [""]

    # ------ Process Each Food Item with Rate Limiting ------

    print("Starting USDA Food Database Serving Size Search 🔍")
    print()

    for food_item in food_list:
        finder.display_servings_for_food(food_item)
        time.sleep(1)  # Be polite to the API

    print("USDA food serving search completed successfully 🎉")
    print("Thanks for exploring the world of food data with us!")
    print("Remember, good nutrition starts with knowing your portions! 🥗")


# -----------------------------------------------------------------------------
# Cell 4: Script Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
