# -----------------------------------------------------------------------------
# USDA Food Data Central API Search Tool
# -----------------------------------------------------------------------------

"""
This script offers a tool for searching and retrieving food data from the 
USDA Food Data Central (FDC) API. It allows users to query multiple food 
items across various data types and displays the results in a structured 
format. It supports both generic food data (Survey FNDDS, Foundation, 
SR Legacy) as well as branded food products.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Dependencies
# -----------------------------------------------------------------------------

import requests
import time

# -----------------------------------------------------------------------------
# Cell 2: Core Function Definitions
# -----------------------------------------------------------------------------

# ------ USDA Food Search Function ------

def list_usda_foods(query: str,
                    data_type: str,
                    api_key: str,
                    page_size: int = 200,
                    polite_pause: float = 0.25) -> None:
    """
    Stream all USDA-FDC search results for a query in a given data type.
    
    This function retrieves and displays all available search results from
    the USDA Food Data Central API for a specific query and data type. Results
    are fetched page by page with automatic pagination until no more results
    are available.
    
    Args:
        query: Search term for food items
        data_type: Type of food data to search (Survey FNDDS, Foundation, 
                  SR Legacy, Branded)
        api_key: USDA API key for authentication
        page_size: Number of results per page (default 200, maximum 200)
        polite_pause: Delay between API requests in seconds
    
    Returns:
        None: Results are printed directly to console
    """
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    total_count = 0
    page_number = 1

    print(f"\n=== {query!r}  |  {data_type} ===")

    while True:
        params = {
            "api_key": api_key,
            "query": query,
            "dataType": [data_type],
            "pageSize": page_size,
            "pageNumber": page_number,
        }

        try:
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            foods = payload.get("foods", [])

            # Display total matches on first page
            if page_number == 1:
                total_hits = payload.get('totalHits', 0)
                print(f"Found {total_hits} matches in the database 🔍")

            if not foods:
                break

            # ------ Process and Display Food Results ------
            
            for food in foods:
                total_count += 1
                desc = food.get("description", "—").title()
                fdc_id = food.get("fdcId")
                
                if data_type == "Branded":
                    brand = food.get("brandName") or food.get("brandOwner", "N/A")
                    print(f"{total_count:3d}. {desc}   (Brand: {brand})  – FDC {fdc_id}")
                else:
                    print(f"{total_count:3d}. {desc}  – FDC {fdc_id}")

            # Check if we have retrieved all available results
            if len(foods) < page_size:
                break

            page_number += 1
            time.sleep(polite_pause)

        except requests.exceptions.RequestException as exc:
            print(f"Error occurred while fetching page {page_number}: {exc} ❌")
            break

# -----------------------------------------------------------------------------
# Cell 3: Main Execution Function
# -----------------------------------------------------------------------------

def main() -> None:
    """
    Execute the main search workflow for USDA food database queries.
    
    This function coordinates the search process by iterating through a
    predefined list of food terms and querying the USDA API for each item.
    It handles API configuration, dataset selection, and manages the overall
    search workflow with appropriate delays between requests.
    """
    
    # ------ Food Search Terms Configuration ------
    
    food_list = [
        "Fruit Juice"
    ]

    # ------ USDA API Configuration ------
    
    api_key = ""

    # ------ Dataset Selection ------
    
    GENERIC_DATASET = "Survey (FNDDS)"

    # ------ Execute Search Queries ------
    
    print("Starting USDA Food Data Central search process 🚀")
    
    for term in food_list:
        print(f"\nSearching for food term: {term}")
        list_usda_foods(term, GENERIC_DATASET, api_key)
        time.sleep(0.5)
    
    print("\nSearch process completed successfully ✅")

# -----------------------------------------------------------------------------
# Cell 4: Script Execution Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
    print("\nThanks for using the USDA Food Search Tool! 🍎")
    print("Hope you found exactly what you were looking for in the food database!")
