# -----------------------------------------------------------------------------
# USDA Food Data Central Search Tool With Customizable Filtering
# -----------------------------------------------------------------------------

"""
This script enables users to query multiple food items across various data types
from the USDA database, including both generic foods (Survey FNDDS, Foundation,
SR Legacy) and branded food products. Users can select from several filtering
options to narrow results, such as filtering by specific keywords, requiring
that results begin with the search terms, or applying a strict term filter that
matches exact phrases or phrases followed by a comma.

The script implements polite API usage with configurable delays between requests
and handles pagination automatically to retrieve all available results.
"""

# -----------------------------------------------------------------------------
# Cell 1: Import Required Libraries and Dependencies
# -----------------------------------------------------------------------------

# ------ Import All Required Modules ------
import requests
import time
import ipywidgets as widgets
from IPython.display import display, clear_output
import re

# -----------------------------------------------------------------------------
# Cell 2: Singular/Plural Processing Functions
# -----------------------------------------------------------------------------

def to_singular(word):
    """
    Convert a word to its singular form using basic English rules.
    
    Args:
        word: The word to convert to singular form
        
    Returns:
        The singular form of the input word
    """
    word = word.strip()
    word_lower = word.lower()
    
    # Handle special cases first
    special_cases = {
        'potatoes': 'potato',
        'tomatoes': 'tomato',
        'mangoes': 'mango',
        'heroes': 'hero',
        'echoes': 'echo',
        'vetoes': 'veto',
        'children': 'child',
        'feet': 'foot',
        'teeth': 'tooth',
        'geese': 'goose',
        'mice': 'mouse',
        'men': 'man',
        'women': 'woman',
        'people': 'person',
        'oxen': 'ox',
    }
    
    if word_lower in special_cases:
        # Preserve original capitalization
        if word.isupper():
            return special_cases[word_lower].upper()
        elif word[0].isupper():
            return special_cases[word_lower].capitalize()
        else:
            return special_cases[word_lower]
    
    # Handle regular plural rules
    if word_lower.endswith('ies') and len(word) > 3:
        # berries -> berry, cherries -> cherry
        return word[:-3] + 'y'
    elif word_lower.endswith('ves'):
        # leaves -> leaf, knives -> knife
        return word[:-3] + 'f'
    elif word_lower.endswith('ses') and len(word) > 3:
        # glasses -> glass, classes -> class
        return word[:-2]
    elif word_lower.endswith('es') and len(word) > 2:
        # Check if it's likely a plural ending
        if word_lower.endswith(('ches', 'shes', 'xes', 'zes')):
            return word[:-2]
        # For words ending in 'es', try removing just 's' first
        elif not word_lower.endswith(('aes', 'ees', 'ies', 'oes', 'ues')):
            return word[:-1]
    elif word_lower.endswith('s') and len(word) > 1 and not word_lower.endswith('ss'):
        # Simple plural: cats -> cat, dogs -> dog
        return word[:-1]
    
    # If no rules apply, return the original word
    return word

def to_plural(word):
    """
    Convert a word to its plural form using basic English rules.
    
    Args:
        word: The word to convert to plural form
        
    Returns:
        The plural form of the input word
    """
    word = word.strip()
    word_lower = word.lower()
    
    # Handle special cases first
    special_cases = {
        'potato': 'potatoes',
        'tomato': 'tomatoes',
        'mango': 'mangoes',
        'hero': 'heroes',
        'echo': 'echoes',
        'veto': 'vetoes',
        'child': 'children',
        'foot': 'feet',
        'tooth': 'teeth',
        'goose': 'geese',
        'mouse': 'mice',
        'man': 'men',
        'woman': 'women',
        'person': 'people',
        'ox': 'oxen',
    }
    
    if word_lower in special_cases:
        # Preserve original capitalization
        if word.isupper():
            return special_cases[word_lower].upper()
        elif word[0].isupper():
            return special_cases[word_lower].capitalize()
        else:
            return special_cases[word_lower]
    
    # Handle regular plural rules
    if word_lower.endswith('y') and len(word) > 1 and word_lower[-2] not in 'aeiou':
        # berry -> berries, cherry -> cherries
        return word[:-1] + 'ies'
    elif word_lower.endswith(('f', 'fe')):
        # leaf -> leaves, knife -> knives
        if word_lower.endswith('fe'):
            return word[:-2] + 'ves'
        else:
            return word[:-1] + 'ves'
    elif word_lower.endswith(('ch', 'sh', 'x', 'z', 's')):
        # glass -> glasses, box -> boxes
        return word + 'es'
    elif word_lower.endswith('o') and len(word) > 1 and word_lower[-2] not in 'aeiou':
        # potato -> potatoes (but photo -> photos, handled by special cases if needed)
        return word + 'es'
    else:
        # Simple plural: cat -> cats, dog -> dogs
        return word + 's'

def process_food_entry(entry):
    """
    Process a single food entry to generate both singular and plural forms.
    
    Args:
        entry: A single food item string to process
        
    Returns:
        A list of terms to search for including singular and plural variants
    """
    entry = entry.strip()
    if not entry:
        return []
    
    # Split multi-word entries and process each significant word
    words = entry.split()
    
    # For single words, create both singular and plural
    if len(words) == 1:
        singular = to_singular(entry)
        plural = to_plural(entry)
        
        # Return unique forms
        if singular.lower() == plural.lower():
            return [entry]  # Word doesn't change form
        elif singular.lower() == entry.lower():
            return [singular, plural]  # Entry was singular
        elif plural.lower() == entry.lower():
            return [singular, plural]  # Entry was plural
        else:
            return [entry, singular, plural]  # Entry was neither standard form
    
    # For multi-word entries, focus on the main noun (usually the last word)
    else:
        main_word = words[-1]
        prefix_words = ' '.join(words[:-1])
        
        singular_main = to_singular(main_word)
        plural_main = to_plural(main_word)
        
        terms = []
        
        # Add the original entry
        terms.append(entry)
        
        # Add singular and plural variants if they're different
        if singular_main.lower() != main_word.lower():
            terms.append(f"{prefix_words} {singular_main}".strip())
        if plural_main.lower() != main_word.lower():
            terms.append(f"{prefix_words} {plural_main}".strip())
        
        # Remove duplicates while preserving order
        unique_terms = []
        seen = set()
        for term in terms:
            if term.lower() not in seen:
                unique_terms.append(term)
                seen.add(term.lower())
        
        return unique_terms

def expand_food_list(food_list_input):
    """
    Process the entire food list to create singular and plural variants.
    
    Args:
        food_list_input: List of food items or comma-separated string
    
    Returns:
        List of processed food terms with singular and plural variants
    """
    # Handle both list and string inputs
    if isinstance(food_list_input, str):
        # Parse comma-separated string, handling "and" as well
        items = re.split(r',|\s+and\s+', food_list_input)
        items = [item.strip() for item in items if item.strip()]
    else:
        items = food_list_input
    
    expanded_list = []
    
    print("=== SINGULAR AND PLURAL PROCESSING ===")
    print("Converting input entries to include both singular and plural forms:\n")
    
    for item in items:
        processed_terms = process_food_entry(item)
        if processed_terms:
            # Display the conversion
            if len(processed_terms) == 1:
                print(f"'{item}' → {processed_terms[0]}")
            else:
                terms_display = '; '.join(processed_terms)
                print(f"'{item}' → {terms_display}")
            
            expanded_list.extend(processed_terms)
    
    print(f"\nExpanded from {len(items)} entries to {len(expanded_list)} search terms.")
    print("=" * 50 + "\n")
    
    return expanded_list

# -----------------------------------------------------------------------------
# Cell 3: Core Function Definitions
# -----------------------------------------------------------------------------

# ------ Define Filter Keywords ------
KEYWORDS = {"Raw", "Frozen", "Canned", "Nfs", "Plain", "Unsalted"}
KEYWORDS_LOWER = {k.lower() for k in KEYWORDS}

# ------ Define the USDA Food Search and Filtering Function ------
def list_usda_foods(
    query: str,
    data_type: str,
    api_key: str,
    enable_keyword_filter: bool,
    enable_leading_words_filter: bool,
    enable_strict_comma_filter: bool,
    page_size: int = 200,
    polite_pause: float = 0.25
) -> list[str]:
    """
    Search the USDA FDC API for food items matching the specified query and data type,
    applying user-selected filters to reorder or restrict results. Implement logic to
    prioritize or strictly filter results based on the presence of keywords, the position
    of query words, and the occurrence of exact or comma-separated phrases.

    Args:
        query: Search term for food items
        data_type: Type of food data to search
        api_key: USDA API key for authentication
        enable_keyword_filter: If True, includes only items containing a keyword
        enable_leading_words_filter: If True, requires all query words to appear at the
            start of the food description
        enable_strict_comma_filter: If True, restricts results to those where the leading
            term is followed by a comma or is an exact match
        page_size: Number of results per page
        polite_pause: Delay between API requests in seconds
    
    Returns:
        A list of strings, where each string is the description of a filtered food item
    """
    base_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    page_number = 1
    all_pages_foods = []

    print(f"\n=== Search Term: {query!r}  |  Data Type: {data_type} ===")

    # ------ Retrieve All Results From the API ------
    while True:
        params = {
            "api_key": api_key,
            "query": query,
            "dataType": [data_type],
            "pageSize": page_size,
            "pageNumber": page_number
        }
        try:
            resp = requests.get(base_url, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            foods = payload.get("foods", [])

            if page_number == 1:
                total_hits = payload.get('totalHits', 0)
                print(f"Total matches found in the USDA database: {total_hits} 🔍")

            if not foods:
                break

            all_pages_foods.extend(foods)

            if len(foods) < page_size:
                break
            page_number += 1
            time.sleep(polite_pause)
        except requests.exceptions.RequestException as exc:
            print(f"An error occurred while fetching page {page_number}: {exc} ❌")
            break

    # ------ Helper Function for Strict Comma or Exact Match Filtering ------
    def is_strict_match(food_item, query_str):
        """
        Determine if a food item matches the strict filtering criteria.
        
        Args:
            food_item: Dictionary containing food information from API
            query_str: The search query string
            
        Returns:
            Boolean indicating whether the item meets strict match criteria
        """
        desc = food_item.get("description", "")
        lower_desc = desc.lower()
        lower_query = query_str.lower()

        if lower_desc == lower_query:
            return True

        query_words = lower_query.split()
        num_query_words = len(query_words)
        leading_desc_words = lower_desc.split()[:num_query_words]
        cleaned_desc_words_set = {re.sub(r'[^a-z0-9]', '', word) for word in leading_desc_words}

        if set(query_words) != cleaned_desc_words_set:
            return False

        desc_words_full = desc.split()
        if len(desc_words_full) < num_query_words:
            return False
        last_word_of_phrase = desc_words_full[num_query_words - 1]
        return last_word_of_phrase.endswith(',')

    # ------ Apply Keyword and Leading Words Filters ------
    pre_filtered_results = []
    for food in all_pages_foods:
        desc = food.get("description", "—")
        lower_desc = desc.lower()

        if enable_keyword_filter:
            desc_words_set = set(re.sub(r'[^a-z0-9\s]', '', lower_desc).split())
            has_keyword = not KEYWORDS_LOWER.isdisjoint(desc_words_set)
            is_perfect_match = (lower_desc == query.lower())
            if not (has_keyword or is_perfect_match):
                continue

        if enable_leading_words_filter:
            query_words_set = set(query.lower().split())
            num_words_to_check = len(query_words_set)
            leading_desc_words = lower_desc.split()[:num_words_to_check]
            cleaned_desc_words_set = {re.sub(r'[^a-z0-9]', '', word) for word in leading_desc_words}
            if query_words_set != cleaned_desc_words_set:
                continue

        pre_filtered_results.append(food)

    # ------ Apply Strict Comma Filter or Reorder Results ------
    final_results = []
    if enable_strict_comma_filter:
        final_results = [food for food in pre_filtered_results if is_strict_match(food, query)]
    elif enable_leading_words_filter:
        pre_filtered_results.sort(key=lambda food: is_strict_match(food, query), reverse=True)
        final_results = pre_filtered_results
    else:
        final_results = pre_filtered_results

    # ------ Display the Filtered Results and Prepare Return List ------
    filtered_descriptions = []
    if not final_results:
        print("--> No results found that match the selected filter criteria for this search term.")
    else:
        for i, food in enumerate(final_results, 1):
            desc_raw = food.get("description", "—")
            desc_title = desc_raw.title()
            fdc_id = food.get("fdcId")
            
            # Add the raw description to the list that will be returned
            filtered_descriptions.append(desc_raw)

            if data_type == "Branded":
                brand = food.get("brandName") or food.get("brandOwner", "N/A")
                print(f"{i:3d}. {desc_title} (Brand: {brand}) – FDC ID: {fdc_id}")
            else:
                print(f"{i:3d}. {desc_title} – FDC ID: {fdc_id}")

        print(f"--> Displayed {len(final_results)} filtered results for this search term.")
    
    # Return the list of filtered descriptions
    return filtered_descriptions

# -----------------------------------------------------------------------------
# Cell 4: User Interface Widget Setup and Main Execution
# -----------------------------------------------------------------------------

# ------ Configure Food Search Terms ------
food_list_input = [""]

# Alternative: You can also use a string format like this:
# food_list_input = "Eggs, Greek Yogurt, Potatoes, and Olive Oil"

# ------ Set Up USDA API Configuration ------
api_key = ""
GENERIC_DATASET = "Survey (FNDDS)"

# ------ Create Interactive UI Widgets ------
keyword_filter_checkbox = widgets.Checkbox(
    value=False,
    description='Filter by Keywords',
    indent=False,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='auto')
)

leading_words_filter_checkbox = widgets.Checkbox(
    value=True,
    description='Only show results that start with the search terms',
    indent=False,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='auto')
)

strict_comma_filter_checkbox = widgets.Checkbox(
    value=True,
    description='Apply strict matching',
    indent=False,
    tooltip='Requires the second filter. Finds exact terms followed by a comma or exact description matches.',
    style={'description_width': 'initial'},
    layout=widgets.Layout(margin='0 0 0 25px', width='auto')
)

# Add checkbox for enabling singular and plural processing
singular_plural_checkbox = widgets.Checkbox(
    value=False,
    description='Automatically generate singular and plural variants of search terms',
    indent=False,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='auto')
)

# Add a new checkbox for printing the final list
print_list_checkbox = widgets.Checkbox(
    value=True,
    description='Print a combined Python list of all results at the end',
    indent=False,
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='auto')
)

search_button = widgets.Button(
    description='Search USDA Database',
    button_style='success',
    tooltip='Click to start the search based on the selected filters',
    icon='search',
    layout=widgets.Layout(width='300px', margin='15px 0 0 0')
)

output_area = widgets.Output()

# ------ Link Strict Filter Checkbox to Leading Words Filter ------
def on_leading_words_change(change):
    """
    Enable or disable the strict filter checkbox based on the state of the leading words filter.
    If the leading words filter is unchecked, the strict filter is also unchecked.
    
    Args:
        change: The change event from the checkbox widget
    """
    strict_comma_filter_checkbox.disabled = not change.new
    if not change.new:
        strict_comma_filter_checkbox.value = False

leading_words_filter_checkbox.observe(on_leading_words_change, names='value')

# ------ Define Search Callback for Button Click ------
def on_search_button_clicked(b):
    """
    Handle the search button click event and execute the food search process.
    
    Args:
        b: The button widget that triggered the event
    """
    with output_area:
        clear_output(wait=True)
        print("Beginning the USDA Food Data Central search process 🚀\n")

        use_singular_plural = singular_plural_checkbox.value
        use_keyword_filter = keyword_filter_checkbox.value
        use_leading_words_filter = leading_words_filter_checkbox.value
        use_strict_comma_filter = strict_comma_filter_checkbox.value
        should_print_list = print_list_checkbox.value

        # Conditionally process the food list to include singular and plural variants
        if use_singular_plural:
            processed_food_list = expand_food_list(food_list_input)
        else:
            # Use original food list without processing
            if isinstance(food_list_input, str):
                # Parse comma-separated string, handling "and" as well
                items = re.split(r',|\s+and\s+', food_list_input)
                processed_food_list = [item.strip() for item in items if item.strip()]
            else:
                processed_food_list = food_list_input
            
            print("Using original search terms without singular and plural processing:")
            for i, term in enumerate(processed_food_list, 1):
                print(f"{i}. {term}")
            print("=" * 50 + "\n")

        all_filtered_foods = []  # Master list to hold all results

        for term in processed_food_list:
            # Capture the list of descriptions returned by the function
            results_for_term = list_usda_foods(
                query=term,
                data_type=GENERIC_DATASET,
                api_key=api_key,
                enable_keyword_filter=use_keyword_filter,
                enable_leading_words_filter=use_leading_words_filter,
                enable_strict_comma_filter=use_strict_comma_filter,
            )
            # Add the results for the current term to the master list
            all_filtered_foods.extend(results_for_term)
            time.sleep(0.5)

        print("\nThe search process has finished successfully ✅")

        # If the user requested it and results were found, print the formatted list
        if should_print_list and all_filtered_foods:
            print("\n" + "="*50)
            print("--- Combined List of Filtered Foods ---")
            print("="*50 + "\n")
            print("food_list = [")
            for food_item in all_filtered_foods:
                # Use repr() to handle quotes inside the string properly
                print(f'    {repr(food_item)},')
            print("]")
            print(f"\nSaved comprehensive list containing {len(all_filtered_foods)} food items 📋")

        print("\nThank you for using the USDA Food Search Tool! Keep exploring nutritious choices and have a delicious day! 🍎")

search_button.on_click(on_search_button_clicked)

# -----------------------------------------------------------------------------
# Cell 5: Display User Interface and Run Tool
# -----------------------------------------------------------------------------

# ------ Display User Interface Components ------
print("Please configure your search filters below and click the button to begin your search.")
display(widgets.VBox([
    widgets.Label("Search Options:"),
    singular_plural_checkbox,
    widgets.HTML("<hr style='margin-top:10px; margin-bottom:10px;'>"),
    widgets.Label("Filter Options:"),
    keyword_filter_checkbox,
    leading_words_filter_checkbox,
    strict_comma_filter_checkbox,
    widgets.HTML("<hr style='margin-top:10px; margin-bottom:10px;'>"),
    widgets.Label("Output Options:"),
    print_list_checkbox,
    search_button,
    widgets.HTML("<hr>"),
    widgets.Label("Search Results:")
]))
display(output_area)
