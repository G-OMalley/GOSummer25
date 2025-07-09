import icepython as ice

def find_available_filters():
    """
    This function discovers the available search facets and then
    retrieves the specific filters for the 'types' facet.
    """
    print("--- Discovering Search Facets ---")
    try:
        # Get high-level categories for filtering
        facets = ice.get_search_facets()
        print("Available facets:", facets)

        # Get the specific filter codes for the 'types' category
        # The guide shows 'types' is a valid facet [cite: 1166]
        print("\n--- Discovering Filters for 'types' ---")
        type_filters = ice.get_search_filters('types')
        
        print("Available 'type' filters:")
        for f in type_filters:
            print(f"  - Code: {f[0]}, Name: {f[1]}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    find_available_filters()