import requests
import json
import os
import datetime 
from dotenv import load_dotenv 
import shutil 
import mimetypes

# --- Configuration ---
PLATTS_AUTH_API_URL = "https://api.ci.spglobal.com/auth/api"
PLATTS_NEWS_INSIGHTS_API_URL = "https://api.ci.spglobal.com/news-insights" 

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) 
# This path assumes .env is in TraderHelper/Platts/ and this script is in TraderHelper/Platts/Fundy/
ENV_FILE_PATH = os.path.join(SCRIPT_DIR, '..', '.env') 
USER_REQUESTED_DOWNLOAD_FOLDER = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\INFO"

# --- Platts API Authentication (Bearer Token) ---
def get_platts_access_token(username, password):
    """Generates an access token for api.ci.spglobal.com APIs."""
    payload = {"username": username, "password": password}
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    print(f"\nAttempting to authenticate to Platts API (for Bearer Token) with username: {username[:5]}...")
    try:
        response = requests.post(PLATTS_AUTH_API_URL, data=payload, headers=headers)
        response.raise_for_status() 
        token_data = response.json()
        access_token = token_data.get("access_token")
        if access_token:
            print("Platts API Authentication successful. Token received.")
            return access_token
        else:
            print(f"Access token not found in Platts API response. Content: {response.text}")
            return None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error during Platts API authentication: {http_err}, Content: {response.text}")
    except json.JSONDecodeError as json_err:
        print(f"JSON decode error during Platts API authentication: {json_err}\nResponse content: {response.text}")
    except Exception as e:
        print(f"Request error during Platts API authentication: {e}")
    return None

# --- News & Insights API Function ---
def find_latest_package_flexible(token, primary_publication_name, alternative_search_terms=None):
    """
    Finds the latest package ID using multiple search strategies with improved relevance checking.
    Returns the content ID, publicationDate, and fileName.
    """
    if not token:
        print("No access token provided for News & Insights API search.")
        return None, None, None

    search_endpoint = f"{PLATTS_NEWS_INSIGHTS_API_URL}/v1/search/packages"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    
    search_strategies = [
        {"type": "filter", "value": primary_publication_name, "params": {"filter": f'publication:"{primary_publication_name}"', "PageSize": 1, "field": "publication"}},
        {"type": "q", "value": primary_publication_name, "params": {"q": primary_publication_name, "PageSize": 1}},
    ]
    if alternative_search_terms:
        for term in alternative_search_terms: # Assumes alternative_search_terms is a list of strings
            search_strategies.append({"type": "q", "value": term, "params": {"q": term, "PageSize": 1}})

    for strategy in search_strategies:
        params = strategy["params"]
        search_value_used = strategy["value"] 
        search_type_desc = f"filter='publication:{search_value_used}' (field=publication)" if strategy["type"] == "filter" else f"keyword q='{search_value_used}'"

        print(f"\nSearching for package using {search_type_desc}...")
        print(f"Endpoint: {search_endpoint}")
        print(f"Params: {params}")

        try:
            response = requests.get(search_endpoint, headers=headers, params=params)
            response.raise_for_status()
            search_results_json = response.json()
            
            results = search_results_json.get("results", [])
            if results:
                latest_package = results[0] 
                package_id = latest_package.get('id')
                package_pub_date = latest_package.get('publicationDate') 
                package_updated_date = latest_package.get('updatedDate')
                package_title = str(latest_package.get('title', '')).lower()
                package_filename_from_meta = latest_package.get('fileName')
                found_publication_name_meta = str(latest_package.get('publicationName', '')).lower()
                
                # Consolidate text fields for relevance checking
                text_to_search_in = (found_publication_name_meta + " " + package_title + " " + str(package_filename_from_meta).lower())


                print(f"Found package: ID={package_id}, PubDate={package_pub_date}, UpdatedDate={package_updated_date}, Title='{package_title}', MetaFileName='{package_filename_from_meta}', MetaPubName='{found_publication_name_meta}'")
                
                is_relevant = False
                # Strategy 1: Exact filter match (lenient on metadata if filter itself matches)
                if strategy["type"] == "filter":
                    if primary_publication_name.lower() in found_publication_name_meta or \
                       primary_publication_name.lower() in package_title:
                        is_relevant = True
                    elif not found_publication_name_meta and not package_title: # If filter matched but no name/title
                        print("Filter matched but package has no title/publication name in search result. Assuming relevant due to filter match.")
                        is_relevant = True 
                
                # Strategy 2: Keyword 'q' search
                else: 
                    # If searching for a known acronym like "MDFD"
                    if search_value_used.lower() == "mdfd":
                        if "mdfd" in text_to_search_in or "megawatt daily market fundamentals" in text_to_search_in:
                            is_relevant = True
                    # General keyword relevance for primary publication name
                    elif search_value_used.lower() == primary_publication_name.lower():
                        primary_keywords = ["megawatt", "daily", "fundamentals", "market"]
                        match_count = 0
                        for kw in primary_keywords:
                            if kw in text_to_search_in:
                                match_count +=1
                        if match_count >= 3: # Require at least 3 of the core keywords to match
                            is_relevant = True
                    # For other alternative terms, check if the term itself is in the text
                    elif search_value_used.lower() in text_to_search_in:
                        is_relevant = True


                if is_relevant:
                    print(f"Found package IS considered relevant for search term '{search_value_used}'.")
                    return package_id, package_pub_date, package_filename_from_meta
                else:
                    print(f"Warning: Found package (Title: '{package_title}', PubName: '{found_publication_name_meta}') "
                          f"NOT deemed a strong match for search term '{search_value_used}'. Continuing search...")
            else: 
                print(f"No packages found with {search_type_desc}.")
                if strategy["type"] == "filter":
                     print("API Response for filter strategy:", json.dumps(search_results_json, indent=2))

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error with {search_type_desc}: {http_err}, URL: {response.url}, Content: {response.text}")
        except Exception as e:
            print(f"An error occurred with {search_type_desc}: {e}")
        
    print(f"All search strategies exhausted. Could not find a definitive package for '{primary_publication_name}'.")
    return None, None, None


# --- News & Insights Content Download Function ---
def download_news_insights_content(platts_access_token, content_id, download_folder, publication_name, meta_filename=None, pub_date=None):
    """Downloads the content, inspecting Content-Type for correct file extension."""
    if not content_id:
        print("No content ID provided for download.")
        return None

    endpoint = f"{PLATTS_NEWS_INSIGHTS_API_URL}/v1/content/{content_id}"
    headers = {"Authorization": f"Bearer {platts_access_token}", "Accept": "application/octet-stream"}
    print(f"\nDownloading News & Insights content for ID {content_id} from: {endpoint}")
    
    try:
        response = requests.get(endpoint, headers=headers, stream=True) 
        response.raise_for_status()
        
        content_type = response.headers.get('content-type')
        print(f"Received Content-Type: {content_type}")

        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
            print(f"Created download folder: {download_folder}")

        safe_publication_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in publication_name).rstrip()
        base_name = safe_publication_name.replace(" ", "_")
        date_suffix = (datetime.datetime.strptime(pub_date, '%Y-%m-%dT%H:%M:%SZ').strftime('%Y%m%d') 
                       if pub_date and isinstance(pub_date, str) else datetime.date.today().strftime('%Y%m%d'))
        
        file_extension = ".xlsx" 
        
        if meta_filename and isinstance(meta_filename, str):
            _name_part, ext_part = os.path.splitext(meta_filename)
            if ext_part and ext_part.lower() in ['.xls', '.xlsx', '.zip', '.csv', '.pdf', '.txt']:
                file_extension = ext_part.lower()
                print(f"Using extension from meta_filename: {file_extension}")
            elif not ext_part : 
                 print(f"Warning: meta_filename '{meta_filename}' has no extension, will use Content-Type or default.")

        # If meta_filename didn't provide a good extension, try to guess from Content-Type
        # Only override if current file_extension is still the default .xlsx and meta_filename didn't provide a valid one
        if file_extension == ".xlsx" and not (meta_filename and os.path.splitext(meta_filename)[1] and os.path.splitext(meta_filename)[1].lower() in ['.xls', '.xlsx', '.zip', '.csv', '.pdf', '.txt']):
            if content_type:
                guessed_extension = mimetypes.guess_extension(content_type.split(';')[0].strip())
                if guessed_extension:
                    known_extensions = ['.xls', '.xlsx', '.zip', '.csv', '.pdf', '.txt']
                    if guessed_extension in known_extensions:
                         file_extension = guessed_extension
                         print(f"Using extension from Content-Type '{content_type}': {file_extension}")
                    else: 
                        print(f"Warning: Guessed extension '{guessed_extension}' from Content-Type '{content_type}' is not in known list. Defaulting to .xlsx.")
                        # file_extension remains ".xlsx"
                else: 
                    print(f"Warning: Could not guess extension from Content-Type '{content_type}'. Defaulting to .xlsx.")
            else: 
                 print(f"Warning: No Content-Type header. Defaulting to .xlsx.")
        
        filename_to_save = f"{base_name}_{date_suffix}{file_extension}"
        file_path = os.path.join(download_folder, filename_to_save)
        
        with open(file_path, "wb") as f:
            shutil.copyfileobj(response.raw, f)
        print(f"Successfully downloaded and saved content to: {file_path}")
        return file_path
        
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error downloading content: {http_err}, URL: {response.url}")
        try: print(f"Error Response content: {response.json()}")
        except json.JSONDecodeError: print(f"Error Response content (non-JSON): {response.text}")
    except Exception as e: print(f"An error occurred downloading content: {e}")
    return None

# --- Main Script ---
if __name__ == "__main__":
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    env_file_actual_path = os.path.join(current_script_path, '..', '.env')

    print(f"Script directory: {current_script_path}")
    print(f"Attempting to load .env file from: {env_file_actual_path}") 

    if not load_dotenv(dotenv_path=env_file_actual_path): 
        print(f"Warning: Could not find .env file at {env_file_actual_path}")
        exit("Exiting: Credentials .env file not found.")
    else:
        print(f"Successfully loaded .env file from {env_file_actual_path}")

    PLATTS_API_USERNAME = os.getenv("PLATTS_USERNAME") 
    PLATTS_API_PASSWORD = os.getenv("PLATTS_PASSWORD")

    if not (PLATTS_API_USERNAME and PLATTS_API_PASSWORD):
        print("PLATTS_USERNAME or PLATTS_PASSWORD for S&P Global API not found in .env file.")
        exit("Exiting: Credentials incomplete.")
    
    print(f"Using Platts API Username (for token): {PLATTS_API_USERNAME[:5]}...")

    if not os.path.exists(USER_REQUESTED_DOWNLOAD_FOLDER):
        try:
            os.makedirs(USER_REQUESTED_DOWNLOAD_FOLDER)
            print(f"Created directory: {USER_REQUESTED_DOWNLOAD_FOLDER}")
        except OSError as e:
            print(f"Error creating directory {USER_REQUESTED_DOWNLOAD_FOLDER}: {e}")
            exit(f"Exiting: Could not create target download directory.")

    print("\n--- Step 1: Authenticate to Platts API (Get Bearer Token) ---")
    platts_access_token = get_platts_access_token(PLATTS_API_USERNAME, PLATTS_API_PASSWORD)

    if platts_access_token:
        publication_to_find = "Megawatt Daily Market Fundamentals Data" 
        # Re-ordered alternative_terms to prioritize MDFD
        alternative_terms = [
            "MDFD", 
            "Megawatt Daily Fundamentals", 
            "Market Fundamentals Data Megawatt", # More specific than just "Megawatt Daily"
            "Megawatt Daily" 
            # Removed "Electricity Daily Demand Report" as it was a distractor
        ]
        print(f"\n--- Step 2: Find latest '{publication_to_find}' Package ID (trying multiple strategies) ---")
        
        content_id, found_pub_date, meta_filename = find_latest_package_flexible(
            platts_access_token, 
            publication_to_find,
            alternative_search_terms=alternative_terms
        )

        if content_id:
            print(f"\n--- Step 3: Download Content ID {content_id} (published on {found_pub_date}) ---")
            
            downloaded_file_path = download_news_insights_content(
                platts_access_token,
                content_id,
                USER_REQUESTED_DOWNLOAD_FOLDER, 
                publication_name=publication_to_find, 
                meta_filename=meta_filename,
                pub_date=found_pub_date 
            )

            if downloaded_file_path:
                print(f"\n'{publication_to_find}' (or best match) file downloaded to: {downloaded_file_path}")
            else:
                print(f"Failed to download the content for the found package ID '{content_id}'.")
        else:
            print(f"Could not find the package/content ID for '{publication_to_find}' after trying multiple search strategies.")
    else:
        print("Could not obtain Platts API access token. Halting.")

    print("\n--- Script finished ---")