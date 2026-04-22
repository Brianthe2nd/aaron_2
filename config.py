import json
import os
def Print(txt):
    print(txt)
# from std_out import Print
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")


# Assuming CONFIG_PATH is defined globally, e.g., CONFIG_PATH = "config.json"
# Please ensure this constant is defined in your actual code.

def get_config(key=None, default=None, name=None):
    """
    Reads config.json and returns the full dict or a specific key's value.
    If the file doesn't exist it creates it.
    If the key is missing, returns default.
    """
    
    # Ensure the file exists
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)
    else:
        pass
    # Try to read the file
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        data = {}
        with open(CONFIG_PATH, "w", encoding="utf-8") as fw:
            json.dump(data, fw)
    except FileNotFoundError:
        data = {}
    
    
    if not name:
        trader_data = data.copy() # Gets full config
    else:
        trader_data = data.get(name)
        if not trader_data:
            data[name] = {}
            trader_data = {}
        else:
            pass
    
    if not key:
        # Returning the full section (trader_data) for consistency if key is not provided
        return trader_data 
    else:
        result = trader_data.get(key, default)
        return result
    
if __name__ == "__main__":
    print(get_config("best_search_logo_height", name = "jd"))

import os
import json
from typing import Any, Optional, Dict, Union


def _load_config_data() -> Dict:
    """Helper function to load data from CONFIG_PATH or return an empty dict."""
    if not os.path.exists(CONFIG_PATH):
        return {}
        
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file exists but is corrupted, treat it as empty
        return {}
    except Exception:
        # Catch other potential file errors
        return {}

def update_config(key: str, value: Any, name: Optional[str] = None) -> None:
    """
    Updates a key-value pair within the configuration file.
    
    The update occurs at the root level if 'name' is None, or within the 
    sub-section identified by 'name' if provided.
    
    Args:
        key (str): The configuration key to update.
        value (Any): The new value for the key.
        name (Optional[str]): The sub-section name (e.g., trader, user) to update.
    """
    
    # 1. Load the existing data
    data: Dict = _load_config_data()
    # print("--- UPDATE CONFIG START ---")
    # print(f"Loading data from: {CONFIG_PATH}")
    # print(f"Key: {repr(key)}, Value: {repr(value)}, Name: {repr(name)}")

    # 2. Determine the target dictionary to modify
    target_dict: Dict

    if not name:

        # print("Target: Root Configuration.")
        target_dict = data
    else:
        # print(f"Target: Sub-section '{name}'.")
        
        if name not in data or not isinstance(data.get(name), dict):
            # print(f"Creating new sub-section '{name}'.")
            data[name] = {}  
 
        target_dict = data[name]


    target_dict[key] = value
    # print(f"Updated: target_dict['{key}'] = {repr(value)}")
 
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        # print(f"Successfully SAVED all data to {CONFIG_PATH}.")
    except Exception as e:
        print(f"ERROR: Could not save config file: {e}")

    # print("--- UPDATE CONFIG END ---")

 

if __name__ == "__main__":
    update_config({"mode": "test", "retry_count": 3})
    update_config("last_value", 42)
    Print(get_config())
    Print(get_config("last_value"))
