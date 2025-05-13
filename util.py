import json

def load_json_file(filepath):
    """Load a JSON file and return its content"""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_json_file(data, filepath):
    """Save data to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
