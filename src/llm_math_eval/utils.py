import json
import logging
from pathlib import Path

# Set up logger
logger = logging.getLogger(__name__)

def write_json(data: dict | list, filename: str | Path):
    """
    Writes data to a JSON file with UTF-8 encoding.

    Args:
        data: The data (dict or list) to be written.
        filename: The path to the output JSON file.
    """
    try:
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        logger.debug(f"Successfully wrote data to {filename}")
    except IOError as e:
        logger.error(f"Error writing to JSON file {filename}: {e}")
    except TypeError as e:
        logger.error(f"Error serializing data to JSON: {e}")

def read_json(filename: str | Path) -> dict | list:
    """
    Reads and loads data from a JSON file with UTF-8 encoding.

    Args:
        filename: The path to the JSON file.

    Returns:
        The parsed JSON content (dict or list), or an empty dict if an error occurs.
    """
    try:
        with open(filename, "r", encoding="utf-8") as file:
            data = json.load(file)
        logger.debug(f"Successfully read data from {filename}")
        return data
    except FileNotFoundError:
        logger.error(f"JSON file not found: {filename}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from file: {filename}")
        return {}
    except IOError as e:
        logger.error(f"Error reading from JSON file {filename}: {e}")
        return {}

def read_from_txt(file_path: str | Path) -> str:
    """
    Reads the entire content of a .txt file.

    Args:
        file_path: Path to the text file.

    Returns:
        The content of the file as a string, or an empty string if an error occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        logger.debug(f"Successfully read content from {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"Text file not found: {file_path}")
        return ""
    except IOError as e:
        logger.error(f"Error reading from text file {file_path}: {e}")
        return ""
