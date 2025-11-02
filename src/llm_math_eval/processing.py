import re
import logging

# Set up logger
logger = logging.getLogger(__name__)

def extract_answer(generated_text: str) -> str | None:
    """
    Extracts the last numerical expression from the generated text.

    Args:
        generated_text: The text output from which to extract a numeric answer.

    Returns:
        The extracted numerical expression if found, otherwise None.
    """
    if not generated_text:
        return None
        
    # Regex to find numeric values (integers and floats, possibly with signs)
    # This will match numbers like '123', '3.14', '-5', '+2.5'
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", generated_text)
    
    if matches:
        logger.debug(f"Extracted answer '{matches[-1]}' from text.")
        return matches[-1]
    else:
        logger.warning(f"No numerical answer found in text: '{generated_text[:50]}...'")
        return None

def postprocess_final_answer(numeric_expression: str | None) -> str | None:
    """
    Cleans and evaluates a numeric expression to obtain a final answer.

    Args:
        numeric_expression: A string containing a numeric expression.

    Returns:
        The evaluated result as a string if computation is successful,
        otherwise returns the original (cleaned) expression or None.
    """
    if numeric_expression is None:
        return None

    try:
        # Remove commas to handle numbers like '1,000'
        cleaned_up = numeric_expression.replace(',', '')
        
        # Use eval to compute simple arithmetic if present (e.g., "21-15")
        # Note: Using eval() can be a security risk if the input is not
        # controlled. Here it's assumed to be run on model-generated
        # numeric expressions.
        result = eval(cleaned_up)
        
        processed_result = str(result)
        logger.debug(f"Postprocessed '{numeric_expression}' to '{processed_result}'")
        return processed_result
    except Exception as e:
        logger.warning(f"Could not eval numeric expression '{numeric_expression}': {e}. Returning as is.")
        # If eval fails (e.g., it's just a number), return the cleaned string
        return numeric_expression.replace(',', '')
