import logging
import random
from pathlib import Path
from .utils import read_json

# Set up logger
logger = logging.getLogger(__name__)

def get_accuracy(json_path: str | Path) -> float:
    """
    Computes the accuracy based on a JSON file containing correctness labels.

    Args:
        json_path: Path to the JSON file containing the evaluation data.

    Returns:
        The accuracy (0.0 to 1.0), or 0.0 if the file is empty or invalid.
    """
    logger.info(f"Calculating accuracy for {json_path}")
    data = read_json(json_path)
    
    if not isinstance(data, dict) or not data:
        logger.warning(f"No data found or invalid format in {json_path}. Returning 0.0 accuracy.")
        return 0.0

    try:
        correct = sum(1 for item in data.values() if item.get('is_correct', False))
        total = len(data)
        
        if total == 0:
            logger.warning(f"JSON file {json_path} contains no entries. Returning 0.0 accuracy.")
            return 0.0
            
        accuracy = correct / total
        logger.info(f"Accuracy for {json_path}: {correct}/{total} = {accuracy:.2%}")
        return accuracy
        
    except Exception as e:
        logger.error(f"Error calculating accuracy for {json_path}: {e}")
        return 0.0

def print_sample_responses(json_path: str | Path, num_samples: int = 3):
    """
    Prints random sample questions and responses from a results JSON file.

    Args:
        json_path: Path to the results JSON file.
        num_samples: Number of samples to print.
    """
    logger.info(f"Printing {num_samples} samples from {json_path}")
    data = read_json(json_path)
    
    if not isinstance(data, dict) or not data:
        logger.warning(f"Cannot print samples: No data found in {json_path}")
        return

    try:
        keys = list(data.keys())
        if not keys:
            logger.warning("Cannot print samples: Data is empty.")
            return
            
        samples = random.sample(keys, min(num_samples, len(keys)))

        for idx, key in enumerate(samples, 1):
            item = data[key]
            print(f"\n{'=' * 20} Sample {idx} (ID: {key}) {'=' * 20}")
            print(f"Question: {item.get('question', 'N/A')}")
            
            # Handle different output formats based on experiment type
            
            # Single generated answer (Direct, Zero/Few-Shot)
            if "generated_answer" in item:
                print(f"Answer (Generated):\n{item['generated_answer']}")
            
            # Multiple generated answers (Self-Consistency, Verbalized)
            if "generated_answers" in item:
                print("Answers (Generated):")
                for i, gen in enumerate(item["generated_answers"], 1):
                    print(f"  {i}. {gen[:150]}...") # Truncate for readability
            
            # Subquestion decomposition
            if "subquestions" in item:
                print("Questions (Decomposed) + Answers (Intermediate):")
                for sq, ans in zip(item.get("subquestions", []), item.get("intermediate_answers", [])):
                    print(f"  Q: {sq}")
                    print(f"  A: {ans[:100]}...")
                print(f"Answer (Final Generated): {item.get('final_response', 'N/A')}")

            # Final extracted answer
            print("-" * 50)
            print(f"Answer (Final Extracted): {item.get('final_answer', 'N/A')}")
            print(f"Answer (Ground Truth): {item.get('ground_truth', 'N/A')}")
            print(f"Correct: {item.get('is_correct', 'N/A')}")
            print("=" * (42 + len(str(idx)) + len(str(key))))

    except Exception as e:
        logger.error(f"Error printing samples: {e}")
