import logging
from datasets import load_dataset, Dataset, DatasetDict

# Set up logger
logger = logging.getLogger(__name__)

def load_gsm8k(sample_number: int) -> Dataset:
    """
    Loads a subset of the GSM8K dataset for evaluation.

    Args:
        sample_number: The number of samples to select from the test set.

    Returns:
        A subset of the GSM8K test dataset.
    """
    try:
        logger.info(f"Loading 'openai/gsm8k' dataset...")
        dataset: DatasetDict = load_dataset("openai/gsm8k", 'main')
        
        test_dataset: Dataset = dataset['test']
        
        if sample_number > len(test_dataset):
            logger.warning(
                f"Requested sample number ({sample_number}) is larger than "
                f"test set size ({len(test_dataset)}). Using full test set."
            )
            sample_number = len(test_dataset)
            
        dataset_samples = test_dataset.select(range(sample_number))
        logger.info(f"Loaded {len(dataset_samples)} samples from GSM8K test set.")
        return dataset_samples
        
    except Exception as e:
        logger.critical(f"Failed to load GSM8K dataset: {e}")
        raise
