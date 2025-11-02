import argparse
import logging
import sys
from pathlib import Path
from llm_math_eval.model import LLM
from llm_math_eval.dataset import load_gsm8k
from llm_math_eval import experiments

# --- Logger Setup ---
def setup_logging(log_level="INFO", log_dir="logs"):
    """Configures the root logger."""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    log_file = log_path / "experiment.log"
    
    # Get numeric log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Remove existing handlers to avoid duplication
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # File Handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    console_handler.setLevel(numeric_level) # Console level can be same as file
    logger.addHandler(console_handler)

    logging.info("--- New Experiment Run ---")

# --- Argument Parsing ---
def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Mathematical Reasoning Experiments on LLMs")
    
    parser.add_argument(
        "--experiment", 
        type=str, 
        required=True, 
        choices=[
            'direct', 'zero_shot', 'few_shot', 'wrong_shot', 
            'self_consistency', 'verbalized', 'subquestion'
        ],
        help="The name of the experiment to run."
    )
    
    parser.add_argument("--model_id", type=str, default="mistralai/Mistral-7B-Instruct-v0.3", help="Hugging Face model ID.")
    parser.add_argument("--sample_numbers", type=int, default=50, help="Number of samples from GSM8K to use.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Directory to save JSON results.")
    parser.add_argument("--prompt_dir", type=Path, default=Path("data/prompts"), help="Directory containing few-shot prompt files.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Directory to save log files.")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (e.g., DEBUG, INFO, WARNING).")
    
    # Experiment-specific args
    parser.add_argument("--shots", type=int, choices=[2, 4, 8], help="Number of shots for few_shot/wrong_shot (2, 4, or 8).")
    parser.add_argument("--k_samples", type=int, help="Number of samples (K) for self_consistency/verbalized.")
    parser.add_argument("--variant", type=int, choices=[1, 2], default=1, help="Variant number for direct (1, 2) or zero_shot (1, 2) experiments.")
    
    return parser.parse_args()

# --- Main Execution ---
def main():
    args = parse_args()
    
    # Setup directories and logging
    args.output_dir.mkdir(exist_ok=True)
    setup_logging(args.log_level, args.log_dir)
    
    logging.info(f"Arguments received: {args}")

    try:
        # 1. Load Model
        logging.info("Initializing model...")
        llm = LLM(model_id=args.model_id)
        
        # 2. Load Dataset
        logging.info("Loading dataset...")
        dataset = load_gsm8k(sample_number=args.sample_numbers)

        # 3. Run selected experiment
        logging.info(f"Dispatching to experiment: {args.experiment}")
        
        if args.experiment == 'direct':
            if args.variant == 1:
                prompt = "Provide a direct and concise answer to the question without additional commentary."
                save_path = args.output_dir / "direct_prompt_v1.json"
            else:
                prompt = "What is the answer to the problem? Only provide the final numeric answer."
                save_path = args.output_dir / "direct_prompt_v2.json"
            experiments.run_direct_prompt(llm, prompt, dataset, save_path)
            
        elif args.experiment == 'zero_shot':
            sys_prompt = "Be helpful, answer the question."
            if args.variant == 1:
                prefix = "Let's think step by step."
                save_path = args.output_dir / "zero_shot_v1.json"
            else:
                prefix = "Reason through it carefully and stepwise."
                save_path = args.output_dir / "zero_shot_v2.json"
            experiments.run_zero_shot(llm, sys_prompt, prefix, dataset, save_path)
            
        elif args.experiment == 'few_shot' or args.experiment == 'wrong_shot':
            if not args.shots:
                logging.critical("--shots (2, 4, or 8) is required for few_shot/wrong_shot.")
                sys.exit(1)
            is_negative = args.experiment == 'wrong_shot'
            shot_str = "wrong_shot" if is_negative else "few_shot"
            save_path = args.output_dir / f"{shot_str}_{args.shots}.json"
            experiments.run_few_shot(
                llm, "Be helpful, answer the question.", 
                args.shots, dataset, save_path, args.prompt_dir, 
                negative_shots=is_negative
            )
            
        elif args.experiment == 'self_consistency':
            if not args.k_samples:
                logging.critical("--k_samples is required for self_consistency.")
                sys.exit(1)
            save_path = args.output_dir / f"self_consistency_{args.k_samples}.json"
            experiments.run_self_consistency(
                llm, "Be helpful and answer the question concisely.", 
                args.k_samples, dataset, save_path
            )
            
        elif args.experiment == 'verbalized':
            if not args.k_samples:
                logging.critical("--k_samples is required for verbalized.")
                sys.exit(1)
            save_path = args.output_dir / f"verbalized_confidence_{args.k_samples}.json"
            experiments.run_verbalized_confidence(
                llm, "Be helpful and answer the question concisely.",
                args.k_samples, dataset, save_path
            )
            
        elif args.experiment == 'subquestion':
            save_path = args.output_dir / "subquestion.json"
            experiments.run_subquestion(
                llm, "Be helpful, answer the question.", 
                dataset, save_path
            )
            
        logging.info(f"Experiment {args.experiment} completed successfully.")

    except Exception as e:
        logging.critical(f"An unhandled error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
