import logging
import re
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from .model import LLM
from .utils import write_json, read_from_txt
from .processing import extract_answer, postprocess_final_answer

# Set up logger
logger = logging.getLogger(__name__)

def run_direct_prompt(llm: LLM, system_prompt: str, dataset, save_path: Path):
    """Runs direct prompting experiment."""
    logger.info(f"Starting 'Direct Prompting' experiment. Saving to {save_path}")
    llm.do_sample = False
    
    total_questions = len(dataset)
    results = {}
    correct_answers = 0

    with tqdm(total=total_questions, desc="Direct Prompting") as pbar:
        for query_id, data in enumerate(dataset):
            question = data['question']
            ground_truth = extract_answer(data['answer'])

            generated = llm.generate(system_prompt=system_prompt, user_prompt=question)
            extracted = extract_answer(generated)
            predicted = postprocess_final_answer(extracted)
            
            is_correct = predicted == ground_truth
            if is_correct:
                correct_answers += 1

            results[str(query_id)] = {
                "question": question,
                "ground_truth": ground_truth,
                "system_prompt": system_prompt,
                "generated_answer": generated,
                "extracted_answer": extracted,
                "final_answer": predicted,
                "is_correct": is_correct,
            }
            
            write_json(results, save_path)
            running_accuracy = (correct_answers / (query_id + 1)) * 100
            pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)
            
    logger.info(f"Direct Prompting finished. Final Accuracy: {running_accuracy:.2f}%")

def run_zero_shot(llm: LLM, system_prompt: str, cot_prefix: str, dataset, save_path: Path):
    """Runs Zero-Shot CoT experiment."""
    logger.info(f"Starting 'Zero-Shot CoT' experiment. Saving to {save_path}")
    llm.do_sample = False
    
    total_questions = len(dataset)
    results = {}
    correct_answers = 0

    with tqdm(total=total_questions, desc="Zero-Shot CoT") as pbar:
        for query_id, data in enumerate(dataset):
            question = data['question']
            ground_truth = extract_answer(data['answer'])

            prompt = f"{cot_prefix.strip()}\n\nQuestion: {question}"
            
            generated = llm.generate(system_prompt=system_prompt, user_prompt=prompt)
            extracted = extract_answer(generated)
            predicted = postprocess_final_answer(extracted)

            is_correct = predicted == ground_truth
            if is_correct:
                correct_answers += 1

            results[str(query_id)] = {
                "question": question,
                "ground_truth": ground_truth,
                "cot_prefix": cot_prefix,
                "generated_answer": generated,
                "extracted_answer": extracted,
                "final_answer": predicted,
                "is_correct": is_correct,
            }

            write_json(results, save_path)
            running_accuracy = (correct_answers / (query_id + 1)) * 100
            pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    logger.info(f"Zero-Shot CoT finished. Final Accuracy: {running_accuracy:.2f}%")

def run_few_shot(llm: LLM, system_prompt: str, shots: int, dataset, save_path: Path, prompt_dir: Path, negative_shots: bool = False):
    """Runs Few-Shot (positive or negative) experiment."""
    shot_type = "Negative" if negative_shots else "Positive"
    shot_file = f"negative_{shots}_shots.txt" if negative_shots else f"{shots}_shots.txt"
    prompt_file_path = prompt_dir / shot_file
    
    logger.info(f"Starting '{shot_type} Few-Shot ({shots}-shot)' experiment. Saving to {save_path}")
    logger.info(f"Loading prompts from {prompt_file_path}")
    
    demonstrations = read_from_txt(prompt_file_path)
    if not demonstrations:
        logger.critical(f"Prompt file {prompt_file_path} is empty or not found. Aborting.")
        return

    llm.do_sample = False
    
    total_questions = len(dataset)
    results = {}
    correct_answers = 0

    with tqdm(total=total_questions, desc=f"{shot_type} Few-Shot ({shots}-shot)") as pbar:
        for query_id, data in enumerate(dataset):
            question = data['question']
            ground_truth = extract_answer(data['answer'])
            
            prompt = demonstrations.strip().replace("{question}", question)
            
            generated = llm.generate(system_prompt=system_prompt, user_prompt=prompt)
            extracted = extract_answer(generated)
            predicted = postprocess_final_answer(extracted)

            is_correct = predicted == ground_truth
            if is_correct:
                correct_answers += 1

            results[str(query_id)] = {
                "question": question,
                "ground_truth": ground_truth,
                "few_shot_examples": shot_file,
                "generated_answer": generated,
                "extracted_answer": extracted,
                "final_answer": predicted,
                "is_correct": is_correct,
            }

            write_json(results, save_path)
            running_accuracy = (correct_answers / (query_id + 1)) * 100
            pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    logger.info(f"{shot_type} Few-Shot finished. Final Accuracy: {running_accuracy:.2f}%")

def run_self_consistency(llm: LLM, system_prompt: str, k: int, dataset, save_path: Path):
    """Runs Self-Consistency experiment."""
    logger.info(f"Starting 'Self-Consistency' (K={k}) experiment. Saving to {save_path}")
    llm.do_sample = True
    llm.temperature = 1.0
    
    total_questions = len(dataset)
    results = {}
    correct_answers = 0

    with tqdm(total=total_questions, desc=f"Self-Consistency (K={k})") as pbar:
        for query_id, data in enumerate(dataset):
            question = data['question']
            ground_truth = extract_answer(data['answer'])
            
            votes = []
            generated_texts = []
            
            for _ in range(k):
                prompt = f"Q: {question}\nA:"
                response = llm.generate(system_prompt=system_prompt, user_prompt=prompt)
                generated_texts.append(response)

                extracted = extract_answer(response)
                final = postprocess_final_answer(extracted)
                if final:
                    votes.append(final)
            
            if votes:
                most_common = Counter(votes).most_common(1)[0][0]
            else:
                most_common = None
                logger.warning(f"No valid answers generated for Q_ID {query_id}. Votes list is empty.")

            is_correct = most_common == ground_truth
            if is_correct:
                correct_answers += 1

            results[str(query_id)] = {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answers": generated_texts,
                "extracted_answers": votes,
                "final_answer": most_common,
                "is_correct": is_correct,
            }

            write_json(results, save_path)
            running_accuracy = (correct_answers / (query_id + 1)) * 100
            pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    logger.info(f"Self-Consistency finished. Final Accuracy: {running_accuracy:.2f}%")

def run_verbalized_confidence(llm: LLM, system_prompt: str, k: int, dataset, save_path: Path):
    """Runs Verbalized Confidence experiment."""
    logger.info(f"Starting 'Verbalized Confidence' (K={k}) experiment. Saving to {save_path}")
    llm.do_sample = True
    llm.temperature = 1.0

    total_questions = len(dataset)
    results = {}
    correct_answers = 0

    with tqdm(total=total_questions, desc=f"Verbalized Confidence (K={k})") as pbar:
        for query_id, data in enumerate(dataset):
            question = data['question']
            ground_truth = extract_answer(data['answer'])

            candidates = []

            for _ in range(k):
                prompt = (
                    f"Q: Think step by step to solve this, then state a confidence score (0-100). {question}\n"
                    f"A (Format): Reasoning: <your_answer>. Confidence: <score>%. Final Answer: <value>."
                )
                output = llm.generate(system_prompt=system_prompt, user_prompt=prompt)
                
                extracted_answer = extract_answer(output)
                postprocessed = postprocess_final_answer(extracted_answer)
                
                # Extract confidence
                match = re.search(r'Confidence[:\-]?\s*(\d{1,3})\s*%?', output, re.IGNORECASE)
                if match:
                    confidence = int(match.group(1))
                    confidence = max(0, min(confidence, 100)) # Clamp to [0, 100]
                else:
                    confidence = 0 # Default if not found
                    logger.warning(f"Could not parse confidence for Q_ID {query_id}. Defaulting to 0.")

                candidates.append({
                    "generated_text": output,
                    "answer": postprocessed,
                    "confidence": confidence
                })
            
            if candidates:
                best_candidate = max(candidates, key=lambda x: x['confidence'])
                final_answer = best_candidate['answer']
            else:
                final_answer = None
                logger.error(f"No candidates generated for Q_ID {query_id}.")

            is_correct = final_answer == ground_truth
            if is_correct:
                correct_answers += 1

            results[str(query_id)] = {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answers": [c['generated_text'] for c in candidates],
                "extracted_answers": [c['answer'] for c in candidates],
                "confidences": [c['confidence'] for c in candidates],
                "final_answer": final_answer,
                "is_correct": is_correct,
            }

            write_json(results, save_path)
            running_accuracy = (correct_answers / (query_id + 1)) * 100
            pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    logger.info(f"Verbalized Confidence finished. Final Accuracy: {running_accuracy:.2f}%")

def run_subquestion(llm: LLM, system_prompt: str, dataset, save_path: Path):
    """Runs Subquestion (Least-to-Most) experiment."""
    logger.info(f"Starting 'Subquestion Decomposition' experiment. Saving to {save_path}")
    llm.do_sample = False

    total_questions = len(dataset)
    results = {}
    correct_answers = 0

    with tqdm(total=total_questions, desc="Subquestion Decomposition") as pbar:
        for query_id, data in enumerate(dataset):
            question = data['question']
            ground_truth = extract_answer(data['answer'])

            # 1. Decompose the question
            decomposition_prompt = (
                f"Decompose the following question into a sequence of easier subquestions to solve it step by step.\n"
                f"Original Question: {question}\n"
                f"List the subquestions one by one, starting with '1. '."
            )
            decomposition_output = llm.generate(system_prompt=system_prompt, user_prompt=decomposition_prompt)
            
            # Clean up subquestions
            subquestions = [
                line.strip() for line in decomposition_output.split("\n")
                if line.strip() and re.match(r'^\d+\.\s*', line)
            ]
            if not subquestions:
                logger.warning(f"Failed to decompose Q_ID {query_id}. Using full question as subquestion.")
                subquestions = [question]

            intermediate_answers = []
            context = f"Original Question: {question}"

            # 2. Answer each subquestion
            for subq in subquestions:
                answer_prompt = f"Context:\n{context}\n\nSub-Question: {subq}\nAnswer:"
                sub_answer = llm.generate(system_prompt=system_prompt, user_prompt=answer_prompt)
                intermediate_answers.append(sub_answer)
                # Update context with the new information
                context += f"\nSub-Q: {subq}\nSub-A: {sub_answer}"

            # 3. Get final answer
            final_prompt = f"Based on the following reasoning:\n{context}\n\nWhat is the final answer to the original question?\nFinal Answer:"
            final_response = llm.generate(system_prompt=system_prompt, user_prompt=final_prompt)
            final_answer = postprocess_final_answer(extract_answer(final_response))

            is_correct = final_answer == ground_truth
            if is_correct:
                correct_answers += 1

            results[str(query_id)] = {
                "question": question,
                "ground_truth": ground_truth,
                "subquestions": subquestions,
                "intermediate_answers": intermediate_answers,
                "final_response": final_response,
                "final_answer": final_answer,
                "is_correct": is_correct
            }
            
            write_json(results, save_path)
            running_accuracy = (correct_answers / (query_id + 1)) * 100
            pbar.set_postfix(accuracy=f"{running_accuracy:.2f}%")
            pbar.update(1)

    logger.info(f"Subquestion Decomposition finished. Final Accuracy: {running_accuracy:.2f}%")
