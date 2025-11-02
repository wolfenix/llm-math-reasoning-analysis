# LLM Mathematical Reasoning Analysis

This project, developed for the Deep Learning (M.S.) course, provides a systematic analysis of the mathematical reasoning capabilities of the `mistralai/Mistral-7B-Instruct-v0.3` model. It evaluates 7 different prompting techniques on the GSM8K benchmark dataset to compare their effectiveness in solving multi-step math word problems.

## Features

* **7 Prompting Strategies:** Implements and benchmarks Direct, Zero-Shot CoT, Few-Shot (Positive), Few-Shot (Negative), Self-Consistency, Verbalized Confidence, and Subquestion Decomposition.
* **Comprehensive Evaluation:** Provides a scriptable framework to run experiments, generate reproducible results, and calculate accuracy for each method.
* **Modular Codebase:** All logic is modularized into `src/` for easy reuse and extension (model, dataset, processing, and experiments).
* **In-Depth Analysis:** Includes an analysis of how prompt structure, the number of examples (shots), and reasoning paths (CoT) impact model performance.

## Core Concepts & Techniques

* **Prompt Engineering:** Zero-Shot, Few-Shot, and Chain-of-Thought (CoT) prompting.
* **LLM Evaluation:** Benchmarking on the GSM8K dataset to measure mathematical reasoning.
* **Advanced Reasoning Methods:** Implementation of Self-Consistency (majority voting) and Subquestion Decomposition (Least-to-Most).
* **Hugging Face Integration:** Utilizes the `transformers` library for model loading and `datasets` for data handling.

---

## How It Works

This project is structured to run a series of controlled experiments. The core logic is modular, and a main script dispatches tasks based on command-line arguments.

### 1. Core Logic & Experiment Framework

* **`src/llm_math_eval/model.py`**: Contains the `LLM` class, a wrapper around the `transformers` text-generation pipeline to simplify interaction with the Mistral model.
* **`src/llm_math_eval/experiments.py`**: This module is the heart of the project, containing separate functions for all 7 evaluation methods (e.g., `run_direct_prompt`, `run_few_shot`, `run_self_consistency`). Each function implements the specific logic for its methodology.
* **`scripts/run_experiment.py`**: The main entry point. It uses `argparse` to select an experiment and its parameters (e.g., number of shots, `K` samples, prompt variants). It initializes the model, loads the dataset, and calls the appropriate function from the `experiments` module.
* **`src/llm_math_eval/processing.py`**: Contains the utility functions `extract_answer` (which uses regex to find the last numerical value in a text) and `postprocess_final_answer` (which cleans and evaluates the extracted number).

### 2. Prompting Methodologies Explained

Each experiment tests a different strategy for eliciting mathematical reasoning from the LLM.

#### 1. Direct Prompting (Baseline)
* **Algorithm:** This is the simplest baseline. The model is given the system prompt and the user prompt (the math problem) directly. It is expected to output the final answer, sometimes with a brief explanation.
* **Purpose:** To measure the model's raw, "out-of-the-box" reasoning capability without any guiding techniques.

#### 2. Zero-Shot Chain-of-Thought (CoT)
* **Algorithm:** This method appends a simple "magic phrase" to the user's question, such as "Let's think step by step." The model is not given any examples, but this simple instruction is enough to trigger it to produce a sequential, step-by-step reasoning process before stating the final answer.
* **Reference:** This technique was famously introduced by **Kojima et al. (2022)** in the paper *"Large Language Models are Zero-Shot Reasoners"*.

#### 3. Few-Shot Chain-of-Thought (CoT)
* **Algorithm:** This method provides the model with a few complete examples of solved problems before presenting the new problem. Each example includes a question, a step-by-step reasoning chain (the CoT), and the final answer. The model learns this *format* (in-context learning) and applies it to the new question.
* **Our Test:** We test this with correct examples (**Positive Shots**) and, unconventionally, with examples that have flawed logic and incorrect answers (**Negative Shots**).
* **Reference:** The concept of in-context learning with examples was formalized by **Brown et al. (2020)** in the GPT-3 paper, *"Language Models are Few-Shot Learners"*.

#### 4. Self-Consistency
* **Algorithm:** This method aims to improve on standard CoT by replacing greedy decoding (picking the single "best" next word) with a majority vote.
    1.  Sampling is enabled (`do_sample=True`), and the model generates *K* different reasoning paths for the *same* question.
    2.  The final answer is extracted from all *K* outputs.
    3.  The most frequent (modal) answer is chosen as the final answer.
* **Reference:** This method was introduced by **Wang et al. (2022)** in *"Self-Consistency Improves Chain of Thought Reasoning in Language Models"*.

#### 5. Verbalized Confidence
* **Algorithm:** This technique attempts to have the model "self-evaluate" its answers.
    1.  The model is prompted to generate *K* different solutions (similar to Self-Consistency).
    2.  Crucially, the prompt also asks the model to provide a numerical "Confidence: X%" score *for its own answer*.
    3.  Instead of a majority vote, we select the single answer that the model rated with the *highest confidence*.
* **Reference:** This approach is based on the findings of **Kadavath et al. (2022)** in *"Language Models (Mostly) Know What They Know"*.

#### 6. Subquestion Decomposition (Least-to-Most)
* **Algorithm:** This is a multi-step process that breaks a complex problem into simpler parts.
    1.  **Decompose:** First, the model is given the main question and asked to *only* break it down into a list of simpler subquestions.
    2.  **Solve Sequentially:** The script then feeds these subquestions back to the model *one by one*, including the previous sub-answers as context for the next step.
    3.  **Final Answer:** After all subquestions are answered, the model is given the full context (original question + all sub-Q&As) and asked to provide the final answer.
* **Reference:** This method is known as "Least-to-Most Prompting," introduced by **Zhou et al. (2022)** in *"Least-to-Most Prompting Enables Complex Reasoning in Large Language Models"*.

### 3. Analysis of Results

After running the experiments, the `outputs/` directory contains a JSON file for each test, detailing every question, the model's full response, the extracted answer, and whether it was correct.

Based on an analysis of a 50-sample run, the key findings are:

* **Direct Prompting is Brittle (18% Avg. Accuracy):** Simply asking for the answer yields poor results. Forcing the model to *only* provide a number (`v2`) is even worse (14%) as it prevents any internal reasoning.
* **CoT is a Strong Baseline (39% Avg. Accuracy):** Adding "Let's think step by step" (Zero-Shot CoT) dramatically improves performance, more than doubling the accuracy over direct prompting.
* **Few-Shot is Effective (40%-52% Accuracy):**
    * **Positive Shots:** Performance peaked at 4 shots (44%), beating Zero-Shot CoT. Adding too many examples (8 shots) decreased performance (36%), likely due to exceeding the model's context or attention limits.
    * **Negative Shots:** Surprisingly, this was the **best-performing method**. Using 4 *incorrect* examples (52% accuracy) outperformed 4 *correct* examples (44%). This suggests the model is adept at learning the *problem-solving format* from the examples, even if the final answers in those examples are wrong.
* **Advanced Methods are Costly & Unstable:**
    * **Self-Consistency (29% Avg. Accuracy):** Was very time-consuming (30-40 min) and performed worse than simple CoT or Few-Shot methods. The diversity of answers was high, making a clear majority vote difficult.
    * **Verbalized Confidence (25% Avg. Accuracy):** This method performed poorly, with a major drop at K=5 (14%). The model was frequently "overconfident" (e.g., 100% confidence) in its incorrect answers.
    * **Subquestion Decomposition (34% Accuracy):** While better than direct prompting, this extremely slow (2-hour) method was overly complex and its performance was hampered by long, convoluted context chains, ultimately failing to beat simpler CoT.

---

## Project Structure

```
llm-math-reasoning-analysis/
├── .gitignore                    # Standard Python gitignore
├── LICENSE                       # MIT License
├── README.md                     # This README file
├── requirements.txt              # Project dependencies
├── data/
│   └── prompts/                  # Stores .txt files for few-shot examples
│       ├── 2_shots.txt
│       ├── 4_shots.txt
│       ├── 8_shots.txt
│       ├── negative_2_shots.txt
│       ├── negative_4_shots.txt
│       └── negative_8_shots.txt
├── logs/
│   ├── .gitkeep                  # Placeholder for log files
│   └── experiment.log            # Log file generated by the script
├── notebooks/
│   └── run_experiments.ipynb     # Notebook to run scripts and analyze results
├── outputs/
│   └── .gitkeep                  # Placeholder for JSON experiment results
├── scripts/
│   └── run_experiment.py         # Main script to run all experiments via CLI
└── src/
    └── llm_math_eval/
        ├── __init__.py           # Makes 'llm_math_eval' a Python package
        ├── dataset.py            # Logic for loading the GSM8K dataset
        ├── evaluation.py         # Functions for calculating accuracy and printing samples
        ├── experiments.py        # Core logic for all 7 prompting methods
        ├── model.py              # LLM wrapper class
        ├── processing.py         # Helper functions for answer extraction
        └── utils.py              # Utility functions for file I/O (JSON, TXT)
```

## How to Use

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/msmrexe/llm-math-reasoning-analysis.git
    cd llm-math-reasoning-analysis
    ```

2.  **Setup Environment and Log In:**
    * Install the required packages:
        ```bash
        pip install -r requirements.txt
        ```
    * You will need a Hugging Face account and token to access the Mistral model.
        ```bash
        huggingface-cli login
        ```

3.  **Run an Experiment:**
    * Use the `scripts/run_experiment.py` script to run any experiment.
    * All results will be saved to the `outputs/` directory and logs to `logs/`.

    **Example Usage:**

    ```bash
    # Run the default Zero-Shot CoT experiment (variant 1)
    python scripts/run_experiment.py --experiment zero_shot --variant 1

    # Run the 4-shot (Positive) experiment
    python scripts/run_experiment.py --experiment few_shot --shots 4

    # Run the 4-shot (Negative) experiment
    python scripts/run_experiment.py --experiment wrong_shot --shots 4

    # Run Self-Consistency with K=3 samples
    python scripts/run_experiment.py --experiment self_consistency --k_samples 3

    # Run Subquestion Decomposition (This will take a long time!)
    python scripts/run_experiment.py --experiment subquestion
    ```
    *To see all options, run:* `python scripts/run_experiment.py --help`

4.  **Analyze Results:**
    * Once experiments are complete, check the JSON files in the `outputs/` directory.
    * For a comprehensive summary and to view sample outputs, open and run the `notebooks/Run_All_Experiments.ipynb` notebook using Jupyter.

---

## Author

Feel free to connect or reach out if you have any questions!

* **Maryam Rezaee**
* **GitHub:** [@msmrexe](https://github.com/msmrexe)
* **Email:** [ms.maryamrezaee@gmail.com](mailto:ms.maryamrezaee@gmail.com)

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for full details.
