# /// script
# dependencies = [
#     "datasets>=3.2.0",
#     "diskcache>=5.6.3",
#     "huggingface-hub>=0.28.1",
#     "litellm>=1.60.8",
#     "openinference-instrumentation-litellm",
#     "arize-phoenix-otel",
# ]
# ///

"""
MMLU Dataset Translation Script

This script loads a dataset from the MMLU evaluation suite, translates each question-choices
pair into Persian and Tajik using a language model, and pushes the resulting dataset to the
Hugging Face Hub.

Dependencies:
    - datasets>=3.2.0
    - diskcache>=5.6.3
    - fastparquet>=2024.11.0
    - huggingface-hub>=0.28.1
    - litellm>=1.60.8
    - openinference-instrumentation-litellm
    - arize-phoenix-otel
"""

import os
import random
import logging
from typing import Optional, List

from datasets import load_dataset, DatasetDict, Dataset
import litellm
from litellm import batch_completion
from litellm.caching.caching import Cache
from tqdm import tqdm

# Set a fixed random seed for reproducibility
random.seed(42)

# ------------------------------
# Setup: Caching and Logging
# ------------------------------

# Initialize disk-based caching to avoid redundant API calls.
Cache(type="disk")

# Configure logging to track the progress and issues during translation.
logging.basicConfig(
    filename="translation_progress.log",
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

# Reduce verbosity of logs from the LiteLLM package.
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)

# Disable logging of raw requests and responses from LiteLLM.
litellm.log_raw_request_response = False

# ------------------------------
# Constants and Environment Variables
# ------------------------------

BATCH_SIZE = 20  # Batch size for processing dataset items.
# Check if LLM_TRACING is enabled via environment variable.
LLM_TRACING = os.getenv("LLM_TRACING", "false").lower() == "true"
# Ensure OPENROUTER_API_KEY environment variable is set
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY environment variable is not set"


# ------------------------------
# Optional: LLM Tracing Setup
# ------------------------------

if LLM_TRACING:
    # Requires arize-phoenix-otel and openinference-instrumentation-litellm packages.
    logger.info("LLM_TRACING is enabled")
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from phoenix.otel import register

    # Get the tracing endpoint from environment or use a default.
    LLM_TRACING_ENDPOINT = os.getenv("LLM_TRACING_ENDPOINT", "http://localhost:6060/v1/traces")

    # Register and initialize the tracer provider.
    tracer_provider = register(project_name="mmlu", endpoint=LLM_TRACING_ENDPOINT, batch=True)
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

# ------------------------------
# Helper Functions
# ------------------------------


def create_translation_prompt(subject: str, question: str, choices: str) -> str:
    """
    Create a translation prompt for converting a question-choices pair
    into Persian and Tajik.

    Parameters:
        subject (str): The subject of the question (optional).
        question (str): The original question text.
        choices (str): The formatted choices text.

    Returns:
        str: A formatted prompt to be used by the language model.
    """
    subject_details = f" FYI, the subject is {subject}." if subject else ""
    prompt = f"""
Translate this question-choices pair into Persian and Tajik.{subject_details}

<question>
{question}
</question>
<choices>
{choices}
</choices>

Follow these rules when translating choices (for Tajik):
  - True -> Дуруст
  - False -> Нодуруст
  - Wrong -> Нодуруст
  - Not Wrong -> Дуруст
  - Correct -> Дуруст
  - Incorrect -> Нодуруст

Output in the same original format. Do not add any additional text. Do not add explanations or commentaries.

Output format:
<persian>
<question>
</question>
<choices>
</choices>
</persian>

<tajik>
<question>
</question>
<choices>
</choices>
</tajik>
""".strip()
    return prompt


def parse_choices(choices_block: str) -> List[str]:
    """
    Parse the translated choices block using a regular expression.

    The choices block is expected to have the format:
        [0] Choice text 1
        [1] Choice text 2
        ...

    Parameters:
        choices_block (str): The block of text containing the choices.

    Returns:
        List[str]: A list of parsed choice strings.
    """
    import re

    pattern = r"^\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)"
    return [match.group(2).strip() for match in re.finditer(pattern, choices_block, re.MULTILINE | re.DOTALL)]


def parse_translation_output(translated_text: str) -> (str, List[str]):
    """
    Parse the translated text to extract the Tajik question and choices.

    Assumes the output format includes <tajik> section with nested <question> and <choices> tags.

    Parameters:
        translated_text (str): The full response text from the language model.

    Returns:
        tuple: (translated_question, translated_choices)
    """
    tajik_block = translated_text.split("<tajik>")[1].split("</tajik>")[0].strip()
    question_tj = tajik_block.split("<question>")[1].split("</question>")[0].strip()
    choices_block_tj = tajik_block.split("<choices>")[1].split("</choices>")[0].strip()
    choices_tj = parse_choices(choices_block_tj)
    return question_tj, choices_tj


def build_messages(dataset: Dataset) -> List[list]:
    """
    Build a list of message prompts from the dataset.

    Parameters:
        dataset (Dataset): The dataset containing items to be translated.

    Returns:
        List[list]: A list where each element is a message list to be used for batch_completion.
    """
    messages = []
    for item in dataset:
        # Format the choices with indices.
        choices_formatted = "\n".join([f"[{i}] {choice}" for i, choice in enumerate(item["choices"], start=1)])
        # Create the translation prompt using the helper function.
        prompt = create_translation_prompt(item["subject"], item["question"], choices_formatted)
        messages.append([{"role": "user", "content": prompt}])
    return messages


def process_response(response, original_item, global_index) -> dict:
    """
    Process an individual response from the language model.

    Parameters:
        response: The response object returned by batch_completion.
        original_item: The original dataset item.
        global_index: The index of the item in the dataset.

    Returns:
        dict: A dictionary with keys "question_tj" and "choices_tj".
    """
    translated_text = response.choices[0].message.content
    try:
        question_tj, choices_tj = parse_translation_output(translated_text)
    except IndexError as parse_error:
        logger.warning(f"Parsing error for item {global_index}: {parse_error}. Response text: {translated_text}")
        question_tj, choices_tj = None, None

    # Validate the number of choices.
    if choices_tj is not None and len(choices_tj) != len(original_item["choices"]):
        logger.warning(
            f"Choice count mismatch for item {global_index}: expected {len(original_item['choices'])}, got {len(choices_tj)}"
        )

    # Log translation summary.
    logger.info(
        f"Item {global_index} translated successfully:\n"
        f"Original Question: {original_item['question']}\n"
        f"Original Choices: {original_item['choices']}\n"
        f"Translated Question: {question_tj}\n"
        f"Translated Choices: {choices_tj}"
    )
    return {"question_tj": question_tj, "choices_tj": choices_tj}


def translate_dataset(dataset: Dataset) -> List[dict]:
    """
    Translate the dataset by processing items in batches. For each item, a translation
    prompt is created and sent to the language model. The response is then parsed,
    validated, and logged.

    Parameters:
        dataset (Dataset): The dataset containing the items to be translated.

    Returns:
        List[dict]: A list of dictionaries with keys "question_tj" and "choices_tj"
                    containing the translated question and choices.
    """
    messages = build_messages(dataset)
    translated_data = []
    num_batches = (len(messages) + BATCH_SIZE - 1) // BATCH_SIZE
    logger.info(f"Starting translation in {num_batches} batches; total items: {len(messages)}")

    # Process messages in batches with a progress bar.
    for batch_index in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_index * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(messages))
        batch_messages = messages[start_idx:end_idx]
        logger.info(f"Processing batch {batch_index + 1}/{num_batches} (items {start_idx} to {end_idx - 1})")

        # Set max_tokens heuristically based on the longest message in the batch.
        max_tokens = max(len(msg[0]["content"]) * 4 for msg in batch_messages)

        try:
            responses = batch_completion(
                model="openrouter/google/gemini-2.0-flash-001",
                messages=batch_messages,
                temperature=0,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.warning(f"Batch {batch_index + 1} failed with error: {e}")
            # Append placeholders for failed items.
            for _ in range(start_idx, end_idx):
                translated_data.append({"question_tj": None, "choices_tj": None})
            continue

        if len(responses) != len(batch_messages):
            logger.warning(
                f"Batch {batch_index + 1}: Expected {len(batch_messages)} responses but got {len(responses)}"
            )

        # Process each response in the batch.
        for i, response in enumerate(responses):
            global_index = start_idx + i
            original_item = dataset[global_index]
            result = process_response(response, original_item, global_index)
            translated_data.append(result)

    return translated_data


def estimate_tokens(dataset_dict: DatasetDict) -> None:
    """
    Estimate the number of tokens required for each dataset split based on the
    length of questions and choices. This is used for logging and monitoring.

    Parameters:
        dataset_dict (DatasetDict): A dictionary containing dataset splits.
    """
    logger.info("Dataset splits and token estimation:")
    for split, ds in dataset_dict.items():
        input_chars = sum(len(item["question"] + "\n".join(item["choices"])) for item in ds)
        approx_input_tokens = input_chars * 4  # Rough estimate: 4 tokens per character
        approx_output_tokens = approx_input_tokens * 3
        logger.info(
            f"{split:16s}: {input_chars:14,} chars, {approx_input_tokens:14,} tokens, {approx_output_tokens:14,} output tokens"
        )


# ------------------------------
# Main Function
# ------------------------------


def main(limit: Optional[int] = None, splits: Optional[List[str]] = None) -> None:
    """
    Main entry point for the translation script.

    Parameters:
        limit (Optional[int]): Limit the number of samples per split (default: None).
        splits (Optional[List[str]]): List of dataset splits to process.
            Defaults to ["test", "validation", "dev"].
    """
    if splits is None:
        splits = ["test", "validation", "dev"]

    logger.info("Loading MMLU dataset...")
    dataset_dict = DatasetDict()

    # Load each specified split of the dataset.
    for split in splits:
        logger.info(f"Loading {split} split...")
        dataset_dict[split] = load_dataset("cais/mmlu", "all", split=split)

    # Estimate tokens for the loaded dataset splits.
    estimate_tokens(dataset_dict)

    # If a limit is set, randomly shuffle and select a subset of samples.
    if limit is not None:
        logger.info(f"Limiting dataset to {limit} samples per split")
        for split in dataset_dict:
            dataset_dict[split] = (
                dataset_dict[split].shuffle(seed=42).select(range(min(limit, len(dataset_dict[split]))))
            )

    # Process translation for each split.
    logger.info("Starting translation process...")
    for split in dataset_dict:
        logger.info(f"Translating {split} split...")
        translated_data = translate_dataset(dataset_dict[split])
        # Add the translated question and choices as new columns.
        dataset_dict[split] = dataset_dict[split].add_column(
            "question_tj", [item["question_tj"] for item in translated_data]
        )
        dataset_dict[split] = dataset_dict[split].add_column(
            "choices_tj", [item["choices_tj"] for item in translated_data]
        )

    logger.info("Creating new dataset with translated columns...")

    # --- Push to Hugging Face Hub ---
    logger.info("Pushing translated dataset to Hugging Face Hub...")
    try:
        dataset_dict.push_to_hub(
            "tajik-evals/MMLU-dev",
            commit_message="Translate to Tajik.",
        )
        logger.info("Dataset successfully pushed to Hugging Face Hub.")
    except Exception as e:
        logger.error(f"Failed to push dataset to hub: {e}")


# ------------------------------
# Script Execution: Argument Parsing
# ------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MMLU dataset translation script")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of samples per split (default: None)")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["test", "validation", "dev", "auxiliary_train"],
        default=["test", "validation", "dev"],
        help="Dataset splits to process (default: test validation dev)",
    )
    args = parser.parse_args()
    main(limit=args.limit, splits=args.splits)
