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
pair into Tajik using a language model, and merges the translations with an existing dataset
if one exists. The resulting dataset is then pushed to the Hugging Face Hub.
"""

import copy
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
import litellm
from litellm import batch_completion
from litellm.caching.caching import Cache
from tqdm import tqdm

# ------------------------------
# Setup: Caching and Logging
# ------------------------------

# Set a fixed random seed for reproducibility.
random.seed(42)

cache = Cache(type="disk")


# this function takes in *args, **kwargs and returns the key you want to use for caching
def custom_get_cache_key(*args, **kwargs):
    # return key to use for your cache:
    key = kwargs.get("model", "") + str(kwargs.get("messages", "")) + str(kwargs.get("temperature", ""))
    return key


cache.get_cache_key = custom_get_cache_key  # set get_cache_key function for your cache

# Initialize disk-based caching to avoid redundant API calls.
litellm.cache = cache

# Configure logging.
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/mmlu_tajik_translation_progress.log",
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
    level=logging.INFO,
    encoding="utf-8",
)
logger = logging.getLogger(__name__)
logging.getLogger("LiteLLM").setLevel(logging.INFO)

# ------------------------------
# Constants and Environment Variables
# ------------------------------

SOURCE_MMLU_REPO_ID = "cais/mmlu"
TARGET_REPO_ID = "tajik-evals/MMLU"
BATCH_SIZE = 500  # Number of items to process per batch.
LLM_TRACING = os.getenv("LLM_TRACING", "false").lower() == "true"
assert os.environ.get("OPENROUTER_API_KEY"), "OPENROUTER_API_KEY environment variable is not set"

# ------------------------------
# Patch LiteLLM for gemini-2.0-flash model
# ------------------------------
litellm.log_raw_request_response = False
litellm.disable_end_user_cost_tracking = True
litellm.model_cost.update(
    {
        "openrouter/google/gemini-2.0-flash-001": {
            "max_tokens": 8192,
            "max_input_tokens": 2000000,
            "max_output_tokens": 8192,
            "input_cost_per_token": 0.10 / 1000000,
            "output_cost_per_token": 0.40 / 1000000,
            "litellm_provider": "openrouter",
            "mode": "chat",
            "supports_function_calling": True,
            "supports_vision": True,
            "supports_tool_choice": True,
        },
    }
)
litellm.model_list.append("google/gemini-2.0-flash-001")
litellm.openrouter_models.append("google/gemini-2.0-flash-001")

# ------------------------------
# Optional: LLM Tracing Setup
# ------------------------------

if LLM_TRACING:
    logger.info("LLM_TRACING is enabled")
    from openinference.instrumentation.litellm import LiteLLMInstrumentor
    from phoenix.otel import register

    LLM_TRACING_ENDPOINT = os.getenv("LLM_TRACING_ENDPOINT", "http://localhost:6060/v1/traces")
    tracer_provider = register(project_name="mmlu", endpoint=LLM_TRACING_ENDPOINT, batch=True)
    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)

# ------------------------------
# Statistics Collector
# ------------------------------


class TranslationStats:
    def __init__(self):
        self.total_items = 0
        self.successes = 0
        self.failures = 0
        self.empty_responses = 0
        self.parse_errors = 0
        self.choice_mismatches = 0
        self.total_batches = 0
        self.failed_batches = 0
        # Merge statistics:
        self.replaced = 0
        self.appended = 0
        self.kept = 0

    def log(self) -> None:
        logger.info("======== Translation Statistics ========")
        logger.info(f"Total items processed: {self.total_items}")
        logger.info(f"Successful translations: {self.successes}")
        logger.info(f"Failed translations: {self.failures}")
        logger.info(f"Empty responses: {self.empty_responses}")
        logger.info(f"Parse errors: {self.parse_errors}")
        logger.info(f"Choice mismatches: {self.choice_mismatches}")
        logger.info(f"Total batches processed: {self.total_batches}")
        logger.info(f"Failed batches: {self.failed_batches}")
        logger.info("======== Merge Statistics =========")
        logger.info(f"Replaced examples: {self.replaced}")
        logger.info(f"Appended examples: {self.appended}")
        logger.info(f"Kept unchanged examples: {self.kept}")
        logger.info("=====================================")


# ------------------------------
# Hash Function and Merge Helper
# ------------------------------


def hash_example(example: dict) -> str:
    """
    Compute a hash key for an example using its original columns.
    """
    question_hash = str(example["question"])
    subject_hash = str(example["subject"])
    choices_hash = str(example["choices"])
    answer_hash = str(example["answer"])
    return "<<>>".join([question_hash, subject_hash, choices_hash, answer_hash])


def merge_datasets(new_ds: Dataset, target_ds: Optional[Dataset], stats: TranslationStats) -> Dataset:
    """
    Merge the new (translated) dataset with the target dataset.
    - Replace examples in the target dataset if a matching hash is found and the new translation is valid.
    - Append new examples (with valid translations) that are not present in the target dataset.
    - Retain target examples when new translation is missing.
    """
    # If target dataset is empty or None, return the new dataset
    if target_ds is None or len(target_ds) == 0:
        return new_ds

    target_ds = copy.deepcopy(target_ds)
    target_dict = {hash_example(ex): i for i, ex in enumerate(target_ds)}

    to_replace = []
    to_append = []

    for ex in tqdm(new_ds):
        key = hash_example(ex)
        new_valid = ex.get("question_tj") is not None and ex.get("choices_tj") is not None

        if key in target_dict:
            if new_valid:
                to_replace.append((target_dict[key], ex))
                stats.replaced += 1
            else:
                stats.kept += 1
        elif new_valid:
            to_append.append(ex)
            stats.appended += 1

    # Batch replace
    for idx, ex in to_replace:
        target_ds[idx].update(ex)

    # Batch append
    if to_append:
        append_ds = Dataset.from_list(
            to_append, features=target_ds.features, info=target_ds.info, split=target_ds.split
        )
        target_ds = concatenate_datasets([target_ds, append_ds], info=target_ds.info, split=target_ds.split)

    return target_ds


# ------------------------------
# Helper Functions
# ------------------------------


def create_translation_prompt(subject: str, question: str, choices: str) -> str:
    """
    Build a prompt for translating a question and its choices.
    """
    subject = subject.replace("_", " ").strip() if subject else ""
    subject_details = f"FYI, the subject is {subject}." if subject else ""
    prompt = f"""
Тарҷумаи ин савол ва вариантҳои ҷавобро ба забонҳои зерин анҷом диҳед.
(Translate the following question and answer choices into specified languages)

<question>
{question}
</question>
<choices>
{choices}
</choices>

{subject_details}.

Follow these rules when translating choices (for Tajik):
  - True -> Дуруст
  - False -> Нодуруст
  - Wrong -> Нодуруст
  - Not Wrong -> Дуруст
  - Correct -> Дуруст
  - Incorrect -> Нодуруст

Do not add any extra text to questions or choices. Avoid explanations, comments, or anything in parentheses.

Output format (avoid JSON) (Fill in ...'s):
<persian>
<question>
...
</question>
<choices>
...
</choices>
</persian>

Note: For uncertain Tajik terms, prefer Persian transliteration for guidance. Use established Tajik terms when confident.

<tajik>
<question>
...
</question>
<choices>
...
</choices>
</tajik>
    """.strip()
    return prompt


def parse_choices(choices_block: str) -> List[str]:
    """
    Parse a block of choices into a list of choice strings.
    Expected format:
        [0] Choice text 1
        [1] Choice text 2
        ...
    """
    pattern = r"^\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)"
    return [match.group(2).strip() for match in re.finditer(pattern, choices_block, re.MULTILINE | re.DOTALL)]


def parse_translation_output(translated_text: str) -> Tuple[str, List[str]]:
    """
    Extract the Tajik question and choices from the model output.
    """
    try:
        tajik_block = translated_text.split("<tajik>")[1].split("</tajik>")[0].strip()
        question_tj = tajik_block.split("<question>")[1].split("</question>")[0].strip()
        choices_block = tajik_block.split("<choices>")[1].split("</choices>")[0].strip()
        choices_tj = parse_choices(choices_block)
        return question_tj, choices_tj
    except (IndexError, ValueError) as e:
        raise ValueError(f"Error parsing translated text: {e}")


def build_messages(dataset: Dataset) -> List[List[Dict[str, str]]]:
    """
    Convert dataset items into a list of messages suitable for batch_completion.
    """
    messages = []
    for item in dataset:
        # Format choices with indices.
        choices_formatted = "\n".join(f"[{i}] {choice}" for i, choice in enumerate(item["choices"], start=1))
        prompt = create_translation_prompt(item.get("subject", ""), item["question"], choices_formatted)
        messages.append([{"role": "user", "content": prompt}])
    return messages


def append_failure_placeholders(translated_data: List[dict], count: int, stats: TranslationStats) -> None:
    """
    Append placeholder entries for a failed batch or response.
    """
    for _ in range(count):
        translated_data.append({"question_tj": None, "choices_tj": None})
        stats.total_items += 1
        stats.failures += 1


def process_response(response: Any, original_item: dict, global_index: int, stats: TranslationStats) -> dict:
    """
    Process an individual translation response and update statistics.
    """
    stats.total_items += 1

    if isinstance(response, Exception):
        logger.warning(f"Error processing item {global_index}: {response}")
        stats.failures += 1
        return {"question_tj": None, "choices_tj": None}

    translated_text = response.choices[0].message.content
    if not translated_text:
        logger.warning(f"Empty response for item {global_index}.")
        stats.empty_responses += 1
        stats.failures += 1
        return {"question_tj": None, "choices_tj": None}

    try:
        question_tj, choices_tj = parse_translation_output(translated_text)
    except Exception as e:
        logger.warning(f"Parsing error for item {global_index}: {e}. Response text: {translated_text}", exc_info=True)
        stats.parse_errors += 1
        stats.failures += 1
        return {"question_tj": None, "choices_tj": None}

    if not question_tj or not choices_tj:
        logger.warning(f"Empty translation for item {global_index}. Translated text:{translated_text}")
        stats.failures += 1
        return {"question_tj": None, "choices_tj": None}

    # Validate that the number of choices matches the original.
    if len(choices_tj) != len(original_item["choices"]):
        logger.warning(
            f"Choice count mismatch for item {global_index}: expected {len(original_item['choices'])}, got {len(choices_tj)}"
        )
        stats.choice_mismatches += 1
        stats.failures += 1
        return {"question_tj": None, "choices_tj": None}

    # If we get here, the translation is considered successful.
    stats.successes += 1
    logger.info(
        f"Item {global_index} translated successfully:\n"
        f"Original Question: {original_item['question']}\n"
        f"Original Choices: {original_item['choices']}\n"
        f"Translated Question: {question_tj}\n"
        f"Translated Choices: {choices_tj}"
    )
    return {"question_tj": question_tj, "choices_tj": choices_tj}


def translate_dataset(dataset: Dataset, stats: TranslationStats) -> List[dict]:
    """
    Translate dataset items in batches and update statistics.
    """
    messages = build_messages(dataset)
    translated_data = []
    num_batches = (len(messages) + BATCH_SIZE - 1) // BATCH_SIZE
    stats.total_batches += num_batches
    logger.info(f"Starting translation in {num_batches} batches; total items: {len(messages)}")

    for batch_index in tqdm(range(num_batches), desc="Processing batches"):
        start_idx = batch_index * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(messages))
        batch_messages = messages[start_idx:end_idx]
        logger.info(f"Processing batch {batch_index + 1}/{num_batches} (items {start_idx} to {end_idx - 1})")

        # Estimate max_tokens (heuristic: 10 tokens per character of the longest prompt).
        max_tokens = max(len(msg[0]["content"]) for msg in batch_messages) * 10

        # log messages
        logger.info(f"Batch {batch_index + 1} messages:\n{batch_messages}")

        try:
            responses = batch_completion(
                model="openrouter/google/gemini-2.0-flash-001",
                messages=batch_messages,
                temperature=0,
                max_tokens=max_tokens,
                max_workers=BATCH_SIZE,
                request_timeout=300,
            )
        except Exception as e:
            logger.warning(f"Batch {batch_index + 1} failed with error: {e}")
            stats.failed_batches += 1
            append_failure_placeholders(translated_data, end_idx - start_idx, stats)
            continue

        if len(responses) != len(batch_messages):
            logger.warning(
                f"Batch {batch_index + 1}: Expected {len(batch_messages)} responses but got {len(responses)}"
            )
            stats.failed_batches += 1
            append_failure_placeholders(translated_data, end_idx - start_idx, stats)
            continue

        # Process each response in the batch.
        for i, response in enumerate(responses):
            global_index = start_idx + i
            original_item = dataset[global_index]
            result = process_response(response, original_item, global_index, stats)
            translated_data.append(result)

    return translated_data


def estimate_tokens(dataset_dict: DatasetDict) -> None:
    """
    Estimate and log token counts, item counts, and associated costs for each dataset split.
    """
    logger.info("Dataset splits, item counts, and token estimation:")
    for split, ds in dataset_dict.items():
        item_count = len(ds)
        input_chars = sum(len(item["question"] + "\n".join(item["choices"])) for item in ds)
        approx_input_tokens = input_chars / 4  # Heuristic: 4 characters per token.
        approx_output_tokens = approx_input_tokens * 3
        approx_input_cost = approx_input_tokens * 0.10 / 1_000_000  # $0.10 per million tokens.
        approx_output_cost = approx_output_tokens * 0.40 / 1_000_000  # $0.40 per million tokens.
        estimated_cost = approx_input_cost + approx_output_cost

        logger.info(
            f"{split:16s}: {item_count:8,} items, "
            f"{input_chars:12,} chars, "
            f"{approx_input_tokens:12,.0f} tokens, "
            f"{approx_output_tokens:12,.0f} output tokens, "
            f"{estimated_cost:5.2f} USD (Gemini 2.0 Flash)"
        )


# ------------------------------
# Main Function
# ------------------------------


def main(limit: Optional[int] = None, splits: Optional[List[str]] = None) -> None:
    """
    Main entry point for translation.
    """
    if splits is None:
        splits = ["test", "validation", "dev"]

    logger.info("Loading MMLU dataset...")
    dataset_dict = DatasetDict()

    for split in splits:
        logger.info(f"Loading {split} split...")
        dataset_dict[split] = load_dataset(SOURCE_MMLU_REPO_ID, "all", split=split)

    estimate_tokens(dataset_dict)

    if limit is not None:
        logger.info(f"Limiting dataset to {limit} samples per split")
        for split in dataset_dict:
            ds = dataset_dict[split]
            dataset_dict[split] = ds.shuffle(seed=42).select(range(min(limit, len(ds))))

    # Instantiate global statistics.
    global_stats = TranslationStats()

    logger.info("Starting translation process...")
    for split in dataset_dict:
        logger.info(f"Translating {split} split...")
        translated_data = translate_dataset(dataset_dict[split], global_stats)
        # Add new translated columns.
        dataset_dict[split] = dataset_dict[split].add_column(
            "question_tj", [item["question_tj"] for item in translated_data]
        )
        dataset_dict[split] = dataset_dict[split].add_column(
            "choices_tj", [item["choices_tj"] for item in translated_data]
        )

    logger.info("Creating new dataset with translated columns...")

    # ------------------------------
    # Merge with Target Repository Dataset
    # ------------------------------
    # Load full target repository
    try:
        logger.info(f"Attempting to load existing target dataset from {TARGET_REPO_ID} ...")
        target_dataset_dict = load_dataset(TARGET_REPO_ID)
        logger.info(f"Loaded existing target dataset.")
    except Exception as e:
        logger.info(f"Target dataset not found. Treating as empty. Error: {e}")
        target_dataset_dict = dataset_dict.copy()
        target_dataset_dict.clear()

    merged_dataset_dict: DatasetDict = DatasetDict()
    all_splits = set(dataset_dict.keys()) | set(target_dataset_dict.keys())

    for split in all_splits:
        logger.info(f"Merging {split} split...")
        new_ds = dataset_dict.get(split, Dataset.from_dict({}))
        target_ds = target_dataset_dict.get(split, None)

        # Merge new data with target data
        merged_ds = merge_datasets(new_ds, target_ds, global_stats)
        # Sort the merged dataset by subject
        merged_ds = merged_ds.sort("subject")
        merged_dataset_dict[split] = merged_ds

    # ------------------------------
    # Push to Hugging Face Hub
    # ------------------------------
    commit_message = (
        f"Merge translation results. Splits: {splits}, Total items processed: {global_stats.total_items}. "
        f"Replaced: {global_stats.replaced}, Appended: {global_stats.appended}, Kept unchanged: {global_stats.kept}. "
        f"Successes: {global_stats.successes}, Failures: {global_stats.failures}, "
        f"Empty responses: {global_stats.empty_responses}, Parse errors: {global_stats.parse_errors}, "
        f"Choice mismatches: {global_stats.choice_mismatches}, Total batches: {global_stats.total_batches}, "
        f"Failed batches: {global_stats.failed_batches}"
    )

    logger.info("Pushing merged dataset to Hugging Face Hub...")
    try:
        merged_dataset_dict.push_to_hub(
            TARGET_REPO_ID,
            commit_message=commit_message,
            commit_description=commit_message,
        )
        logger.info("Merged dataset successfully pushed to Hugging Face Hub with merge and translation statistics.")
    except Exception as e:
        logger.error(f"Failed to push merged dataset to hub: {e}")

    # Log aggregated statistics.
    global_stats.log()


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
        help="Dataset splits to process (default: test validation dev). Example: --splits test validation",
    )
    args = parser.parse_args()
    main(limit=args.limit, splits=args.splits)
