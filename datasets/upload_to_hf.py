from datasets import Dataset, DatasetDict
from pathlib import Path
import json
from loguru import logger

logger.add(
    "logs/upload_{time}.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def load_translated_data(data_dir="translated_data_mmlu"):
    """Load translated data from JSON files"""
    data_dir = Path(data_dir)
    datasets = {}
    
    for split in ['dev', 'validation', 'test', 'auxiliary_train']:
        file_path = data_dir / f"{split}_translated_full.json"
        if not file_path.exists():
            continue
            
        logger.info(f"Loading {split} data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to format expected by Datasets
        formatted_data = {
            'question': [item['original_question'] for item in data],
            'question_tj': [item['translated_question'] for item in data],
            'choices': [item['original_choices'] for item in data],
            'choices_tj': [item['translated_choices'] for item in data],
            'answer': [item['answer'] for item in data],
            'answer_tj': [item['answer_tj'] for item in data],
            'subject': [item['subject'] for item in data]
        }
        
        datasets[split] = Dataset.from_dict(formatted_data)
        logger.info(f"Loaded {len(data)} examples for {split} split")
    
    return DatasetDict(datasets)

def main():
    # Load and prepare combined dataset for upload
    dataset_dict = load_translated_data()
    
    # Add dataset metadata
    metadata = {
        "description": """
        MMLU (Massive Multitask Language Understanding) translated to Tajik.
        Original dataset: https://huggingface.co/datasets/cais/mmlu
        
        Contains both original English and Tajik translations for questions and choices.
        Includes all 57 subjects from the original MMLU dataset.
        
        Splits:
        - dev: Development set
        - validation: Validation set
        - test: Test set
        - auxiliary_train: Additional training data from ARC, MC_TEST, OBQA, RACE, etc.
        """,
        "license": "MIT",
        "homepage": "https://github.com/muhtasham/tajik-nlp",
        "citation": "@misc{tajik-mmlu,\n  author = {Muhtasham Oblokulov},\n  title = {MMLU Dataset Translated to Tajik},\n  year = {2024}\n}"
    }
    
    # Set metadata for each split
    for split in dataset_dict.values():
        split.info.description = metadata["description"]
        split.info.license = metadata["license"]
        split.info.homepage = metadata["homepage"]
        split.info.citation = metadata["citation"]
    
    # Push to hub
    logger.info("Pushing dataset to Hugging Face Hub")
    dataset_dict.push_to_hub(
        "muhtasham/mmlu-tajik",
        private=False,
        commit_message="Added Tajik translations for MMLU"
    )
    
    logger.info("Successfully uploaded dataset!")

if __name__ == "__main__":
    main() 