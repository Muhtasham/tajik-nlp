import json
import asyncio

from datasets import Dataset, DatasetDict
from dotenv import load_dotenv
from pathlib import Path
from loguru import logger
from openai import OpenAI

load_dotenv()

# Setup logging
logger.add(
    "logs/upload_{time}.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def validate_and_fix_answers(data_dir: str = "translated_data_socratic"):
    """Check and fix answer formatting to ensure #### is present"""
    logger.info("Validating answer formats...")
    data_dir = Path(data_dir)
    fixes_made = 0
    
    for split in ['train', 'test']:
        file_path = data_dir / f"{split}_translated_full.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        modified = False
        for item in data:
            orig_answer = item['original_answer']
            trans_answer = item['translated_answer']
            
            if trans_answer:  # Skip None values
                # Extract the final number from original answer
                if '####' in orig_answer:
                    orig_number = orig_answer.split('####')[-1].strip()
                    
                    # Check if translation has #### and the number
                    if '####' not in trans_answer:
                        item['translated_answer'] = f"{trans_answer}\n#### {orig_number}"
                        modified = True
                        fixes_made += 1
                        logger.info(f"Fixed answer format for item {item['id']}")
        
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved fixes to {file_path}")
    
    return fixes_made

def load_translated_data(data_dir: str = "translated_data_socratic"):
    """Load translated data from JSON files"""
    data_dir = Path(data_dir)
    datasets = {}
    
    for split in ['train', 'test']:
        file_path = data_dir / f"{split}_translated_full.json"
        logger.info(f"Loading {split} data from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to format expected by Datasets
        formatted_data = {
            'question': [item['original_question'] for item in data],
            'question_tj': [item['translated_question'] for item in data],
            'answer': [item['original_answer'] for item in data],
            'answer_tj': [item['translated_answer'] for item in data],
        }
        
        datasets[split] = Dataset.from_dict(formatted_data)
        logger.info(f"Loaded {len(data)} examples for {split} split")
    
    return DatasetDict(datasets)

async def fix_missing_translations(missing_data):
    """Fix missing translations by calling OpenAI directly"""
    logger.info("Fixing missing translations...")
    client = OpenAI()
    
    for item in missing_data:
        text_to_translate = item['original_answer'] if item['missing'] == 'answer' else item['original_question']
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": f"Translate the following text to Tajik, keep the original format: \n {text_to_translate}"}
                ],
                temperature=0.0
            )
            translation = response.choices[0].message.content
            
            # Update the JSON file
            file_path = Path("translated_data_socratic") / f"{item['split']}_translated_full.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update the missing translation
            if item['missing'] == 'answer':
                data[item['index']]['translated_answer'] = translation
            else:
                data[item['index']]['translated_question'] = translation
            
            # Save back
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Fixed translation for index {item['index']} in {item['split']} split")
            
        except Exception as e:
            logger.error(f"Failed to fix translation for index {item['index']}: {e}")

def main():
    # Login to Hugging Face
    logger.info("Logging in to Hugging Face")
    
    # Validate and fix answer formats before loading dataset
    fixes = validate_and_fix_answers()
    if fixes:
        logger.info(f"Made {fixes} format fixes to answers")
    
    # Load and convert data
    logger.info("Loading translated data")
    dataset_dict = load_translated_data("translated_data_socratic")
    
    # Add dataset metadata for each split
    metadata = {
        "description": """
        GSM8K (Grade School Math 8K) Socratic translated to Tajik.
        Original dataset: https://huggingface.co/datasets/openai/gsm8k
        
        Contains both original English and Tajik translations for questions and answers.
        """,
        "license": "MIT",
        "homepage": "https://github.com/muhtasham/tajik-nlp",
        "citation": "@misc{tajik-gsm8k,\n  author = {Muhtasham Oblokulov},\n  title = {GSM8K Math Dataset Translated to Tajik},\n  year = {2024}\n}"
    }
    
    # Set metadata for each split
    for split in dataset_dict.values():
        split.info.description = metadata["description"]
        split.info.license = metadata["license"]
        split.info.homepage = metadata["homepage"]
        split.info.citation = metadata["citation"]
    
    # Collect all missing translations with split info
    all_missing = []
    for split_name, ds in dataset_dict.items():
        for i, (q, a) in enumerate(zip(ds['question_tj'], ds['answer_tj'])):
            if q is None or a is None:
                all_missing.append({
                    'index': i,
                    'split': split_name,
                    'missing': 'question' if q is None else 'answer',
                    'original_question': ds['question'][i],
                    'original_answer': ds['answer'][i]
                })
    
    if all_missing:
        logger.info(f"Found {len(all_missing)} missing translations")
        # Fix missing translations
        asyncio.run(fix_missing_translations(all_missing))
        
        # Reload and retry upload
        logger.info("Reloading dataset with fixed translations")
        dataset_dict = load_translated_data()
        
        # Push to hub again
        logger.info("Pushing updated dataset to Hugging Face Hub")
        dataset_dict.push_to_hub(
            "muhtasham/gsm8k-socratic-tajik",
            private=False,
            commit_message="Fixed missing translations"
        )
    
    logger.info("Dataset uploaded successfully!")
    
    # Print some stats
    logger.info("\nDataset Statistics:")
    for split_name, ds in dataset_dict.items():
        logger.info(f"{split_name}: {len(ds)} examples")
        
        # Check for any missing translations and show the problematic examples
        missing_translations = []
        for i, (q, a) in enumerate(zip(ds['question_tj'], ds['answer_tj'])):
            if q is None or a is None:
                missing_translations.append({
                    'index': i,
                    'missing': 'question' if q is None else 'answer',
                    'original_question': ds['question'][i],
                    'original_answer': ds['answer'][i]
                })
        
        if missing_translations:
            logger.warning(f"\nFound {len(missing_translations)} missing translations in {split_name}:")
            for item in missing_translations:
                logger.warning(f"""
                Index: {item['index']}
                Missing: {item['missing']}
                Original Question: {item['original_question'][:100]}...
                Original Answer: {item['original_answer'][:100]}...
                """)

    # Optional: Save problematic examples to a file for later fixing
    if any(missing_translations):
        with open('missing_translations.json', 'w', encoding='utf-8') as f:
            json.dump(missing_translations, f, ensure_ascii=False, indent=2)
        logger.info("Saved missing translations to missing_translations.json")

if __name__ == "__main__":
    main()

