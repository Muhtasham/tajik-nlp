import json
import asyncio

from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI
from pathlib import Path
from tqdm.asyncio import tqdm
from loguru import logger

load_dotenv()

# Set up logger
logger.add(
    "logs/mmlu_translation_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Constants for rate limiting
CHUNK_SIZE = 1000
MAX_CONCURRENT = 500

async def translate_text(client, text):
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": f"Translate the following text to Tajik, keep the original format: \n {text}"}
            ],
            temperature=0.0,
            store=True
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Translation error: {e}\nText: {text[:100]}...")
        return None

async def process_chunk(client, items, start_idx, semaphore):
    # Map numeric answers to Tajik alphabet
    answer_map = {
        0: 'А',  # 0 -> А
        1: 'Б',  # 1 -> Б
        2: 'В',  # 2 -> В
        3: 'Г'   # 3 -> Г
    }
    
    tasks = []
    for i, item in enumerate(items):
        async with semaphore:
            # Translate question and all choices
            tasks.append(asyncio.create_task(translate_text(client, item['question'])))
            for choice in item['choices']:
                tasks.append(asyncio.create_task(translate_text(client, choice)))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i in range(0, len(results), 5):  # 5 because 1 question + 4 choices
        q_result = results[i] if not isinstance(results[i], Exception) else None
        c_results = [
            results[i+j] if not isinstance(results[i+j], Exception) else None
            for j in range(1, 5)
        ]
        
        item = items[i//5]
        processed_results.append({
            'id': start_idx + i//5,
            'original_question': item['question'],
            'translated_question': q_result,
            'original_choices': item['choices'],
            'translated_choices': c_results,
            'answer': item['answer'],
            'answer_tj': answer_map[item['answer']]  # Map numeric answer to Tajik letter
        })
        
        if not q_result or any(c is None for c in c_results):
            logger.warning(f"Failed translation for item {start_idx + i//5}")
    
    return processed_results

async def translate_dataset():
    logger.info("Starting translation process for MMLU")
    client = AsyncOpenAI()
    ds = load_dataset("cais/mmlu", "all")
    
    output_dir = Path("translated_data_mmlu")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    for split in ['dev', 'validation', 'test', 'auxiliary_train']:
        if split not in ds:
            continue
            
        logger.info(f"Processing {split} split")
        all_results = []
        dataset = ds[split]
        
        # Convert to list of dicts
        dataset = [{'question': item['question'], 
                   'choices': item['choices'],
                   'answer': item['answer'],
                   'subject': item['subject']} 
                  for item in dataset]
        
        # Process in chunks
        chunks = [dataset[i:i + CHUNK_SIZE] 
                 for i in range(0, len(dataset), CHUNK_SIZE)]
        
        logger.info(f"Split {split} into {len(chunks)} chunks of {CHUNK_SIZE} items")
        
        with tqdm(total=len(dataset), desc=f"Processing {split}") as pbar:
            for chunk_idx, chunk in enumerate(chunks):
                logger.debug(f"Processing chunk {chunk_idx + 1}/{len(chunks)}")
                
                results = await process_chunk(
                    client, 
                    chunk, 
                    chunk_idx * CHUNK_SIZE,
                    semaphore
                )
                
                all_results.extend(results)
                
                # Save chunk results
                output_file = output_dir / f"{split}_chunk_{chunk_idx}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                pbar.update(len(chunk))
        
        # Save complete split
        final_output = output_dir / f"{split}_translated_full.json"
        with open(final_output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Completed {split} split, saved to {final_output}")
        logger.info(f"Processed {len(all_results)} items in {split} split")

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
            'answer_tj': [item['answer_tj'] for item in data],  # Add Tajik answer
            'subject': [item['subject'] for item in data]
        }
        
        datasets[split] = Dataset.from_dict(formatted_data)
        logger.info(f"Loaded {len(data)} examples for {split} split")
    
    return DatasetDict(datasets)

async def fix_missing_translations(missing_data):
    """Fix missing translations by calling OpenAI directly"""
    logger.info("Fixing missing translations...")
    client = OpenAI()
    
    for item in missing_data:
        if item['missing'] == 'question':
            text_to_translate = item['original_question']
        else:  # missing choice
            text_to_translate = item['original_choices'][item['choice_idx']]
        
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
            file_path = Path("translated_data_mmlu") / f"{item['split']}_translated_full.json"
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Update the missing translation
            if item['missing'] == 'question':
                data[item['index']]['translated_question'] = translation
            else:
                data[item['index']]['translated_choices'][item['choice_idx']] = translation
            
            # Save back
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Fixed translation for index {item['index']} in {item['split']} split")
            
        except Exception as e:
            logger.error(f"Failed to fix translation for index {item['index']}: {e}")

async def main():
    # Step 1: Translate dataset
    await translate_dataset()
    
    # Step 2: Load and prepare combined dataset for upload
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
    
    # Collect all missing translations
    all_missing = []
    for split_name, ds in dataset_dict.items():
        for i, (q, choices) in enumerate(zip(ds['question_tj'], ds['choices_tj'])):
            if q is None:
                all_missing.append({
                    'index': i,
                    'split': split_name,
                    'missing': 'question',
                    'original_question': ds['question'][i]
                })
            for choice_idx, choice in enumerate(choices):
                if choice is None:
                    all_missing.append({
                        'index': i,
                        'split': split_name,
                        'missing': 'choice',
                        'choice_idx': choice_idx,
                        'original_choices': ds['choices'][i]
                    })
    
    if all_missing:
        logger.info(f"Found {len(all_missing)} missing translations")
        # Fix missing translations
        await fix_missing_translations(all_missing)
        
        # Reload and retry upload
        logger.info("Reloading dataset with fixed translations")
        dataset_dict = load_translated_data()
    
    # Push to hub as one dataset
    logger.info("Pushing combined dataset to Hugging Face Hub")
    dataset_dict.push_to_hub(
        "muhtasham/mmlu-tajik",
        private=False,
        commit_message="Added Tajik translations for all MMLU subjects"
    )
    
    logger.info("Successfully processed and uploaded all subjects!")

if __name__ == "__main__":
    asyncio.run(main()) 