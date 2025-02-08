import asyncio
import json

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
from tqdm.asyncio import tqdm
from loguru import logger

load_dotenv()

# Set up logger
logger.add(
    "logs/translation_{time}.log",
    rotation="100 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Constants for rate limiting - optimized for 10K RPM
CHUNK_SIZE = 1000  # Process in chunks of 1000 items
MAX_CONCURRENT = 500  # Allow up to 500 concurrent requests

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
    tasks = []
    for i, item in enumerate(items):
        async with semaphore:
            tasks.append(asyncio.create_task(translate_text(client, item['question'])))
            tasks.append(asyncio.create_task(translate_text(client, item['answer'])))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    processed_results = []
    for i in range(0, len(results), 2):
        q_result = results[i] if not isinstance(results[i], Exception) else None
        a_result = results[i+1] if not isinstance(results[i+1], Exception) else None
        
        item = items[i//2]
        processed_results.append({
            'id': start_idx + i//2,
            'original_question': item['question'],
            'translated_question': q_result,
            'original_answer': item['answer'],
            'translated_answer': a_result
        })
        
        if not q_result or not a_result:
            logger.warning(f"Failed translation for item {start_idx + i//2}")
    
    return processed_results

async def main():
    logger.info("Starting translation process")
    client = AsyncOpenAI()
    ds = load_dataset("openai/gsm8k", "socratic")
    
    output_dir = Path("translated_data_socratic")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    
    for split in ['train', 'test']:
        logger.info(f"Processing {split} split")
        all_results = []
        dataset = ds[split]
        
        # Convert to list of dicts
        dataset = [{'question': item['question'], 'answer': item['answer']} 
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
    
    logger.info("Translation process completed")

if __name__ == "__main__":
    asyncio.run(main())