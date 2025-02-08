from datasets import load_dataset
from loguru import logger
import random

# Setup logging
logger.add(
    "logs/test_{time}.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def load_and_verify_datasets():
    """Load both datasets and verify their contents"""
    
    datasets = {
        'main': "muhtasham/gsm8k-tajik",
        'socratic': "muhtasham/gsm8k-socratic-tajik"
    }
    
    for name, repo in datasets.items():
        logger.info(f"\nTesting {name} dataset from {repo}")
        try:
            ds = load_dataset(repo)
            
            # Print basic stats
            logger.info("\nDataset Statistics:")
            for split, data in ds.items():
                logger.info(f"{split}: {len(data)} examples")
                
                # Verify all required columns exist
                columns = data.column_names
                required_cols = ['question', 'question_tj', 'answer', 'answer_tj']
                missing_cols = [col for col in required_cols if col not in columns]
                if missing_cols:
                    logger.error(f"Missing columns in {split}: {missing_cols}")
                
                # Check for missing translations
                missing_q = sum(1 for q in data['question_tj'] if q is None)
                missing_a = sum(1 for a in data['answer_tj'] if a is None)
                if missing_q or missing_a:
                    logger.warning(f"{split} has {missing_q} missing questions and {missing_a} missing answers")
                
                # Sample and display random examples
                logger.info("\nRandom Examples:")
                for _ in range(2):
                    idx = random.randint(0, len(data)-1)
                    example = data[idx]
                    logger.info(f"\nExample {idx}:")
                    logger.info(f"Question (EN): {example['question']}")
                    logger.info(f"Question (TJ): {example['question_tj']}")
                    logger.info(f"Answer (EN): {example['answer']}")
                    logger.info(f"Answer (TJ): {example['answer_tj']}")
                    
                    # Verify #### format in answers
                    if '####' not in example['answer']:
                        logger.warning(f"Missing #### in original answer at index {idx}")
                    if '####' not in example['answer_tj']:
                        logger.warning(f"Missing #### in translated answer at index {idx}")
        
        except Exception as e:
            logger.error(f"Error loading {name} dataset: {e}")

if __name__ == "__main__":
    load_and_verify_datasets() 