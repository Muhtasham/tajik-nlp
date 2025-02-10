import json
import re
import asyncio
from tqdm import tqdm

from loguru import logger
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

logger.add(
    "logs/fix_translations_{time}.log",
    rotation="100 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def is_numeric_choice(text):
    """Check if the choice is numeric, money, or special patterns"""
    # Skip if text starts with common English words
    if re.match(r'^(by|the|a|an|in|on|at|to|for|of|from)\b', text.lower()):
        return False
        
    # Skip if text is too long (likely a sentence)
    if len(text.split()) > 5:
        return False
    
    # Basic numbers: "0", "1", "0,1", "0.5", "-1", "3/4"
    if bool(re.match(r'^-?\d+(\.\d+)?([,/]\d+)?$', text)):
        return True
        
    # Money: "$100", "100$", "£50", "€20", "100 USD"
    if bool(re.match(r'^[$£€]\d+(\.\d+)?$|^\d+(\.\d+)?[$£€]$|\d+\s*(USD|EUR|GBP)$', text)):
        return True
        
    # Percentages and ranges: "10%", "50.5%", "10-15%", "40-60%"
    if bool(re.match(r'^\d+(\.\d+)?%$|^\d+\s*-\s*\d+%$', text)):
        return True
        
    # Ratios and proportions: "1 : 1", "50 : 1", "1:1", "3:2:1"
    if bool(re.match(r'^\d+(\.\d+)?(\s*:\s*\d+(\.\d+)?)+$', text)):
        return True
        
    # Comma-separated numbers and ranges: "1,2,3", "1,2,3,4,5", "1,3,4"
    if bool(re.match(r'^(\d+,\s*)*\d+$', text)):
        return True
        
    # Units with variations: "21000 km/h", "32000 км/с", "10 g/day", "20 м/с²", "GeV/c^2"
    if bool(re.match(r'^\d+(\.\d+)?\s*(km/h|км/с|g/day|г/день|м/с|м/с²|GeV/c\^2|МэВ)$', text)):
        return True
        
    # Scientific notation with variations: "6.345 x 10^-4", "1.148 x 10^-6", "10^3"
    if bool(re.match(r'^\d+(\.\d+)?\s*x\s*10\^-?\d+|^10\^-?\d+', text)):
        return True
        
    # Chemical formulas: "H2O", "CO2", "BaH2", "Na+"
    if bool(re.match(r'^[A-Z][a-z]?\d*|\([A-Za-z\d]+\)\d+|\+|\-$', text)):
        return True
        
    # Roman numerals and combinations: "I", "II", "III", "I and II", "II and III only"
    if bool(re.match(r'^[IVX]+(\s+(and|only)\s+[IVX]+)*(\s+only)?$', text)):
        return True
        
    # Between ranges: "between 0.16 and 0.50", "between 0.02 and 0.16"
    if bool(re.match(r'^between\s+\d+(\.\d+)?\s+and\s+\d+(\.\d+)?$', text)):
        return True
        
    # Mathematical expressions: "π^2", "π^2 / 4", "π^2 / 2"
    if bool(re.match(r'^π(\^2)?(\s*\/\s*[24])?$', text)):
        return True

    return False

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def translate_with_context(client, text, context=None, is_choice=False, other_choices=None, choice_idx=None):
    """Translate question and choices in single call"""
    # Always do combined translation now
    choices_text = '\n'.join(f"[{i}] {c}" for i,c in enumerate(context))
    prompt = f"""Translate this question and choices to Tajik Cyrillic:

Question: {text}

Choices:
{choices_text}

Rules:
1. Return EXACTLY in this format (no extra text):
<question>
[Tajik question]
</question>
<choices>
[0] [Choice in Tajik]
[1] [Choice in Tajik]
[2] [Choice in Tajik]
[3] [Choice in Tajik]
</choices>

2. Keep all numbers and special characters exactly as they are
3. Use ONLY Tajik Cyrillic script - NO Persian/Arabic script
4. Translate COMPLETE sentences - don't leave sentences unfinished
5. Don't preserve English text unless it's a proper noun or technical term"""

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {"role": "system", "content": "You are a precise translator that returns ONLY the translation in the exact format requested."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        text = response.choices[0].message.content.strip()
        
        # Clean up common patterns
        text = re.sub(r'^(Here is|Translation:|Here\'s) (the )?translation:?\s*', '', text, flags=re.IGNORECASE)
        text = text.split('\n\n')[0]  # Take first block if multiple translations
        
        # Validate format
        if not (text.startswith('<question>') and '</choices>' in text):
            logger.error(f"❌ Invalid translation format:\n{text}")
            return None
            
        return text
    except Exception as e:
        logger.error(f"❌ Translation error: {e}\nText: {text[:100]}...")
        return None

def clean_translation(text):
    """Clean translation by removing English text if both English and Tajik exist"""
    if text and "\n" in text:
        parts = text.split("\n")
        # Take the last part which is usually Tajik
        return parts[-1].strip()
    return text

def needs_translation(text, original_text=None):
    """Check if text needs translation"""
    if text is None:
        return True
    
    # Clean the text first
    text = clean_translation(text)
    
    # If translated text is exactly same as original (and not numeric)
    if original_text and text == original_text and not is_numeric_choice(text):
        return True
    
    # Check for bot-like responses and other issues
    bad_patterns = [
        r'^please\b',
        r'^sure\b', 
        r'let me',
        r'i will',
        r'i can',
        r'if you',
        r'here is',
        r'the text you provided',
        r'would you like',
        r'translation of',
        r'remains the same',
        # Add Persian/Arabic script check
        r'[\\u0600-\\u06FF]',  # Persian/Arabic characters
        r'^none of',  # Common untranslated phrases
        r'^all of',
        r'^both',
        # Add more untranslated patterns
        r'^by\b',
        r'^the\b',
        r'^a\b',
        r'^an\b'
    ]
    
    for pattern in bad_patterns:
        if re.search(pattern, text.lower()):
            return True
            
    # Check for incomplete sentences (ending in prepositions/etc)
    if re.search(r'\b(of|in|by|to|for|from|with)\s*$', text.lower()):
        return True
    
    # If it's empty after cleaning
    if not text.strip():
        return True
        
    # Check if contains non-Tajik characters (basic check)
    # Count ratio of ASCII to non-ASCII chars
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    total_chars = len(text.strip())
    if total_chars == 0 or ascii_chars / total_chars > 0.5:  # If more than 50% ASCII
        return True
        
    return False

def parse_translation_response(text):
    """Parse combined question and choices translation response"""
    try:
        # Get question part
        question_match = re.search(r'<question>\s*(.*?)\s*</question>', text, re.DOTALL)
        question = question_match.group(1).strip() if question_match else None
        
        # Get choices part
        choices_match = re.search(r'<choices>\s*(.*?)\s*</choices>', text, re.DOTALL)
        choices_text = choices_match.group(1) if choices_match else None
        
        # Parse each choice with index
        choices = []
        if choices_text:
            choice_pattern = r'\[(\d+)\]\s*(.+?)(?=\n\[\d+\]|\Z)'
            choices = [match.group(2).strip() for match in re.finditer(choice_pattern, choices_text)]
        
        return question, choices
    except Exception as e:
        logger.error(f"Failed to parse translation response: {e}\nText: {text}")
        return None, None

def has_bot_response(text):
    """Check if text contains bot-like responses"""
    bot_patterns = [
        r'^please\b',
        r'^sure\b', 
        r'let me',
        r'i will',
        r'i can',
        r'if you',
        r'here is',
        r'the text you provided',
        r'would you like',
        r'translation of',
        r'remains the same'
    ]
    
    if not text:
        return False
        
    text = text.lower().strip()
    return any(re.search(pattern, text) for pattern in bot_patterns)

def is_english_text(text):
    """Check if text is primarily English (non-numeric, non-bot response)"""
    if not text or is_numeric_choice(text):
        return False
        
    # Skip checking if it's a bot response
    if has_bot_response(text):
        return False
        
    # Count ratio of ASCII to non-ASCII chars
    ascii_chars = sum(1 for c in text if ord(c) < 128 and c.isalpha())
    total_chars = sum(1 for c in text if c.isalpha())
    
    # If more than 70% of alphabetic chars are ASCII, likely English
    return total_chars > 0 and (ascii_chars / total_chars > 0.7)

async def fix_translations(file_path):
    logger.info(f"Processing {file_path}")
    
    # Load original MMLU dataset for subjects
    try:
        ds = load_dataset("cais/mmlu", "all")
        original_data = {}
        for split in ds:
            for item in ds[split]:
                # Fix subject mapping - use question as key since id may not match
                key = item['question'].strip()
                original_data[key] = item['subject']
        logger.info("✅ Loaded original MMLU subjects")
    except Exception as e:
        logger.error(f"❌ Failed to load original MMLU: {e}")
        original_data = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    client = AsyncOpenAI()
    fixed_count = 0
    total_items = len(data)
    
    # Enhanced progress bar
    pbar = tqdm(
        data,
        desc=f"Processing {Path(file_path).name}",
        unit="items",
        total=total_items,
        dynamic_ncols=True,  # Better terminal handling
        position=0  # Keep at bottom
    )
    
    for item in pbar:
        needs_save = False
        
        # FIRST check for bot responses in choices
        has_bot_choices = False
        for trans in item['translated_choices']:
            if has_bot_response(trans):
                has_bot_choices = True
                needs_choice_translation = True
                logger.info(f"🤖 Found bot response in choice: {trans}")
                break
        
        # THEN check for English text in choices
        has_english_choices = False
        if not has_bot_choices:  # Only check if not already marked for bot response
            for i, trans in enumerate(item['translated_choices']):
                if is_english_text(trans):
                    has_english_choices = True
                    needs_choice_translation = True
                    logger.info(f"🔤 Found English text in choice {i}: {trans}")
                    break
        
        # Handle both bot responses and English text
        if has_bot_choices or has_english_choices:
            try:
                translation = await translate_with_context(
                    client,
                    item['original_question'],
                    context=item['original_choices'],
                    is_choice=False
                )
                
                if translation:
                    question, choices = parse_translation_response(translation)
                    if choices and len(choices) == len(item['translated_choices']):
                        for i, new_trans in enumerate(choices):
                            old_trans = item['translated_choices'][i]
                            item['translated_choices'][i] = new_trans
                            if has_bot_choices:
                                logger.info(f"[{i}] 🤖 Fixed bot response:\n    📝 Before: {old_trans}\n    ✨ After: {new_trans}")
                            else:
                                logger.info(f"[{i}] 🔤 Fixed English text:\n    📝 Before: {old_trans}\n    ✨ After: {new_trans}")
                        needs_save = True
                        fixed_count += 1
                continue  # Skip other checks for this item
            except Exception as e:
                logger.error(f"❌ Translation error: {e}")
        
        # Then proceed with numeric checks and other fixes
        # Update progress with more info
        pbar.set_postfix({
            'ID': item.get('id', 'N/A'),
            'Fixed': fixed_count,
            'Needs Save': needs_save
        })
        
        # Add subject if missing using question as key
        if 'subject' not in item and item['original_question'].strip() in original_data:
            item['subject'] = original_data[item['original_question'].strip()]
            needs_save = True
            logger.info(f"📚 Added subject: {item['subject']}")
        
        # First handle numeric choices
        has_numeric_choices = False
        for i, orig in enumerate(item['original_choices']):
            if is_numeric_choice(orig):
                has_numeric_choices = True
                if item['translated_choices'][i] != orig:
                    item['translated_choices'][i] = orig
                    needs_save = True
                    logger.info(f"🔢 Preserved numeric choice: {orig}")
        
        if has_numeric_choices:
            if needs_save:
                logger.info("✅ All choices were numeric, skipping translation")
            continue
        
        # Check if any choices are identical to original and need translation
        needs_choice_translation = False
        for orig, trans in zip(item['original_choices'], item['translated_choices']):
            if needs_translation(trans, orig):
                needs_choice_translation = True
                break
        
        # Combined translation for both question and choices
        if needs_translation(item['translated_question']) or needs_choice_translation:
            try:
                translation = await translate_with_context(
                    client,
                    item['original_question'],
                    context=item['original_choices'],
                    is_choice=False
                )
                
                if translation:
                    question, choices = parse_translation_response(translation)
                    
                    # Log the full before/after state in one message
                    log_msg = "🔄 Fixed translations:\n"
                    
                    # Single injection block with logging
                    if question:
                        old_q = item['translated_question']
                        item['translated_question'] = question  # Inject question
                        log_msg += f"\nQuestion:\n📝 Before: {old_q}\n✨ After: {question}\n"
                        needs_save = True
                        fixed_count += 1
                    
                    if choices and len(choices) == len(item['translated_choices']):
                        log_msg += "\nChoices:\n"
                        for i, (orig, new_trans) in enumerate(zip(item['original_choices'], choices)):
                            if not is_numeric_choice(orig):  # Skip numerics
                                old_trans = item['translated_choices'][i]
                                item['translated_choices'][i] = new_trans  # Inject choice
                                log_msg += f"[{i}] 📝 Before: {old_trans}\n    ✨ After: {new_trans}\n"
                        needs_save = True
                        fixed_count += 1
                    
                    logger.info(log_msg)
                    
            except Exception as e:
                logger.error(f"❌ Translation error: {e}\nText: {item['original_question'][:100]}...")
        
        # Clean any remaining mixed translations
        if "\n" in str(item['translated_question']):
            old_q = item['translated_question']
            item['translated_question'] = clean_translation(old_q)
            logger.info(f"🧹 Cleaned mixed translation in question:\n📝 Before: {old_q}\n✨ After: {item['translated_question']}")
            needs_save = True
        
        for i, trans in enumerate(item['translated_choices']):
            if "\n" in str(trans):
                old_trans = trans
                item['translated_choices'][i] = clean_translation(trans)
                logger.info(f"🧹 Cleaned mixed translation in choice {i}:\n📝 Before: {old_trans}\n✨ After: {item['translated_choices'][i]}")
                needs_save = True
    
    if fixed_count > 0 or needs_save:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"💾 Saved {fixed_count} fixes in {file_path}")

async def main():
    data_dir = Path("translated_data_mmlu")

    files_to_fix = [
        data_dir / "validation_translated_full.json",
        data_dir / "test_translated_full.json",
        data_dir / "dev_translated_full.json",
        data_dir / "auxiliary_train_translated_full.json"
    ]
    
    # Add progress bar for files
    for file_path in tqdm(files_to_fix, desc="Processing files", unit="file"):
        if file_path.exists():
            await fix_translations(file_path)

if __name__ == "__main__":
    asyncio.run(main()) 