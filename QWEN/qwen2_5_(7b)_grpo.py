# -*- coding: utf-8 -*-

import re
from datasets import load_dataset, Dataset
from unsloth import FastLanguageModel, PatchFastRL
from unsloth import is_bfloat16_supported
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams
from loguru import logger
from dotenv import load_dotenv
load_dotenv()

HF_TOKEN = "load here"

PatchFastRL("GRPO", FastLanguageModel)

max_seq_length = 512 # Can increase for longer reasoning traces
lora_rank = 128 # Larger rank = smarter, but slower
logger.info(f"Setting parameters: max_seq_length = {max_seq_length}, lora_rank = {lora_rank}")

MODEL_NAME = "muhtasham/tajik-qwen2.5-7b_finetuned_merged-16bit"

logger.info(f"Loading model from {MODEL_NAME}")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = max_seq_length,
    load_in_4bit = False, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.7, # Reduce if out of memory
)

logger.info("Creating LoRA model")
model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 3407,
)


logger.info("Loading dataset")

# Load and prep dataset
SYSTEM_PROMPT = """
Ба тарзи зерин ҷавоб диҳед:
<андеша>
...
</андеша>
<ҷавоб>
...
</ҷавоб>
"""
logger.debug(f"System prompt: {SYSTEM_PROMPT}")

XML_COT_FORMAT = """\
<андеша>
{reasoning}
</андеша>
<ҷавоб>
{answer}
</ҷавоб>
"""
logger.debug(f"XML COT format: {XML_COT_FORMAT}")

def extract_xml_answer(text: str) -> str:
    answer = text.split("<ҷавоб>")[-1]
    answer = answer.split("</ҷавоб>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:
    logger.info(f"Loading {split} split")
    data = load_dataset('muhtasham/gsm8k-tajik')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_tj']}
        ],
        'answer': extract_hash_answer(x['answer_tj'])
    }) # type: ignore
    logger.info(f"Loaded {split} split")
    return data # type: ignore

dataset = get_gsm8k_questions()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<андеша>\n.*?\n</андеша>\n<ҷавоб>\n.*?\n</ҷавоб>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<андеша>.*?</андеша>\s*<ҷавоб>.*?</ҷавоб>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<андеша>\n") == 1:
        count += 0.125
    if text.count("\n</андеша>\n") == 1:
        count += 0.125
    if text.count("\n<ҷавоб>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</ҷавоб>\n")[-1])*0.001
    if text.count("\n</ҷавоб>") == 1:
        count += 0.125
        count -= (len(text.split("\n</ҷавоб>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

logger.info("Setting training arguments")
training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 256,
    max_completion_length = 200,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 250,
    max_grad_norm = 0.1,
    report_to = "none", # Can use Weights & Biases
    output_dir = "outputs",
)
logger.info("Creating trainer")
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
    ],
    args = training_args,
    train_dataset = dataset,
)
logger.info("Training model")
trainer.train()

logger.info("Generating output")
text = tokenizer.apply_chat_template([
    {"role" : "user", "content" : "How many r's are in strawberry?"},
], tokenize = False, add_generation_prompt = True)
logger.info("Setting sampling parameters")
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
logger.info("Generating output")
output = model.fast_generate(
    [text],
    sampling_params = sampling_params,
    lora_request = None,
)[0].outputs[0].text
logger.info(f"Output: {output}")

logger.info("Saving LoRA")
model.save_lora("grpo_saved_lora")
logger.info("Generating output with LoRA")
text = tokenizer.apply_chat_template([
    {"role" : "system", "content" : SYSTEM_PROMPT},
    {"role" : "user", "content" : "How many r's are in strawberry?"},
], tokenize = False, add_generation_prompt = True)

logger.info("Setting sampling parameters")
sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
logger.info("Generating output")
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text
logger.info(f"Output: {output}")

logger.info("Saving LoRA")
model.save_lora("grpo_saved_lora")

# Merge to 16bit
logger.info("Saving merged 16bit model")
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
logger.info("Pushing merged 16bit model to Hugging Face")
if True: model.push_to_hub_merged("muhtahsam/tajik-r1", tokenizer, save_method = "merged_16bit", token = HF_TOKEN)

# Merge to 4bit
logger.info("Saving merged 4bit model")
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",)
logger.info("Pushing merged 4bit model to Hugging Face")
if True: model.push_to_hub_merged("muhtahsam/tajik-r1-4bit", tokenizer, save_method = "merged_4bit", token = HF_TOKEN)

# Just LoRA adapters
logger.info("Saving LoRA adapters")
if True: model.save_pretrained_merged("model", tokenizer, save_method = "lora",)
logger.info("Pushing LoRA adapters to Hugging Face")
if True: model.push_to_hub_merged("muhtahsam/tajik-r1-lora", tokenizer, save_method = "lora", token = HF_TOKEN)

logger.info("Saving GGUF model")
if True:
    model.push_to_hub_gguf(
        "muhtahsam/tajik-r1-gguf", # Change muhtahsam to your username!
        tokenizer,
        quantization_method = ["q4_k_m", "q8_0", "q5_k_m", "f16"],
        token = HF_TOKEN,
    )
logger.info("Pushing GGUF model to Hugging Face")