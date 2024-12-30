import torch
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString
import re

# Quantization related settings
load_in_4bit = False
load_in_8bit = False

quantization_config = None
if load_in_4bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
elif load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )

# Initialize the translation model
device = 'cuda'  # change to 'cpu' if you don't have GPU
model_name = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
# model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Let's increase the max length of the model from 512 to 1024
tokenizer.model_max_length = 1024
model.config.max_length = 1024


def translate_text(text, to_lang='ru'):
    """Translates English text to Russian using the T5 model."""
    if not text.strip():
        return text  # Return empty or whitespace strings as is
    prefix = f'translate to {to_lang}: '
    src_text = prefix + text
    input_ids = tokenizer(src_text, return_tensors="pt").input_ids.to(device)
    generated_tokens = model.generate(input_ids)
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    return result[0]


def parse_and_translate(text):
    """
    Parses the text using BeautifulSoup to handle nested tags and translates
    the content inside the tags. Preserves the tags themselves in English.
    """
    # Define the tags to preserve
    tags = [
        'thinking',
        'tokenization',
        'reflection',
        'additional_information',
        'self_improvement',
        'meta_learning',
        'backward_propagation',
        'output',
    ]

    soup = BeautifulSoup(text, 'html.parser')

    def process_node(node):
        for content in node.contents:
            if isinstance(content, NavigableString):
                # Translate the text
                translated_text = translate_text(str(content))
                content.replace_with(translated_text)
            elif content.name in tags:
                # Process nested tags recursively
                process_node(content)
            else:
                # If the tag is not in the preserved list, treat its content as plain text
                if content.string:
                    translated_text = translate_text(str(content.string))
                    content.string.replace_with(translated_text)
                else:
                    process_node(content)

    process_node(soup)
    return str(soup)


def process_cot_dataset(input_path, output_path):
    """
    Reads a CoT-style JSON dataset, translates 'gpt' content with parse_and_translate(),
    and 'human' content with translate_text(). Saves the output to output_path.
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    translated_data = []

    # Process each conversation
    for item in tqdm(data, desc="Translating CoT dataset"):
        new_item = {'conversations': []}
        for convo in item['conversations']:
            new_convo = {'from': convo['from']}
            value = convo['value']

            if convo['from'] == 'gpt':
                # Use parse_and_translate for GPT messages
                translated_value = parse_and_translate(value)
            else:
                # For 'human' entries, translate directly
                translated_value = translate_text(value)

            new_convo['value'] = translated_value
            new_item['conversations'].append(new_convo)

        translated_data.append(new_item)

    # Save the translated data to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)


def process_math_dataset(input_path, output_path):
    """
    Reads the MATH-500 dataset from a JSON file with columns: 'problem',
    'solution', and 'answer'. Translates 'problem' and 'solution' fully,
    and only translates 'answer' if it matches the pattern \\text{...}.
    """
    pattern = re.compile(r'^\\text\{([^}]*)\}$')

    def translate_answer(answer):
        """Translates the text inside \\text{...} if it matches the pattern."""
        match = pattern.match(answer.strip())
        if match:
            content = match.group(1)
            translated_content = translate_text(content)
            return f'\\text{{{translated_content}}}'
        else:
            return answer

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    translated_data = []

    # Process each entry
    for item in tqdm(data, desc="Translating MATH-500 dataset (JSON)"):
        problem = item.get('problem', '')
        solution = item.get('solution', '')
        answer = item.get('answer', '')

        translated_problem = translate_text(problem)
        translated_solution = translate_text(solution)
        translated_answer = translate_answer(answer)

        # Keep extra fields like 'subject', 'level', 'unique_id' as is
        new_item = {
            'problem': translated_problem,
            'solution': translated_solution,
            'answer': translated_answer
        }

        # Preserve any other fields if desired
        for key in item:
            if key not in new_item:
                new_item[key] = item[key]

        translated_data.append(new_item)

    # Save the translated data to a new JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=2)


def process_math_dataset_jsonl(input_path, output_path):
    """
    Reads the MATH-500 dataset from a JSONL file (one JSON per line).
    Translates 'problem' and 'solution' fully,
    and only translates 'answer' if it matches the pattern \\text{...}.
    Writes the result as JSONL line by line to output_path.
    """
    pattern = re.compile(r'^\\text\{([^}]*)\}$')

    def translate_answer(answer):
        """Translates the text inside \\text{...} if it matches the pattern."""
        match = pattern.match(answer.strip())
        if match:
            content = match.group(1)
            translated_content = translate_text(content)
            return f'\\text{{{translated_content}}}'
        else:
            return answer

    with open(input_path, 'r', encoding='utf-8') as fin, \
        open(output_path, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, desc="Translating MATH-500 dataset (JSONL)"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            # Extract fields
            problem = item.get('problem', '')
            solution = item.get('solution', '')
            answer = item.get('answer', '')

            translated_problem = translate_text(problem)
            translated_solution = translate_text(solution)
            translated_answer = translate_answer(answer)

            # Keep extra fields
            new_item = {
                'problem': translated_problem,
                'solution': translated_solution,
                'answer': translated_answer
            }
            # Preserve any other fields if desired
            for key in item:
                if key not in new_item:
                    new_item[key] = item[key]

            # Write the result as a single line of JSON
            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")


# process_cot_dataset(
#     'chain-of-thought-sharegpt/chain_of_thought_sharegpt.json',
#     'translated_chain_of_thought_sharegpt.json'
# )

process_math_dataset_jsonl(
    'MATH-500/test.jsonl',
    'MATH-500-Russian.jsonl'
)
