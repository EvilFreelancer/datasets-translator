import torch
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString


# Quantization related settings
load_in_4bit = True
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
device = 'cuda'
model_name = 'utrobinmv/t5_translate_en_ru_zh_small_1024'
# model_name = 'utrobinmv/t5_translate_en_ru_zh_large_1024'
model = T5ForConditionalGeneration.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quantization_config
)
tokenizer = T5Tokenizer.from_pretrained(model_name)

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
    Parses the text using BeautifulSoup to handle nested tags and translates the content inside the tags.
    Preserves the tags themselves in English.
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

    # Parse the text
    soup = BeautifulSoup(text, 'html.parser')

    def process_node(node):
        for content in node.contents:
            if isinstance(content, NavigableString):
                # Translate the text
                translated_text = translate_text(str(content))
                content.replace_with(translated_text)
            elif content.name in tags:
                # Process nested tags
                process_node(content)
            else:
                # If the tag is not in the list, treat its content as plain text
                if content.string:
                    translated_text = translate_text(str(content.string))
                    content.string.replace_with(translated_text)
                else:
                    process_node(content)

    process_node(soup)
    # Return the processed text without altering the original tags
    return str(soup)

# Read the JSON file
with open('chain-of-thought-sharegpt/chain_of_thought_sharegpt.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

translated_data = []

# Process each conversation
for item in tqdm(data):
    new_item = {'conversations': []}
    for convo in item['conversations']:
        new_convo = {'from': convo['from']}
        value = convo['value']

        if convo['from'] == 'gpt':
            # Process tags and translate content
            translated_value = parse_and_translate(value)
        else:
            # For 'human' entries, translate directly
            translated_value = translate_text(value)

        new_convo['value'] = translated_value
        new_item['conversations'].append(new_convo)

    translated_data.append(new_item)

# Save the translated data to a new JSON file
with open('translated_chain_of_thought_sharegpt.json', 'w', encoding='utf-8') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=2)
