import json
from typing import List
from tqdm import tqdm
import fire
import ollama

OLLAMA_BASE_URL = "http://localhost:11434"

OLLAMA_MODEL = "llama3.1:8b-q8_0"

DEFAULT_SYSTEM_PROMPT = (
    "Translate provided text to Russian language. "
    "Your reply should contain ONLY the translated text, nothing else. "
    "Please use exactly the same formatting as the original text."
)


def ollama_translate(
    system_prompt: str,
    text_to_translate: str,
    ollama_base_url: str = OLLAMA_BASE_URL,
    ollama_model: str = OLLAMA_MODEL
) -> str:
    """
    Translate provided text using Ollama
    :param system_prompt: str Instruction of how text should be translated
    :param text_to_translate: str What need to translate
    :param ollama_base_url: str Address of Ollama API-server
    :param ollama_model: str Ollama model
    :return:
    """
    if not text_to_translate.strip():
        return text_to_translate

    # Prepare prompt to translate
    prompt = f"{system_prompt}\n\n{text_to_translate}".strip()

    # Call Ollama
    response = ollama.generate(
        base_url=ollama_base_url,
        model=ollama_model,
        prompt=prompt,
        stream=False
    )

    # Response processing
    if isinstance(response, dict) and "content" in response:
        return response["content"].strip()
    elif isinstance(response, str):
        return response.strip()
    else:
        return str(response).strip()


def translate_dataset(
    input_path: str,
    output_path: str,
    fields_to_translate: List = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ollama_base_url: str = OLLAMA_BASE_URL,
    ollama_model: str = OLLAMA_MODEL,
):
    """
    Translate JSONL dataset
    :param input_path: str Path to input JSONL dataset
    :param output_path: str Path to output JSONL dataset
    :param fields_to_translate: List of fields to translate, if None all fields will be translated
    :param system_prompt: str Instruction of how text should be translated
    :param ollama_base_url: str Address of Ollama API-server
    :param ollama_model: str Model which will be used for translation
    :return:
    """
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:

        for line in tqdm(fin, desc=f"Translating JSONL dataset: {input_path}"):
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            new_item = {}

            for key, value in item.items():
                if key in fields_to_translate or fields_to_translate is None:
                    new_item[key] = ollama_translate(
                        system_prompt=system_prompt,
                        text_to_translate=value,
                        ollama_base_url=ollama_base_url,
                        ollama_model=ollama_model
                    )
                else:
                    new_item[key] = value

            fout.write(json.dumps(new_item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    fire.Fire(translate_dataset)
