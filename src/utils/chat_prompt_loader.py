import json
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import logging
import requests
import imghdr
import base64

logger = logging.getLogger(__name__)

def _load_url_as_base64(image_url: str) -> tuple[str, str]:
    resp = requests.get(image_url, timeout=15)
    resp.raise_for_status()
    image_bytes = resp.content
    fmt = imghdr.what(None, h=image_bytes)
    media_type_map = {
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    media_type = media_type_map.get(fmt, "image/jpeg")
    return base64.b64encode(image_bytes).decode("utf-8"), media_type


def load_prompt_from_json(file_path: str) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt JSON not found at {file_path}. "
            "Run the prompt .py file first to generate it."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_chat_prompt_from_file(file_path: str) -> ChatPromptTemplate:
    path = Path(file_path)

    if path.suffix == ".json":
        with open(path, "r") as f:
            data = json.load(f)
    elif path.suffix == ".py":
        scope = {}
        exec(path.read_text(), scope)
        data = scope["PROMPT_DATA"]
    else:
        raise ValueError("Unsupported prompt file format. Use .json or .py.")

    system_message = SystemMessagePromptTemplate.from_template(data["system"])

    # ✅ Handle both string and list (multimodal) human templates
    human_template = data["human"]
    if isinstance(human_template, list):
        human_message = HumanMessagePromptTemplate.from_template(human_template)
    else:
        human_message = HumanMessagePromptTemplate.from_template(human_template)

    return ChatPromptTemplate.from_messages([system_message, human_message])


def load_other_json(file_path):
    path = Path(file_path)
    with open(path, "r") as f:
        data = json.load(f)
    return data

def safe_json_parse(self, response_text, context=""):
    """Safely parse JSON with proper error handling"""
    try:
        # Strip any whitespace/newlines that might cause issues
        cleaned_text = response_text.strip()
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {context}: {e}")
        logger.error(f"Raw response: {repr(response_text)}")
        # Try to extract JSON if response has extra content
        try:
            # Look for JSON object boundaries
            start = response_text.find('{')
            if start != -1:
                # Find the matching closing brace
                brace_count = 0
                end = start
                for i, char in enumerate(response_text[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                json_part = response_text[start:end]
                logger.info(f"Attempting to parse extracted JSON: {json_part}")
                return json.loads(json_part)
        except:
            pass
        
        return None
