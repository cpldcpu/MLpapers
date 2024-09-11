import os
import PyPDF2
import requests
import json
import re
import logging

# Global variables for debugging
debug_msg = True
num_of_papers = 0  # Set to 0 to process all papers, or a positive integer to limit the number of papers

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_arxiv_number(text):
    arxiv_pattern = r'arXiv:(\d+\.\d+)'
    match = re.search(arxiv_pattern, text)
    return match.group(1) if match else "N/A"

def generate_json_completion(query, max_depth=5):
    pydantic_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "authors": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "arxiv_number": {"type": "string"}
        },
        "required": ["title", "authors", "summary", "arxiv_number"]
    }

    sys_prompt = f"You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n<schema>\n{json.dumps(pydantic_schema, indent=2)}\n</schema>"
    prompt = [{"role": "system", "content": sys_prompt},
              {"role": "user", "content": query}]

    def run_inference(prompt):
        response = requests.post(
            "http://localhost:1234/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "messages": prompt,
                "temperature": 0.7,
                "max_tokens": 500
            }
        )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Error: {response.status_code}, {response.text}")
            return None

    def get_assistant_message(completion):
        if completion and 'choices' in completion:
            return completion['choices'][0]['message']['content']
        return None

    def validate_json_data(data, schema):
        try:
            json_object = json.loads(data)
            # Implement JSON schema validation here
            # For simplicity, we'll just check if all required fields are present
            for field in schema['required']:
                if field not in json_object:
                    return False, None, f"Missing required field: {field}"
            return True, json_object, None
        except json.JSONDecodeError as e:
            return False, None, str(e)

    def recursive_loop(prompt, depth):
        nonlocal max_depth
        if depth >= max_depth:
            logger.warning(f"Maximum recursion depth reached ({max_depth}). Stopping recursion.")
            return None

        completion = run_inference(prompt)
        if not completion:
            return None

        assistant_message = get_assistant_message(completion)
        if assistant_message is None:
            logger.warning("Assistant message is None")
            return None

        validation, json_object, error_message = validate_json_data(assistant_message, pydantic_schema)
        
        if debug_msg:
            logger.info(f"Assistant Message:\n{assistant_message}")
            logger.info(f"JSON schema validation: {'passed' if validation else 'failed'}")
            if validation:
                logger.info(f"Parsed JSON object:\n{json.dumps(json_object, indent=2)}")
            else:
                logger.info(f"Error message: {error_message}")

        if validation:
            return json_object

        tool_message = f"Agent iteration {depth} to assist with user query: {query}\n"
        tool_message += f"<tool_response>\nJSON schema validation failed\nHere's the error: {error_message}\nPlease return correct JSON object\n</tool_response>"
        
        prompt.append({"role": "assistant", "content": assistant_message})
        prompt.append({"role": "tool", "content": tool_message})
        
        return recursive_loop(prompt, depth + 1)

    return recursive_loop(prompt, 0)

def get_paper_info(text):
    query = f"""Extract the following information from the given text and return it as a JSON object:
    1. Title of the paper
    2. Authors (as a list. Do **NOT** include email addresses or other information, only names.)
    3. A one-sentence summary of the paper
    4. ArXiv number (if present, otherwise "N/A")

    Text: {text[:2000]}...
    """
    
    paper_info = generate_json_completion(query)
    
    if paper_info:
        # If ArXiv number wasn't found by LLM, try to extract it from the text
        if paper_info['arxiv_number'] == "N/A":
            paper_info['arxiv_number'] = extract_arxiv_number(text)
        return paper_info
    else:
        logger.error("Failed to extract paper information")
        return None

def create_markdown(results, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, paper_info in enumerate(results, 1):
            f.write(f"#### {i}. {paper_info['title']}\n\n")
            f.write(f"*{', '.join(paper_info['authors'])}*\n\n")
            f.write(f"**Summary:** {paper_info['summary']}\n\n")
            f.write(f"**ArXiv:** {paper_info['arxiv_number']}, ")
            f.write(f"[Link]({paper_info['file_path']})\n")

def main(folder_path, output_file):
    results = []
    for i, filename in enumerate(os.listdir(folder_path)):
        if num_of_papers > 0 and i >= num_of_papers:
            break
        if filename.endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            logger.info(f"Processing file: {file_path}")
            text = extract_text_from_pdf(file_path)
            paper_info = get_paper_info(text)
            if paper_info:
                paper_info['file_path'] = file_path
                results.append(paper_info)
    
    create_markdown(results, output_file)

if __name__ == "__main__":
    folder_path = "Papers/"
    output_file = "summary.md"
    main(folder_path, output_file)