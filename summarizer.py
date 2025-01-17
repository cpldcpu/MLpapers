import os
import PyPDF2
import requests
import json
import re
import logging
import hashlib
import urllib.parse
from datetime import datetime

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
    1. Title of the paper (Sometimes random spaces are added or removed in the title due to OCR errors. Correct titles to use proper spacing and proper english. Convert all caps to title case.)
    2. Authors (as a list. Do **NOT** include email addresses, affiliation or other information, only names.)
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

def get_file_hash(file_path):
    """Generate a hash for the file to use as a unique identifier."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as file:
        buf = file.read()
        hasher.update(buf)
    return hasher.hexdigest()

def read_existing_markdown(file_path):
    """Read the existing markdown file and extract information about processed papers."""
    processed_papers = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            pattern = r' \[Local link\]\((.+?)\), Hash: (.+?), \*Added: (.+?)\*'
            matches = re.findall(pattern, content)
            for file_path_escaped, file_hash, date_added in matches:
                # Unescape the file path and normalize it
                file_path = urllib.parse.unquote(file_path_escaped)
                file_path = os.path.normpath(file_path)
                processed_papers[file_path] = file_hash
                logger.debug(f"Found existing paper: {file_path}, Hash: {file_hash}")
    return processed_papers

def generate_section_name(folder_path):
    """Generate a descriptive section name from a folder path using the LLM."""
    if folder_path == "":
        return "Unsorted Papers"
        
    query = f"""Given the folder name '{os.path.basename(folder_path)}', generate a short but descriptive section title 
    that would categorize AI/ML research papers in this folder. Return ONLY the title, nothing else.
    The title should be in title case and should not contain any special characters."""
    
    prompt = [
        {"role": "system", "content": "You are a helpful assistant that generates concise section titles."},
        {"role": "user", "content": query}
    ]
    
    response = requests.post(
        "http://localhost:1234/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "messages": prompt,
            "temperature": 0.7,
            "max_tokens": 50
        }
    )
    
    if response.status_code == 200:
        title = response.json()['choices'][0]['message']['content'].strip()
        return title
    return os.path.basename(folder_path).replace('_', ' ').title()

def create_or_update_markdown(results, output_file, processed_papers):
    # Group papers by their sections
    sections = {}
    for paper_info in results:
        section = paper_info.get('section', 'Unsorted Papers')
        if section not in sections:
            sections[section] = []
        sections[section].append(paper_info)
    
    # Write the markdown file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Research Paper Summaries\n\n")
        
        # Generate Table of Contents
        f.write("## Table of Contents\n\n")
        for section_name in sorted(sections.keys()):
            # Create a link-friendly version of the section name
            section_link = section_name.lower().replace(' ', '-')
            f.write(f"- [{section_name}](#{section_link})\n")
            # Add paper titles as sub-items in TOC
            for paper_info in sections[section_name]:
                paper_link = paper_info['title'].lower().replace(' ', '-')
                # Remove special characters from link
                paper_link = re.sub(r'[^a-z0-9-]', '', paper_link)
                f.write(f"  - [{paper_info['title']}](#{paper_link})\n")
        f.write("\n---\n\n")  # Add a horizontal line after TOC
        
        # Process each section
        for section_name in sorted(sections.keys()):
            f.write(f"## {section_name}\n\n")
            for paper_info in sections[section_name]:
                f.write(f"### {paper_info['title']}\n\n")
                f.write(f"*{', '.join(paper_info['authors'])}*\n\n")
                f.write(f"**Summary:** {paper_info['summary']}\n\n")
                
                arxiv_number = paper_info['arxiv_number']
                if arxiv_number != "N/A":
                    arxiv_link = f"https://arxiv.org/abs/{arxiv_number}"
                    f.write(f"**ArXiv:** [{arxiv_number}]({arxiv_link}), ")
                else:
                    f.write(f"**ArXiv:** {arxiv_number}, ")

                date_added = datetime.now().strftime("%Y-%m-%d")            
                
                rel_path = os.path.relpath(paper_info['file_path'], start=os.path.dirname(output_file))
                escaped_link = urllib.parse.quote(rel_path)
                f.write(f"[Local link]({escaped_link}), Hash: {paper_info['file_hash']}, *Added: {date_added}* \n\n")

def main(folder_path, output_file):
    # Normalize paths
    folder_path = os.path.abspath(folder_path)
    output_file = os.path.abspath(output_file)
    logger.info(f"Starting to process papers in: {folder_path}")
    
    processed_papers = read_existing_markdown(output_file)
    new_results = []
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        logger.info(f"Traversing directory: {root}")
        logger.debug(f"Found subdirectories: {dirs}")
        logger.debug(f"Found files: {files}")
        
        if num_of_papers > 0 and len(new_results) >= num_of_papers:
            break
            
        # Get relative path for section name
        rel_path = os.path.relpath(root, folder_path)
        if rel_path == ".":
            rel_path = ""
        section_name = generate_section_name(rel_path)
        logger.info(f"Processing section: {section_name} ({rel_path})")
            
        for filename in files:
            if filename.endswith('.pdf'):
                file_path = os.path.join(root, filename)
                file_path = os.path.normpath(file_path)
                file_hash = get_file_hash(file_path)
                
                # Use normalized path for comparison
                if file_path in processed_papers and processed_papers[file_path] == file_hash:
                    logger.info(f"Skipping already processed file: {file_path}")
                    continue
                
                logger.info(f"Processing file: {file_path}")
                text = extract_text_from_pdf(file_path)
                paper_info = get_paper_info(text)
                if paper_info:
                    paper_info['file_path'] = file_path
                    paper_info['file_hash'] = file_hash
                    paper_info['section'] = section_name
                    new_results.append(paper_info)
    
    if new_results:
        create_or_update_markdown(new_results, output_file, processed_papers)
        logger.info(f"Added {len(new_results)} new papers to {output_file}")
    else:
        logger.info("No new papers to add.")

if __name__ == "__main__":
    # Use absolute paths for better reliability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, "Papers")
    output_file = os.path.join(script_dir, "summary.md")
    main(folder_path, output_file)