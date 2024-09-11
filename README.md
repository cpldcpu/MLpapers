# MLpapers

Personal collection of machine learning papers. Just to keep track of papers I've read and want to read.

## Files and Directories

**[Papers/](./Papers/)**: Directory containing PDF files of the ML papers.

**[summary.md](./summary.md)**: Generated markdown file containing summaries of the processed papers.

## Summarizer

**[summarizer.py](./summarizer.py)**: Python script to process the papers and generate summaries.

Requires a LLM that is able to generate valid JSON output. I used *NousResearch/Hermes-3-Llama-3.1-8B-GGUF/Hermes-3-Llama-3.1-8B.Q8_0.gguf*. Model card [here](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-8B), information about function calling using this model [here](https://github.com/NousResearch/Hermes-Function-Calling/blob/main/jsonmode.py). 

The script was three-shotted with Claude-3.5-Sonnet (I had to feed in some of the hermes function calling examples to get properly evaluated JSON) plus some fine tuning of the prompt. I used LM studio as local server while running the script to generate summary.md.
