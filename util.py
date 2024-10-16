import os
import torch
import re
import json
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import ollama

def list_files_in_folder_no_extension(folder_path):
    try:
        files_and_dirs = os.listdir(folder_path)
        files = [os.path.splitext(f)[0] for f in files_and_dirs if os.path.isfile(os.path.join(folder_path, f))]
        return files
    except FileNotFoundError:
        return f"Error: The folder '{folder_path}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"

def get_summarization(source, file_path):
    def process_paragraphs(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        paragraphs = data.get("paragraphs", [])
        cleaned_paragraphs = []
        for paragraph in paragraphs:
            no_html = re.sub(r'<[^>]*>', '', paragraph)
            if not re.match(r'^\d+$', no_html.strip()):
                cleaned_paragraphs.append(no_html)
        cleaned_content = '\n'.join(cleaned_paragraphs)
        items = re.split(r'(?i)item\s+\d+[a-zA-Z]?', cleaned_content)
        items = [item.strip() for item in items if item.strip()]
        result = " ".join(items)
        return result
    def Bart_Summary(file_path):
        def chunk_text(text, max_length):
            return [text[i:i + max_length] for i in range(0, len(text), max_length)]
        def generate_summary(text, chunk_size=1000, summary_max_length=150):
            device = 0 if torch.cuda.is_available() else -1
            summarizer = pipeline("summarization", device=device)
            chunks = chunk_text(text, chunk_size)
            summaries = []
            for chunk in chunks:
                summary = summarizer(chunk, max_length=summary_max_length, min_length=30, do_sample=False)
                summaries.append(summary[0]['summary_text'])
            return "\n\n".join(summaries)
        text = process_paragraphs(file_path)
        summary = generate_summary(text)
        return summary
    def ChatGPT_summary(file_path):
        text = process_paragraphs(file_path)
        client = OpenAI()
        role = "You are a great helper in fields of reading financial reports in 10-K and 10-Q filing, and you are good at give great summary of these financial reports with great detail in numeric data."
        prompt = f'Here is a financial report that need your help to get a great summarization in great detail in numeric data.\nYou need to be in great detail since your summary will be later used as the source for comparision with other corporation.\nHere is the financial report, keep in mind that I need a detailed financial report summary in numeric data.\n{text}'
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    def Ollama_summary(file_path):
        text = process_paragraphs(file_path)
        role = "You are a great helper in fields of reading financial reports in 10-K and 10-Q filing, and you are good at give great summary of these financial reports with great detail in numeric data."
        prompt = f'Here is a financial report that need your help to get a great summarization in great detail in numeric data.\nYou need to be in great detail since your summary will be later used as the source for comparision with other corporation.\nHere is the financial report, keep in mind that I need a detailed financial report summary in numeric data.\n{text}'
        output = ollama.chat(
            model="llama3.1",
            messages=[
                {"role":"system", "content":role},
                {"role":"user", "content":prompt}
            ],
            stream=False,
        )
        return output['message']['content']
    if source == "ChatGPT":
        return ChatGPT_summary(file_path)
    elif source == "Bart-GPU-Generating":
        return Bart_Summary(file_path)
    elif source == "LlaMA3.1-8B":
        return Ollama_summary(file_path)


def handle_similarity(file_path, output):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def json_to_plain_text(file_path):
        def remove_html_tags(text):
            clean = re.compile('<.*?>')
            return re.sub(clean, '', text)
        def is_numeric_only(text):
            return text.strip().isdigit()
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        paragraphs = data.get('paragraphs', [])
        plain_text_list = []
        for paragraph in paragraphs:
            clean_paragraph = remove_html_tags(paragraph)
            if not is_numeric_only(clean_paragraph):
                plain_text_list.append(clean_paragraph)
        return plain_text_list
    corpus = json_to_plain_text(file_path)
    save_path = "sentence-transformers/all-mpnet-base-v2"
    trained_model = SentenceTransformer(save_path)
    corpus_embeddings = trained_model.encode(corpus, show_progress_bar=False, convert_to_tensor=True).to(device)
    processed_output = output.split('\n\n')
    similarity_output = ""
    for query in processed_output:
        if query[0] == '#':
            continue
        else:
            query_embeddings = trained_model.encode(query, show_progress_bar=False, convert_to_tensor=True).to(device)
            result = util.semantic_search(query_embeddings, corpus_embeddings)[0][0]
            similarity_output += f"Original sentence:{corpus[result['corpus_id']]}\n\nGenerated sentence: {query}\n\nSimilarity rate: {result['score']}\n\n"
    return similarity_output

def handle_analyst(summary_A, summary_B):
    client = OpenAI()
    role = "You are a financial assistant for a investing corporation, skilled in comparing different 10-K filing financial reprots in a great detail, you can also say which one is better in specific perspective."
    prompt = f"Please base on the two summaries I give you after the prompt, they are very detailed in numeric data. Please write a comparison analysis report between the two corporations' financial reports.\nIf the corporation I give you are in total different fields, just tell me.\nHere is the first summary:\n{summary_A}\n\nHere is the second summary:\n{summary_B}\n\n"
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": role},
            {"role": "user", "content": prompt}
        ]
    )
    comparison = completion.choices[0].message.content
    return comparison

def handle_perspective(perspective, summary_A, summary_B):
    def chat_GPT(summary,perspective):
        client = OpenAI()
        role = "You are a financial assistant for a investing corporation, skilled in comparing different 10-K filing financial reprots in a great detail, you can also say which one is better in specific perspective."
        prompt = f"Please base on the summaries I give you after the prompt, it is very detailed in numeric data. Please tell me all the detailed informations which is related to the perspective {perspective}\nIf the corporation I give you have nothing in that perspective, just tell me.\nHere is the summary:\n{summary}"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    def compare_perspectives(summary_A, summary_B, perspective):
        client = OpenAI()
        role = "You are a financial assistant for a investing corporation, skilled in comparing different 10-K filing financial reprots in a great detail, you can also say which one is better in specific perspective."
        prompt = f"Please base on the summaries I give you after the prompt, they are very detailed in numeric data. Please compare the detailed informations which is related to the perspective {perspective} between the co\nIf the corporation I give you have nothing in that perspective, just tell me.\nHere is the first summary:\n{summary_A}\nThe second summary:\n{summary_B}"
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content
    perspective_A = chat_GPT(summary_A, perspective)
    perspective_B = chat_GPT(summary_B, perspective)
    perspective_comparison = compare_perspectives(summary_A, summary_B, perspective)
    return (perspective_A, perspective_B, perspective_comparison)

def read_cache():
    source_path = './static/cache/source.txt'
    outputA_path = './static/cache/outputA.txt'
    outputB_path = './static/cache/outputB.txt'
    similarity_A_path = './static/cache/similarity_A.txt'
    similarity_B_path = './static/cache/similarity_B.txt'
    optionA_path = './static/cache/optionA.txt'
    optionB_path = './static/cache/optionB.txt'
    comparison_path = './static/cache/comparison.txt'
    perspective_path = './static/cache/perspective.txt'
    perspective_A_path = './static/cache/perspective_A.txt'
    perspective_B_path = './static/cache/perspective_B.txt'
    perspective_comparison_path = './static/cache/perspective_comparison.txt'
    with open(source_path,'r') as f:
        source = f.read()
    with open(outputA_path, 'r') as f:
        outputA = f.read()
    with open(outputB_path, 'r') as f:
        outputB = f.read()
    with open(similarity_A_path, 'r') as f:
        similarity_A = f.read()
    with open(similarity_B_path, 'r') as f:
        similarity_B = f.read()
    with open(optionA_path, 'r') as f:
        optionA = f.read()
    with open(optionB_path, 'r') as f:
        optionB = f.read()
    with open(comparison_path, 'r')as f:
        comparison = f.read()
    with open(perspective_path, 'r')as f:
        perspective = f.read()
    with open(perspective_A_path, 'r')as f:
        perspective_A = f.read()
    with open(perspective_B_path, 'r')as f:
        perspective_B = f.read()
    with open(perspective_comparison_path, 'r')as f:
        perspective_comparison = f.read()
    return (source, outputA, outputB, similarity_A, similarity_B, optionA, optionB, comparison, perspective, perspective_A, perspective_B, perspective_comparison)

def write_source(source):
    source_path = './static/cache/source.txt'
    with open(source_path,'w') as f:
        f.write(source)
def write_outputA(summary_A):
    outputA_path = './static/cache/outputA.txt'
    summary_A = re.sub(r'[^\x00-\x7F]', '', summary_A)
    with open(outputA_path, 'w') as f:
        f.write(summary_A)

def write_outputB(summary_B):
    outputB_path = './static/cache/outputB.txt'
    summary_B = re.sub(r'[^\x00-\x7F]', '', summary_B)
    with open(outputB_path, 'w') as f:
        f.write(summary_B)

def write_similarity_A(similarity_A):
    similarity_A_path = './static/cache/similarity_A.txt'
    similarity_A = re.sub(r'[^\x00-\x7F]', '', similarity_A)
    with open(similarity_A_path, 'w') as f:
        f.write(similarity_A)

def write_similarity_B(similarity_B):
    similarity_B_path = './static/cache/similarity_B.txt'
    similarity_B = re.sub(r'[^\x00-\x7F]', '', similarity_B)
    with open(similarity_B_path, 'w') as f:
        f.write(similarity_B)

def write_optionA(optionA):
    optionA_path = './static/cache/optionA.txt'
    with open(optionA_path, 'w') as f:
        f.write(optionA)

def write_optionB(optionB):
    optionB_path = './static/cache/optionB.txt'
    with open(optionB_path, 'w') as f:
        f.write(optionB)

def write_comparison(comparison):
    comparison_path = './static/cache/comparison.txt'
    with open(comparison_path, 'w') as f:
        f.write(comparison)

def write_perspective(perspective):
    perspective_path = './static/cache/perspective.txt'
    with open(perspective_path, 'w') as f:
        f.write(perspective)

def write_perspective_A(perspectpve_A):
    perspective_A_path = './static/cache/perspective_A.txt'
    with open(perspective_A_path, 'w') as f:
        f.write(perspectpve_A)

def write_perspective_B(perspectpve_B):
    perspective_B_path = './static/cache/perspective_B.txt'
    with open(perspective_B_path, 'w') as f:
        f.write(perspectpve_B)

def write_perspective_comparison(perspective_comparison):  
    perspective_comparison_path = './static/cache/perspective_comparison.txt'
    with open(perspective_comparison_path, 'w') as f:
        f.write(perspective_comparison)

def initialization():
    init =''
    write_source(init)
    write_outputA(init)
    write_outputB(init)
    write_similarity_A(init)
    write_similarity_B(init)
    write_optionA(init)
    write_optionB(init)
    write_comparison(init)
    write_perspective(init)
    write_perspective_A(init)
    write_perspective_B(init)
    write_perspective_comparison(init)