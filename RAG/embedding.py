import os
import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

def extract_text_from_file(file_path):
    """Extract the entire text from a .txt file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    return text

def chunk_files_in_directory(directory_path):
    """Reads .txt files in a directory, treating each file as a single chunk."""
    chunks = []
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        if file_name.endswith('.txt'):
            text = extract_text_from_file(file_path)
            chunks.append(text)
            print(f"Chunk {len(chunks)}: {file_name} (size: {len(text.split())} words)")
    return chunks

def create_embeddings(chunks, model_name='BAAI/bge-large-en-v1.5'):
    """Generate embeddings for text chunks using BAAI/bge-large-en-v1.5."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        chunk_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().detach().numpy()
        embeddings.append(chunk_embedding)
    return embeddings

def save_to_json(embeddings, chunks, output_file):
    """Save the embeddings and text chunks to a JSON file."""
    json_objects = [{
        "id": idx,
        "text": chunk,
        "vector": row.tolist()
    } for idx, (chunk, row) in enumerate(zip(chunks, embeddings))]

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_objects, f, indent=4)
    print(f"Saved: {output_file}")

def process_directories(directory_paths, output_folder):
    """Process multiple directories and save separate JSON files for each."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for directory_path in directory_paths:
        dir_name = os.path.basename(directory_path.rstrip('/'))
        output_file = os.path.join(output_folder, f"{dir_name}.json")

        chunks = chunk_files_in_directory(directory_path)
        if chunks:
            embeddings = create_embeddings(chunks)
            save_to_json(embeddings, chunks, output_file)
        else:
            print(f"No text files found in {directory_path}, skipping.")

# Example Usage
directories = [
    "../text_data"
]
output_folder = "../processed_data"
process_directories(directories, output_folder)
