from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, AutoModelForSeq2SeqLM
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from xml.etree import ElementTree

# Load ClinicalBERT for embedding generation
tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
model = AutoModel.from_pretrained("medicalai/ClinicalBERT")

gen_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Function to search PubMed and retrieve article IDs
def search_pmc(query, max_results=5):
    url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pmc",  # Use 'pmc' for PubMed Central (full-text)
        "term": query,
        "retmax": max_results,
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    tree = ElementTree.fromstring(response.content)
    article_ids = [id_elem.text for id_elem in tree.findall(".//Id")]
    return article_ids

# Function to fetch PubMed articles based on article IDs
def fetch_full_text_pmc(article_ids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": ",".join(article_ids),
        "retmode": "xml"
    }
    response = requests.get(url, params=params)
    
    tree = ElementTree.fromstring(response.content)
    full_text_articles = []
    
    for article in tree.findall(".//article"):
        title = article.find(".//article-title").text
        body = article.find(".//body").text  
        if body is not None:
            full_text_articles.append({"title": title, "body": body})
    
    return full_text_articles

def create_embeddings(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

def extract_relevant_sentences(query, article_body, top_k=5):
    sentences = article_body.split(". ")
    sentence_embeddings = create_embeddings(sentences)  
    query_embedding = create_embeddings([query])  
    
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    
    most_similar_indices = np.argsort(similarities[0])[-top_k:][::-1]
    
    relevant_sentences = [sentences[i] for i in most_similar_indices]
    return ". ".join(relevant_sentences) if relevant_sentences else article_body 

def generate_response(query, relevant_sentences):
    combined_input = query + " Relevant info: " + relevant_sentences

    inputs = gen_tokenizer(combined_input, return_tensors="pt")

    outputs = gen_model.generate(inputs["input_ids"], max_length=200, num_return_sequences=1)

    response = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    user_query = input("Please enter your medical query: ")

    article_ids = search_pmc(user_query)
    
    if not article_ids:
        print("No articles found for the query.")
        return

    articles = fetch_full_text_pmc(article_ids)

    for article in articles:
        relevant_sentences = extract_relevant_sentences(user_query, article['body'])

        response = generate_response(user_query, relevant_sentences)

        print(f"Response from Article '{article['title']}':\n{response}\n")

if __name__ == "__main__":
    main()
