from openai import OpenAI
import numpy as np


def get_embedding(text, model="text-embedding-3-small"):
    client = OpenAI()
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_embeddings_batch(texts, model="text-embedding-3-small"):
    client = OpenAI()
    
    # Clean all texts (replace newlines)
    texts = [t.replace("\n", " ") for t in texts]
    # Make the API call for all at once (or a manageable batch)
    response = client.embeddings.create(input=texts, model=model)
    # Extract embeddings
    embeddings = [d.embedding for d in response.data]
    return np.array(embeddings)

# Auto-theme generation using GPT
def generate_theme_label(top_terms):
    client = OpenAI()
    prompt = (
        f"Given these top TF-IDF keywords and phrases from Yelp reviews: "
        f"{', '.join(top_terms)}.\n"
        "Suggest a concise, human-readable theme name (3–6 words)."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
