from flask import Flask, request, jsonify, render_template
import pickle
import os
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import hdbscan
import numpy as np
import time
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Initialize Flask app
app = Flask(__name__)

# Path to the local data file
DATA_FILE = 'zotero_data.csv'  # File to store titles, abstracts, and DOIs

# Cache file to store processed data
CACHE_FILE = 'zotero_cache.pkl'

# Load a transformer model
embedding_model_name = 'all-MiniLM-L6-v2' # 'paraphrase-multilingual-MiniLM-L12-v2'
model = SentenceTransformer(embedding_model_name)

# Function to save processed data to cache
def save_cache(data):
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(data, f)

# Function to load processed data from cache
def load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                data = pickle.load(f)
                # Verify that the cache data is valid
                if isinstance(data, tuple) and len(data) == 4:  # titles, abstracts, DOIs, embeddings
                    return data
                else:
                    print("Cache file is corrupted. Deleting it...")
                    os.remove(CACHE_FILE)
        except Exception as e:
            print(f"Error loading cache: {e}. Deleting it...")
            os.remove(CACHE_FILE)
    return None

# Function to load data from a local CSV file
def load_local_data():
    if os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
        # fill NANS
        df['Title'] = df['Title'].fillna("")
        df['Abstract Note'] = df['Abstract Note'].fillna("")
        df['DOI'] = df['DOI'].fillna("")
        df['Publication Year'] = df['Publication Year'].fillna("")
        df['Publication Title'] = df['Publication Title'].fillna("")
        df['Author'] = df['Author'].fillna("")
        # get columns
        titles = df['Title'].tolist()
        abstracts = df['Abstract Note'].tolist()
        dois = df['DOI'].tolist()
        return titles, abstracts, dois
    return None

# Function to fetch data from the local file
def fetch_zotero_data():
    try:
        # verify if cache is already present
        if os.path.exists(CACHE_FILE):
            print("Loading data from cache...")
            return

        print("Fetching data from local file...", end="", flush=True)
        data = load_local_data()
        if not data:
            print("\nNo local data found. Please load data first.")
            return

        titles, abstracts, dois = data
        print(f"\nTotal items fetched: {len(titles)}")
        print("Data retrieval complete.")

        # Verify data
        if not titles or not abstracts or not dois:
            print("Error: Titles, abstracts, or DOIs are missing or empty.")
            return

        # Phase 2: Embedding generation
        embeddings = []
        batch_size = 32  # Process abstracts in batches for better performance
        valid_abstracts = [abstract if abstract.strip() else title for abstract, title in zip(abstracts, titles)]  # Use titles as fallback
        if not valid_abstracts:
            print("No valid abstracts or titles found. Skipping embedding generation.")
            return

        print("Generating embeddings...", end="", flush=True)
        total_items = len(valid_abstracts)
        for i in range(0, total_items, batch_size):
            batch = valid_abstracts[i:i + batch_size]
            batch_embeddings = model.encode(batch)  # Generate embeddings for the batch
            embeddings.extend(batch_embeddings)

            # calculate progress
            progress = int((i + len(batch)) / total_items * 100)
            print(f"embedding:{progress}")  # send progress

        print("\nEmbedding generation complete.")

        # save data in cache
        save_cache((titles, abstracts, dois, embeddings))
        print("Data fetching and embedding generation completed.")
    except Exception as e:
        print(f"\nError in fetch_zotero_data: {e}")

#function to create APA type citation
def format_apa_citation(title, author, year, publication_title):
    # APA style: AuthorLastName, AuthorFirstInitial. (Year). Title. PublicationTitle.
    return f"{author} ({year}). {title}. {publication_title}."

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream progress updates
@app.route('/progress')
def progress():
    return app.response_class(fetch_zotero_data(), mimetype='text/event-stream')

# Route to reload data (clear cache and restart the process)
@app.route('/reload', methods=['POST'])
def reload_data():
    try:
        if os.path.exists(CACHE_FILE):
            os.remove(CACHE_FILE)
            print("Cache file deleted.")
        return jsonify({"status": "success", "message": "Cache cleared. Data will be reloaded on next request."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# Route for semantic search API
@app.route('/search', methods=['GET'])
def search_api():
    query = request.args.get('q')  # Get query from URL parameters
    if not query:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    # Load cached data
    cached_data = load_cache()
    if not cached_data:
        return jsonify({"error": "No data available. Please fetch data first."}), 400
    titles, abstracts, dois, embeddings = cached_data

    # Load authors from the CSV file
    df = pd.read_csv(DATA_FILE)
    authors = df['Author'].fillna("N/A").tolist()

    # Generate embedding for the query
    query_embedding = model.encode(query)

    # Perform semantic search
    results = []
    for i, (title, abstract, doi, embedding) in enumerate(zip(titles, abstracts, dois, embeddings)):
        score = cosine_similarity([query_embedding], [embedding])[0][0]
        # Convert score to a native Python float
        score = float(score)
        results.append({
            "title": title,
            "author": authors[i],
            "abstract": abstract,
            "doi": doi,
            "score": score
        })

    # Sort results by score in descending order
    results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Limit results to the top 10
    results = results[:10]

    return jsonify(results)

@app.route('/topics', methods=['GET'])
def topics_api():
    # Load cached data
    cached_data = load_cache()
    if not cached_data:
        return jsonify({"error": "No data available. Please fetch data first."}), 400
    titles, abstracts, dois, embeddings = cached_data

    # Debug: print the shape of the embeddings
    print("Sample embeddings shape:", np.array(embeddings).shape)

    # Load additional metadata from the CSV file
    df = pd.read_csv(DATA_FILE)
    authors = df['Author'].fillna("N/A").tolist()
    years = df['Publication Year'].fillna("N/A").tolist()
    publication_titles = df['Publication Title'].fillna("N/A").tolist()

    # Filter out empty abstracts
    valid_abstracts = [abstract if abstract.strip() else title for abstract, title in zip(abstracts, titles)]  # Use titles as fallback
    if not valid_abstracts:
        return jsonify({"error": "No valid abstracts or titles found for topic modeling"})

    #reduce dimensionality of embeddings
    reducer = UMAP(n_components=5, metric='cosine', n_neighbors=15)
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Perform clustering using HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean', cluster_selection_epsilon=0.5)  # Tunable parameters
    cluster_labels = clusterer.fit_predict(reduced_embeddings)

    # Debug: print the cluster labels
    print("Cluster labels:", cluster_labels)
    print("Number of clusters:", len(set(cluster_labels)) - 1)  # remove noise points
    print("Percentage of noise points:", list(cluster_labels).count(-1) / len(cluster_labels))

    # Extract keywords using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(valid_abstracts)
    feature_names = vectorizer.get_feature_names_out()

    # Create a dictionary to store cluster details
    topics = {}
    for cluster_id in set(cluster_labels):
        if cluster_id == -1:
            continue  # Ignore noise points
        # Get indices of documents in the current cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        # Sum TF-IDF scores for the cluster
        cluster_tfidf = tfidf_matrix[cluster_indices].sum(axis=0).A1
        # Get indices of top 5 keywords
        top_keywords_indices = cluster_tfidf.argsort()[-5:][::-1]
        # Extract top keywords
        top_keywords = [feature_names[i] for i in top_keywords_indices]
        # Get the number of papers in the cluster
        num_papers = len(cluster_indices)
        # Get the list of papers with author, year, and title
        papers = [
            f"{authors[i]} ({years[i]}). {titles[i]}"
            for i in cluster_indices
        ]
        # Store cluster details
        topics[f"Cluster {cluster_id}"] = {
            "keywords": top_keywords,
            "num_papers": num_papers,
            "papers": papers
        }
    return jsonify(topics)

#route to render the topics page
@app.route('/topics-page')
def topics_page():
    return render_template('topics.html')

@app.route('/statistics', methods=['GET'])
def statistics_api():
    # Load cached data
    cached_data = load_cache()
    if not cached_data:
        return jsonify({"error": "No data available. Please fetch data first."}), 400
    titles, abstracts, dois, embeddings = cached_data

    # Load additional metadata from the CSV file
    df = pd.read_csv(DATA_FILE)
    num_entries = len(df)

    return jsonify({
        "num_entries": num_entries,
        "embedding_model": embedding_model_name,
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)