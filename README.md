# Zotero Semantic Search Web App

A simple web application for semantic search and topic modeling of academic papers stored in Zotero. This app allows users to:
- Perform semantic searches on paper titles and abstracts using a lightweight transformer model.
- Visualize the main topics of the papers using a topic modeling pipeline based on HDBSCAN clustering and TF-IDF keyword extraction.
- View search results in a table format with authors, abstracts, and DOIs.

## Features

- **Semantic Search**: Find relevant papers based on a query using cosine similarity of embeddings.
- **Topic Modeling**: Discover the main topics in your Zotero library using HDBSCAN clustering and TF-IDF keyword extraction.
- **Progress Bars**: Track the progress of data retrieval from Zotero and embedding generation in real-time.
- **Caching**: Processed data is cached to avoid reprocessing on subsequent runs.
- **User-Friendly Interface**: Simple and intuitive web interface with progress bars and interactive tables.

## Technologies Used

- **Backend**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Clustering**: HDBSCAN
- **Keyword Extraction**: TF-IDF

## How It Works

1. **Data Retrieval**: The app fetches papers from an export of your Zotero Library.
2. **Embedding Generation**: Abstracts are converted into embeddings using a lightweight transformer model (standard model is all-MiniLM-L6-v2 but you can use any model you prefer).
3. **Semantic Search**: Users can search for papers by entering a query. The app calculates the cosine similarity between the query embedding and paper embeddings to rank results.
4. **Topic Modeling**: The app clusters papers based on their embeddings and extracts the most relevant keywords for each cluster.

## Setup Instructions

1. Clone the repository:
   git clone https://github.com/a-meneghini/zotero-semantic-search.git
   cd zotero-semantic-search
2. Install the required dependencies:
   pip install -r requirements.txt
3. Export your Zotero library as zotero_data.csv and put it into the project folder
4. Run the Flask app:
   python app.py
Open your browser and go to http://127.0.0.1:5000

## License

This project is licensed under the GPL3 License. See the LICENSE file for details.
