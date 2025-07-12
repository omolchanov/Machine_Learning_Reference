"""
Semantic Search & Entity Linking

Store embeddings or semantic fingerprints in Neo4j (e.g., vector embeddings with Neo4j’s vector search plugin).
Combine vector similarity search with graph queries.

Enables rich search like:
“Find documents about ‘climate change’ related to ‘policy’ authored by ‘X’” — mixing semantics and graph filters.
"""

import warnings
warnings.filterwarnings('ignore')

from sentence_transformers import SentenceTransformer
from together import Together
from neo4j import GraphDatabase

# --- Neo4j connection ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Together()

# --- Embedder (local model) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Together API ---
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"


# Create vector index (run once) ---
def create_vector_index():
    with driver.session() as session:
        session.run("""
        CREATE VECTOR INDEX document_vector_index
        FOR (d:Document) ON (d.embedding)
        OPTIONS {
            indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: "cosine"
            }
        }
        """)


def store_documents_with_embeddings():
    # --- Sample documents ---
    docs = [
        {"id": "1", "text": "Climate change affects global temperature patterns.", "author": "Alice"},
        {"id": "2", "text": "Policy changes are needed to combat global warming.", "author": "Bob"},
        {"id": "3", "text": "Climate and policy are tightly connected.", "author": "Alice"},
        {"id": "4", "text": "Economic trends influence stock markets.", "author": "Charlie"},
    ]
    with driver.session() as session:
        for doc in docs:
            embedding = embedder.encode(doc["text"]).tolist()  # Convert to the plain list
            session.run("""
                            MERGE (d:Document {id: $id})
                            SET d.text = $text,
                                d.author = $author,
                                d.embedding = $embedding
                        """, id=doc["id"], text=doc["text"], author=doc["author"], embedding=embedding)


# Semantic + filtered search ---
def semantic_search(query: str, author: str = None, top_k: int = 2):
    query_emb = embedder.encode(query).tolist()

    with driver.session() as session:
        result = session.run("""
            CALL db.index.vector.queryNodes('document_vector_index', $top_k, $query_emb)
            YIELD node, score
            WHERE ($author IS NULL OR node.author = $author)
            RETURN node.text AS text, node.author AS author, score
            ORDER BY score DESC
        """, query_emb=query_emb, author=author, top_k=top_k)

        return [record.data() for record in result]


# Enriching with LLM
def ask_llm_with_context(question, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering based on provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5,
        max_tokens=500
    )
    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    # try:
    #     create_vector_index()
    # except Exception as e:
    #     print("Index might already exist:", e)
    store_documents_with_embeddings()

    query = "climate change and policy"

    # Perform semantic search
    results = semantic_search(query, author='Alice', top_k=3)
    for r in results:
        print(f"\nText: {r['text']} \nAuthor: {r['author']} \nScore: {r['score']:.3f}")

    answer = ask_llm_with_context(query, results)
    print(f"LLM answer: \n{answer}")
