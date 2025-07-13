"""
Data Provenance & Audit Trails

Track how LLM outputs relate to source data by storing references and linkages in Neo4j.
Enables transparent, auditable AI systems.

We’ll capture:
Which source documents were used in the prompt
Which LLM generated the output
What was the question, answer, and timestamp
"""

import datetime
import random
import re

from sentence_transformers import SentenceTransformer

from neo4j import GraphDatabase
from together import Together

# --- Neo4j connection ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# --- Together API ---
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Together()
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- Store documents with embeddings ---
def store_documents():

    # --- Sample documents ---
    docs = [
        "Alice wrote a policy proposal on climate action in 2023.",
        "Bob discussed AI alignment with policy makers.",
        "A report on carbon taxes was released by Alice.",
        "Charlie wrote a research article on AI ethics and bias.",
    ]

    with driver.session() as session:
        for i, text in enumerate(docs):
            vector = embedder.encode(text).tolist()
            session.run("""
                MERGE (d:Document {id: $id})
                SET d.text = $text,
                    d.embedding = $vector
            """, id=str(i), text=text, vector=vector)


# --- Create vector index ---
def create_vector_index():
    with driver.session() as session:
        try:
            session.run("""
                CREATE VECTOR INDEX document_vector_index
                IF NOT EXISTS
                FOR (d:Document)
                ON (d.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: "cosine"
                    }
                }
            """)
        except Exception as e:
            print("Index may already exist or error:", e)


# --- Semantic search ---
def semantic_search(query: str, top_k: int = 3):
    query_vector = embedder.encode(query).tolist()
    cypher = """
        CALL db.index.vector.queryNodes('document_vector_index', $top_k, $vector)
        YIELD node, score
        RETURN node.id AS doc_id, node.text AS text, score
    """
    with driver.session() as session:
        result = session.run(cypher, vector=query_vector, top_k=top_k)
        return [r.data() for r in result]


# --- Generate  LLM response ---
def generate_response(query: str, context_docs: list):
    context = "\n".join([f"- {doc['text']}" for doc in context_docs])

    messages = [
        {"role": "system", "content": f"You are an expert assistant. Use the context below to answer the question."},
        {"role": "user", "content": f"Context:\n{context} \nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=150
    )

    def clean_llm_response(text: str) -> str:
        # Remove <think>...</think> blocks or any similar
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    return clean_llm_response(response.choices[0].message.content.strip())


# --- Save provenance data ---
def record_provenance(user_id: int, query: str, answer: str, docs: list, model_name: str):
    try:
        timestamp = datetime.datetime.now().isoformat()

        with driver.session() as session:
            # Create query and answer
            session.run("""
                MERGE (u:User {id: $user_id})
                CREATE (q:Query {
                    text: $query_text,
                    timestamp: datetime($timestamp)
                })
                CREATE (a:Answer {
                    text: $answer,
                    timestamp: datetime($timestamp)
                })
                CREATE (m:LLM {name: $model_name})
                MERGE (u)-[:ASKED]->(q)
                MERGE (q)-[:GENERATED]->(a)
                MERGE (q)-[:USED_MODEL]->(m)
            """, user_id=user_id, query_text=query, answer=answer, model_name=model_name, timestamp=timestamp)

            # Link to source documents
            for doc in docs:
                session.run("""
                    MATCH (q:Query {text: $query_text})
                    MATCH (d:Document {id: $doc_id})
                    MERGE (q)-[:USED_SOURCE {score: $score}]->(d)
                """, query_text=query, doc_id=doc["doc_id"], score=doc["score"])
    except Exception as e:
        print(e)


if __name__ == '__main__':
    print("Ingesting documents into Neo4j...")
    store_documents()
    create_vector_index()

    user_id = random.randint(1, 100)
    question = "Is there anything that Bob has written on AI ?"

    print("\nRunning semantic search...")
    relevant_docs = semantic_search(question, top_k=2)
    for doc in relevant_docs:
        print(f" - {doc['doc_id']}: {doc['text']} (score={doc['score']:.4f})")

    print("\nGenerating answer from LLM...")
    answer = generate_response(question, relevant_docs)
    print(f"\nAnswer:\n{answer}")

    print("\nRecording provenance data in Neo4j...")
    record_provenance(user_id, question, answer, relevant_docs, model_name=MODEL)
