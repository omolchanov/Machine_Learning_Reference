"""
Prompt Engineering & Context Retrieval

Neo4j can store document chunks, metadata, embeddings, or entity graphs.

For an input question, you can:
Search the graph for relevant context nodes or documents
Retrieve related facts/entities to build a contextual prompt fed to the LLM
This improves LLM relevance and reduces hallucination.

We simulate storing document chunks with metadata in Neo4j, then:

Given a question, we match documents by keyword (or embedding similarity, if enabled).
Retrieved chunks are fed as context into the LLM (Together.ai) for answering.
"""

from together import Together
from neo4j import GraphDatabase

# --- Neo4j connection ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

# --- Together API ---
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
client = Together()


def insert_corpus():
    docs = [
        {"id": "doc1", "content": "Neo4j is a graph database used for connected data."},
        {"id": "doc2", "content": "LLMs can be enhanced by retrieving contextual knowledge from graphs."},
        {"id": "doc3", "content": "Vector embeddings enable semantic search of unstructured data."},
        {"id": "doc4", "content": "Together.ai provides access to powerful open-weight language models."},
    ]
    with driver.session() as session:
        for doc in docs:
            session.run("""
                MERGE (d:Document {id: $id, content: $content})
            """, id=doc["id"], content=doc["content"])


def find_context(question: str) -> str:
    # Simulate a basic keyword match (replace with embedding similarity later)
    keywords = [word.lower() for word in question.split()]
    with driver.session() as session:
        result = session.run("""
            MATCH (d:Document)
            WHERE any(word IN $keywords WHERE toLower(d.content) CONTAINS word)
            RETURN d.content AS content
            LIMIT 3
        """, keywords=keywords)
        return "\n".join([record["content"] for record in result])


def ask_llm_with_context(question, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering based on provided context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content.strip()


if __name__ == '__main__':
    insert_corpus()

    question = "How can Neo4j help LLMs?"
    context = find_context(question)
    print(f"Retrieved context:\n{context}\n")

    answer = ask_llm_with_context(question, context)
    print(f"LLM Answer:\n{answer}")
