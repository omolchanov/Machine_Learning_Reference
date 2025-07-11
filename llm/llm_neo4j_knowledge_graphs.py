"""
Knowledge Graphs for Context and Reasoning

LLMs generate text but often lack structured, explicit knowledge.
Neo4j stores entities and their relationships as a graph — ideal for representing rich context.
You can build a knowledge graph from documents, facts, user data, or domain ontologies.
At query time, combine LLM generation with graph queries to:
Ground LLM answers in verified facts
Perform complex multi-hop reasoning via graph traversal
Provide explanations or provenance by tracing graph paths
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


def setup_knowledge_graph():
    with driver.session() as session:
        session.run("""
            MERGE (alice:Person {name: "Alice"})
            MERGE (laptop:Product {name: "Laptop"})
            SET laptop.price = 1500
            MERGE (mouse:Product {name: "Mouse"})
            SET mouse.price = 25
            MERGE (electronics:Category {name: "Electronics"})
            MERGE (laptop)-[:BELONGS_TO]->(electronics)
            MERGE (mouse)-[:BELONGS_TO]->(electronics)
            MERGE (alice)-[:PURCHASED]->(laptop)
            MERGE (alice)-[:PURCHASED]->(mouse)
        """)


def get_context_for_person(name):
    with driver.session() as session:
        result = session.run("""
        MATCH (p:Person {name: $name})-[:PURCHASED]->(prod:Product)
        RETURN prod.name AS product, prod.price AS price
        """, name=name)

        facts = []

        for record in result:
            facts.append(f"{name} purchased {record['product']} for ${record['price']}.")
        return " ".join(facts)


def ask_llm_with_context(question, context):
    message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=message,
        max_tokens=120,
        temperature=0.5,
    )

    return response.choices[0].message.content


if __name__ == '__main__':
    setup_knowledge_graph()

    question = "What did Alice buy?"
    context = get_context_for_person("Alice")

    answer = ask_llm_with_context(question, context)
    print(answer)
