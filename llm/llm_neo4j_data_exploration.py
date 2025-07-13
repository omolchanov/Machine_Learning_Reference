"""
Training Data as a Graph

You’ll model:
Documents (samples, e.g. text or image references)
Labels (categories, outputs)
Concepts (keywords, entities)
Relationships like HAS_LABEL, MENTIONS, etc.
"""

import datetime
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

# --- Neo4j connection ---
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "testpassword"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- Ingest training documents into Neo4j ---
def ingest_training_data(samples):
    with driver.session() as session:
        for s in samples:
            emb = embedder.encode(s["text"]).tolist()

            session.run("""
                MERGE (d:Document {id: $id})
                SET d.text = $text,
                    d.embedding = $embedding,
                    d.timestamp = datetime($timestamp)

                MERGE (l:Label {name: $label})
                MERGE (d)-[:HAS_LABEL]->(l)
            """, {
                "id": s["id"],
                "text": s["text"],
                "embedding": emb,
                "timestamp": datetime.datetime.now().isoformat(),
                "label": s["label"]
            })

            for concept in s["concepts"]:
                session.run("""
                    MERGE (c:Concept {name: $concept})
                    MERGE (d:Document {id: $id})-[:MENTIONS]->(c)
                """, {"id": s["id"], "concept": concept})


# --- Create a vector index for documents ---
def create_vector_index():
    with driver.session() as session:
        try:
            session.run("""
                CREATE VECTOR INDEX training_data_vector_index
                FOR (d:Document)
                ON (d.embedding)
                OPTIONS {
                    indexConfig: {
                        `vector.dimensions`: 384,
                        `vector.similarity_function`: "cosine"
                    }
                }
            """)
            print("Vector index created.")
        except Exception as e:
            print("Index may already exist or error:", e)


# --- Example dataset ---
samples = [
    {
        "id": "doc1",
        "text": "AI in medicine is transforming diagnosis.",
        "label": "health",
        "concepts": ["AI", "medicine", "diagnosis"]
    },
    {
        "id": "doc2",
        "text": "Climate policy must address renewable energy.",
        "label": "environment",
        "concepts": ["climate", "policy", "renewable energy"]
    },
    {
        "id": "doc3",
        "text": "AI is being used to model climate change.",
        "label": "environment",
        "concepts": ["AI", "climate"]
    },
    {
        "id": "doc4",
        "text": "New diagnostic tools are being powered by AI.",
        "label": "health",
        "concepts": ["AI", "diagnostic tools"]
    }
]


# --- Explore data insights (run in browser or from script) ---
exploration_queries = {
    "Label Distribution": """
        MATCH (l:Label)<-[:HAS_LABEL]-(d:Document)
        RETURN l.name AS label, count(*) AS count
        ORDER BY count DESC
    """,

    "Concepts per Label": """
        MATCH (d:Document)-[:HAS_LABEL]->(l:Label),
              (d)-[:MENTIONS]->(c:Concept)
        RETURN l.name AS label, c.name AS concept, count(*) AS frequency
        ORDER BY frequency DESC
    """,

    "Ambiguous Concepts": """
        MATCH (c:Concept)<-[:MENTIONS]-(d:Document)-[:HAS_LABEL]->(l:Label)
        WITH c.name AS concept, collect(DISTINCT l.name) AS labels
        WHERE size(labels) > 1
        RETURN concept, labels
    """
}


# --- Run exploration queries from Python (optional) ---
def run_exploratory_queries():
    with driver.session() as session:
        for title, query in exploration_queries.items():
            print(f"\n {title}")
            result = session.run(query)
            for row in result:
                print(dict(row))


# --- Main ---
if __name__ == "__main__":
    print("Ingesting training data...")
    ingest_training_data(samples)

    print("Creating vector index...")
    create_vector_index()

    print("Running data exploration...")
    run_exploratory_queries()
