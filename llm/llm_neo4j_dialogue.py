"""
Interactive Dialogue & Personalization

Model user profiles, conversation history, preferences as graphs.
Use Neo4j to dynamically manage dialogue state or recommend next actions.
LLMs can query graph state to provide personalized, context-aware responses.
"""

import re

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


# --- Create user and preferences ---
def setup_user(user_id: str, interests: list):
    with driver.session() as session:
        # Step 1: Create the user
        session.run("""
            MERGE (u:User {id: $user_id})
            SET u.created_at = datetime()
        """, user_id=user_id)

        # Step 2: Attach interests using WITH to reuse `u`
        for topic in interests:
            session.run("""
                MATCH (u:User {id: $user_id})
                MERGE (t:Topic {name: $topic})
                MERGE (u)-[:INTERESTED_IN]->(t)
            """, user_id=user_id, topic=topic)


# --- Save message to conversation graph ---
def save_message(user_id: str, message: str, role: str):
    with driver.session() as session:
        result = session.run("""
            MATCH (u:User {id: $user_id})
            CREATE (m:Message {
                text: $text,
                role: $role,
                timestamp: datetime()
            })
            CREATE (u)-[:SENT]->(m)
            RETURN elementId(m) AS message_id
        """, user_id=user_id, text=message, role=role)
        return result.single()["message_id"]


# --- Get recent conversation history + user interests ---
def get_user_context(user_id: str, limit=5):
    with driver.session() as session:
        messages = session.run("""
            MATCH (u:User {id: $user_id})-[:SENT]->(m:Message)
            RETURN m.text AS text, m.role AS role
            ORDER BY m.timestamp DESC
            LIMIT $limit
        """, user_id=user_id, limit=limit).data()

        interests = session.run("""
            MATCH (u:User {id: $user_id})-[:INTERESTED_IN]->(t:Topic)
            RETURN t.name AS topic
        """, user_id=user_id).data()

        return messages[::-1], [i["topic"] for i in interests]  # oldest to newest


# --- Generate personalized LLM response ---
def generate_response(user_id: str, user_input: str):
    # Save user message
    user_msg_id = save_message(user_id, user_input, role="user")

    # Get context from graph
    messages, interests = get_user_context(user_id)

    history = "\n".join([f"{m['role']}: {m['text']}" for m in messages])
    interest_text = ", ".join(interests)

    messages = [
        {"role": "system", "content": f"You are a helpful assistant. The use is interested in {interest_text}."},
        {"role": "user", "content": f"Conversation:\n{history}\n\nUser input: {user_input}"}
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

    answer = clean_llm_response(response.choices[0].message.content.strip())

    # Save assistant response and get its ID
    assistant_msg_id = save_message(user_id, answer, role="assistant")

    # Link the messages with ANSWERED relationship
    with driver.session() as session:
        session.run("""
             MATCH (a:Message), (q:Message)
             WHERE elementId(a) = $assistant_msg_id AND elementId(q) = $user_msg_id
             MERGE (a)-[:ANSWERED]->(q)
         """, assistant_msg_id=assistant_msg_id, user_msg_id=user_msg_id)

    return answer


if __name__ == '__main__':
    user_id = "user123"
    setup_user(user_id, interests=["climate", "AI", "policy"])

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        reply = generate_response(user_id, user_input)
        print(f"\nLLM Answer: {reply}")
