"""
Research Assistant Agent with Long-Term Memory using Endee
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional

import httpx
from openai import OpenAI

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
ENDEE_URL = os.getenv("ENDEE_URL", "http://localhost:8080")
INDEX_NAME = "research_memory"
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-3.5-turbo"
TOP_K = 5  # how many memories to retrieve per query

client = OpenAI(api_key=OPENAI_API_KEY)


# ──────────────────────────────────────────────
# 1. Endee Helpers
# ──────────────────────────────────────────────

def create_index_if_needed():
    """Create the Endee vector index on first run."""
    resp = httpx.post(f"{ENDEE_URL}/indexes", json={
        "name": INDEX_NAME,
        "dimension": 1536,          # dimension for text-embedding-3-small
        "metric": "cosine"
    })
    if resp.status_code in (200, 201):
        print(f"[Endee] Index '{INDEX_NAME}' created.")
    elif resp.status_code == 409:
        print(f"[Endee] Index '{INDEX_NAME}' already exists — OK.")
    else:
        raise RuntimeError(f"[Endee] Failed to create index: {resp.text}")


def embed(text: str) -> list[float]:
    """Convert text to a vector embedding using OpenAI."""
    response = client.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def store_memory(text: str, metadata: dict):
    """Store a memory (text + metadata) into Endee."""
    vector = embed(text)
    doc = {
        "id": str(uuid.uuid4()),
        "vector": vector,
        "payload": {
            "text": text,
            "timestamp": datetime.utcnow().isoformat(),
            **metadata
        }
    }
    resp = httpx.post(f"{ENDEE_URL}/indexes/{INDEX_NAME}/documents", json=doc)
    if resp.status_code not in (200, 201):
        print(f"[Endee] Warning: could not store memory: {resp.text}")
    else:
        print(f"[Endee] Memory stored: '{text[:60]}...'")


def retrieve_memories(query: str, top_k: int = TOP_K) -> list[dict]:
    """Retrieve the most relevant memories from Endee for a given query."""
    vector = embed(query)
    resp = httpx.post(f"{ENDEE_URL}/indexes/{INDEX_NAME}/search", json={
        "vector": vector,
        "top_k": top_k
    })
    if resp.status_code != 200:
        print(f"[Endee] Warning: search failed: {resp.text}")
        return []
    results = resp.json().get("results", [])
    return [r["payload"] for r in results]


# ──────────────────────────────────────────────
# 2. Agent Logic
# ──────────────────────────────────────────────

SYSTEM_PROMPT = """You are ResearchBot, an intelligent research assistant with long-term memory.

You help users:
- Understand and summarize research papers
- Remember topics, concepts, and papers they've discussed before
- Connect ideas across different research areas
- Answer questions using both your knowledge and their stored memories

When you receive context from memory, use it to give personalized, accurate responses.
Always be concise, clear, and cite what you remember when relevant.
"""


def build_prompt_with_memory(user_message: str, memories: list[dict]) -> list[dict]:
    """Build the message list for the OpenAI API, injecting relevant memories."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if memories:
        memory_text = "\n\n".join([
            f"[Memory from {m.get('timestamp', 'unknown time')}] {m.get('text', '')}"
            for m in memories
        ])
        messages.append({
            "role": "system",
            "content": f"Relevant memories from our past conversations:\n{memory_text}"
        })

    messages.append({"role": "user", "content": user_message})
    return messages


def chat(user_message: str) -> str:
    """Main chat function: retrieve memory → respond → store new memory."""

    # Step 1: Retrieve relevant memories
    print(f"\n[Agent] Searching memory for: '{user_message[:50]}...'")
    memories = retrieve_memories(user_message)
    print(f"[Agent] Found {len(memories)} relevant memories.")

    # Step 2: Build prompt with memory context
    messages = build_prompt_with_memory(user_message, memories)

    # Step 3: Get response from OpenAI
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=800
    )
    reply = response.choices[0].message.content

    # Step 4: Store the conversation turn in memory
    store_memory(
        text=f"User asked: {user_message}\nAgent replied: {reply}",
        metadata={"type": "conversation"}
    )

    return reply


def add_paper(title: str, abstract: str, authors: str = "", year: str = ""):
    """Add a research paper to the agent's memory."""
    text = f"Paper: {title}\nAuthors: {authors}\nYear: {year}\nAbstract: {abstract}"
    store_memory(text, metadata={
        "type": "paper",
        "title": title,
        "authors": authors,
        "year": year
    })
    print(f"[Agent] Paper '{title}' added to memory.")


# ──────────────────────────────────────────────
# 3. CLI Interface
# ──────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  ResearchBot — AI Research Assistant with Memory")
    print("  Powered by Endee + OpenAI")
    print("=" * 60)
    print("Commands:")
    print("  /add   — Add a research paper to memory")
    print("  /quit  — Exit")
    print("  anything else — Chat with the agent")
    print("=" * 60)

    create_index_if_needed()

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "/quit":
            print("Goodbye!")
            break

        elif user_input.lower() == "/add":
            title = input("  Paper title: ").strip()
            authors = input("  Authors: ").strip()
            year = input("  Year: ").strip()
            abstract = input("  Abstract (paste and press Enter): ").strip()
            add_paper(title, abstract, authors, year)
            print(f"  ✓ Paper added to memory!")

        else:
            print("\nResearchBot: ", end="", flush=True)
            reply = chat(user_input)
            print(reply)


if __name__ == "__main__":
    main()
