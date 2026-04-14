# ResearchBot — AI Research Assistant with Long-Term Memory

> An AI agent that remembers research papers, past conversations, and topics — powered by [Endee](https://github.com/endee-io/endee) vector database and OpenAI GPT-3.5.

---

## Project Overview

ResearchBot is an AI-powered research assistant that uses **long-term semantic memory** to:

- Remember research papers you add (title, authors, abstract)
- Recall past conversations and questions
- Answer questions by combining AI knowledge with your personal memory store
- Connect ideas across papers and research topics

Unlike a regular chatbot, ResearchBot **never forgets**. Every conversation and every paper you add is stored as a vector embedding in Endee, and retrieved intelligently when relevant.

---

## System Design

```
User Input
    │
    ▼
[ Embed query ]  ←──── OpenAI text-embedding-3-small
    │
    ▼
[ Search Endee ]  ←─── Semantic vector search (cosine similarity)
    │                   Returns top-5 most relevant memories
    ▼
[ Build Prompt ]  ←─── System prompt + retrieved memories + user message
    │
    ▼
[ OpenAI GPT-3.5 ]  ── Generates response using memory context
    │
    ▼
[ Store to Endee ]  ── Save conversation turn as new memory vector
    │
    ▼
Agent Response
```

### How Endee is used

| Operation | Endee API | Purpose |
|---|---|---|
| Store memory | `POST /indexes/{name}/documents` | Save paper or conversation as vector |
| Retrieve memory | `POST /indexes/{name}/search` | Find most relevant past context |
| Create index | `POST /indexes` | Set up vector store on first run |

Memory types stored:
- `paper` — research paper title + abstract + authors
- `conversation` — user question + agent reply pair

---


## Example Usage

```
You: What is the Transformer architecture?

ResearchBot: The Transformer is a neural network architecture introduced in
"Attention Is All You Need" (Vaswani et al., 2017). It replaces recurrence
and convolutions with attention mechanisms entirely...

You: /add
  Paper title: Constitutional AI
  Authors: Anthropic
  Year: 2022
  Abstract: We propose a method for training AI systems to be helpful,
  harmless, and honest using a set of principles...

You: How does Constitutional AI relate to alignment?

ResearchBot: Based on the paper you just added, Constitutional AI is
Anthropic's approach to AI alignment where...
```

---

## Running with Docker (full stack)

```bash
# Run both Endee and the agent together
docker-compose up

# In another terminal, attach to the agent
docker attach research-agent
```

---


## Technologies Used

| Tool | Purpose |
|---|---|
| [Endee](https://endee.io) | Vector database for storing and searching memories |
| [OpenAI GPT-3.5](https://openai.com) | Language model for generating responses |
| [OpenAI Embeddings](https://openai.com) | Converting text to vectors |
| Python 3.11 | Application logic |
| Docker | Running Endee locally |

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
