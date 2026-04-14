"""
Load sample research papers into the agent's memory.
Run this once to pre-populate the agent with example papers.

Usage:
    python src/load_samples.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.agent import create_index_if_needed, add_paper

SAMPLE_PAPERS = [
    {
        "title": "Attention Is All You Need",
        "authors": "Vaswani et al.",
        "year": "2017",
        "abstract": (
            "We propose a new simple network architecture, the Transformer, based solely on "
            "attention mechanisms, dispensing with recurrence and convolutions entirely. "
            "The Transformer generalizes well to other tasks and achieves state-of-the-art "
            "results on machine translation tasks."
        )
    },
    {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        "authors": "Devlin et al.",
        "year": "2018",
        "abstract": (
            "We introduce BERT, a new language representation model which stands for "
            "Bidirectional Encoder Representations from Transformers. Unlike recent language "
            "representation models, BERT is designed to pre-train deep bidirectional "
            "representations by jointly conditioning on both left and right context in all layers."
        )
    },
    {
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": "Lewis et al.",
        "year": "2020",
        "abstract": (
            "We explore a general-purpose fine-tuning recipe for retrieval-augmented generation "
            "(RAG) — models which combine pre-trained parametric and non-parametric memory "
            "for language generation. Our RAG models achieve state-of-the-art results on "
            "open-domain question answering and fact verification."
        )
    },
    {
        "title": "LangChain: Building Applications with LLMs through Composability",
        "authors": "Chase et al.",
        "year": "2022",
        "abstract": (
            "LangChain is a framework for developing applications powered by language models. "
            "It enables applications that are data-aware and agentic, connecting language models "
            "to other sources of data and allowing them to interact with their environment."
        )
    },
    {
        "title": "Generative Agents: Interactive Simulacra of Human Behavior",
        "authors": "Park et al.",
        "year": "2023",
        "abstract": (
            "We introduce generative agents, computational software agents that simulate "
            "believable human behavior. Generative agents wake up, cook breakfast, and head "
            "to work; they form opinions, notice each other, and initiate conversations. "
            "They remember, reflect on, and plan based on long-term memory stored in a "
            "natural language memory stream."
        )
    }
]


def main():
    print("Loading sample research papers into Endee memory...")
    create_index_if_needed()
    for paper in SAMPLE_PAPERS:
        add_paper(**paper)
    print(f"\nDone! {len(SAMPLE_PAPERS)} papers loaded into memory.")
    print("You can now run: python src/agent.py")


if __name__ == "__main__":
    main()
