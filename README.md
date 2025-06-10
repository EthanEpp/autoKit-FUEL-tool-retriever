
# AutoKit FUEL

**Feedback-driven Update & Enrichment through Lookup for LLM Agent Tool Discovery**

Created by Ethan Epp and Jonathan Cheng


---

## 🌟 Overview

**AutoKit FUEL** is a modular, self-improving system for discovering, retrieving, and maintaining high-quality tool metadata ("toolcards") for use by LLM agents. It acts as both:

* 🔍 A **toolfinder agent** that can interpret natural-language queries and return the best-fit tool for the task, and
* 🛠️ A **self-healing infrastructure** that maintains an up-to-date, verifiable tool knowledgebase by fixing broken links, enhancing metadata, and continuously enriching content with new discoveries.

It solves a common pain point in LLM-based applications: **keeping tool references correct, current, and easy to integrate**—without manual intervention.

---

## 💡 Motivation

With the growing ecosystem of agent tools and APIs, developers and agents alike struggle to:

* Discover new or suitable tools for their task
* Interpret or integrate tool APIs with limited or broken documentation
* Avoid “link rot” and obsolete tool metadata
* Maintain consistent, standardized, and searchable tool descriptions

**AutoKit FUEL** addresses these problems by creating a centralized, evolving toolcard repository—kept fresh via autonomous feedback loops and human-in-the-loop support.

---

## 🧠 Key Features

* **Tool Discovery Agent (AutoKit)**
  Uses a hybrid of RAG and ReAct strategies to retrieve the best LangChain or external tools based on natural language prompts.

* **FUEL Pipeline (Feedback-driven Update & Enrichment through Lookup)**
  Continuously:

  * Verifies documentation URLs
  * Repairs broken links
  * Enhances tool descriptions and metadata
  * Adds new tools from scratch when discovered via search

* **Self-Healing Toolcards**
  Simulates real-world decay by injecting broken URLs and automatically recovering 85%+ of them through a ReAct fixer agent.

* **Modular, Extensible Architecture**
  Built using LangChain, LangGraph, Anthropic Claude, and Tavily Search API with reusable and composable nodes.

* **User Feedback Loop**
  Collects input from users to either:

  * Generate stub code tailored to their use case, or
  * Reattempt search if the suggested tool was unsatisfactory.

---

## 🔧 How It Works

### Workflow Overview

```
User Prompt ➝ Query Rewriting ➝ Toolcard Retrieval ➝ Generation ➝ 
Evaluation (Grounding + Relevance) ➝ Web Fallback (if needed) ➝ 
Human Feedback ➝ Tool Addition ➝ Verification ➝ End
```

### Core Components

* `document_search`: Retrieves documents using a vectorstore (Chroma + OpenAI embeddings)
* `generate`: Selects a tool from retrieved docs using a custom RAG prompt ("ToolFinderGPT")
* `transform_query`: Improves vague queries for higher-recall retrieval
* `web_search`: Uses ReAct agent + Tavily to search the web and extract new toolcards
* `add_tool_to_database`: Adds new tools in standardized JSON format
* `verify_tool_entry`: Validates tool metadata against live documentation pages
* `human_feedback_satisfaction`: Interactive feedback collection
* `handle_positive_feedback`: Generates custom code stub for the use case
* `handle_negative_feedback`: Reattempts discovery with refined query
* `react_fixer_agent`: Repairs broken toolcards
* `verifier_chain`: Ensures tool metadata matches retrieved documentation

---

## 🚀 Getting Started

### 🧱 Requirements

* Python 3.10+
* OpenAI API key
* Anthropic Claude API key
* Tavily Search API key

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export TAVILY_API_KEY=...
```

### 🛠️ Run the Agent

To start the pipeline:

```python
from main import graph, pretty_print_graph_stream

inputs = {"messages": [("human", "I need a tool to summarize a PDF")]}
pretty_print_graph_stream(graph, inputs)
```

The agent will:

1. Search the vectorstore for a suitable tool
2. Use a ReAct web search if retrieval fails
3. Output a recommended tool, optionally generate starter code, and update the database

---

## 📈 Results Summary

### Toolcard Recovery

* 85% repair success rate across 40 corrupted toolcards
* Tool descriptions improved with more accurate class/module paths and detailed summaries

### Retrieval Quality

* Consistently produced coherent, grounded suggestions
* Rare hallucinations due to effective RAG grounding + hallucination grading

---

## 🔄 Future Work

* Integrate **Model Context Protocol (MCP)** for tool sharing across agents
* Implement tool **execution and validation**
* Add benchmarks for **retrieval accuracy and latency**
* Integrate **GitHub/Hub-type tool discovery** for broader ecosystem reach
* Enable agent self-improvement via **Reflexion-style loops**

---

## 📖 Citation

If you use this project in academic work:

```
@misc{autokit2025,
  title={AutoKit FUEL: Tool Retrieval Agent with Feedback-driven Update & Enrichment through Lookup},
  author={Epp, Ethan and Cheng, Jonathan},
  year={2025},
  howpublished={\url{https://github.com/EthanEpp/autoKit-FUEL-tool-retriever}},
  note={CMPSC 291A - UCSB}
}
```

---

## 📬 Contact

Feel free to reach out:

* Ethan Epp: [eepp@ucsb.edu](mailto:eepp@ucsb.edu)
* Jonathan Cheng: [jonathancheng@ucsb.edu](mailto:jonathancheng@ucsb.edu)

Or just ask chatGPT, it probably knows. Shoutout chatGPT
---
