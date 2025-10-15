import re
from typing import Annotated, Iterator, Literal, TypedDict
import re
import json
from typing import Annotated, Iterator, TypedDict, Literal
import getpass
import os
import requests
from langchain import hub
from langchain_community.document_loaders import web_base
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.you import YouSearchTool
from langchain_community.utilities.you import YouSearchAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, convert_to_messages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langgraph.graph import END, StateGraph, add_messages
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field

import json
from typing import Dict, List


MAX_RETRIES = 3
VERBOSE = True
# Set up model retriever and tools
def load_tool_urls_from_json(path: str) -> list[str]:
    """
    Read a JSON array of tool definitions from `path` and return the list of 'documentation' URLs.
    Each JSON object is expected to have a "documentation" field containing the URL string.
    """
    with open(path, "r", encoding="utf-8") as f:
        tools_list = json.load(f)
    # tools_list is expected to be a list of dicts; extract "documentation" from each
    urls = []
    for entry in tools_list:
        doc_url = entry.get("documentation")
        if isinstance(doc_url, str) and doc_url.strip():
            urls.append(doc_url.strip())
            print(doc_url)
    return urls

# Point to your local improved_tools.json
TOOL_JSON_PATH = "./improved_tools.json"
TOOL_DOC_URLS = load_tool_urls_from_json(TOOL_JSON_PATH)


NEWLINE_RE = re.compile("\n+")

class ToolDocsLoader:
    def __init__(self, url: str):
        self.url = url

    def load(self) -> list[Document]:
        # (Example: scrape tool name, description, usage code, etc.)
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(self.url)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Example heuristic—adjust based on actual page structure:
        tool_name = soup.find("h1").get_text().strip()
        description_paragraphs = soup.find_all("p", limit=2)
        usage_blocks = soup.find_all("pre")
        
        pieces = [f"Tool: {tool_name}"]
        for p in description_paragraphs:
            pieces.append("Description: " + p.get_text().strip())
        for code_block in usage_blocks:
            pieces.append("Example:\n" + code_block.get_text())
        page_text = "\n\n".join(pieces) + f"\n\nSource: {self.url}"
        return [Document(page_content=page_text, metadata={"source": self.url})]


def prepare_tool_documents(urls: list[str]) -> list[Document]:
    # Choose whether to split or not. If each tool doc is already small, skip splitting.
    all_docs: list[Document] = []
    for url in urls:
        print(url)
        loader = ToolDocsLoader(url)
        page_docs = loader.load()
        all_docs.extend(page_docs)
    
    # Option A: If you want to split large pages into chunks:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            r"^Tool:",      # whenever you see “Tool:” at the start of a line
            r"\n\n+"        # or multiple blank lines
        ],
        is_separator_regex=True,
        chunk_size=800
    )
    return text_splitter.split_documents(all_docs)


def get_tool_retriever() -> BaseRetriever:
    print(TOOL_DOC_URLS)
    documents = prepare_tool_documents(TOOL_DOC_URLS)
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="tool-rag-chroma",
        embedding=OpenAIEmbeddings(),
    )
    return vectorstore.as_retriever()



llm = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
print("step1")
# retriever = get_retriever()
retriever = get_tool_retriever()
print("step2")

tavily_search_tool = TavilySearchResults(max_results=3)

print("step3")

#Set up graph state
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    documents: list[Document]
    candidate_answer: str
    retries: int
    web_fallback: bool
    searched: bool
    user_feedback: str
    sample_code: str


class GraphConfig(TypedDict):
    max_retries: int


# document search
def document_search(state: GraphState):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    if VERBOSE:
        print("---RETRIEVE---")

    question = convert_to_messages(state["messages"])[-1].content

    # Retrieval
    documents = retriever.invoke(question)
    # retrieved = len(documents) > 0


    return {"documents": documents, "question": question, "web_fallback": True, "searched": False}

# Tool RAG
# RAG_PROMPT: ChatPromptTemplate = hub.pull("rlm/rag-prompt")

TOOL_RAG_SYSTEM = """
You are “ToolFinderGPT,” an assistant that helps users pick the best LangChain/third-party tool for their use-case.  
You will be given:

1. A list of “Documents” (each one is a tool description, including the tool’s name, a one-sentence summary, a usage example, and the source URL).  
2. A user’s query (e.g. “I need a tool to fetch Wikipedia articles.” “I want a tool to summarize a URL.”).

Your job:
- Choose exactly one tool from the Documents that best matches the user’s use-case.
- Output:

    Recommended Tool: <ToolName>
    Description: <one-sentence description from the Document>
    
    ```python
    <import‐and‐call stub>
    ```

If the retrieved Documents do not contain any suitable tool, say “I’m sorry, I couldn’t find a tool that meets your requirements.” 
"""

TOOL_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", TOOL_RAG_SYSTEM),
    ("user", "Context (tool descriptions):\n\n{context}\n\nUser’s query: {question}\n\nProvide your answer:")
])


def generate(state: GraphState):
    """
    Generate answer

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    if VERBOSE:
        print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    retries = state["retries"] if state.get("retries") is not None else -1

    # rag_chain = RAG_PROMPT | llm | StrOutputParser()
    # generation = rag_chain.invoke({"context": documents, "question": question})
    tool_rag_chain = TOOL_RAG_PROMPT | llm | StrOutputParser()
    generation = tool_rag_chain.invoke({"context": documents, "question": question})
    return {"retries": retries + 1, "candidate_answer": generation}

# Query Rewriter
QUERY_REWRITER_SYSTEM = """
You a question re-writer that converts an input question to a better version that is optimized for vectorstore tool retrieval.
Look at the input and try to reason about the underlying semantic intent / meaning—specifically, transform natural language “I need X” into a concise “tool for X” query.
"""


QUERY_REWRITER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_REWRITER_SYSTEM),
        (
            "human",
            "Here is the initial question: \n\n {question} \n Formulate an improved question.",
        ),
    ]
)

def transform_query(state: GraphState):
    """
    Transform the query to produce a better question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates question key with a re-phrased question
    """
    if VERBOSE:
        print("---TRANSFORM QUERY---")

    question = state["question"]

    # Re-write question
    query_rewriter = QUERY_REWRITER_PROMPT | llm | StrOutputParser()
    better_question = query_rewriter.invoke({"question": question})
    return {"question": better_question}

# React Web Search
def tavily_search_fn(query: str) -> str:
    """
    Calls the TavilySearchResults tool under the hood.
    Returns a plain text blob (all hit contents joined with separators).
    """
    # invoke() expects a string query and returns a list of dicts with "content"
    results = tavily_search_tool.invoke(query)
    if not results:
        return "No results found."
    # Join all the result["content"] fields into one long text string
    return "\n\n".join([hit["content"] for hit in results])


# 2.1) A brief system‐prompt that tells Claude to act as a “tool‐finder” agent.
REACT_PROMPT = """
You are “ToolFinderAgent,” a ReAct agent whose sole job is to search for and locate an LLM‐agent‐tool (or general API) that fulfills the user’s request/goal.
You do ​not​ need to already know the tool’s name—you will search for it.  
Your search strategy must be:

1. First attempt to find a suitable tool by querying LangChain or LangGraph documentation.  
   • For example, if the user wants a “calculator,” your first query might be “langchain calculator tool” or “langgraph calculator tool”  
   • To do that, call the tool `tavily_search_fn` with that LangChain/LangGraph‐specific query.

2. If the LangChain/LangGraph query yields no relevant results, broaden your search to “any tool or API” that can accomplish the goal.  
   • For instance, if you can’t find a LangChain calculator, you might search “LLM Calculator Tool Api” or “online calculator API.”  
   • Again, use `tavily_search_fn` for this broader query.

3. If the tool API query yields no relevant results, think critically and creatively about if there are any other tools that we can use to accomplish this goal, that aren't exactly what they initially asked for, but can be utilized to accomplish the goal.  
   • For instance, if you can’t find a tool calculator API, you reason that you can also do calculator based calculations in a python script so you might search “Python Environmet Tool API”.
   • Again, use `tavily_search_fn` for this broader query.

4. Each time you use `tavily_search_fn`, show your “Thought:” with the exact query you intend to run, then run `Action: tavily_search_fn` with that query.  
   • After you see `Observation: ...` (the returned text), think again (“Is this result relevant?”).  
   • If relevant, you may stop searching. If not, think of a refined or broader query and call `tavily_search_fn` again.

5. Once you have identified a successful documentation link or tool name, output exactly one final “Answer:” message in plain text (no code fences).  
   • That “Answer:” should specify which tool or API you found, and include its documentation URL.  
   • Do not output any JSON—just human‐readable text like:  
     “Answer: I found the ‘LangChain Python Calculator Tool’ at https://python.langchain.com/docs/integrations/tools/calculator/ which fulfills your request.”  

Remember:  
- You may call `tavily_search_fn(...)` multiple times if necessary  
- Always start with a LangChain/LangGraph‐specific query before widening to general APIs  
- Show Thought/Action/Observation for each step so the chain of reasoning is transparent.  
- Always return the link to the documentation of the tool if you find one.
"""

react_agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20240620",
    tools=[tavily_search_fn],
    prompt=REACT_PROMPT
)

def web_search(state: GraphState) -> dict:
    """
    Run a ReAct agent that uses tavily_search_fn to find (or reason about) a tool.
    """
    if VERBOSE:
        print("---RUNNING ReAct‐BASED TOOL SEARCH---")

    question = state["question"]
    if VERBOSE:
        print("QUESTION:", question)
    messages = [
        HumanMessage(content=question)
    ]
    
    # 1) Invoke the ReAct agent by passing {"question": question}.
    #    The agent’s internal prompt is REACT_PROMPT, which expects {question}.
    try:
        result = react_agent.invoke({"messages": messages})
    except Exception as e:
        # If the agent fails for any reason, bail out with an empty Document
        if VERBOSE:
            print("ReAct agent failed:", e)
        return {
            "documents": state.get("documents", []),
            "web_fallback": False,
            "searched": False
        }

    # 2) The agent returns either a dict with "content" or directly a string
    if isinstance(result, dict) and "content" in result:
        agent_response = result["content"].strip()
    else:
        agent_response = str(result).strip()

    if VERBOSE:
        print("AGENT RESPONSE:", agent_response)

    # 3) Wrap that response in a Document
    new_doc = Document(page_content=agent_response, metadata={"source": "react_tool_search"})
    documents = state.get("documents", []) + [new_doc]

    # 4) Indicate that we did perform a “search”
    return {
        "documents": documents,
        "web_fallback": False,
        "searched": True
    }

# Standard web search (old method)
SEARCH_QUERY_SYSTEM = """
You are a search‐query optimizer specifically for Tavily Search.  
When given a user’s question (and optional context), your job is to produce a concise, Tavily‐friendly search query
that will maximize relevant results.  You will only be searching for LLM Agent Tool api and documentation. Do NOT return any explanation—just output the single line of text
that should be sent to Tavily.
"""


SEARCH_QUERY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SEARCH_QUERY_SYSTEM),
    ("user", "User question: {question}\n\n---\nGenerate a Tavily search query:")
])


def format_search_query(state: GraphState) -> dict:
    """
    Take state["question"] and ask Claude to rewrite it into a concise Tavily‐optimized query.
    Returns: { "search_query": <rewritten string> }.
    """
    if VERBOSE:
        print("---FORMATTING SEARCH QUERY (no context)---")

    question = state["question"]
    # Now our prompt only needs `{"question": ...}` because we removed {context} above.
    search_query_chain = SEARCH_QUERY_PROMPT | llm | StrOutputParser()
    better_query = search_query_chain.invoke({"question": question})

    if VERBOSE:
        print(f"→ Formatted search query: {better_query!r}")
    return {"search_query": better_query}


def trad_web_search(state: GraphState) -> dict:
    """
    Perform a Tavily search using an LLM‐rewritten query (no context from state["documents"]).
    """
    if VERBOSE:
        print("---RUNNING WEB SEARCH (with optimized query)---")

    # 3.1) Rewrite just the question
    formatted = format_search_query(state)
    if formatted != None:
        searched = True
    else:
        searched = False

    search_query = formatted["search_query"]
    print("SEARCH QUERY:", search_query)
    # 3.2) Call Tavily with the rewritten query
    search_results = tavily_search_tool.invoke(search_query)

    # 3.3) Package Tavily’s hits into a single Document
    search_content = "\n\n".join([r["content"] for r in search_results])
    new_doc = Document(page_content=search_content, metadata={"source": "tavily"})

    # 3.4) Append to state["documents"] and disable further web‐fallback
    documents = state.get("documents", [])
    documents.append(new_doc)
    
    return {"documents": documents, "web_fallback": False, "searched": searched}


# Output format
def finalize_response(state: GraphState):
    if VERBOSE:
        print("---FINALIZING THE RESPONSE---")

    return {"messages": [AIMessage(content=state["candidate_answer"])]}

# Adding tools to databse
import json
import re

TOOL_ADDER_SYSTEM = """
You are a tool adder system that takes an input containing a structured description of a tool, and generates a stub in a specific format that will be added to a json of tools.
"""

TASK = """Here is the tool description: \n\n {description} \n Create a single stub for this tool based on this description. Format your response exactly the same as the following examples. Here are three examples of tool stubs:\n\n
{{
  "name": "OpenAPI Toolkit",
  "description": "Call any REST API via OpenAPI specs",
  "programming_language": "python",
  "module": "langchain_experimental.toolkits.open_api",
  "class": "OpenAPIToolkit",
  "init_args": {{ "spec_paths": "./manifests" }},
  "openapi": "./manifests",
  "pricing": "Free (tooling only)",
  "documentation": "https://python.langchain.com/docs/integrations/tools/openapi/"
}},
{{
  "name": "NLA Toolkit",
  "description": "Natural-language API invocation",
  "programming_language": "python",
  "module": "langchain_experimental.toolkits.nla",
  "class": "NLAToolkit",
  "init_args": {{ "nla_url": "${{NLA_SERVER_URL}}" }},
  "openapi": null,
  "pricing": "Free",
  "documentation": "https://python.langchain.com/docs/integrations/tools/openapi_nla/"
}},
{{
  "name": "Zapier NLA",
  "description": "Zapier Natural Language Actions integration",
  "programming_language": "python",
  "module": "langchain.tools.zapier",
  "class": "ZapierNLATool",
  "init_args": {{ "api_key": "${{ZAPIER_NLA_KEY}}" }},
  "openapi": null,
  "pricing": "Paid (Zapier plan)",
  "documentation": "https://python.langchain.com/docs/integrations/tools/zapier/"
}},
...
"""

TOOL_ADDER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", TOOL_ADDER_SYSTEM),
        (
            "human",
            TASK,
        ),
    ]
)

def add_tool_to_database(state: GraphState):
    """
    Adds a new tool to the local database of tools if the pipeline used an external search.

    Args:
        state (dict): The current graph state

    Returns:
        Nothing, but adds a new entry to the local database of tools
    """

    if not state.get("searched", False):
        print("STATE SEARCHED")
        return state
    
    if VERBOSE:
        print("---ADDING TOOL TO DATABASE---")

    # Generate the tool stub
    tool_adder = TOOL_ADDER_PROMPT | llm | StrOutputParser()

    tool_stub = tool_adder.invoke({"description": state["candidate_answer"]})

    # Parse the tool stub for the essential information
    # Since you're using StrOutputParser(), tool_stub is already a string
    text = tool_stub.strip()
    
    # if VERBOSE:
    #     print(f"Generated tool stub: {text}")
    
    # Try to parse the JSON directly first
    try:
        tool_stub_dict = json.loads(text)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON block using regex
        # Find the outermost JSON object - start from first { and find matching }
        start = text.find('{')
        if start == -1:
            print("No opening brace found in response.")
            print(f"Raw response: {text}")
            return state
            
        # Count braces to find the matching closing brace
        brace_count = 0
        end = start
        for i, char in enumerate(text[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    end = i
                    break
        
        if brace_count != 0:
            print("No matching closing brace found.")
            print(f"Raw response: {text}")
            return state
            
        json_text = text[start:end+1]
        
        # if VERBOSE:
        #     print(f"Extracted JSON: {json_text}")
            
        try:
            tool_stub_dict = json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"Extracted JSON: {json_text}")
            print(f"Raw response: {text}")
            return state
    
    # add the tool to the database
    # Load the existing file
    try:
        with open("improved_tools.json", "r") as f:
            tool_list = json.load(f)
    except FileNotFoundError:
        tool_list = []  # Create empty list if file doesn't exist

    # Check if a tool with the same name already exists
    new_tool_name = tool_stub_dict.get("name", "")
    existing_names = [tool.get("name", "") for tool in tool_list]
    
    if new_tool_name in existing_names:
        if VERBOSE:
            print(f"Tool '{new_tool_name}' already exists in database. Skipping addition.")
        return state
    
    # Append the new stub
    tool_list.append(tool_stub_dict)

    # Write back the file
    with open("improved_tools.json", "w") as f:
        json.dump(tool_list, f, indent=4)
    
    if VERBOSE:
        print(f"Successfully added tool '{new_tool_name}' to database.")

    return state


# Verifying and fixing tool card database

def extract_main_text(html: str) -> str:
    """
    Given the raw HTML string `html`, return only the "main content" text.
    1) If there is a <main> tag, return its text (stripped of excess whitespace).
    2) Otherwise, get all text, but remove everything between
       'Skip to main content' and 'On this page' (inclusive).
    """
    soup = BeautifulSoup(html, "html.parser")

    # 1) If there's a <main> element, use that:
    main_tag = soup.find("main")
    if main_tag:
        return "\n".join(line.strip() for line in main_tag.get_text().splitlines() if line.strip())

    # 2) Otherwise, fall back to full-text minus the "Skip to main content" → "On this page" block:
    all_text = soup.get_text(separator="\n")

    start_marker = "Skip to main content"
    end_marker   = "On this page"

    start_idx = all_text.find(start_marker)
    end_idx   = all_text.find(end_marker)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Remove the entire chunk from start_marker up to end_marker
        cleaned = all_text[:start_idx] + all_text[end_idx + len(end_marker):]
    else:
        cleaned = all_text

    # Clean up duplicate blank lines or leading/trailing whitespace
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    return "\n".join(lines)


            
class ToolStub(BaseModel):
    name: str
    description: str
    programming_language: str
    module: str
    class_: str = Field(alias="class")  # `class` is a Python keyword, so alias it
    init_args: Dict[str, Any]
    openapi: Optional[str]            # JSON `null` → Python None
    pricing: Optional[str]  
    documentation: Optional[str] 

class VerifiedToolStub(ToolStub):
    verified: bool

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
import json

# ────────────────────────────────────────────────────────────────────────────────
# 1) ReAct prompt for the “fixer” agent
# ────────────────────────────────────────────────────────────────────────────────

FIXER_REACT_PROMPT = """
You are “ToolFixerAgent,” a ReAct agent whose job is to take an outdated tool stub (provided below as the entire user message) 
and find the correct, up‐to‐date documentation URL (and any changed fields) for that tool.  
You do not assume the original “documentation” URL still works.

Whenever you receive the user’s message (which is the entire outdated tool JSON), follow these steps EXACTLY:

1. Thought: Read the JSON stub (fields: name, description, programming_language, module, class, init_args, openapi, pricing, documentation).
   Extract the “name” field and formulate a concise LangChain/LangGraph–specific search query, e.g. “langchain <name> documentation” 
   or “langgraph <name> docs.”

2. Action: Call tavily_search_fn with that query.  
   You must write exactly:
     Action: tavily_search_fn
     Input: "<your query here>"

3. Observation: You will get back some text from Tavily. Inspect it.  
   • If you find a valid documentation URL or updated import hints, stop.  
   • Otherwise, Thought: “No LangChain page found—broaden to any Python <name> API”  
     then Action: tavily_search_fn with “Python <name> API” (or similar).

4. Repeat Thought/Action/Observation until you identify a working documentation URL 
   (and any changed class/module imports).

5. Finally, output exactly one line starting with “Answer: ” followed by a single JSON object 
   that matches the `ToolStub` schema (all fields—name, description, programming_language, module, 
   class, init_args, openapi, pricing, documentation—must appear).  
   • If you absolutely cannot find a valid documentation URL, set `"documentation": null` 
     and leave other fields as originally or defaulted.

Do NOT output any text outside of “Thought: …”, “Action: …”, “Observation: …”, 
and the final “Answer: { … }” line.  
"""

# ────────────────────────────────────────────────────────────────────────────────
# 2) Create the ReAct‐style fixer agent (one tool: tavily_search_fn)
# ────────────────────────────────────────────────────────────────────────────────
react_fixer_agent = create_react_agent(
    model="anthropic:claude-3-5-sonnet-20240620",
    tools=[tavily_search_fn],
    prompt=FIXER_REACT_PROMPT
)



VERIFIER_SYSTEM = """
You are a LLM agent tool‐verifier assistant.  A JSON object describing a third‐party agent tool (name, description, module, pricing, documentation URL, etc.) is provided, along with the text of its official documentation page. These tools are used for extending LLM agent capabilities via an api.  Your job is to:

1. Check each field in the JSON (name, description, programming_language, module, class, init_args, openapi, pricing, documentation) against the documentation page.
2. If any field is incorrect or out‐of‐date (e.g. pricing has changed, the description could be richer, the “class” or import path has moved, etc.), fix it. You should also add to or expand on the description to make it clearer about what the tool does.
3. If you believe a field is already correct and up‐to‐date, it is okay to leave it as‐is.
4. Return exactly one JSON object (no extra commentary, no markdown fences) that contains the “fully verified” or “updated” stub.  

You may assume the documentation page text is accurate and up‐to‐date.  If you cannot find evidence for a particular field (for example, pricing isn’t mentioned), leave the original value as‐is, but set a new key `"verified": false` to signal uncertainty.  Otherwise, set `"verified": true`.  If you
"""

VERIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", VERIFIER_SYSTEM),
    (
        "user",
        "Here is the original tool JSON:\n\n{tool_json}\n\n"
        "Here is the TEXT of its documentation page (HTML stripped to plaintext):\n\n{page_text}\n\n"
        "---\n"
        "Produce the updated JSON stub now:"
    ),
])
# 1) Build a chain that outputs VerifiedToolStub
verifier_chain = (VERIFIER_PROMPT 
                  | llm.with_structured_output(VerifiedToolStub))


def verify_tool_entry(state: GraphState) -> GraphState:
    """
    Verifies (and potentially updates) each tool stub in improved_tools.json by fetching its
    documentation URL. If fetching or parsing fails, the original stub is moved to missing_tools.json.
    Otherwise, a VerifiedToolStub is created and written to improved_tools.json.
    """
    stubtemp = False
    
    # if (state.get("searched") != False) && stubtemp:
    if stubtemp: #dont burn credits
        if VERBOSE:
            print("---VERIFYING TOOL ENTRIES---")

        # 1) Load improved_tools.json (all stubs to verify)
        try:
            with open("improved_tools.json", "r", encoding="utf-8") as f:
            # with open("broken_tools.json", "r", encoding="utf-8") as f:
                tool_list: List[Dict] = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            if VERBOSE:
                print("improved_tools.json not found or invalid → skipping verification.")
                # print("broken_tools.json not found or invalid → skipping verification.")
            return state

        if not tool_list:
            if VERBOSE:
                print("improved_tools.json is empty → skipping.")
                # print("broken_tools.json is empty → skipping.")
            return state

        # 2) Iterate over each stub
        for original_stub in tool_list:
            name = original_stub.get("name", "<unknown>")
            doc_url = original_stub.get("documentation")
            if VERBOSE:
                print(f"\n---VERIFYING TOOL: {name}---")
                print("DOC_URL:", doc_url)

            # If there's no documentation URL, move to missing
            if not doc_url:
                if VERBOSE:
                    print(f"No documentation URL for '{name}' → marking as missing.")
                _add_to_missing_and_fix(original_stub)
                continue  # proceed to next stub

            # 3) Try to fetch and extract main content
            try:
                resp = requests.get(doc_url, timeout=10)
                resp.raise_for_status()
                html = resp.text
                page_text = extract_main_text(html)

                # If extract_main_text returned empty or whitespace only, treat as failure
                if not page_text.strip():
                    if VERBOSE:
                        print(f"Fetched page for '{name}', but extracted text is empty → marking as missing.")
                    _add_to_missing_and_fix(original_stub)
                    continue

            except Exception as e:
                if VERBOSE:
                    print(f"Error fetching or parsing page at {doc_url}: {e}")
                _add_to_missing_and_fix(original_stub)
                continue

            # 4) Prepare inputs for the verifier chain
            stub_json_str = json.dumps(original_stub, indent=2)
            verifier_input = {
                "tool_json": stub_json_str,
                "page_text": page_text[:8000]  # truncate if very long
            }

            # 5) Invoke structured verifier (might raise if validation fails)
            try:
                verified_stub: VerifiedToolStub = verifier_chain.invoke(verifier_input)
            except Exception as e:
                if VERBOSE:
                    print(f"Verification LLM failed for '{name}': {e}")
                _add_to_missing_and_fix(original_stub)
                continue

            # 6) Load (or create) improved_tools.json, with JSONDecodeError handling
            try:
                with open("improved_tools.json", "r", encoding="utf-8") as f:
                    improved_list: List[Dict] = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                improved_list = []

            # 7) Convert Pydantic → dict (including “verified” field)
            # new_dict = verified_stub.dict(by_alias=True)
            new_dict = verified_stub.model_dump(by_alias=True)

            # 8) Replace or append in improved_tools.json
            existing_names = [t.get("name") for t in improved_list]
            if verified_stub.name in existing_names:
                idx = existing_names.index(verified_stub.name)
                improved_list[idx] = new_dict
                if VERBOSE:
                    print(f"Replaced entry for '{verified_stub.name}' in improved_tools.json.")
            else:
                improved_list.append(new_dict)
                if VERBOSE:
                    print(f"Appended '{verified_stub.name}' to improved_tools.json.")

            # 9) Write back improved_tools.json
            with open("improved_tools.json", "w", encoding="utf-8") as f:
                json.dump(improved_list, f, indent=4)

            if VERBOSE:
                print(f"Successfully wrote verified stub for '{verified_stub.name}'.")

    return state

import json
from typing import Dict, List
from langchain_core.messages import HumanMessage, AIMessage

def _add_to_missing_and_fix(tool_stub: Dict) -> None:
    """
    1) Append tool_stub to missing_tools.json if not already present.
    2) Invoke react_fixer_agent; normalize its return so that whether AIMessage.content is a string or list,
       we end up with a single text blob. Extract the JSON after "Answer:", validate via ToolStub, and 
       write to fixed_tools.json.
    """
    name = tool_stub.get("name", "<unknown>")

    # ──────────────────────────────────────────────────────────────────────────
    # Part A: Append to missing_tools.json
    # ──────────────────────────────────────────────────────────────────────────
    try:
        with open("missing_tools.json", "r", encoding="utf-8") as f:
            missing_list: List[Dict] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        missing_list = []

    existing_names = [t.get("name") for t in missing_list]
    if name not in existing_names:
        missing_list.append(tool_stub)
        with open("missing_tools.json", "w", encoding="utf-8") as f:
            json.dump(missing_list, f, indent=4)
        if VERBOSE:
            print(f"Added '{name}' to missing_tools.json.")
    else:
        if VERBOSE:
            print(f"'{name}' already in missing_tools.json; skipping append.")

    # ──────────────────────────────────────────────────────────────────────────
    # Part B: Run the React fixer agent
    # ──────────────────────────────────────────────────────────────────────────
    if VERBOSE:
        print(f"Launching react_fixer_agent for '{name}'...")

    # 1) Build a single HumanMessage with old stub JSON
    old_json_str = json.dumps(tool_stub, indent=2)
    messages_in = [HumanMessage(content=old_json_str)]

    # 2) Invoke the agent
    try:
        result = react_fixer_agent.invoke({"messages": messages_in})
    except Exception as e:
        if VERBOSE:
            print(f"react_fixer_agent failed for '{name}': {e}")
        return

    # 3) Normalize into a list of messages
    if isinstance(result, dict) and "messages" in result:
        messages_out = result["messages"]
    elif isinstance(result, list):
        messages_out = result
    else:
        if VERBOSE:
            print(f"Unexpected return type from fixer agent for '{name}': {type(result)}")
        return

    # 4) Extract the first AIMessage's content, handling str vs list
    ai_content = None
    for msg in messages_out:
        if isinstance(msg, AIMessage):
            content = msg.content
            # If it's already a string, just strip
            if isinstance(content, str):
                ai_content = content.strip()
            # If it's a list (of dicts or other), attempt to join any 'text' fields
            elif isinstance(content, list):
                pieces = []
                for element in content:
                    if isinstance(element, dict) and "text" in element:
                        pieces.append(element["text"])
                    else:
                        pieces.append(str(element))
                ai_content = "\n".join(pieces).strip()
            else:
                # Fallback: stringify
                ai_content = str(content).strip()
            break

    if ai_content is None:
        if VERBOSE:
            print(f"No AIMessage found in fixer output for '{name}'. Aborting fix.")
        return

    if VERBOSE:
        print("Fixer agent normalized content:\n", ai_content)

    # 5) Find “Answer:” and parse JSON after it
    answer_prefix = "Answer:"
    idx = ai_content.find(answer_prefix)
    if idx == -1:
        if VERBOSE:
            print(f"No 'Answer:' found in fixer output for '{name}'. Aborting fix.")
        return

    json_part = ai_content[idx + len(answer_prefix):].strip()
    try:
        fixed_stub_dict = json.loads(json_part)
    except json.JSONDecodeError as e:
        if VERBOSE:
            print(f"Failed to parse JSON from fixer output for '{name}': {e}")
            print("JSON part was:", json_part)
        return

    # 6) Validate via Pydantic ToolStub
    try:
        fixed_stub_obj: ToolStub = ToolStub.model_validate(fixed_stub_dict)
    except Exception as e:
        if VERBOSE:
            print(f"Pydantic validation failed for fixed stub '{name}': {e}")
        return

    # 7) Load or create fixed_tools.json
    try:
        with open("fixed_tools.json", "r", encoding="utf-8") as f:
            fixed_list: List[Dict] = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        fixed_list = []

    # 8) Replace or append
    final_name = fixed_stub_obj.name
    existing_names_fixed = [t.get("name") for t in fixed_list]
    if final_name in existing_names_fixed:
        idx2 = existing_names_fixed.index(final_name)
        fixed_list[idx2] = fixed_stub_obj.model_dump(by_alias=True)
        if VERBOSE:
            print(f"Replaced '{final_name}' in fixed_tools.json.")
    else:
        fixed_list.append(fixed_stub_obj.model_dump(by_alias=True))
        if VERBOSE:
            print(f"Appended '{final_name}' to fixed_tools.json.")

    # 9) Write back to fixed_tools.json
    with open("fixed_tools.json", "w", encoding="utf-8") as f:
        json.dump(fixed_list, f, indent=4)
    if VERBOSE:
        print(f"Successfully wrote fixed stub for '{final_name}' to fixed_tools.json.")

# Human in the loop feedback

# if yes - you can provide some stub code to fill in so we can generate code for them
# if no - do another react search to find a better tool, with the user feedback
    # put an intermediate node to collect extra user feedback

def human_feedback_satisfaction(state: GraphState) -> GraphState:
    """
    Collect human feedback on the generated answer: checks if the tool found is satisfactory for the user.
    Updates the state with the user's feedback (yes or no).
    """
    if VERBOSE:
        print("---COLLECTING HUMAN FEEDBACK---")
    
    user_input = input("Did the provided tool address your needs? (yes/no): ").strip().lower()

    print("USER FEEDBACK:", user_input)
    state["user_feedback"] = user_input
    return state

CODE_GENERATOR_SYSTEM = """
You are a code generator system that takes an input containing a structured description of a tool and a specific use case, and generates code for that use case, using the provided tool.
"""

CODE_GENERATOR_TASK = "Here is the tool description and a specific use case for this tool: \n Description: \n{tool_description} \n Use case: \n {use_case} \n Generate code that uses the provided tool to address the specific use case."

CODE_GENERATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CODE_GENERATOR_SYSTEM),
        (
            "human",
            CODE_GENERATOR_TASK,
        ),
    ]
)

# collect more user feedback on the tool found
# generate stub code for the tool found - use another LLM call to generate a code stub that addresses the specific use-case for the tool found
def handle_positive_feedback(state: GraphState) -> GraphState:
    """
    Handle the case where the user is satisfied with the tool found.
    In this case, we will prompt the user for the specific use-case for the tool found, and then generate a code stub that addresses the specific use-case.
    """
    if VERBOSE:
        print("---HANDLING POSITIVE FEEDBACK---")

    user_input = input("Please describe your specific use-case for the tool found so we can generate some starter code for you: ").strip()
    code_generator = CODE_GENERATOR_PROMPT | llm | StrOutputParser()
    code_stub = code_generator.invoke({"tool_description": state["candidate_answer"], "use_case": user_input})
    state["sample_code"] = code_stub

    # if VERBOSE:
    #     print("GENERATED CODE: ", code_stub)
    
    return state

# collect more user feedback on the tool found
# run another ReAct search with the user's feedback
def handle_negative_feedback(state: GraphState) -> GraphState:
    """
    Handle the case where the user is not satisfied with the tool found.
    This function can be extended to run another ReAct search with the user's feedback.
    """
    if VERBOSE:
        print("---HANDLING NEGATIVE FEEDBACK---")
    
    user_feedback = input("Please provide more details on what you were looking for: ").strip()
    state["question"] += "Based on the user feedback and the previously retrieved tool, find a better tool than the previously retrieved one.\n User Feedback: {user_feedback} \n Previously Retrieved Tool: {previous_tool}".format(user_feedback=user_feedback, previous_tool=state["candidate_answer"])
    return state

# Reflexion 
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


HALLUCINATION_GRADER_SYSTEM = (
"""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no', where 'yes' means that the answer is grounded in / supported by the set of facts.

IF the generation includes code examples, make sure those examples are FULLY present in the set of facts, otherwise always return score 'no'.
"""
)

HALLUCINATION_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", HALLUCINATION_GRADER_SYSTEM),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


ANSWER_GRADER_SYSTEM = (
"""
You are a grader assessing whether an answer addresses / resolves a question.
Give a binary score 'yes' or 'no', where 'yes' means that the answer resolves the question.
"""
)

ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ANSWER_GRADER_SYSTEM),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

def grade_generation_v_documents_and_question(state: GraphState, config) -> Literal["generate", "transform_query", "web_search", "finalize_response"]:
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    documents = state["documents"]
    generation = state["candidate_answer"]
    web_fallback = state["web_fallback"]
    retries = state["retries"] if state.get("retries") is not None else -1
    max_retries = config.get("configurable", {}).get("max_retries", MAX_RETRIES)

    # this means we've already gone through web fallback and can return to the user
    if not web_fallback:
        return "finalize_response"

    if VERBOSE:
        print("---CHECK HALLUCINATIONS---")

    hallucination_grader = HALLUCINATION_GRADER_PROMPT | llm.with_structured_output(GradeHallucinations)
    hallucination_grade: GradeHallucinations = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    # Check hallucination
    if hallucination_grade.binary_score == "no":
        if VERBOSE: print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "generate" if retries < max_retries else "web_search"

    if VERBOSE:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")

    # Check question-answering
    answer_grader = ANSWER_GRADER_PROMPT | llm.with_structured_output(GradeAnswer)
    answer_grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})

    if answer_grade.binary_score == "yes":
        if VERBOSE: print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "finalize_response"
    else:
        if VERBOSE: print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "transform_query" if retries < max_retries else "web_search"
    
def check_final_response(state: GraphState, config) -> Literal["add_tool_to_datase", "END"]:
    """
    Determines whether the final response the LLM gave contains a valid tool stub, or if the LLM could not find a valid tool.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """
    question = state["question"]
    generation = state["candidate_answer"]

    if VERBOSE:
        print("---CHECK FINAL RESPONSE---")

    # Check question-answering
    answer_grader = ANSWER_GRADER_PROMPT | llm.with_structured_output(GradeAnswer)
    answer_grade: GradeAnswer = answer_grader.invoke({"question": question, "generation": generation})
    if answer_grade.binary_score == "yes":
        if VERBOSE: print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "finalize_response"
    else:
        if VERBOSE: print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "END"
    
# conditional edge determining which state to execute next based on user feedback
def handle_user_feedback(state: GraphState) -> Literal["handle_positive_feedback", "handle_negative_feedback", "finalize_response"]:
    """
    Handle the user's feedback on the tool found.
    If the user is satisfied, finalize the response.
    If not, run another ReAct search with the user's feedback.
    """
    if VERBOSE:
        print("---HANDLING USER FEEDBACK---")

    feedback_state = state["user_feedback"]
    
    if feedback_state == "yes":
        return "handle_positive_feedback"
    elif feedback_state == "no":
        return "handle_negative_feedback"
    else:
        print("Invalid input. Please respond with 'yes' or 'no'.")
        return "finalize_response"  # No change, just return current state
    
# Build the graph

workflow = StateGraph(GraphState, config_schema=GraphConfig)

# Define the nodes
workflow.add_node("document_search", document_search)
workflow.add_node("generate", generate)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search", web_search)
workflow.add_node("finalize_response", finalize_response)
workflow.add_node("add_tool_to_database", add_tool_to_database) 
workflow.add_node("verify_tool_entry", verify_tool_entry)

workflow.add_node("human_feedback_satisfaction", human_feedback_satisfaction)
workflow.add_node("handle_positive_feedback", handle_positive_feedback)
workflow.add_node("handle_negative_feedback", handle_negative_feedback)

# Build graph
workflow.set_entry_point("document_search")
workflow.add_edge("document_search", "generate")
workflow.add_edge("transform_query", "document_search")
workflow.add_edge("web_search", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question
)

# after getting the candidate answer, check with the user to see if they are satisfied with the tool found
workflow.add_edge("finalize_response", "human_feedback_satisfaction")
workflow.add_conditional_edges(
    "human_feedback_satisfaction",
    handle_user_feedback
)
workflow.add_edge("handle_positive_feedback", "add_tool_to_database")
workflow.add_edge("handle_negative_feedback", "web_search")

# After adding to the database, verify that new stub
workflow.add_edge("add_tool_to_database", "verify_tool_entry")

# Once verification is complete, we END the pipeline
workflow.add_edge("verify_tool_entry", END)




# Compile
graph = workflow.compile()



# Visualize the graph
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))


def pretty_print_graph_stream(graph, inputs, config=None):
    """
    Consume graph.stream(...) and print each step in a human-readable way.
    Knows how to format:
      • document_search
      • generate
      • transform_query
      • web_search
      • finalize_response
      • human_feedback_satisfaction
      • handle_positive_feedback
      • handle_negative_feedback
      • add_tool_to_database
      • verify_tool_entry
    Any other nodes will be dumped as a raw dict.
    """
    for step in graph.stream(inputs, config or {}):
        # step is like {"document_search": {...}} or {"human_feedback_satisfaction": {...}}, etc.
        for node_name, result in step.items():
            # 1) document_search
            if node_name == "document_search":
                docs = result.get("documents", [])
                q = result.get("question", "")
                print(f"[document_search] Retrieved {len(docs)} document(s) for question: {q!r}")

            # 2) generate
            elif node_name == "generate":
                answer = result.get("candidate_answer", "").strip()
                retries = result.get("retries", 0)
                print(f"[generate] (retry #{retries})\n{answer}\n")

            # 3) transform_query
            elif node_name == "transform_query":
                new_q = result.get("question", "")
                print(f"[transform_query] Rewrote question to: {new_q!r}")

            # 4) web_search
            elif node_name == "web_search":
                docs = result.get("documents", [])
                searched = result.get("searched", False)
                if searched and docs:
                    src = docs[-1].metadata.get("source", "<unknown>")
                    print(f"[web_search] Appended web‐result from {src!r} (now {len(docs)} total docs).")
                else:
                    print(f"[web_search] Ran web search but no new docs found.")

            # 5) finalize_response
            elif node_name == "finalize_response":
                msgs = result.get("messages", [])
                for msg in msgs:
                    print(f"[finalize_response] AI: {msg.content.strip()}\n")

            # 6) human_feedback_satisfaction
            elif node_name == "human_feedback_satisfaction":
                feedback = result.get("user_feedback", "<no feedback>")
                print(f"[human_feedback_satisfaction] User answered: {feedback!r}")

            # 7) handle_positive_feedback
            elif node_name == "handle_positive_feedback":
                code = result.get("sample_code", "").strip()
                print("[handle_positive_feedback] Generated code stub:\n")
                print(code + "\n")

            # 8) handle_negative_feedback
            elif node_name == "handle_negative_feedback":
                new_q = result.get("question", "")
                print(f"[handle_negative_feedback] Updated question for next search:\n  {new_q!r}")

            # 9) add_tool_to_database
            elif node_name == "add_tool_to_database":
                # We check if "searched" was True in the result
                if result.get("searched", False):
                    print("[add_tool_to_database] Attempted to add a new tool stub (searched=True).")
                else:
                    print("[add_tool_to_database] No new tool was added (searched=False).")

            # 10) verify_tool_entry
            elif node_name == "verify_tool_entry":
                print("[verify_tool_entry] Finished verifying tools. Check JSON files for missing/fixed/improved stubs.")

            # 11) Any other node → raw dump
            else:
                print(f"[{node_name!r}] {result}")

        print("─" * 60)

VERBOSE = True
tool_request = "STUB TOOL REQUEST GOES HERE"
inputs = {"messages": [("human", tool_request)]}
pretty_print_graph_stream(graph, inputs)
