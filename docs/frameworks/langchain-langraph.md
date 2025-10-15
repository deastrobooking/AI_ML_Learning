# LangChain & LangGraph Guide

A comprehensive guide to building LLM applications with LangChain and LangGraph.

## Table of Contents
1. [Introduction](#introduction)
2. [LangChain Basics](#langchain-basics)
3. [LangGraph Fundamentals](#langgraph-fundamentals)
4. [Building AI Agents](#building-ai-agents)
5. [RAG (Retrieval Augmented Generation)](#rag-retrieval-augmented-generation)
6. [Memory and State Management](#memory-and-state-management)
7. [Advanced Patterns](#advanced-patterns)
8. [Production Best Practices](#production-best-practices)
9. [Resources](#resources)

---

## Introduction

### What is LangChain?

LangChain is a framework for developing applications powered by language models. It provides:
- **Chains**: Sequences of calls to LLMs or other utilities
- **Agents**: LLMs that make decisions about which actions to take
- **Memory**: Persist state between calls
- **Document Loaders**: Load documents from various sources
- **Vector Stores**: Store and retrieve embeddings

### What is LangGraph?

LangGraph is a library for building stateful, multi-actor applications with LLMs. Key features:
- **State Management**: Built-in state persistence
- **Graphs**: Define complex workflows as directed graphs
- **Checkpointing**: Save and resume application state
- **Human-in-the-Loop**: Add human approval steps

### Installation

```bash
# LangChain
pip install langchain langchain-openai langchain-community

# LangGraph
pip install langgraph

# Additional dependencies
pip install chromadb  # Vector store
pip install faiss-cpu  # Alternative vector store
pip install tiktoken  # Token counting
pip install openai anthropic  # LLM providers
```

---

## LangChain Basics

### Simple LLM Call

```python
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7,
    openai_api_key="your-api-key"
)

# Simple call
messages = [
    SystemMessage(content="You are a helpful AI assistant."),
    HumanMessage(content="What is the capital of France?")
]

response = llm.invoke(messages)
print(response.content)
```

### Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate, PromptTemplate

# Simple template
template = PromptTemplate(
    input_variables=["topic"],
    template="Write a short poem about {topic}."
)

prompt = template.format(topic="artificial intelligence")

# Chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specializing in {domain}."),
    ("human", "{question}")
])

messages = chat_template.format_messages(
    domain="machine learning",
    question="What is gradient descent?"
)
```

### Chains

```python
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0.7)

# Simple chain
chain = LLMChain(
    llm=llm,
    prompt=template
)

result = chain.run(topic="Python programming")
print(result)

# Using LCEL (LangChain Expression Language)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

chain = (
    {"topic": RunnablePassthrough()}
    | template
    | llm
    | StrOutputParser()
)

result = chain.invoke("machine learning")
```

### Document Loaders

```python
from langchain.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load text file
loader = TextLoader("document.txt")
documents = loader.load()

# Load PDF
pdf_loader = PyPDFLoader("document.pdf")
pdf_docs = pdf_loader.load()

# Load directory
dir_loader = DirectoryLoader("./docs", glob="**/*.md")
all_docs = dir_loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_documents(documents)
```

---

## RAG (Retrieval Augmented Generation)

### Basic RAG Implementation

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# 1. Load documents
loader = TextLoader("knowledge_base.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# 4. Create retrieval chain
llm = ChatOpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 5. Query
query = "What is the main topic of the document?"
result = qa_chain.run(query)
print(result)
```

### Advanced RAG with Sources

```python
from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

result = qa_with_sources({"question": query})
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Multi-Query Retrieval

```python
from langchain.retrievers import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# Automatically generates multiple query variations
docs = retriever.get_relevant_documents(query)
```

---

## LangGraph Fundamentals

### Simple Graph

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# Define state
class State(TypedDict):
    messages: Annotated[list, operator.add]
    counter: int

# Define nodes (functions)
def node_1(state: State) -> State:
    return {"messages": ["Node 1 executed"], "counter": state["counter"] + 1}

def node_2(state: State) -> State:
    return {"messages": ["Node 2 executed"], "counter": state["counter"] + 1}

# Create graph
workflow = StateGraph(State)

# Add nodes
workflow.add_node("node1", node_1)
workflow.add_node("node2", node_2)

# Add edges
workflow.set_entry_point("node1")
workflow.add_edge("node1", "node2")
workflow.add_edge("node2", END)

# Compile
app = workflow.compile()

# Run
result = app.invoke({"messages": [], "counter": 0})
print(result)
```

### Conditional Routing

```python
from langgraph.graph import StateGraph, END

def router(state: State) -> str:
    """Decide which node to execute next"""
    if state["counter"] < 5:
        return "continue"
    else:
        return "end"

# Create workflow
workflow = StateGraph(State)

workflow.add_node("process", process_node)
workflow.set_entry_point("process")

# Add conditional edges
workflow.add_conditional_edges(
    "process",
    router,
    {
        "continue": "process",  # Loop back
        "end": END
    }
)

app = workflow.compile()
```

### Agent with LangGraph

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

# Define tools
def calculator(expression: str) -> str:
    """Calculate mathematical expressions"""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

def search(query: str) -> str:
    """Search for information"""
    # Implement your search logic
    return f"Search results for: {query}"

tools = [
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for mathematical calculations"
    ),
    Tool(
        name="Search",
        func=search,
        description="Useful for finding information"
    )
]

# Create agent
llm = ChatOpenAI(temperature=0)
agent = create_react_agent(llm, tools)

# Run agent
result = agent.invoke({
    "messages": [("user", "What is 25 * 17?")]
})

print(result["messages"][-1].content)
```

---

## Building AI Agents

### ReAct Agent Pattern

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.tools import Tool

# Define custom tools
def get_weather(location: str) -> str:
    """Get weather for a location"""
    return f"The weather in {location} is sunny and 72°F"

def get_news(topic: str) -> str:
    """Get news about a topic"""
    return f"Latest news about {topic}: [news content]"

tools = [
    Tool(
        name="WeatherTool",
        func=get_weather,
        description="Get current weather for a location. Input should be a city name."
    ),
    Tool(
        name="NewsTool",
        func=get_news,
        description="Get latest news about a topic. Input should be a topic or keyword."
    )
]

# Create agent
llm = ChatOpenAI(temperature=0, model="gpt-4")

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=PromptTemplate.from_template("""
    You are a helpful assistant. Use the available tools to answer questions.
    
    Tools: {tools}
    Tool Names: {tool_names}
    
    Question: {input}
    {agent_scratchpad}
    """)
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=5
)

# Execute
result = agent_executor.invoke({
    "input": "What's the weather in San Francisco and give me news about AI?"
})
```

### Multi-Agent System

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class MultiAgentState(TypedDict):
    messages: list
    current_agent: str
    result: str

def researcher_agent(state: MultiAgentState):
    """Research agent gathers information"""
    # Implement research logic
    return {
        "messages": state["messages"] + ["Research completed"],
        "current_agent": "writer",
        "result": "Research findings..."
    }

def writer_agent(state: MultiAgentState):
    """Writer agent creates content"""
    # Implement writing logic
    return {
        "messages": state["messages"] + ["Writing completed"],
        "current_agent": "done",
        "result": state["result"] + "\nWritten content..."
    }

# Create multi-agent workflow
workflow = StateGraph(MultiAgentState)

workflow.add_node("researcher", researcher_agent)
workflow.add_node("writer", writer_agent)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)

app = workflow.compile()

result = app.invoke({
    "messages": [],
    "current_agent": "researcher",
    "result": ""
})
```

---

## Memory and State Management

### Conversation Memory

```python
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationChain

# Buffer memory (stores all messages)
memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

conversation.predict(input="Hi, my name is Alice")
conversation.predict(input="What's my name?")

# Summary memory (summarizes old messages)
summary_memory = ConversationSummaryMemory(llm=llm)

conversation_with_summary = ConversationChain(
    llm=llm,
    memory=summary_memory
)
```

### Persistent Memory with LangGraph

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Create checkpointer
checkpointer = SqliteSaver.from_conn_string("./checkpoints.db")

# Compile with checkpointer
app = workflow.compile(checkpointer=checkpointer)

# Run with thread ID for persistence
config = {"configurable": {"thread_id": "user-123"}}
result = app.invoke(initial_state, config)

# Resume later with same thread_id
continued = app.invoke(next_input, config)
```

### Custom Memory

```python
from langchain.memory import BaseChatMemory
from langchain.schema import BaseMessage

class CustomMemory(BaseChatMemory):
    messages: list = []
    max_messages: int = 10
    
    def save_context(self, inputs, outputs):
        # Save conversation
        self.messages.append(inputs)
        self.messages.append(outputs)
        
        # Keep only recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def load_memory_variables(self, inputs):
        return {"history": self.messages}
    
    def clear(self):
        self.messages = []
```

---

## Advanced Patterns

### Streaming Responses

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    temperature=0
)

# Stream response
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### Parallel Execution

```python
from langchain_core.runnables import RunnableParallel

# Execute multiple chains in parallel
parallel_chain = RunnableParallel(
    summary=summarize_chain,
    translation=translate_chain,
    sentiment=sentiment_chain
)

results = parallel_chain.invoke({"text": "Your input text"})
```

### Error Handling

```python
from langchain_core.runnables import RunnableLambda

def safe_llm_call(input_text):
    try:
        return llm.invoke(input_text)
    except Exception as e:
        return f"Error: {str(e)}"

safe_chain = RunnableLambda(safe_llm_call)
```

### Custom Output Parsers

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class MovieRecommendation(BaseModel):
    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Movie genre")
    rating: float = Field(description="Rating out of 10")

parser = PydanticOutputParser(pydantic_object=MovieRecommendation)

prompt = PromptTemplate(
    template="Recommend a movie.\n{format_instructions}\n",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | llm | parser
result = chain.invoke({})
print(result.title, result.year)
```

---

## Production Best Practices

### 1. API Key Management

```python
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)
```

### 2. Rate Limiting

```python
from langchain.llms.base import BaseLLM
import time

class RateLimitedLLM(BaseLLM):
    def __init__(self, base_llm, calls_per_minute=10):
        self.base_llm = base_llm
        self.calls_per_minute = calls_per_minute
        self.last_call_time = 0
    
    def _call(self, prompt, stop=None):
        # Rate limiting logic
        time_since_last = time.time() - self.last_call_time
        if time_since_last < 60 / self.calls_per_minute:
            time.sleep((60 / self.calls_per_minute) - time_since_last)
        
        self.last_call_time = time.time()
        return self.base_llm._call(prompt, stop)
```

### 3. Caching

```python
from langchain.cache import InMemoryCache, SQLiteCache
from langchain.globals import set_llm_cache

# In-memory cache
set_llm_cache(InMemoryCache())

# Persistent cache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))
```

### 4. Logging and Monitoring

```python
from langchain.callbacks import StdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

callback_manager = CallbackManager([StdOutCallbackHandler()])

llm = ChatOpenAI(
    callback_manager=callback_manager,
    verbose=True
)
```

### 5. Testing

```python
def test_chain():
    # Test with known inputs
    test_input = "What is 2+2?"
    result = chain.invoke(test_input)
    assert "4" in result.lower()

def test_agent_tools():
    # Test tool execution
    result = calculator("2+2")
    assert result == "4"
```

---

## Resources

### Official Documentation
- [LangChain Docs](https://python.langchain.com/docs/get_started/introduction)
- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangChain API Reference](https://api.python.langchain.com/)

### Tutorials
- [LangChain Crash Course](https://python.langchain.com/docs/tutorials/)
- [Building RAG Applications](https://python.langchain.com/docs/tutorials/rag/)
- [LangGraph Tutorials](https://langchain-ai.github.io/langgraph/tutorials/)

### Community
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangChain Discord](https://discord.gg/langchain)
- [LangChain Twitter](https://twitter.com/LangChainAI)

### Example Projects
- [LangChain Templates](https://github.com/langchain-ai/langchain/tree/master/templates)
- [LangGraph Examples](https://github.com/langchain-ai/langgraph/tree/main/examples)

---

**Next Steps**: 
- Try the [Creating AI Agents](../tutorials/ai-agents.md) tutorial
- Explore [Building RAG Applications](../tutorials/rag-application.md)
- Learn about [LLMs Guide](../guides/llms.md)
