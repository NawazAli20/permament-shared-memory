[![New Chat, Same Memory! LangChain + PostgreSQL](assets/yt-banner.png)](https://www.youtube.com/playlist?list=PLfpB0rPkNb_8)

# New Chat, Same Memory!

### Build an AI chatbot that remembers you across sessions with LangChain and PostgreSQL

[This Lecture video link](https://youtu.be/S71Ahw6pKb0?si=rfTtaDjrAOF0z2sC)
[Watch the LLM & Agentic AI lecture series](https://www.youtube.com/playlist?list=PLfpB0rPkNb_8)


## About this class

In the previous lecture, we used `PostgresSaver` to resume a conversation with the same `thread_id`. This lecture adds `PostgresStore` so a chatbot can save information about a user and recall it in a different session.

Our example uses one user, `nawaz_ali`, and two conversations, `session1` and `session2`. The conversations keep separate message histories while sharing saved user facts through the same namespace.

## What you will learn

- Distinguish thread-scoped conversation state from shared user memory.
- Pass a `user_id` through a context dataclass.
- Access the store and user context through `ToolRuntime`.
- Save and retrieve facts with custom tools.
- Recall saved information from another thread.
- Keep database connections open during agent execution.

## Two kinds of memory

| Component | Scope in this example | Purpose |
| --- | --- | --- |
| `PostgresSaver` | A conversation identified by `thread_id` | Persist messages and graph state so the conversation can resume |
| `PostgresStore` | A namespace containing `user_id` | Persist selected user facts that multiple conversations can access |

Both components use PostgreSQL for durable storage. Their scopes serve different purposes.

**Shared memory does not automatically merge chat histories.** This example explicitly saves facts such as a name, location and workplace. A chat history interface also needs application logic to list and authorize access to a user’s conversations.

## The demonstration

| Step | User | Thread | Action |
| --- | --- | --- | --- |
| 1 | `nawaz_ali` | `session1` | Save: “My name is Nawaz. I live in Illinois. I work at Bradley University.” |
| 2 | `nawaz_ali` | `session2` | Ask: “Find my information from the saved memory.” |

Both memory tools use this namespace:

```python
namespace = ("users", runtime.context.user_id, "memories")
```

For this user, it becomes:

```python
("users", "nawaz_ali", "memories")
```

`"users"` and `"memories"` are application-defined labels. The save and read tools must use the same namespace to access the same records. The thread ID does not appear in this namespace, which allows both sessions to share it.

## Setup

You need Python 3.10 or newer, a running PostgreSQL database, and access to a model that supports tool calling. The lecture uses Groq.

Install the packages in your project’s virtual environment:

```bash
python -m pip install langchain langchain-groq langgraph langgraph-checkpoint-postgres "psycopg[binary,pool]" python-dotenv
```

Create a local `.env` file:

```dotenv
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://your_user:your_password@localhost:5432/your_database
```

Create the database and grant your database user permission to create the required tables before running the example. Keep `.env` out of Git.

## Complete lecture example

Copy the code below into a Python file or VS Code notebook cell. Choose a tool-capable model available in your Groq account if the lecture’s model is unavailable.

```python
import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("Set DATABASE_URL in your .env file")

model = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="groq",
)


@dataclass
class Context:
    user_id: str


@tool
def save_memory(key: str, value: str, runtime: ToolRuntime[Context]) -> str:
    """Save a user fact under a descriptive key. Replace its previous value."""
    if runtime.store is None:
        raise RuntimeError("No memory store is configured")

    namespace = ("users", runtime.context.user_id, "memories")
    runtime.store.put(namespace, key, {"value": value})
    return f"Saved memory: {key}={value}"


@tool
def read_memory(runtime: ToolRuntime[Context]) -> str:
    """Read saved facts for the current user."""
    if runtime.store is None:
        raise RuntimeError("No memory store is configured")

    namespace = ("users", runtime.context.user_id, "memories")
    items = runtime.store.search(namespace, limit=100)
    memories = {item.key: item.value["value"] for item in items}
    return json.dumps(memories) if memories else "No saved memories"


with (
    PostgresSaver.from_conn_string(DB_URL) as checkpointer,
    PostgresStore.from_conn_string(DB_URL) as store,
):
    # Initialize the required tables on first use.
    checkpointer.setup()
    store.setup()

    agent = create_agent(
        model=model,
        tools=[save_memory, read_memory],
        checkpointer=checkpointer,
        store=store,
        context_schema=Context,
        system_prompt=(
            "You are a helpful assistant. Use save_memory when the user "
            "asks you to save information. Save each fact under a descriptive "
            "key. Use read_memory to answer questions about saved information."
        ),
    )

    user_context = Context(user_id="nawaz_ali")

    response_1 = agent.invoke(
        {"messages": [HumanMessage(content=(
            "My name is Nawaz and I live in Illinois. "
            "I work at Bradley University. Save my information."
        ))]},
        config={"configurable": {"thread_id": "session1"}},
        context=user_context,
    )
    response_1["messages"][-1].pretty_print()

    response_2 = agent.invoke(
        {"messages": [HumanMessage(
            content="Find my information from the saved memory"
        )]},
        config={"configurable": {"thread_id": "session2"}},
        context=user_context,
    )
    response_2["messages"][-1].pretty_print()

# Both database connections close after the with block finishes.
```

### Expected behavior

After the first call successfully saves the facts, the second session can retrieve them and answer with the user’s name, location and workplace. Exact wording and memory key names depend on the model’s tool calls.

To inspect those calls during the demonstration:

```python
for message in response_1["messages"]:
    message.pretty_print()
```

The `read_memory` tool has no `key` or `value` parameters because this version reads all saved facts, up to its limit of 100 items. Larger stores need pagination or a more selective retrieval strategy.

## Key code explained

| Code | Role |
| --- | --- |
| `Context(user_id="nawaz_ali")` | Supplies the current user identity to the agent’s runtime context |
| `config={"configurable": {"thread_id": "session1"}}` | Selects the conversation whose checkpoints the agent uses |
| `ToolRuntime[Context]` | Supplies runtime access to the context and store without asking the model to generate them |
| `store.put(namespace, key, {"value": value})` | Creates or updates a fact at the given namespace and key |
| `store.search(namespace, limit=100)` | Retrieves stored items from the user’s namespace |
| `checkpointer=checkpointer, store=store` | Gives the agent both conversation persistence and shared memory |

## Common error: “the connection is closed”

Keep `agent.invoke()` inside the `with` block. Exiting the block closes the database connections, even when PostgreSQL itself is still running. An agent created inside the block continues to reference those connections afterward.

In VS Code, rerun the full block to create fresh connections. Restarting PostgreSQL does not reopen an old Python connection object.

## Try it yourself

1. Add a favorite course in `session1`, then ask about it in `session2`.
2. Change the location and save it under the same key. Retrieve the updated value.
3. Create a third thread for the same user and ask about the workplace.
4. Use another `user_id` with a fresh thread and check that its namespace has no saved facts yet.
5. Restart the Python process, reconnect to the same database, and recall facts in a new thread using the original user ID.

For a production chatbot, derive `user_id` from the authenticated session and enforce conversation ownership. A namespace organizes records; it does not implement access control by itself.

## References

- [LangGraph: Add and manage memory](https://docs.langchain.com/oss/python/langgraph/add-memory)
- [LangChain: Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [PostgreSQL documentation](https://www.postgresql.org/docs/)

