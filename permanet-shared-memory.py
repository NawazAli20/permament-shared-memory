# %% [markdown]
# # Create permanent memory shared with sessions

# %%
import os 
from dotenv import load_dotenv
load_dotenv()
# ## Get the llm model

# %%
from langchain.chat_models import init_chat_model
model = init_chat_model(model="openai/gpt-oss-120b", model_provider="groq")
DB_URL = os.getenv("DATABASE_URL")

# %%
# import 
import json 
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore 
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

# %%
@dataclass
class Context:
    user_id:str 

# %% [markdown]
# ## create save_memory tool

# %%
@tool 
def save_memory(key:str, value:str,runtime:ToolRuntime[Context])->str:
    """Save user information in description key format like name, course, id like that.
    replace previous value for the same key with new value
    """
    if runtime.store is None: 
        raise RuntimeError("No memory store is created")
    
    namespace = ("users",runtime.context.user_id,"memories")

    runtime.store.put(
        namespace=namespace,
        key=key,
        value={"value":value}
    )

    return f"Saved memory {key}={value}"

# %% [markdown]
# ## Create read_memory tool

# %%
@tool 
def read_memory(key:str, value:str,runtime:ToolRuntime[Context])->str:
    """Retriev the saved memory
    """
    if runtime.store is None: 
        raise RuntimeError("No memory store is created")
    
    namespace = ("users",runtime.context.user_id,"memories")

    items = runtime.store.search(namespace,limit=100)

    memories = {
        item.key: item.value["value"] for item in items
    }

    return json.dumps(memories) if memories else "NO saved memories"

# %% [markdown]
# ## Langchain agent 

# %%
from langchain.agents import create_agent

# %%
with (PostgresSaver.from_conn_string(DB_URL) as checkpointer, 
PostgresStore.from_conn_string(DB_URL) as store):
    #initalize the memory 
    
    checkpointer.setup() 
    store.setup() 

    agent = create_agent(
        model=model,
        tools=[save_memory,read_memory],
        checkpointer=checkpointer,
        store=store,
        context_schema=Context,
        system_prompt="""You are a helpful assistant. Save memory using save_memory tool 
        when you are asked to save information. Retrieve information using read_memory tool."""
    )

    from langchain.messages import HumanMessage

    context = Context(user_id="nawaz_ali")
    thread_id1 = {"configurable":{"thread_id":"1"}}
    thread_id2 = {"configurable":{"thread_id":"2"}}

    response1 = agent.invoke(
        {
            "messages":[
                HumanMessage(content="My name is Dr. Nawaz Ali, I live in Illinois and work at Bradley University")
            ]
        },
        config=thread_id1,
        context=context
    ) 
    response1["messages"][-1].pretty_print()


    response2 = agent.invoke(
        {
            "messages":[
                HumanMessage(content="Retrieve my updated information.")
            ]
        },
        config=thread_id2,
        context=context
    ) 
    response2["messages"][-1].pretty_print()



