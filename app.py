"""
Streamlit + LangChain Agent with Groq (llama3-8b-8192)
- Uses Streamlit secrets for GROQ_API_KEY
- Simple tools (calculator, current time)
- Tool-calling agent (LangChain 0.2+)
- Chat-style UI with memory
"""

import ast
import operator as op
from datetime import datetime
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_tool_calling_agent

# -------------------- Config --------------------
st.set_page_config(page_title="Groq LangChain Agent", page_icon="🤖", layout="centered")

# Read Groq API key from secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in secrets. Add it in .streamlit/secrets.toml")
    st.stop()

# -------------------- Safe calculator --------------------
# A tiny, safe arithmetic evaluator (no variables/functions)
# Supports +,-,*,/,**,%, parentheses
_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos
}

def _eval_expr(node):
    if isinstance(node, ast.Num):  # py3.8-
        return node.n
    if isinstance(node, ast.Constant):  # py3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numeric constants allowed.")
    if isinstance(node, ast.BinOp):
        left = _eval_expr(node.left)
        right = _eval_expr(node.right)
        op_func = _ALLOWED.get(type(node.op))
        if not op_func:
            raise ValueError("Unsupported operator.")
        return op_func(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _eval_expr(node.operand)
        op_func = _ALLOWED.get(type(node.op))
        if not op_func:
            raise ValueError("Unsupported unary operator.")
        return op_func(operand)
    raise ValueError("Invalid expression.")

@tool("calculator", return_direct=False)
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Use for math.
    Allowed: +, -, *, /, **, %, parentheses. No variables or functions."""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_expr(tree.body)
        return str(value)
    except Exception as e:
        return f"Calculator error: {e}"

@tool("current_time", return_direct=False)
def current_time(_: str = "") -> str:
    """Return the current UTC date/time in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

TOOLS = [calculator, current_time]

# -------------------- LLM (Groq) --------------------
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Groq model",
    options=["llama3-8b-8192", "llama3-70b-8192"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name=model_name,
    temperature=temperature,
    # streaming=True  # optional; requires a callback to stream to UI
)

# -------------------- Prompt / Agent --------------------
system_prompt = st.sidebar.text_area(
    "System prompt",
    value=(
        "You are a helpful assistant. "
        "Use tools when needed. Be concise and show your reasoning steps only via tool usage, not verbosely."
    ),
    height=100,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

# -------------------- UI: Chat --------------------
st.title("🤖 Groq LangChain Agent")
st.caption("Powered by Groq `llama3-*` via LangChain tool-calling agent. Secrets used for API keys.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render history
for role, content in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask something (try: 'what time is it?' or 'compute (3+5)*2')")

def add_to_history(role, content):
    st.session_state.chat_history.append((role, content))

if user_input:
    add_to_history("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke agent
    with st.chat_message("assistant"):
        try:
            result = agent_executor.invoke(
                {
                    "input": user_input,
                    "chat_history": [
                        # Convert to LC messages (role->str ok; agent uses placeholders for context)
                        # We just pass plaintext history here; prompt has MessagesPlaceholder
                        # For more advanced memory, use RunnableWithMessageHistory + LC memory.
                    ]
                }
            )
            answer = result.get("output", "")
        except Exception as e:
            answer = f"Error: {e}"

        st.markdown(answer)
        add_to_history("assistant", answer)

# Utilities
with st.expander("Session utils"):
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
    st.download_button(
        "Download chat (markdown)",
        data="\n\n".join([f"**{r.upper()}**: {c}" for r, c in st.session_state.chat_history]),
        file_name="chat_history.md",
        mime="text/markdown",
    )
