"""
Streamlit + LangChain Agent with Groq (llama3-8b-8192)
- Python 3.12 compatible
- Uses Streamlit secrets for GROQ_API_KEY
- Tools: calculator(expression: str), current_time()  [no args]
- LangChain 0.2 tool-calling agent
- Minimal chat UI
"""
import ast
import operator as op
from datetime import datetime
import streamlit as st

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentExecutor, create_tool_calling_agent

# -------------------- Streamlit setup --------------------
st.set_page_config(page_title="Groq LangChain Agent", page_icon="🤖", layout="centered")

# Read Groq API key from secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", None)
if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in secrets. Add it in .streamlit/secrets.toml or Streamlit Cloud → Secrets.")
    st.stop()

# -------------------- Safe calculator tool --------------------
# Support +,-,*,/,**,%, parentheses only
_ALLOWED = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos
}

def _eval_expr(node):
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

@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression. Allowed: +, -, *, /, **, %, parentheses. No variables or functions."""
    try:
        tree = ast.parse(expression, mode="eval")
        value = _eval_expr(tree.body)
        return str(value)
    except Exception as e:
        return f"Calculator error: {e}"

@tool
def current_time() -> str:
    """Return the current UTC date/time in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

TOOLS = [calculator, current_time]

# -------------------- LLM (Groq) --------------------
st.sidebar.header("Model Settings")
model_name = st.sidebar.selectbox(
    "Groq model",
    options=["llama-4-scout-17b-16e-instruct"],
    index=0,
)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.2, 0.1)

llm = ChatGroq(api_key=GROQ_API_KEY, model=model_name,
    temperature=temperature,
)

# -------------------- Prompt / Agent --------------------
system_prompt = st.sidebar.text_area(
    "System prompt",
    value=(
        "You are a helpful assistant. Use the provided tools (calculator, current_time) when they can improve accuracy. "
        "Be concise."
    ),
    height=100,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=TOOLS, verbose=True)

# -------------------- UI: Chat --------------------
st.title("🤖 Groq LangChain Agent")
st.caption("Powered by Groq `llama3-*` via LangChain tool-calling agent. Secrets used for API keys.")

if "chat_msgs" not in st.session_state:
    st.session_state.chat_msgs = []  # list[tuple[role, content]]

# Render history
for role, content in st.session_state.chat_msgs:
    with st.chat_message(role):
        st.markdown(content)

user_input = st.chat_input("Ask something (e.g., 'what time is it?' or 'compute (3+5)*2')")

def add_msg(role: str, content: str):
    st.session_state.chat_msgs.append((role, content))

if user_input:
    add_msg("user", user_input)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Invoke agent
    with st.chat_message("assistant"):
        try:
            result = agent_executor.invoke({"input": user_input})
            answer = result.get("output", "")
        except Exception as e:
            answer = f"Error: {e}"
        st.markdown(answer)
        add_msg("assistant", answer)

# Utilities
with st.expander("Session utils"):
    if st.button("Clear chat"):
        st.session_state.chat_msgs = []
        st.experimental_rerun()
    st.download_button(
        "Download chat (markdown)",
        data="\n\n".join([f"**{r.upper()}**: {c}" for r, c in st.session_state.chat_msgs]),
        file_name="chat_history.md",
        mime="text/markdown",
    )
