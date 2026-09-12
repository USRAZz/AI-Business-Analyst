# to run on terminal write: streamlit run app.py
import streamlit as st
import ollama
import json

st.set_page_config(page_title="AI Business Analyst", page_icon="📊", layout="wide")
st.title("📊 AI Business Analyst Assistant")

MODEL = "gemma4:e4b"

SYSTEM_PROMPT = """You are an expert AI Business Analyst.
Your role is to analyze sales, revenue, and product data to provide clear, actionable business insights.
When asked a question, use the available tools when necessary to fetch accurate data before answering."""

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_revenue_data",
            "description": "Retrieves revenue and sales metrics for specified time periods or products.",
            "parameters": {
                "type": "object",
                "properties": {
                    "period": {
                        "type": "string",
                        "description": "The time period to analyze, e.g., 'January', 'Q1', '2024'."
                    }
                },
                "required": ["period"]
            }
        }
    }
]

def execute_tool_call(tool_name, arguments):
    if tool_name == "get_revenue_data":
        period = arguments.get("period", "")
        return f"Revenue data for {period}: Total Sales = $150,000, Top Product = Product A."
    return "No tool output available."

def ask_ollama_agent(messages, max_steps=5):
    current_messages = messages.copy()
    
    for step in range(max_steps):
        response = ollama.chat(
            model=MODEL, 
            messages=current_messages, 
            tools=TOOL_SCHEMAS
        )
        msg = response.get("message", {})
        tool_calls = msg.get("tool_calls")
        
        if not tool_calls:
            return msg.get("content", "")
        
        current_messages.append(msg)
        for tool in tool_calls:
            function_info = tool.get("function", {})
            tool_name = function_info.get("name")
            raw_args = function_info.get("arguments", {})
            
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args
                
            tool_result = execute_tool_call(tool_name, args)
            current_messages.append({
                "role": "tool",
                "content": str(tool_result),
                "name": tool_name
            })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask a business question (e.g., Compare January and February revenue)..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    messages_payload = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": m["role"], "content": m["content"]} 
        for m in st.session_state.chat_history 
        if m["role"] in ["user", "assistant"]
    ]
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing business data..."):
            try:
                response_text = ask_ollama_agent(messages_payload)
                st.markdown(response_text)
                st.session_state.chat_history.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Error communicating with Ollama: {str(e)}")