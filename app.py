import pandas as pd
import json
import ollama
import matplotlib.pyplot as plt
import gradio as gr

# 1. Setup the model and read the data
MODEL = "gemma4:e4b"  # أو "phi3:latest"

df = pd.read_csv("sample_sales.csv")
df["order_date"] = pd.to_datetime(df["order_date"])
df["month"] = df["order_date"].dt.to_period("M").astype(str)

# 2. Define the 5 tools with data type handling
def get_total_revenue():
    return {
        "total_revenue": float(round(df["total_price"].sum(), 2)),
        "orders": int(len(df)),
        "average_order_value": float(round(df["total_price"].mean(), 2)),
    }

def get_monthly_sales():
    grouped = df.groupby("month")["total_price"].sum().round(2)
    return {"monthly_sales": {str(k): float(v) for k, v in grouped.to_dict().items()}}

def get_top_products(n=5, order="top"):
    grouped = df.groupby("product_name")["total_price"].sum()
    grouped = grouped.sort_values(ascending=(order == "bottom")).head(n)
    return {"products": {str(k): float(v) for k, v in grouped.round(2).to_dict().items()}}

def get_customer_stats():
    per_customer = df.groupby("customer_name")["total_price"].sum()
    top_cust = {str(k): float(v) for k, v in per_customer.sort_values(ascending=False).head(5).round(2).to_dict().items()}
    return {
        "total_customers": int(df["customer_name"].nunique()),
        "top_customers": top_cust,
    }

def compare_periods(period1_start, period1_end, period2_start, period2_end):
    p1 = df[(df["order_date"] >= period1_start) & (df["order_date"] <= period1_end)]
    p2 = df[(df["order_date"] >= period2_start) & (df["order_date"] <= period2_end)]
    r1, r2 = float(p1["total_price"].sum()), float(p2["total_price"].sum())
    change = r2 - r1
    pct = round((change / r1) * 100, 1) if r1 else None
    return {
        "period_1_revenue": float(round(r1, 2)),
        "period_2_revenue": float(round(r2, 2)),
        "change": float(round(change, 2)),
        "change_percent": float(pct) if pct is not None else None,
    }

# 3. Link the tools and model schema
TOOLS = {
    "get_total_revenue": get_total_revenue,
    "get_monthly_sales": get_monthly_sales,
    "get_top_products": get_top_products,
    "get_customer_stats": get_customer_stats,
    "compare_periods": compare_periods,
}

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "get_total_revenue", "description": "Get total revenue, order count, and average order value.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_monthly_sales", "description": "Get revenue grouped by month.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "get_top_products", "description": "Get best or worst selling products.", "parameters": {"type": "object", "properties": {"n": {"type": "integer"}, "order": {"type": "string", "enum": ["top", "bottom"]}}}}},
    {"type": "function", "function": {"name": "get_customer_stats", "description": "Get customer count and top spending customers.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "compare_periods", "description": "Compare revenue between two date ranges (YYYY-MM-DD).", "parameters": {"type": "object", "properties": {"period1_start": {"type": "string"}, "period1_end": {"type": "string"}, "period2_start": {"type": "string"}, "period2_end": {"type": "string"}}, "required": ["period1_start", "period1_end", "period2_start", "period2_end"]}}},
]

SYSTEM_PROMPT = """You are an AI Business Analyst.
Never calculate numbers yourself - always call the right tool(s) to get real numbers.
If a tool returns an error, tell the user honestly instead of guessing.
If the question needs more than one tool, call them one after another.
Be concise and explain the numbers in plain business language.
"""

# 4. Interactive query function and comprehensive report
def ask_ollama(question, history=None, max_steps=5):
    history = history or []
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [{"role": "user", "content": question}]

    for step in range(max_steps):
        response = ollama.chat(model=MODEL, messages=messages, tools=TOOL_SCHEMAS)
        msg = response["message"]

        if not msg.get("tool_calls"):
            return msg["content"]

        messages.append(msg)
        for call in msg["tool_calls"]:
            name = call["function"]["name"]
            args = call["function"].get("arguments", {}) or {}
            result = TOOLS[name](**args) if name in TOOLS else {"error": f"unknown tool {name}"}
            messages.append({"role": "tool", "name": name, "content": json.dumps(result, default=str)})

    return "Couldn't finish in time - try a simpler question."

def analyze_business():
    facts = {
        "revenue": get_total_revenue(),
        "monthly_sales": get_monthly_sales(),
        "top_products": get_top_products(5, "top"),
        "worst_products": get_top_products(5, "bottom"),
        "customers": get_customer_stats(),
    }
    prompt = (
        "Here are calculated business facts as JSON. Do NOT recalculate anything. "
        "Write a short report with: Key Findings, Notable Changes, Recommendations.\n\n"
        + json.dumps(facts, indent=2, default=str)
    )
    response = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": "You are a business analyst. Interpret given facts, never invent numbers."},
        {"role": "user", "content": prompt},
    ])
    return response["message"]["content"]

# 5. Gradio interface with tabs
def chat_response(message, history):
    formatted_history = []
    for h in history:
        formatted_history.append({"role": "user", "content": h[0]})
        formatted_history.append({"role": "assistant", "content": h[1]})
    return ask_ollama(message, history=formatted_history)

with gr.Blocks(title="AI Business Analyst") as demo:
    gr.Markdown("# 📊 AI Business Analyst Agent")
    gr.Markdown("Ask questions about your sales data or generate an executive summary report automatically.")
    
    with gr.Tab("💬 Chat Analyst"):
        chatbot = gr.ChatInterface(
            fn=chat_response,
            examples=[
                "What is our total revenue?",
                "Which product sold the most?",
                "Compare sales between January and February"
            ]
        )
        
    with gr.Tab("📝 Executive Report"):
        report_btn = gr.Button("Generate Executive Report", variant="primary")
        report_output = gr.Markdown()
        report_btn.click(fn=analyze_business, outputs=report_output)

if __name__ == "__main__":
    demo.launch(share=False)