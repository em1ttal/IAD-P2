import requests
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Optional, List
import json

# --- OpenRouter config ---
OPENROUTER_API_KEY = "TODO: Add your OpenRouter API key here"
MODEL = "openrouter/polaris-alpha"
URL = "https://openrouter.ai/api/v1/chat/completions"

# JSON schema for merchant decision
JSON_SCHEMA = {
    "name": "auction_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "description": "BUY or NOT_BUY"},
            "reason": {"type": "string", "description": "Short explanation"}
        },
        "required": ["action", "reason"],
        "additionalProperties": False
    }
}

# --- State Definition ---
class AuctionState(TypedDict):
    product: str
    price: float
    bottom_price: float
    merchant_budget: float
    merchant_decision: str
    logs: List[str]


# --- Nodes ---
def operator_agent(state: AuctionState):
    """
    Operator announces the current price.
    """
    state["logs"].append(f"Operator: {state['product']} at {state['price']}€")
    print(f"[Operator] {state['product']} at {state['price']}€")
    return state


def merchant_agent(state: AuctionState):
    """
    Merchant decides whether to buy using OpenRouter LLM with JSON schema.
    """
    prompt = (
        f"You are a merchant in a Dutch auction.\n"
        f"Current product: {state['product']}\n"
        f"Current price: {state['price']}€\n"
        f"Your budget: {state['merchant_budget']}€\n"
        "Decide whether to BUY or NOT_BUY.\n"
        "Respond strictly in JSON format according to the schema."
    )

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": JSON_SCHEMA
            }
        }
    )

    

    data = response.json()
    decision = data["choices"][0]["message"]["content"]

    decision =  json.loads(decision)

    state["merchant_decision"] = decision["action"]
    state["logs"].append(f"Merchant decision: {decision}")
    print(f"[Merchant] {decision}")

    # Update budget if bought
    if decision["action"] == "BUY" and state["merchant_budget"] >= state["price"]:
        state["merchant_budget"] -= state["price"]
        state["logs"].append(f"Merchant bought {state['product']} for {state['price']}€, remaining budget {state['merchant_budget']}€")

    return state


def evaluator_node(state: AuctionState):
    """
    Check decision and continue or end auction.
    """
    if state["merchant_decision"] == "BUY":
        state["logs"].append(f"{state['product']} SOLD at {state['price']}€")
        print(f"[Evaluator] {state['product']} SOLD at {state['price']}€")
        # just return state, edge will direct to END
        return state
    elif state["price"] <= state["bottom_price"]:
        state["logs"].append(f"{state['product']} not sold. Price reached bottom {state['bottom_price']}€")
        print(f"[Evaluator] {state['product']} not sold. Price reached bottom.")
        return state
    else:
        # Reduce price and loop back
        state["price"] *= 0.9
        print(f"[Evaluator] No sale. Lowering price to {state['price']:.2f}€")
        state["merchant_decision"] = ""  # reset for next round
        return state

# --- Build LangGraph ---
graph = StateGraph(AuctionState)
graph.add_node("operator", operator_agent)
graph.add_node("merchant", merchant_agent)
graph.add_node("evaluator", evaluator_node)

graph.add_edge(START, "operator")
graph.add_edge("operator", "merchant")
graph.add_edge("merchant", "evaluator")
graph.add_conditional_edges("evaluator", lambda state: "operator" if state["merchant_decision"] != "BUY" and state["price"] > state["bottom_price"] else END)

auction_app = graph.compile()

# --- Run Example ---
initial_state = AuctionState(
    product="Tuna",
    price=50.0,
    bottom_price=20.0,
    merchant_budget=60.0,
    merchant_decision="",
    logs=[]
)

final_state = auction_app.invoke(initial_state)

print("\n=== AUCTION LOG ===")
for entry in final_state["logs"]:
    print(entry)
