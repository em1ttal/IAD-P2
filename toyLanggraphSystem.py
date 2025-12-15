"""
============================================================================
LANGGRAPH DUTCH FISH AUCTION - ORCHESTRATED AGENT SYSTEM
============================================================================

This implements the Dutch auction system using LangGraph for agent orchestration.
It represents the evolution from peer-to-peer message passing (osBrain) to
a structured graph-based workflow where state is shared and managed centrally.

ARCHITECTURE:
-------------
The system is modeled as a state machine with the following nodes:
1. OPERATOR_NODE: Manages the auction flow, updates prices, and broadcasts items.
2. MERCHANTS_NODE: Represents the collective of buyer agents. Each merchant
   independently evaluates the current item using LLM reasoning.
3. EVALUATOR_NODE: Resolves the round - determines winners, handles transactions,
   and decides whether to continue (lower price) or move to next item.

GRAPH FLOW:
-----------
[START] -> [OPERATOR] -> [MERCHANTS] -> [EVALUATOR] --+
              ^                                       |
              |                                       |
              +-----------------(Loop)----------------+

STATE MANAGEMENT:
-----------------
A shared `AuctionState` object persists across the graph execution, containing:
- Inventory and current item details
- Current market price
- Merchant states (budgets, inventories, preferences)
- Bids placed in the current round
- Transaction logs

DIFFERENCES FROM CLASSICAL/LLM VERSIONS:
----------------------------------------
- Centralized State: Instead of distributed local states, we use a shared graph state.
- Synchronous Rounds: The workflow ensures all merchants bid before the auction proceeds.
- Structured Control Flow: Logic is defined by edges and conditional transitions.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import os
import csv
import json
import time
import random
import requests
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END

# ============================================================================
# CONFIGURATION
# ============================================================================
# LLM Configuration
OPENROUTER_API_KEY = "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"
LLM_MODEL = "mistralai/devstral-2512:free"
LLM_TIMEOUT = 10
LLM_URL = "https://openrouter.ai/api/v1/chat/completions"

# Simulation Parameters
NUM_MERCHANTS = 3
FISH_TYPES = ['H', 'S', 'T']  # Hake, Sole, Tuna
FISH_PER_SESSION = 5          # Number of items to auction

# Price Configuration
START_PRICE_MIN = 40
START_PRICE_MAX = 60
MIN_PRICE_MIN = 5
MIN_PRICE_MAX = 15
PRICE_DECREMENT = 5

# Output Configuration
RESULTS_DIR = "auction_results_langgraph"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATE_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SETUP_CSV = os.path.join(RESULTS_DIR, f"setup_{DATE_STR}.csv")
LOG_CSV = os.path.join(RESULTS_DIR, f"log_{DATE_STR}.csv")

# ============================================================================
# PERSONALITIES
# ============================================================================
PERSONALITIES = {
    'CAUTIOUS': {
        'description': 'Very conservative and risk-averse',
        'system_prompt': """You are a CAUTIOUS merchant in a Dutch fish auction.
You are conservative and risk-averse. You prefer to wait for prices to drop before buying.
You buy when the price is 30% or less of your budget, or when you desperately need a type for diversity.
If the price drops below 20, you should seriously consider buying if you need that type.
Always consider: Is this a reasonable price? Will I miss this opportunity if I wait?"""
    },
    'GREEDY': {
        'description': 'Aggressive and competitive',
        'system_prompt': """You are a GREEDY merchant in a Dutch fish auction.
You are aggressive and highly competitive. You want to buy quickly to beat other merchants.
You don't like losing auctions and prefer to secure items early, even at higher prices.
You're willing to pay premium prices to ensure you win, especially for items you need.
Always consider: Will someone else buy this before me? Should I act NOW?"""
    },
    'PREFERENCE_DRIVEN': {
        'description': 'Obsessed with preferred fish type',
        'system_prompt': """You are a PREFERENCE-DRIVEN merchant in a Dutch fish auction.
You are obsessed with your preferred fish type and prioritize it above all else.
You're willing to pay high prices for your favorite fish but are very reluctant to buy others.
You only buy non-preferred types if they're extremely cheap or absolutely necessary for diversity.
Always consider: Is this my favorite type? If not, is it REALLY worth it?"""
    },
    'BALANCED': {
        'description': 'Moderate and strategic',
        'system_prompt': """You are a BALANCED merchant in a Dutch fish auction.
You take a moderate, strategic approach. You balance price, need, and preference carefully.
You're neither neither too aggressive nor too passive. You make rational decisions based on all factors.
You consider the full picture: current budget, missing types, preferences, and price fairness.
Always consider: What's the most rational decision given all my goals?"""
    }
}

# ============================================================================
# STATE DEFINITION
# ============================================================================
class MerchantState(TypedDict):
    """
    Represents the state of an individual merchant (buyer) agent.
    Maintains their personal inventory, budget, and strategic preferences.
    """
    id: str                 # Unique identifier (e.g., "Merchant_1")
    personality: str        # LLM persona (CAUTIOUS, GREEDY, etc.)
    preference: str         # Preferred fish type (H, S, or T)
    budget: float           # Current remaining funds (starts at 100)
    inventory: List[Dict]   # List of items purchased so far
    types_owned: List[str]  # Set of unique fish types acquired (tracks diversity goal)

class ItemState(TypedDict):
    """
    Represents a single fish item in the auction inventory.
    """
    id: int                 # Unique product ID
    type: str               # Fish type (H, S, or T)
    start_price: int        # Initial high price for the Dutch auction
    min_price: int          # Lowest acceptable price (reserve price)
    current_price: int      # Price in the current tick/round (updates dynamically)
    status: str             # Current state: 'ACTIVE', 'SOLD', or 'DISCARDED'

class AuctionState(TypedDict):
    """
    Global shared state of the entire auction system.
    This object is passed between graph nodes (Operator -> Merchants -> Evaluator)
    and persists changes throughout the simulation.
    """
    # ---- SYSTEM STATUS ----
    inventory: List[ItemState]  # Full list of all items to be auctioned
    current_item_index: int     # Pointer to current item in inventory list
    is_auction_active: bool     # True if auction is running, False if finished
    
    # ---- CURRENT ROUND STATE ----
    current_item: Optional[ItemState] # The specific item currently being bid on
    current_price: float              # Current asking price for the item
    round_messages: List[str]         # Messages generated in this step (for display)
    
    # ---- AGENTS ----
    merchants: List[MerchantState]    # List of all participating buyer agents
    bids: Dict[str, str]              # Map of MerchantID -> Decision ('BUY' or 'WAIT')
                                      # Collected during the Merchants node execution
    
    # ---- LOGGING ----
    logs: List[str]                   # System-wide event log for console output

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def create_initial_state() -> AuctionState:
    """Initialize the auction state with random inventory and merchants."""
    
    # 1. Create Inventory
    inventory = []
    for i in range(FISH_PER_SESSION):
        s_price = random.randint(START_PRICE_MIN, START_PRICE_MAX)
        m_price = random.randint(MIN_PRICE_MIN, MIN_PRICE_MAX)
        if s_price <= m_price:
            s_price = m_price + 2 * PRICE_DECREMENT
            
        inventory.append({
            "id": i + 1,
            "type": random.choice(FISH_TYPES),
            "start_price": s_price,
            "min_price": m_price,
            "current_price": s_price,
            "status": "ACTIVE"
        })

    # 2. Create Merchants
    merchants = []
    personality_names = list(PERSONALITIES.keys())
    
    # Setup CSV logging
    try:
        with open(SETUP_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Merchant', 'Personality', 'Preference', 'Budget'])
            
            for i in range(NUM_MERCHANTS):
                m_name = f"Merchant_{i+1}"
                pers = personality_names[i % len(personality_names)]
                pref = random.choice(FISH_TYPES)
                budg = 100.0
                
                merchants.append({
                    "id": m_name,
                    "personality": pers,
                    "preference": pref,
                    "budget": budg,
                    "inventory": [],
                    "types_owned": []
                })
                
                writer.writerow([m_name, pers, pref, budg])
    except Exception as e:
        print(f"Error creating setup CSV: {e}")

    # Initialize Log CSV
    try:
        with open(LOG_CSV, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operator', 'Product', 'Type', 'Sale Price', 'Merchant'])
    except Exception as e:
        print(f"Error creating log CSV: {e}")

    return {
        "inventory": inventory,
        "current_item_index": 0,
        "is_auction_active": True,
        "current_item": None,
        "current_price": 0,
        "round_messages": [],
        "merchants": merchants,
        "bids": {},
        "logs": ["Auction System Initialized"]
    }

def call_llm_decision(merchant: MerchantState, item: ItemState, price: float) -> Dict:
    """Calls the LLM to decide whether to BUY or WAIT."""
    
    # Construct context for LLM
    types_missing = [t for t in FISH_TYPES if t not in merchant['types_owned']]
    
    system_prompt = PERSONALITIES[merchant['personality']]['system_prompt']
    user_prompt = f"""Current situation:
- Fish Type: {item['type']}
- Current Price: {price}
- My Budget: {merchant['budget']}
- My Preference: {merchant['preference']}
- Types I Own: {merchant['types_owned'] if merchant['types_owned'] else 'None'}
- Types Missing: {types_missing if types_missing else 'None (I have all types!)'}
- My Inventory Count: {len(merchant['inventory'])} fish
- Is Preferred Type: {'YES' if item['type'] == merchant['preference'] else 'NO'}

Should I buy this fish at the current price? Respond with your decision and reasoning."""

    # API Request
    try:
        response = requests.post(
            LLM_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "auction_decision",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["BUY", "WAIT"]},
                                "reason": {"type": "string"}
                            },
                            "required": ["action", "reason"],
                            "additionalProperties": False
                        }
                    }
                },
            },
            timeout=LLM_TIMEOUT
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return json.loads(content)
        else:
            return {"action": "WAIT", "reason": f"API Error: {response.status_code}"}
            
    except Exception as e:
        return {"action": "WAIT", "reason": f"LLM Exception: {str(e)[:30]}"}

def log_transaction(product_id, product_type, price, merchant_id):
    """Append a transaction to the CSV log."""
    try:
        with open(LOG_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            # Operator ID is 1 for this single-threaded orchestration
            writer.writerow([1, product_id, product_type, price, merchant_id])
    except Exception as e:
        print(f"Error logging transaction: {e}")

# ============================================================================
# GRAPH NODES
# ============================================================================
def operator_node(state: AuctionState) -> AuctionState:
    """
    OPERATOR NODE: Manages the start of rounds and item transitions.
    """
    logs = state["logs"]
    idx = state["current_item_index"]
    
    # Check if auction is finished
    if idx >= len(state["inventory"]):
        state["is_auction_active"] = False
        logs.append("Auction Finished: No more items in inventory.")
        return state

    current_item = state["inventory"][idx]
    
    # Initialize item price if starting new item
    if state["current_item"] is None or state["current_item"]["id"] != current_item["id"]:
        state["current_item"] = current_item
        state["current_price"] = current_item["start_price"]
        logs.append(f"--- NEW ITEM: {current_item['type']} (ID: {current_item['id']}) ---")
        logs.append(f"Starting Price: {state['current_price']} | Min Price: {current_item['min_price']}")
    
    # Broadcast (conceptually)
    state["round_messages"] = [
        f"OPERATOR: Selling {current_item['type']} at {state['current_price']}"
    ]
    print(f"\n[Operator] Auctioning {current_item['type']} at {state['current_price']}")
    
    return state

def merchants_node(state: AuctionState) -> AuctionState:
    """
    MERCHANTS NODE: Collects decisions from all merchants for the current price.
    In a real distributed system, these would run in parallel.
    Here we iterate through them to simulate the concurrent decision phase.
    """
    if not state["is_auction_active"]:
        return state
        
    current_item = state["current_item"]
    price = state["current_price"]
    bids = {}
    
    # Each merchant makes a decision
    for merchant in state["merchants"]:
        # Skip if budget insufficient
        if merchant["budget"] < price:
            bids[merchant["id"]] = "WAIT"
            print(f"  > {merchant['id']} ({merchant['personality']}): WAIT | Insufficient budget ({merchant['budget']} < {price})")
            continue
            
        # Call LLM for decision
        decision = call_llm_decision(merchant, current_item, price)
        action = decision.get("action", "WAIT")
        reason = decision.get("reason", "Unknown")
        
        bids[merchant["id"]] = action
        
        # Log decision with reason
        print(f"  > {merchant['id']} ({merchant['personality']}): {action} | {reason}")
        
        # Respect API rate limits (avoid 429) - very conservative for free tier
        time.sleep(20.0)
    
    state["bids"] = bids
    print(f"  [Bids] {bids}")
    return state

def evaluator_node(state: AuctionState) -> AuctionState:
    """
    EVALUATOR NODE: Resolves the auction round.
    Checks bids, determines winner, handles transaction, or updates price.
    """
    if not state["is_auction_active"]:
        return state
        
    bids = state["bids"]
    current_item = state["current_item"]
    price = state["current_price"]
    
    # 1. Check for Buyers
    buyers = [mid for mid, action in bids.items() if action == "BUY"]
    
    if buyers:
        # SOLD!
        # Pick a winner (randomly if multiple simultaneous bids, mimicking race condition)
        winner_id = random.choice(buyers)
        
        # Process Transaction
        for m in state["merchants"]:
            if m["id"] == winner_id:
                m["budget"] -= price
                m["inventory"].append(current_item)
                if current_item["type"] not in m["types_owned"]:
                    m["types_owned"].append(current_item["type"])
                break
        
        # Log
        msg = f"SOLD {current_item['type']} to {winner_id} for {price}"
        state["logs"].append(msg)
        print(f"[Evaluator] {msg}")
        log_transaction(current_item['id'], current_item['type'], price, winner_id)
        
        # Move to next item
        state["current_item_index"] += 1
        state["current_item"] = None # Reset for next cycle
        state["bids"] = {}
        
    else:
        # NO SALE
        # Check if we hit bottom price
        if price - PRICE_DECREMENT < current_item["min_price"]:
            # Discard Item
            msg = f"DISCARDED {current_item['type']} (Price {price} -> Min {current_item['min_price']})"
            state["logs"].append(msg)
            print(f"[Evaluator] {msg}")
            log_transaction(current_item['id'], current_item['type'], 0, "")
            
            # Move to next item
            state["current_item_index"] += 1
            state["current_item"] = None
            state["bids"] = {}
        else:
            # Continue Auction: Decrease Price
            state["current_price"] -= PRICE_DECREMENT
            # Stay on same item
    
    return state

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("="*60)
    print("LANGGRAPH FISH AUCTION SYSTEM")
    print("="*60)
    
    # 1. Build Graph
    workflow = StateGraph(AuctionState)
    
    workflow.add_node("operator", operator_node)
    workflow.add_node("merchants", merchants_node)
    workflow.add_node("evaluator", evaluator_node)
    
    workflow.add_edge(START, "operator")
    workflow.add_edge("operator", "merchants")
    workflow.add_edge("merchants", "evaluator")
    
    # Conditional Logic for Next Step
    def next_step(state: AuctionState):
        if not state["is_auction_active"]:
            return END
        return "operator"
        
    workflow.add_conditional_edges("evaluator", next_step)
    
    app = workflow.compile()
    
    # 2. Run Simulation
    print(f"Initializing {NUM_MERCHANTS} merchants and {FISH_PER_SESSION} items...")
    initial_state = create_initial_state()
    
    print(f"Logs will be saved to: {RESULTS_DIR}")
    
    # Execute Graph
    # We use stream to see progress, but for simplicity invoke is fine.
    # Given the potentially long execution time, we'll just run it.
    final_state = app.invoke(initial_state, config={"recursion_limit": 150})
    
    print("\n" + "="*60)
    print("AUCTION COMPLETED")
    print("="*60)
    
    # 3. Final Summary
    print("\nFinal Merchant Status:")
    for m in final_state["merchants"]:
        print(f"- {m['id']} ({m['personality']}): Budget {m['budget']}, Items: {len(m['inventory'])}, Types: {m['types_owned']}")

if __name__ == "__main__":
    main()
