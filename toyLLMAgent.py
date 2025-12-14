"""
============================================================================
LLM-AUGMENTED DUTCH FISH AUCTION - MULTI-AGENT SYSTEM
============================================================================

This extends the classical Dutch auction with LLM-powered merchant reasoning.
Each merchant now has two decision layers:
1. Reactive layer: Python + osBrain (message passing, state management)
2. Cognitive layer: LLM reasoning (strategic decision-making with personality)

MERCHANT PERSONALITIES:
-----------------------
1. CAUTIOUS: Conservative bidder, waits for low prices, risk-averse
2. GREEDY: Aggressive bidder, buys quickly to beat competitors
3. PREFERENCE-DRIVEN: Focused on preferred fish, ignores others unless bargain
4. BALANCED: Moderate strategy, balances all goals evenly

Each personality is encoded in the LLM system prompt.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import time
import csv
import random
import os
import json
import requests
from datetime import datetime
from osbrain import run_agent, run_nameserver, Agent
import osbrain

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
osbrain.config['SERIALIZER'] = 'json'

# ============================================================================
# LLM CONFIGURATION
# ============================================================================
OPENROUTER_API_KEY = "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"
# Valid free models on OpenRouter (remove 'openrouter/' prefix):
# - "mistralai/devstral-2512:free"
# - "qwen/qwen-2.5-7b-instruct:free"
# - "meta-llama/llama-3.2-3b-instruct:free"
LLM_MODEL = "mistralai/devstral-2512:free"
LLM_TIMEOUT = 10  # seconds (increased for free models)

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
NUM_OPERATORS = 2
NUM_MERCHANTS = 4  # Increased to show different personalities
FISH_TYPES = ['H', 'S', 'T']
FISH_PER_OPERATOR = 5

START_PRICE_MIN = 40
START_PRICE_MAX = 60
MIN_PRICE_MIN = 5
MIN_PRICE_MAX = 15
PRICE_DECREMENT = 5
TICK_INTERVAL = 1.0  # Slightly longer to allow LLM processing

# ============================================================================
# PERSONALITY DEFINITIONS
# ============================================================================
PERSONALITIES = {
    'CAUTIOUS': {
        'description': 'Very conservative and risk-averse',
        'system_prompt': """You are a CAUTIOUS merchant in a Dutch fish auction.
You are very conservative and risk-averse. You prefer to wait for prices to drop significantly before buying.
You only buy when you're getting a good deal, and you're careful not to overspend.
You're patient and don't mind missing opportunities if the price isn't right.
Always consider: Is this price truly a bargain? Can I wait longer for a better deal?"""
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
You're neither too aggressive nor too passive. You make rational decisions based on all factors.
You consider the full picture: current budget, missing types, preferences, and price fairness.
Always consider: What's the most rational decision given all my goals?"""
    }
}

# ============================================================================
# GLOBAL VARIABLES FOR INITIALIZATION
# ============================================================================
# We use global variables to pass initialization data to avoid remote calls
GLOBAL_LOG_FILE = ""
GLOBAL_SETUP_FILE = ""

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
RESULTS_DIR = "auction_results_llm"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATE_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SETUP_CSV = os.path.join(RESULTS_DIR, f"setup_{DATE_STR}.csv")
LOG_CSV = os.path.join(RESULTS_DIR, f"log_{DATE_STR}.csv")

# ============================================================================
# LLM HELPER FUNCTION
# ============================================================================
def call_llm_for_decision(system_prompt, user_prompt, timeout=LLM_TIMEOUT):
    """
    Call the LLM API to get a buying decision.
    
    Returns:
        dict with 'action' ('BUY' or 'WAIT') and 'reason' (explanation)
        Returns {'action': 'WAIT', 'reason': 'LLM error'} on failure
    """
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
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
                                "action": {
                                    "type": "string",
                                    "enum": ["BUY", "WAIT"],
                                    "description": "Decision to buy or wait"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Brief explanation of the decision"
                                }
                            },
                            "required": ["action", "reason"],
                            "additionalProperties": False
                        }
                    }
                }
            },
            timeout=timeout
        )
        
        data = response.json()
        
        # Check for API errors
        if 'error' in data:
            error_msg = data['error'].get('message', 'Unknown API error')
            print(f"⚠️  API Error: {error_msg}")
            return {'action': 'WAIT', 'reason': f'API error: {error_msg[:30]}'}
        
        # Check if response has expected structure
        if 'choices' not in data:
            print(f"⚠️  Unexpected API response: {data}")
            return {'action': 'WAIT', 'reason': 'Invalid API response'}
        
        decision_str = data["choices"][0]["message"]["content"]
        decision = json.loads(decision_str)
        
        # Validate the decision
        if decision.get('action') not in ['BUY', 'WAIT']:
            return {'action': 'WAIT', 'reason': 'Invalid LLM response'}
            
        return decision
        
    except requests.exceptions.Timeout:
        return {'action': 'WAIT', 'reason': 'LLM timeout'}
    except KeyError as e:
        print(f"⚠️  API response missing key: {e}")
        return {'action': 'WAIT', 'reason': f'Missing key: {str(e)[:30]}'}
    except Exception as e:
        print(f"⚠️  LLM Error: {type(e).__name__}: {str(e)[:100]}")
        return {'action': 'WAIT', 'reason': f'{type(e).__name__}: {str(e)[:30]}'}

# ============================================================================
# OPERATOR CLASS
# ============================================================================
class Operator(Agent):
    def on_init(self):
        self.bind('PUB', alias='market')
        self.bind('PULL', alias='sales', handler='handle_buy')
        self.inventory = []
        self.current_item_idx = 0
        self.current_price = 0
        self.auction_active = False
        self.is_sold = False
        # Use global variable for log file
        self.log_filename = GLOBAL_LOG_FILE
        self.operator_id = 1

    def set_operator_id(self, op_id):
        self.operator_id = op_id
    
    def init_inventory(self, num_fish, start_product_id):
        self.inventory = []
        for i in range(num_fish):
            start_price = random.randint(START_PRICE_MIN, START_PRICE_MAX)
            min_price = random.randint(MIN_PRICE_MIN, MIN_PRICE_MAX)
            
            if start_price <= min_price:
                start_price = min_price + 2 * PRICE_DECREMENT
            
            self.inventory.append({
                'id': start_product_id + i,
                'type': random.choice(FISH_TYPES),
                'start_price': start_price,
                'min_price': min_price
            })

    def start_auction(self):
        self.auction_active = True
        if len(self.inventory) > 0:
            self.current_price = self.inventory[0]['start_price']
        self.each(TICK_INTERVAL, 'tick')
        self.log_info("Auction started!")

    def tick(self):
        if not self.auction_active:
            return
        
        if self.current_item_idx >= len(self.inventory):
            self.auction_active = False
            self.log_info("Auction finished. No more items.")
            return

        item = self.inventory[self.current_item_idx]
        
        if self.current_price < item['min_price']:
            self.log_info(f"Item {item['id']} ({item['type']}) discarded (price {self.current_price} below minimum {item['min_price']}).")
            self.log_sale(item['id'], item['type'], 0, "")
            self.next_item()
            return
        
        msg = {
            'type': 'AUCTION_ITEM',
            'operator_id': self.operator_id,
            'product_id': item['id'],
            'product_type': item['type'],
            'price': self.current_price
        }
        self.send('market', msg)
        self.log_info(f"Broadcasting: Item {item['id']} ({item['type']}) at {self.current_price}")
        
        self.current_price -= PRICE_DECREMENT

    def next_item(self):
        self.current_item_idx += 1
        self.is_sold = False
        if self.current_item_idx < len(self.inventory):
            self.current_price = self.inventory[self.current_item_idx]['start_price']
        
    def handle_buy(self, msg):
        if not self.auction_active:
            return
        
        if self.current_item_idx >= len(self.inventory):
            return
        
        current_item = self.inventory[self.current_item_idx]
        req_pid = msg.get('product_id')
        m_id = msg.get('merchant_id', 'Unknown')
        
        if req_pid == current_item['id'] and not self.is_sold:
            sale_price = self.current_price + PRICE_DECREMENT
            self.is_sold = True
            
            self.log_info(f"SOLD item {current_item['id']} to {m_id} for {sale_price}")
            self.log_sale(current_item['id'], current_item['type'], sale_price, m_id)
            
            confirmation = {
                'type': 'SALE_CONFIRMATION',
                'operator_id': self.operator_id,
                'product_id': current_item['id'],
                'product_type': current_item['type'],
                'merchant_id': m_id,
                'price': sale_price,
                'msg': 'SOLD'
            }
            self.send('market', confirmation)
            self.next_item()

    def log_sale(self, product_id, product_type, price, merchant_id):
        try:
            with open(self.log_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.operator_id, product_id, product_type, price, merchant_id])
                f.flush()
        except Exception as e:
            self.log_info(f"Error writing log: {e}")

# ============================================================================
# LLM-AUGMENTED MERCHANT CLASS
# ============================================================================
class LLMMerchant(Agent):
    """
    Enhanced Merchant with LLM reasoning layer.
    
    ARCHITECTURE:
    1. Reactive Layer (Python): Handles messages, maintains state
    2. Cognitive Layer (LLM): Makes strategic buying decisions with personality
    """
    
    def on_init(self):
        # Basic attributes
        self.budget = 100
        self.preference = random.choice(FISH_TYPES)
        self.inventory = []
        self.merchant_id = int(self.name.split('_')[-1]) if '_' in self.name else random.randint(100, 999)
        
        # Goal tracking
        self.types_owned = set()
        
        # State tracking
        self.pending_buy_pids = {}
        self.pending_buy_amounts = {}
        self.operator_connections = {}
        
        # LLM-specific attributes
        self.personality = None
        self.system_prompt = None
        
    def set_personality(self, personality_name):
        """Set the merchant's personality for LLM prompting"""
        if personality_name in PERSONALITIES:
            self.personality = personality_name
            self.system_prompt = PERSONALITIES[personality_name]['system_prompt']
            self.log_info(f"Personality set to: {personality_name}")
        else:
            self.log_info(f"Unknown personality: {personality_name}, using BALANCED")
            self.personality = 'BALANCED'
            self.system_prompt = PERSONALITIES['BALANCED']['system_prompt']
        
    def setup_connections(self, operators):
        """Connect to all operators"""
        for op_id, market_addr, sales_addr in operators:
            self.connect(market_addr, handler='handle_market')
            sales_alias = f'sales_{op_id}'
            self.connect(sales_addr, alias=sales_alias)
            self.operator_connections[op_id] = sales_alias
        
    def handle_market(self, msg):
        """Route messages from operators"""
        msg_type = msg.get('type')
        
        if msg_type == 'SALE_CONFIRMATION':
            self.handle_confirmation(msg)
        elif msg_type == 'AUCTION_ITEM':
            self.handle_auction_item(msg)
            
    def handle_confirmation(self, msg):
        """Process sale confirmations"""
        op_id = msg.get('operator_id')
        product_id = msg.get('product_id')
        product_type = msg.get('product_type')
        
        if msg.get('merchant_id') == self.merchant_id:
            price = msg['price']
            self.budget -= price
            self.inventory.append({'id': product_id, 'price': price, 'type': product_type})
            self.types_owned.add(product_type)
            
            self.log_info(f"[{self.personality}] WON {product_type} (item {product_id}) for {price}. Budget: {self.budget}. Types: {self.types_owned}")
            
            self.pending_buy_pids.pop(op_id, None)
            self.pending_buy_amounts.pop(op_id, None)
        else:
            if self.pending_buy_pids.get(op_id) == product_id:
                self.log_info(f"[{self.personality}] Lost item {product_id} to Merchant {msg.get('merchant_id')}")
                self.pending_buy_pids.pop(op_id, None)
                self.pending_buy_amounts.pop(op_id, None)

    def handle_auction_item(self, msg):
        """
        Process auction items using LLM reasoning.
        
        This is where the COGNITIVE LAYER integrates with the REACTIVE LAYER.
        """
        price = msg['price']
        p_type = msg['product_type']
        p_id = msg['product_id']
        op_id = msg['operator_id']
        
        # Check pending bids
        if op_id in self.pending_buy_pids:
            pending_pid = self.pending_buy_pids[op_id]
            if pending_pid != p_id:
                self.pending_buy_pids.pop(op_id)
                self.pending_buy_amounts.pop(op_id, None)
            else:
                return
        
        # Calculate available budget
        pending_total = sum(self.pending_buy_amounts.values())
        available_budget = self.budget - pending_total
        
        # Basic budget check
        if price > available_budget:
            return
        
        # ========================================
        # LLM REASONING LAYER
        # ========================================
        # Construct prompt with current state
        types_missing = [t for t in FISH_TYPES if t not in self.types_owned]
        
        user_prompt = f"""Current situation:
- Fish Type: {p_type}
- Current Price: {price}
- My Budget: {available_budget} (total: {self.budget}, pending: {pending_total})
- My Preference: {self.preference}
- Types I Own: {list(self.types_owned) if self.types_owned else 'None'}
- Types Missing: {types_missing if types_missing else 'None (I have all types!)'}
- My Inventory Count: {len(self.inventory)} fish
- Is Preferred Type: {'YES' if p_type == self.preference else 'NO'}
- Need for Diversity: {'YES (missing this type!)' if p_type not in self.types_owned else 'NO (already have this type)'}

Should I buy this fish at the current price? Respond with your decision and reasoning."""

        # Call LLM for decision
        decision = call_llm_for_decision(self.system_prompt, user_prompt)
        
        action = decision.get('action', 'WAIT')
        reason = decision.get('reason', 'No reason provided')
        
        self.log_info(f"[{self.personality}] LLM Decision for {p_type}@{price}: {action} - {reason}")
        
        # ========================================
        # ACT ON LLM DECISION
        # ========================================
        if action == 'BUY':
            # Send buy request
            req = {
                'operator_id': op_id,
                'product_id': p_id,
                'merchant_id': self.merchant_id,
                'msg': 'BUY'
            }
            
            sales_alias = self.operator_connections.get(op_id, f'sales_{op_id}')
            self.send(sales_alias, req)
            
            # Track pending bid
            self.pending_buy_pids[op_id] = p_id
            self.pending_buy_amounts[op_id] = price
            
            self.log_info(f"[{self.personality}] BUYING {p_type} for {price} from Op{op_id}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    
    print("=" * 80)
    print("LLM-AUGMENTED DUTCH FISH AUCTION")
    print("=" * 80)
    print(f"Using LLM Model: {LLM_MODEL}")
    print()
    
    # Set global variables for file paths
    abs_setup_csv = os.path.abspath(SETUP_CSV)
    abs_log_csv = os.path.abspath(LOG_CSV)
    GLOBAL_LOG_FILE = abs_log_csv
    GLOBAL_SETUP_FILE = abs_setup_csv
    
    # Initialize nameserver
    print("Initializing NameServer...")
    try:
        ns = run_nameserver()
    except Exception as e:
        print(f"NameServer init error: {e}")
        exit(1)
    
    # Create CSV files
    print(f"Creating log files in {RESULTS_DIR}...")
    try:
        with open(abs_setup_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Merchant', 'Personality', 'Preference', 'Budget'])
            f.flush()
        
        with open(abs_log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operator', 'Product', 'Type', 'Sale Price', 'Merchant'])
            f.flush()
            
    except Exception as e:
        print(f"Error creating CSV files: {e}")
        ns.shutdown()
        exit(1)

    # Create operators
    print(f"\nStarting {NUM_OPERATORS} Operators...")
    operators = []
    product_id_counter = 1
    
    for i in range(1, NUM_OPERATORS + 1):
        op_name = f'Operator_{i}'
        try:
            op = run_agent(op_name, base=Operator)
            op.set_operator_id(i)
            op.init_inventory(FISH_PER_OPERATOR, product_id_counter)
            product_id_counter += FISH_PER_OPERATOR
            operators.append(op)
            print(f"  Created {op_name} with {FISH_PER_OPERATOR} fish")
        except Exception as e:
            print(f"Error creating operator {op_name}: {e}")
            ns.shutdown()
            exit(1)
    
    # Create LLM-augmented merchants with different personalities
    print(f"\nStarting {NUM_MERCHANTS} LLM-Augmented Merchants...")
    merchants = []
    operator_connections = [
        (op.get_attr('operator_id'), op.addr('market'), op.addr('sales')) 
        for op in operators
    ]
    
    # Assign personalities cyclically
    personality_names = list(PERSONALITIES.keys())
    
    for i in range(1, NUM_MERCHANTS + 1):
        m_name = f'Merchant_{i}'
        try:
            m = run_agent(m_name, base=LLMMerchant)
            
            # Assign personality
            personality = personality_names[(i - 1) % len(personality_names)]
            m.set_personality(personality)
            
            m.setup_connections(operator_connections)
            merchants.append(m)
            
            # Log setup
            pref = m.get_attr('preference')
            budg = m.get_attr('budget')
            
            with open(abs_setup_csv, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([m_name, personality, pref, budg])
                f.flush()
            
            print(f"  Created {m_name} | Personality: {personality:20s} | Preference: {pref} | Budget: {budg}")
        except Exception as e:
            print(f"Error creating merchant {m_name}: {e}")
            ns.shutdown()
            exit(1)

    # Start auctions
    print("\nStarting Auctions...")
    for op in operators:
        try:
            op.start_auction()
            print(f"  Operator {op.get_attr('operator_id')} auction started")
        except Exception as e:
            print(f"Error starting auction: {e}")
    
    print("\n" + "=" * 80)
    print("AUCTION IN PROGRESS - LLM agents are making decisions...")
    print("=" * 80)
    
    # Wait for completion
    try:
        while True:
            time.sleep(1)
            all_finished = all(not op.get_attr('auction_active') for op in operators)
            if all_finished:
                print("\n" + "=" * 80)
                print("All auctions finished!")
                print("=" * 80)
                print("Waiting for agents to complete final processing...")
                time.sleep(5)  # Give agents time to finish any pending LLM calls
                break
                 
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        print("\nShutting down agents...")
        print(f"Results saved to:")
        print(f"  - {abs_setup_csv}")
        print(f"  - {abs_log_csv}")
        
        # Gracefully shutdown with longer timeout for LLM processing
        try:
            ns.shutdown(timeout=20)  # Increased timeout to allow LLM calls to complete
            print("✓ Shutdown complete!")
        except TimeoutError as e:
            print(f"⚠️  Shutdown timeout (agents may still be processing): {e}")
            print("   This is usually harmless - the program will exit anyway.")
        except Exception as e:
            print(f"⚠️  Shutdown error: {e}")