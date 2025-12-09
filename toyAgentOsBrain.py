"""
============================================================================
DUTCH FISH AUCTION - MULTI-AGENT SYSTEM
============================================================================

This implements a distributed Dutch auction system using the osBrain framework
for multi-agent communication.

OVERVIEW:
---------
A Dutch auction starts with a high price that decreases over time until a buyer
accepts. If the price drops below the seller's minimum price, the item is
discarded as unsold.

Each fish has randomly assigned starting and minimum prices within configured
ranges, making each auction unique and requiring adaptive bidding strategies.

AGENTS:
-------
1. OPERATORS (Sellers):
   - Each operates independently with their own inventory of fish
   - Broadcasts current item and price at regular intervals (Dutch auction)
   - Decreases price by a fixed amount each tick until sale or discard
   - Processes buy requests first-come-first-served (handles race conditions)
   - Broadcasts sale confirmations to all merchants
   - Logs all transactions to CSV

2. MERCHANTS (Buyers):
   - Each has a fixed budget (100) and a random fish preference (H, S, or T)
   
   GOALS (in priority order):
   1. Obtain at least one fish of each type (diversity goal)
   2. Obtain as many preferred fish as possible (preference satisfaction)
   3. Stay within budget (no negative balance allowed)
   
   STRATEGY:
   - PRIORITY 1 (DIVERSITY): Aggressively bid on missing types
   - PRIORITY 2 (PREFERENCE): Bid on preferred type after diversity achieved
   - PRIORITY 3 (OPPORTUNISTIC): Take bargains (price ≤ 15) on any type
   
   CAPABILITIES:
   - Listen to broadcasts from all operators simultaneously
   - Can bid on items from different operators in parallel
   - Track inventory, types owned, remaining budget, and pending bids
   - Handle race conditions when competing with other merchants

COMMUNICATION PATTERNS:
-----------------------
- PUB-SUB: Operators broadcast auction items and sale confirmations to all merchants
- PUSH-PULL: Merchants send targeted buy requests to specific operators

MESSAGE FLOW EXAMPLE:
--------------------
1. Operator 1 broadcasts: "AUCTION_ITEM: Product 3 (type H) at price 55"
2. Merchant 2 evaluates: "I need H for diversity!" → Decides to buy
3. Merchant 2 sends: "BUY request for product 3" → Operator 1
4. Operator 1 processes: First valid request wins, marks as sold
5. Operator 1 broadcasts: "SALE_CONFIRMATION: Product 3 sold to Merchant 2 for 55"
6. All merchants update state: Winner updates budget/inventory, losers try next item
7. Price decreases by 5, operator moves to next item, repeat...

CONCURRENCY & RACE CONDITIONS:
------------------------------
- Multiple merchants may bid on the same item simultaneously
- Operator processes bids sequentially; first processed bid wins
- Losers learn they lost via SALE_CONFIRMATION broadcast
- Merchants track pending bids per operator to avoid duplicate requests

OUTPUT FILES:
-------------
- setup_[timestamp].csv: Initial merchant configurations (preference, budget)
- log_[timestamp].csv: All auction transactions (sales and discards)
"""

# ============================================================================
# IMPORTS
# ============================================================================
import time
import csv
import random
import os
from datetime import datetime
from osbrain import run_agent, run_nameserver, Agent
import osbrain

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
# Set osBrain to use JSON for message serialization (allows complex data structures)
osbrain.config['SERIALIZER'] = 'json'

# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================
NUM_OPERATORS = 2              # Number of sellers in the auction
NUM_MERCHANTS = 3              # Number of buyers in the auction
FISH_TYPES = ['H', 'S', 'T']   # H=Hake, S=Sole, T=Tuna
FISH_PER_OPERATOR = 4          # How many fish each operator has to sell

# Price Configuration (each fish gets random values within these ranges)
START_PRICE_MIN = 40           # Minimum starting price for any fish
START_PRICE_MAX = 60           # Maximum starting price for any fish
MIN_PRICE_MIN = 5              # Minimum acceptable price (lower bound)
MIN_PRICE_MAX = 15             # Minimum acceptable price (upper bound)
PRICE_DECREMENT = 5            # How much the price drops each tick
TICK_INTERVAL = 0.5            # Time between price updates (seconds)

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
# Create directory for storing auction results if it doesn't exist
RESULTS_DIR = "auction_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

# Generate timestamped filenames for this auction run
DATE_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SETUP_CSV = os.path.join(RESULTS_DIR, f"setup_{DATE_STR}.csv")  # Merchant configurations
LOG_CSV = os.path.join(RESULTS_DIR, f"log_{DATE_STR}.csv")      # Auction transactions

# ============================================================================
# OPERATOR CLASS (SELLER AGENT)
# ============================================================================
# The Operator represents a seller conducting a Dutch auction.
#
# RESPONSIBILITIES:
# - Maintains an inventory of fish, each with random start/min prices
# - Broadcasts current item and price at regular intervals (TICK_INTERVAL)
# - Implements Dutch auction: price decreases until someone buys or min reached
# - Processes buy requests from merchants (first-come-first-served)
# - Broadcasts sale confirmations to all merchants
# - Logs all sales and discards to CSV file
#
# DUTCH AUCTION MECHANISM:
# - Start at high price (random per fish: START_PRICE_MIN to START_PRICE_MAX)
# - Decrease by PRICE_DECREMENT every tick
# - First merchant to bid wins at current price
# - If price falls below minimum (random per fish: MIN_PRICE_MIN to MIN_PRICE_MAX),
#   item is discarded and operator moves to next item
# ============================================================================
class Operator(Agent):
    def on_init(self):
        """
        Initialize the Operator agent when it's created.
        Sets up communication channels and initial state.
        """
        # ---- COMMUNICATION SETUP ----
        # PUB socket: One-way broadcast to all merchants (market updates & confirmations)
        self.bind('PUB', alias='market')
        
        # PULL socket: Receives buy requests from merchants asynchronously
        # When a message arrives, it automatically calls handle_buy()
        self.bind('PULL', alias='sales', handler='handle_buy')
        
        # ---- STATE VARIABLES ----
        self.inventory = []              # List of fish to sell (populated later)
        self.current_item_idx = 0        # Which item we're currently auctioning
        self.current_price = 0           # Current price of active item (set when item starts)
        self.auction_active = False      # Is the auction running?
        self.is_sold = False             # Has current item been sold?
        self.log_filename = "log.csv"    # Where to log transactions
        self.operator_id = 1             # Unique ID for this operator

    def set_log_file(self, filename):
        """Set the CSV file where transactions will be logged"""
        self.log_filename = filename
    
    def set_operator_id(self, op_id):
        """Set the unique identifier for this operator"""
        self.operator_id = op_id
    
    def init_inventory(self, num_fish, start_product_id):
        """
        Initialize the operator's inventory with fish to sell.
        
        Each fish is assigned INDEPENDENT random values:
        - Type: Random choice from FISH_TYPES (H, S, or T)
        - Starting price: Random in range [START_PRICE_MIN, START_PRICE_MAX]
        - Minimum price: Random in range [MIN_PRICE_MIN, MIN_PRICE_MAX]
        
        This randomization makes each auction unique and unpredictable,
        requiring merchants to adapt their bidding strategies dynamically.
        
        Args:
            num_fish: How many fish this operator will sell
            start_product_id: Starting ID to ensure unique IDs across all operators
        """
        self.inventory = []
        for i in range(num_fish):
            # Generate random prices for this specific fish (independent of other fish)
            start_price = random.randint(START_PRICE_MIN, START_PRICE_MAX)
            min_price = random.randint(MIN_PRICE_MIN, MIN_PRICE_MAX)
            
            # Ensure starting price is higher than minimum (allow at least 2 price drops)
            if start_price <= min_price:
                start_price = min_price + 2 * PRICE_DECREMENT
            
            self.inventory.append({
                'id': start_product_id + i,        # Unique product ID across all operators
                'type': random.choice(FISH_TYPES), # Random type: H (Hake), S (Sole), or T (Tuna)
                'start_price': start_price,        # Initial auction price (decreases each tick)
                'min_price': min_price             # Minimum acceptable price (discard if reached)
            })

    def start_auction(self):
        """
        Start the auction process.
        Sets a recurring timer that calls tick() every TICK_INTERVAL seconds.
        Initializes the price from the first item's starting price.
        """
        self.auction_active = True
        
        # Set initial price from first item (if inventory exists)
        if len(self.inventory) > 0:
            self.current_price = self.inventory[0]['start_price']
        
        self.each(TICK_INTERVAL, 'tick')  # Schedule tick() to run repeatedly
        self.log_info("Auction started!")

    def tick(self):
        """
        Called every TICK_INTERVAL seconds during the auction.
        This implements the core Dutch auction mechanism.
        
        DUTCH AUCTION FLOW (per tick):
        1. Check if current price is already below minimum (discard if so)
        2. Broadcast current item and price to all merchants
        3. Decrease price by PRICE_DECREMENT for next tick
        4. Wait for merchants to respond (they have full TICK_INTERVAL)
        
        IMPORTANT:
        - Each fish has its own independent start/min prices (random)
        - Merchants have full TICK_INTERVAL to evaluate and bid
        - First merchant to bid wins (handled in handle_buy)
        - If item sells, next_item() is called from handle_buy()
        - Discard check happens BEFORE broadcast (not after) to avoid race condition
        """
        # Safety check: don't run if auction is stopped
        if not self.auction_active:
            return
        
        # Check if we've sold all items
        if self.current_item_idx >= len(self.inventory):
            self.auction_active = False
            self.log_info("Auction finished. No more items.")
            return

        # Get the current item being auctioned
        item = self.inventory[self.current_item_idx]
        
        # ---- CHECK IF PRICE ALREADY TOO LOW ----
        # If current price is below minimum, discard without broadcasting
        # This prevents broadcasting items that are already below minimum
        if self.current_price < item['min_price']:
            self.log_info(f"Item {item['id']} ({item['type']}) discarded (price {self.current_price} below minimum {item['min_price']}).")
            # Log with price=0 and no merchant to indicate discard
            self.log_sale(item['id'], item['type'], 0, "") 
            self.next_item()  # Move to next item
            return  # Exit this tick, next tick will handle next item
        
        # ---- BROADCAST AUCTION STATE ----
        # Send product info to all merchants via PUB socket
        # Merchants have the full TICK_INTERVAL to respond
        msg = {
            'type': 'AUCTION_ITEM',
            'operator_id': self.operator_id,
            'product_id': item['id'],
            'product_type': item['type'],
            'price': self.current_price
        }
        self.send('market', msg)
        self.log_info(f"Broadcasting: Item {item['id']} ({item['type']}) at {self.current_price}")
        
        # ---- DUTCH AUCTION: DECREASE PRICE ----
        # Decrease price for next tick
        # Merchants can still buy at the broadcasted price during this tick
        self.current_price -= PRICE_DECREMENT

    def next_item(self):
        """
        Move to the next item in inventory.
        Resets price to the new item's starting price and clears sold flag.
        """
        self.current_item_idx += 1
        self.is_sold = False
        
        # Set price to the new item's individual starting price
        if self.current_item_idx < len(self.inventory):
            self.current_price = self.inventory[self.current_item_idx]['start_price']
        
    def handle_buy(self, msg):
        """
        Process buy requests from merchants.
        This is called automatically when a merchant sends a BUY message to the PULL socket.
        
        COMMUNICATION PATTERN:
        - Uses PULL socket (one-way, no direct reply to merchant)
        - Results broadcast via PUB socket so all merchants know the outcome
        
        CONCURRENCY & RACE CONDITIONS:
        When multiple merchants bid on the same item simultaneously:
        1. Messages queue in PULL socket
        2. Processed sequentially by osBrain
        3. First processed: is_sold=False → SALE ACCEPTED → is_sold=True
        4. Second processed: is_sold=True → SALE REJECTED (already sold)
        5. Winner determined by network timing + queue order
        6. All merchants notified via SALE_CONFIRMATION broadcast
        
        Args:
            msg: Dictionary with product_id, merchant_id, operator_id
        """
        # Ignore requests if auction is not active
        if not self.auction_active:
            return 
        
        # ---- BOUNDS CHECK ----
        # Prevent crash from late-arriving buy requests after auction finishes
        # This can happen when a merchant sends a BUY just as the last item sells
        if self.current_item_idx >= len(self.inventory):
            return  # Auction finished, ignore late requests
        
        # Extract information from the buy request
        current_item = self.inventory[self.current_item_idx]
        req_pid = msg.get('product_id')      # Which product they want to buy
        m_id = msg.get('merchant_id', 'Unknown')  # Who wants to buy it
        
        # ---- VALIDATE AND PROCESS PURCHASE ----
        # Check: Is this the current item? Has it not been sold yet?
        # CRITICAL: This check prevents double-selling when multiple bids arrive!
        if req_pid == current_item['id'] and not self.is_sold:
            # SALE ACCEPTED!
            # 
            # PRICE CALCULATION:
            # self.current_price was already decremented in tick() for the NEXT tick,
            # but the merchant is buying at the price we BROADCASTED (before decrement).
            # Therefore: sale_price = current_price + PRICE_DECREMENT
            sale_price = self.current_price + PRICE_DECREMENT
            
            # ⚠️ ATOMIC FLAG: Set immediately to prevent double-selling!
            # Next bid for this item will see is_sold=True and be rejected
            self.is_sold = True
            
            self.log_info(f"SOLD item {current_item['id']} to {m_id} for {sale_price}")
            
            # Log the transaction to CSV
            self.log_sale(current_item['id'], current_item['type'], sale_price, m_id)
            
            # ---- BROADCAST SALE CONFIRMATION ----
            # Tell all merchants about the sale (winner knows they won, losers know they lost)
            confirmation = {
                'type': 'SALE_CONFIRMATION',
                'operator_id': self.operator_id,
                'product_id': current_item['id'],
                'product_type': current_item['type'],  # Include type so merchants can track what they won
                'merchant_id': m_id,
                'price': sale_price,
                'msg': 'SOLD'
            }
            self.send('market', confirmation)
            
            # Move to next item in inventory
            self.next_item()
        else:
            # ---- REJECTED BUY REQUEST ----
            # This happens when:
            # 1. Item already sold (another merchant was faster)
            # 2. Wrong product ID (merchant bidding on old/invalid item)
            # The merchant will learn they lost via the SALE_CONFIRMATION broadcast
            pass

    def log_sale(self, product_id, product_type, price, merchant_id):
        """
        Write a sale transaction to the CSV log file.
        
        Format: Operator, Product, Type, Sale Price, Merchant
        If price=0 and merchant_id="", it means the item was discarded.
        
        Args:
            product_id: Unique ID of the fish
            product_type: Type of fish (H, S, or T)
            price: Final sale price (or 0 if discarded)
            merchant_id: ID of buyer (or "" if discarded)
        """
        fname = getattr(self, 'log_filename', 'log.csv') 
        try:
            with open(fname, 'a', newline='') as f:
                writer = csv.writer(f)
                # Write: [Operator ID, Product ID, Type, Price, Merchant ID]
                writer.writerow([self.operator_id, product_id, product_type, price, merchant_id])
                f.flush()  # Force write to disk immediately
        except Exception as e:
            self.log_info(f"Error writing log: {e}")

# ============================================================================
# MERCHANT CLASS (BUYER AGENT)
# ============================================================================
# The Merchant represents a buyer participating in multiple Dutch auctions.
#
# INITIAL STATE:
# - Budget: 100 (fixed)
# - Preference: Random fish type (H, S, or T)
# - Inventory: Empty
# - Types owned: None
#
# GOALS (in priority order):
# 1. DIVERSITY: Obtain at least one fish of each type (H, S, T)
# 2. PREFERENCE: Obtain as many preferred fish as possible
# 3. BUDGET: Stay within budget (no negative balance allowed)
#
# BIDDING STRATEGY:
# The merchant uses a goal-oriented strategy with three priority levels:
#
# PRIORITY 1 - DIVERSITY (types not yet owned):
#   - Bid aggressively on missing types
#   - Reserve budget for remaining missing types
#   - Willing to spend up to 40% of budget on critical missing types
#
# PRIORITY 2 - PREFERENCE SATISFACTION (preferred type):
#   - After diversity goal met (or for already-owned types)
#   - Bid on preferred fish to maximize collection
#   - Willing to spend up to 60% of remaining budget
#
# PRIORITY 3 - OPPORTUNISTIC (bargains):
#   - Bid on any fish if price is very low (≤ 15)
#   - Takes advantage of good deals regardless of type
#
# MULTI-OPERATOR BIDDING:
# - Connects to all operators simultaneously
# - Can bid on items from different operators in parallel
# - Maintains one pending bid per operator (no spam)
# - Tracks pending bid amounts to prevent overspending
# - Available budget = total budget - sum of pending bid amounts
# - This ensures merchants never spend more than their budget
#
# RACE CONDITION HANDLING:
# - Multiple merchants may bid on same item
# - Operator picks winner (first processed)
# - All merchants receive sale confirmation
# - Winner: Updates budget, inventory, and types owned
# - Losers: Clear pending bid and try next item
# ============================================================================
class Merchant(Agent):
    def on_init(self):
        """
        Initialize the Merchant agent when created.
        Sets up initial budget, preferences, and state.
        
        MERCHANT GOALS (in priority order):
        1. Obtain at least one fish of each type (diversity)
        2. Obtain as many preferred fish as possible (preference satisfaction)
        3. Stay within budget (no negative balance)
        """
        # ---- MERCHANT ATTRIBUTES ----
        self.budget = 100  # Starting money
        self.preference = random.choice(FISH_TYPES)  # Random favorite fish type
        self.inventory = []  # Fish purchased during auction
        
        # Extract merchant ID from agent name (e.g., "Merchant_1" -> 1)
        self.merchant_id = int(self.name.split('_')[-1]) if '_' in self.name else random.randint(100, 999)
        
        # ---- GOAL TRACKING ----
        # Track which types we have acquired (for "at least one of each" goal)
        self.types_owned = set()  # Set of fish types we've acquired
        
        # ---- STATE TRACKING ----
        # pending_buy_pids is CRITICAL for race condition handling!
        # - Maps operator_id -> product_id for each pending bid
        # - Allows bidding on multiple items from different operators simultaneously
        # - Prevents sending multiple bids for same item from same operator
        # - Cleared when we receive SALE_CONFIRMATION (win or lose)
        self.pending_buy_pids = {}  # Maps operator_id -> product_id (prevents duplicate bids per operator)
        self.pending_buy_amounts = {}  # Maps operator_id -> bid amount (for budget tracking)
        self.operator_connections = {}  # Maps operator_id -> sales channel alias
        
    def setup_connections(self, operators):
        """
        Connect to all operators in the auction.
        Each merchant can buy from any operator.
        
        Args:
            operators: List of tuples [(op_id, market_addr, sales_addr), ...]
                      - op_id: Operator's unique ID
                      - market_addr: Address for receiving broadcasts (SUB)
                      - sales_addr: Address for sending buy requests (PUSH)
        """
        for op_id, market_addr, sales_addr in operators:
            # ---- SUBSCRIBE TO MARKET BROADCASTS ----
            # SUB socket: Receive auction updates and sale confirmations
            # All messages go to handle_market() method
            self.connect(market_addr, handler='handle_market')
            
            # ---- SETUP PURCHASE CHANNEL ----
            # PUSH socket: Send buy requests to this specific operator
            # Each operator gets unique alias to send to correct one
            sales_alias = f'sales_{op_id}'
            self.connect(sales_addr, alias=sales_alias)
            self.operator_connections[op_id] = sales_alias
        
    def handle_market(self, msg):
        """
        Router for all messages received from operators.
        This is called whenever ANY operator broadcasts a message.
        
        Args:
            msg: Dictionary with 'type' field indicating message type
        """
        msg_type = msg.get('type')
        
        if msg_type == 'SALE_CONFIRMATION':
            # An item was sold - check if we won or lost
            self.handle_confirmation(msg)
        elif msg_type == 'AUCTION_ITEM':
            # New item being auctioned or price update
            self.handle_auction_item(msg)
            
    def handle_confirmation(self, msg):
        """
        Process sale confirmation messages broadcast by operators.
        This is the KEY to handling race conditions on the buyer side.
        
        BROADCAST PATTERN:
        - ALL merchants receive EVERY sale confirmation (PUB-SUB)
        - Each merchant checks: "Did I win, or did someone else?"
        
        RACE CONDITION RESOLUTION:
        When multiple merchants bid on the same item:
        1. Operator picks one winner (first processed)
        2. Operator broadcasts SALE_CONFIRMATION with winner's merchant_id
        3. Winner (this merchant):
           - Deducts price from budget
           - Adds item to inventory
           - Adds type to types_owned (tracks diversity goal)
           - Clears pending bid for this operator
        4. Losers (other merchants):
           - Clear pending bid for this operator
           - Ready to bid on next item from that operator
        
        Args:
            msg: Dictionary with operator_id, product_id, product_type, merchant_id (winner), price
        """
        op_id = msg.get('operator_id')
        product_id = msg.get('product_id')
        product_type = msg.get('product_type')  # Get the fish type to track what we own
        
        # ---- CHECK IF WE WON ----
        if msg.get('merchant_id') == self.merchant_id:
            # ✅ SUCCESS! We won the auction (we were the fastest/first processed)
            price = msg['price']
            
            # Update our state
            self.budget -= price  # Deduct money spent
            self.inventory.append({'id': product_id, 'price': price, 'type': product_type})
            
            # Track which types we own (for "at least one of each" goal)
            self.types_owned.add(product_type)
            
            self.log_info(f"WON {product_type} (item {product_id}) for {price}. Budget: {self.budget}. Types owned: {self.types_owned}")
            
            # Clear pending bid for this operator (both product ID and amount)
            self.pending_buy_pids.pop(op_id, None)
            self.pending_buy_amounts.pop(op_id, None)
        else:
            # ❌ Someone else won the item
            # If we were trying to buy this item from this operator, we lost the race
            if self.pending_buy_pids.get(op_id) == product_id:
                self.log_info(f"Lost item {product_id} to Merchant {msg.get('merchant_id')}")
                
                # ⚠️ CRITICAL: Clear flags for this operator so we can bid on their next item!
                self.pending_buy_pids.pop(op_id, None)
                self.pending_buy_amounts.pop(op_id, None)

    def handle_auction_item(self, msg):
        """
        Process auction item broadcasts (the core buying decision logic).
        
        GOAL-ORIENTED BIDDING STRATEGY (in priority order):
        1. DIVERSITY FIRST: Get at least one fish of each type
           - Bid aggressively on types we don't have yet
           - Reserve enough budget for remaining types
        2. PREFERENCE SATISFACTION: Get as many preferred fish as possible
           - Once we have one of each, focus on preferred type
        3. OPPORTUNISTIC: Take good deals on any fish
           - If price is very low, buy regardless of type
        
        Merchants can bid on multiple items simultaneously as long as:
        - They're from different operators (vendors)
        - The budget allows for it (no negative balance)
        
        Args:
            msg: Contains operator_id, product_id, product_type, price
        """
        # Extract auction details
        price = msg['price']              # Current asking price
        p_type = msg['product_type']      # Fish type (H, S, or T)
        p_id = msg['product_id']          # Unique product ID
        op_id = msg['operator_id']        # Which operator is selling
        
        # ---- CHECK PENDING BIDS FOR THIS OPERATOR ----
        # We track pending bids PER OPERATOR to:
        # 1. Prevent spam (don't send multiple bids for same item)
        # 2. Allow parallel bidding (can bid on different operators simultaneously)
        
        if op_id in self.pending_buy_pids:
            pending_pid = self.pending_buy_pids[op_id]
            
            if pending_pid != p_id:
                # This is a NEW item from this operator (different from pending bid)
                # This means we lost the previous auction (operator moved to next item)
                # Clear the old pending bid and amount, then evaluate this new item
                self.pending_buy_pids.pop(op_id)
                self.pending_buy_amounts.pop(op_id, None)
            else:
                # Still the SAME item we already bid on
                # Don't send duplicate bids - wait for confirmation
                return
        
        # ---- CALCULATE AVAILABLE BUDGET ----
        # When bidding on multiple items simultaneously, we must account for pending bids
        # to avoid spending more than our total budget
        pending_total = sum(self.pending_buy_amounts.values())
        available_budget = self.budget - pending_total
        
        # ---- BASIC BUDGET CHECK ----
        if price > available_budget:
            return  # Can't afford this item (considering pending bids)
        
        # ---- GOAL-ORIENTED BUYING STRATEGY ----
        should_buy = False
        reason = ""  # For logging/debugging
        
        # Calculate how many types we still need to complete diversity goal
        types_needed = len(FISH_TYPES) - len(self.types_owned)
        
        # ========================================
        # PRIORITY 1: DIVERSITY - Get at least one of each type
        # ========================================
        if p_type not in self.types_owned:
            # We need this type to achieve diversity goal!
            
            # BUDGET RESERVATION STRATEGY:
            # Reserve enough budget for remaining missing types to ensure we can
            # still afford them later. Estimate MIN_PRICE_MAX per missing type.
            budget_to_reserve = (types_needed - 1) * MIN_PRICE_MAX
            affordable_price = available_budget - budget_to_reserve
            
            if price <= affordable_price:
                # Safe to buy: Have enough left for other missing types
                should_buy = True
                reason = f"DIVERSITY (need {p_type}, {types_needed} types left)"
            elif types_needed > 0 and price <= available_budget * 0.4:
                # Price is higher than safe amount, but still reasonable
                # Buy anyway if not too expensive (max 40% of available budget per fish)
                should_buy = True
                reason = f"DIVERSITY_URGENT (need {p_type})"
        
        # ========================================
        # PRIORITY 2: PREFERENCE SATISFACTION - Get more of preferred type
        # ========================================
        elif p_type == self.preference:
            # We already have this type (diversity goal met for this type)
            # Now focus on preference satisfaction: collect more of preferred type
            
            # BUDGET MANAGEMENT:
            # Don't spend too much on non-critical purchases (max 60% of available budget)
            # Keep some budget for diversity if we haven't completed that goal yet
            if price <= available_budget * 0.6:
                should_buy = True
                reason = f"PREFERENCE (love {p_type})"
        
        # ========================================
        # PRIORITY 3: OPPORTUNISTIC - Take bargains
        # ========================================
        elif price <= 15:
            # Price is very low - good deal on any fish type
            # Buy to maximize value even if not needed for goals
            should_buy = True
            reason = f"BARGAIN ({p_type} at {price})"
        
        # ---- SEND BUY REQUEST ----
        if should_buy:
            self.log_info(f"{reason} - Bidding {price} for {p_type} from Op{op_id} (Available: {available_budget})")
            
            # Construct buy request message
            req = {
                'operator_id': op_id,
                'product_id': p_id,
                'merchant_id': self.merchant_id,
                'msg': 'BUY'
            }
            
            # Send request to the correct operator via PUSH socket
            # Each operator has a unique channel alias (e.g., sales_1, sales_2)
            sales_alias = self.operator_connections.get(op_id, f'sales_{op_id}')
            self.send(sales_alias, req)
            
            # Track this pending bid for this operator
            # Store both the product ID and the bid amount (for budget tracking)
            # Will be cleared when we receive SALE_CONFIRMATION (win or lose)
            self.pending_buy_pids[op_id] = p_id
            self.pending_buy_amounts[op_id] = price

# ============================================================================
# MAIN EXECUTION BLOCK
# ============================================================================
# This runs when you execute the script directly (not imported as a module)
# It sets up the entire auction system and runs the simulation
# ============================================================================
if __name__ == '__main__':
    
    # ========================================================================
    # STEP 1: INITIALIZE OSBRAIN NAMESERVER
    # ========================================================================
    # The nameserver is required by osBrain for agent discovery and communication
    # All agents register with the nameserver to find each other
    print("Initializing NameServer...")
    try:
        ns = run_nameserver()
    except Exception as e:
        print(f"NameServer init error: {e}")
    
    # ========================================================================
    # STEP 2: CREATE CSV LOG FILES
    # ========================================================================
    # We create two CSV files to record the auction:
    # 1. setup_*.csv - Initial configuration (merchants' preferences & budgets)
    # 2. log_*.csv - Transaction log (what was sold, to whom, for how much)
    print(f"Creating log files in {RESULTS_DIR}: {SETUP_CSV}, {LOG_CSV}")
    try:
        # Convert to absolute paths to avoid any working directory issues
        abs_setup_csv = os.path.abspath(SETUP_CSV)
        abs_log_csv = os.path.abspath(LOG_CSV)
        
        # Create setup CSV with header
        with open(abs_setup_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Merchant', 'Preference', 'Budget'])
            f.flush()
        
        # Create log CSV with header
        with open(abs_log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Operator', 'Product', 'Type', 'Sale Price', 'Merchant'])
            f.flush()
            
    except Exception as e:
        print(f"Error creating CSV files: {e}")

    # ========================================================================
    # STEP 3: CREATE OPERATOR AGENTS (SELLERS)
    # ========================================================================
    # Create multiple operators, each with their own inventory of fish.
    # Each operator runs an independent Dutch auction simultaneously.
    # 
    # IMPORTANT: Each individual fish gets random prices within configured ranges:
    # - Starting price: {START_PRICE_MIN} to {START_PRICE_MAX}
    # - Minimum price: {MIN_PRICE_MIN} to {MIN_PRICE_MAX}
    # This makes each auction unique and unpredictable.
    print(f"Starting {NUM_OPERATORS} Operators...")
    print(f"  Price ranges: Start ${START_PRICE_MIN}-${START_PRICE_MAX}, Min ${MIN_PRICE_MIN}-${MIN_PRICE_MAX}")
    operators = []
    product_id_counter = 1  # Ensures unique product IDs across all operators
    
    for i in range(1, NUM_OPERATORS + 1):
        op_name = f'Operator_{i}'
        
        # Create the operator agent
        op = run_agent(op_name, base=Operator)
        
        # Configure the operator
        op.set_log_file(abs_log_csv)         # Where to log transactions
        op.set_operator_id(i)                 # Unique operator ID
        op.init_inventory(FISH_PER_OPERATOR, product_id_counter)  # Create fish with random prices
        
        # Update counter so next operator gets different product IDs
        product_id_counter += FISH_PER_OPERATOR
        
        operators.append(op)
        print(f"  Created {op_name} with {FISH_PER_OPERATOR} fish (IDs {product_id_counter - FISH_PER_OPERATOR} to {product_id_counter - 1})")
    
    # ========================================================================
    # STEP 4: CREATE MERCHANT AGENTS (BUYERS)
    # ========================================================================
    # Create multiple merchants with random preferences and fixed starting budget.
    # 
    # Each merchant:
    # - Has budget of 100 (fixed)
    # - Gets random fish preference (H, S, or T)
    # - Connects to ALL operators simultaneously
    # - Uses goal-oriented strategy: diversity > preference > bargains
    # - Can bid on multiple items from different operators in parallel
    print(f"Starting {NUM_MERCHANTS} Merchants...")
    merchants = []
    
    # Gather connection information from all operators.
    # Each merchant needs two addresses per operator:
    # - market address (SUB socket): for receiving auction broadcasts
    # - sales address (PUSH socket): for sending buy requests
    operator_connections = [
        (op.get_attr('operator_id'), op.addr('market'), op.addr('sales')) 
        for op in operators
    ]
    
    for i in range(1, NUM_MERCHANTS + 1):
        m_name = f'Merchant_{i}'
        
        # Create the merchant agent
        m = run_agent(m_name, base=Merchant)
        
        # Connect merchant to all operators
        m.setup_connections(operator_connections)
        
        merchants.append(m)
        
        # Log this merchant's initial configuration to setup CSV
        pref = m.get_attr('preference')  # Get randomly assigned preference
        budg = m.get_attr('budget')      # Get starting budget
        
        with open(abs_setup_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([m_name, pref, budg])
            f.flush()
        
        print(f"  Created {m_name} (Preference: {pref}, Budget: {budg})")

    # ========================================================================
    # STEP 5: START ALL AUCTIONS
    # ========================================================================
    # Start all operators' auctions simultaneously.
    # Each operator runs independently with their own tick timer.
    # Merchants listen to all operators and can bid on any active auction.
    print("Starting Auctions...")
    for op in operators:
        op.start_auction()  # Activates the tick() timer (runs every TICK_INTERVAL)
        print(f"  Operator {op.get_attr('operator_id')} auction started")
    
    # ========================================================================
    # STEP 6: WAIT FOR ALL AUCTIONS TO COMPLETE
    # ========================================================================
    # The main thread waits here while agents run autonomously.
    # 
    # Agents communicate asynchronously via osBrain:
    # - Operators broadcast auction items and confirmations
    # - Merchants send buy requests
    # - All happens in parallel, no blocking
    # 
    # Auctions finish when (per operator):
    # - All items sold, OR
    # - All items discarded (price fell below minimum)
    # 
    # Main thread polls until ALL operators finish.
    try:
        while True:
            time.sleep(1)  # Check every second
            
            # Check if ALL operators have finished their auctions
            all_finished = all(not op.get_attr('auction_active') for op in operators)
            
            if all_finished:
                 print("All auctions finished.")
                 time.sleep(2)  # Brief pause to ensure all final messages processed
                 break
                 
    except KeyboardInterrupt:
        # User pressed Ctrl+C to stop early
        print("\nInterrupted by user.")
    except Exception as e:
        # Catch any unexpected errors during execution
        print(f"Runtime error: {e}")
    finally:
        # Always clean up: shutdown nameserver and all agents
        print("Shutting down...")
        ns.shutdown()
