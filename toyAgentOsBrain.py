import time
import csv
import random
import os
from datetime import datetime
from osbrain import run_agent, run_nameserver, Agent
import osbrain

# System Config
osbrain.config['SERIALIZER'] = 'json'

# Simulation Config
NUM_MERCHANTS = 3
FISH_TYPES = ['H', 'S', 'T']
START_PRICE = 50
MIN_PRICE = 10
PRICE_DECREMENT = 5
TICK_INTERVAL = 0.5

# Paths
RESULTS_DIR = "auction_results"
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

DATE_STR = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SETUP_CSV = os.path.join(RESULTS_DIR, f"setup_{DATE_STR}.csv")
LOG_CSV = os.path.join(RESULTS_DIR, f"log_{DATE_STR}.csv")

class Operator(Agent):
    def on_init(self):
        # Bind sockets
        # PUB for broadcasting auction state and results
        self.bind('PUB', alias='market')
        
        # PULL for receiving buy requests asynchronously (no reply needed)
        self.bind('PULL', alias='sales', handler='handle_buy')
        
        self.inventory = []
        for i in range(1, 11): 
            self.inventory.append({
                'id': i, 
                'type': random.choice(FISH_TYPES),
                'min_price': MIN_PRICE
            })
        
        self.current_item_idx = 0
        self.current_price = START_PRICE
        self.auction_active = False
        self.is_sold = False
        self.log_filename = "log.csv"
        self.operator_id = 1

    def set_log_file(self, filename):
        self.log_filename = filename

    def start_auction(self):
        self.auction_active = True
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
        
        # Broadcast Product
        msg = {
            'type': 'AUCTION_ITEM',
            'operator_id': self.operator_id,
            'product_id': item['id'],
            'product_type': item['type'],
            'price': self.current_price
        }
        self.send('market', msg)
        self.log_info(f"Broadcasting: Item {item['id']} ({item['type']}) at {self.current_price}")
        
        # Decrement for next tick
        self.current_price -= PRICE_DECREMENT
        
        # Check discard condition
        if self.current_price < item['min_price']:
            self.log_info(f"Item {item['id']} ({item['type']}) discarded (unsold).")
            # Log discard
            self.log_sale(item['id'], item['type'], 0, "") 
            self.next_item()

    def next_item(self):
        self.current_item_idx += 1
        self.current_price = START_PRICE
        self.is_sold = False
        
    def handle_buy(self, msg):
        """
        Handles incoming BUY messages from Merchants via PULL socket.
        Does not return a value (PULL-PUSH is one-way).
        Instead, broadcasts the result via PUB.
        """
        if not self.auction_active:
            return 
            
        current_item = self.inventory[self.current_item_idx]
        req_pid = msg.get('product_id')
        m_id = msg.get('merchant_id', 'Unknown')
        
        # Validation Logic
        if req_pid == current_item['id'] and not self.is_sold:
            sale_price = self.current_price + PRICE_DECREMENT
            self.is_sold = True
            
            self.log_info(f"SOLD item {current_item['id']} to {m_id} for {sale_price}")
            self.log_sale(current_item['id'], current_item['type'], sale_price, m_id)
            
            # Broadcast Sale Confirmation
            confirmation = {
                'type': 'SALE_CONFIRMATION',
                'operator_id': self.operator_id,
                'product_id': current_item['id'],
                'merchant_id': m_id,
                'price': sale_price,
                'msg': 'SOLD'
            }
            self.send('market', confirmation)
            
            # Prepare next item
            self.next_item()
        else:
            # Optionally broadcast that it's already sold or ignore
            pass

    def log_sale(self, product_id, product_type, price, merchant_id):
        fname = getattr(self, 'log_filename', 'log.csv') 
        try:
            # Using absolute path to ensure writing to correct location even if working directory differs
            with open(fname, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([product_id, product_type, price, merchant_id])
                f.flush() # Ensure immediate write
        except Exception as e:
            self.log_info(f"Error writing log: {e}")

class Merchant(Agent):
    def on_init(self):
        self.budget = 100
        self.preference = random.choice(FISH_TYPES)
        self.inventory = []
        self.merchant_id = int(self.name.split('_')[-1]) if '_' in self.name else random.randint(100, 999)
        self.pending_buy_pid = None # Track what we are trying to buy
        
    def setup_connections(self, operator_addr_market, operator_addr_sales):
        # SUB to Market (Auction + Confirmations)
        self.connect(operator_addr_market, handler='handle_market')
        
        # PUSH to Sales (Buy Requests)
        self.connect(operator_addr_sales, alias='sales')
        
    def handle_market(self, msg):
        msg_type = msg.get('type')
        
        if msg_type == 'SALE_CONFIRMATION':
            self.handle_confirmation(msg)
        elif msg_type == 'AUCTION_ITEM':
            self.handle_auction_item(msg)
            
    def handle_confirmation(self, msg):
        # Did we win?
        if msg.get('merchant_id') == self.merchant_id:
            price = msg['price']
            p_type = 'Unknown' # We could track this or trust logic
            # Deduct budget
            self.budget -= price
            self.inventory.append({'id': msg['product_id'], 'price': price})
            self.log_info(f"WON item {msg['product_id']} for {price}. Budget: {self.budget}")
            self.pending_buy_pid = None
        else:
            # Someone else won
            if self.pending_buy_pid == msg.get('product_id'):
                self.log_info(f"Lost item {self.pending_buy_pid} to Merchant {msg.get('merchant_id')}")
                self.pending_buy_pid = None

    def handle_auction_item(self, msg):
        # Check if we are already trying to buy this item or another
        if self.pending_buy_pid is not None:
            # If we were trying to buy PREVIOUS item and now seeing NEW item, we failed silently
            if self.pending_buy_pid != msg['product_id']:
                self.pending_buy_pid = None
            else:
                return # Still seeing the item we want, waiting for confirmation or lower price?
                # Actually, if we already sent a buy, we shouldn't send again immediately for the same price step
                # unless we want to spam. Let's be polite.
        
        price = msg['price']
        p_type = msg['product_type']
        p_id = msg['product_id']
        op_id = msg['operator_id']
        
        should_buy = False
        
        if price > self.budget:
            return

        # Simple Strategy
        if p_type == self.preference:
            if price <= self.budget:
                should_buy = True
        elif price <= 25:
            should_buy = True
            
        if should_buy and self.pending_buy_pid is None:
            self.log_info(f"Decided to buy {p_type} at {price}")
            req = {
                'operator_id': op_id,
                'product_id': p_id,
                'merchant_id': self.merchant_id,
                'msg': 'BUY'
            }
            
            # Send Async
            self.send('sales', req)
            self.pending_buy_pid = p_id

if __name__ == '__main__':
    print("Initializing NameServer...")
    try:
        ns = run_nameserver()
    except Exception as e:
        print(f"NameServer init error: {e}")
    
    # Init CSVs
    print(f"Creating log files in {RESULTS_DIR}: {SETUP_CSV}, {LOG_CSV}")
    try:
        # Use absolute path for safety
        abs_setup_csv = os.path.abspath(SETUP_CSV)
        abs_log_csv = os.path.abspath(LOG_CSV)
        
        with open(abs_setup_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Merchant', 'Preference', 'Budget'])
            f.flush()
        
        with open(abs_log_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Product', 'Type', 'Sale Price', 'Merchant'])
            f.flush()
            
    except Exception as e:
        print(f"Error creating CSV files: {e}")

    # Start Operator
    print("Starting Operator...")
    operator = run_agent('Operator', base=Operator)
    operator.set_log_file(abs_log_csv) # Pass absolute path
    
    # Start Merchants
    print("Starting Merchants...")
    merchants = []
    for i in range(1, NUM_MERCHANTS + 1):
        m_name = f'Merchant_{i}'
        m = run_agent(m_name, base=Merchant)
        m.setup_connections(operator.addr('market'), operator.addr('sales'))
        merchants.append(m)
        
        # Log setup information
        pref = m.get_attr('preference')
        budg = m.get_attr('budget')
        
        with open(abs_setup_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([m_name, pref, budg])
            f.flush()

    # Start Auction
    print("Starting Auction...")
    operator.start_auction()
    
    # Wait loop
    try:
        while True:
            time.sleep(1)
            if not operator.get_attr('auction_active'):
                 print("Auction finished.")
                 time.sleep(2) 
                 break
    except KeyboardInterrupt:
        print("Interrupted.")
    except Exception as e:
        print(f"Runtime error: {e}")
    finally:
        ns.shutdown()
