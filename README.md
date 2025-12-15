# Dutch Fish Auction - Multi-Agent System

This project implements a Dutch auction system using three different approaches:
1. **Classical Distributed System** (osBrain)
2. **LLM-Augmented Agents** (osBrain + LLM)
3. **Graph-Based Orchestration** (LangGraph)

## Setup Instructions

### 1. Python Version

This project requires **Python 3.12** due to compatibility constraints with the osBrain framework.

```bash
python --version  # Should show Python 3.12.x
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API Key

The LLM-augmented implementations require an OpenRouter API key.

#### Step 1: Get your API key
Visit [OpenRouter](https://openrouter.ai/keys) and create an account to get your API key.

#### Step 2: Set credit limit
**Important:** In your OpenRouter account settings, set a positive credit limit (e.g., 5€). If you set it to 0€, your API key will be automatically deactivated, causing 401 authentication errors.

#### Step 3: Create .env file
Copy the example environment file and add your API key:

```bash
cp env.example .env
```

Edit `.env` and replace `your_openrouter_api_key_here` with your actual API key:

```
OPENROUTER_API_KEY=sk-or-v1-your-actual-key-here
```

**Security Note:** The `.env` file is in `.gitignore` and will not be committed to version control.

## Running the Simulations

### Classical osBrain Implementation
```bash
python toyAgentOsBrain.py
```

### LLM-Augmented Implementation
```bash
python toyLLMAgent.py
```

### LangGraph Implementation
```bash
python toyLanggraphSystem.py
```

**Note:** The LangGraph implementation includes 20-second delays between API calls to avoid rate limiting (429 errors). A full simulation may take 15-20 minutes.

## Output

Each implementation generates CSV logs in its respective results directory:
- `auction_results/` - Classical osBrain
- `auction_results_llm/` - LLM-augmented
- `auction_results_langgraph/` - LangGraph

Each run creates two files:
- `setup_TIMESTAMP.csv` - Initial merchant configuration
- `log_TIMESTAMP.csv` - Transaction history

## Troubleshooting

### 401 Unauthorized Errors
**Cause:** Your API key is deactivated, likely due to having a 0€ credit limit.  
**Solution:** Set a positive credit limit (e.g., 5€) in your OpenRouter account settings.

### 429 Rate Limit Errors
**Cause:** Too many API requests in a short time.  
**Solution:** The code already includes delays. If you still see these errors, increase the `time.sleep()` values in the merchant nodes.

### osBrain Compatibility Issues
**Cause:** Wrong Python version or pyzmq version.  
**Solution:** Ensure you're using Python 3.12 and pyzmq 25.1.1 (specified in requirements.txt).

### Zombie Processes
**Cause:** Previous simulation was interrupted without proper shutdown.  
**Solution:** 
```bash
# On Linux/Mac
killall python
# On Windows
taskkill /F /IM python.exe
```

## Project Structure

```
.
├── toyAgentOsBrain.py          # Classical distributed implementation
├── toyLLMAgent.py              # LLM-augmented implementation
├── toyLanggraphSystem.py       # LangGraph orchestration
├── env.example                 # Example environment configuration
├── requirements.txt            # Python dependencies
├── main.tex                    # Project report (LaTeX)
└── README.md                   # This file
```

## Authors

- Eshaan Mittal
- Adrià Gasull

## License

Academic project for Intel·ligència Artificial Distribuïda course.
