# Multi-Agent Systems: Message Passing Assignment

**Distributed Artificial Intelligence 2025/2026**  
Facultat de Matemàtiques i Informàtica

## 📋 Overview

This assignment explores the evolution from classical distributed AI to modern multi-agent systems through three progressive implementations:

1. **Classical Implementation**: Traditional message-passing multi-agent system using Python and osBrain
2. **LLM-Augmented Agents**: Enhanced agents with Large Language Model reasoning capabilities
3. **Agent Orchestration**: Modern agent coordination using LangGraph framework

## 🎯 Learning Objectives

- Implement direct communication between agents through message passing
- Integrate LLM reasoning into agent decision-making
- Explore modern agent orchestration frameworks
- Compare classical vs AI-augmented multi-agent systems

## 🐟 Project Context: Dutch Fish Auction

We model **la Subhasta del peix** (the fish auction), a traditional Catalan Dutch auction where:
- Fish are displayed with an initial high price
- Price gradually decreases until a buyer accepts
- Each fish has a bottom price - if not sold, it's discarded

### Fish Types
- **H**: Hake
- **S**: Sole  
- **T**: Tuna

## Part 1: Classical Implementation

### 🔧 Technology Stack
- **Language**: Python
- **Library**: [osBrain](https://osbrain.readthedocs.io/)
- **Documentation**: [API Reference](https://osbrain.readthedocs.io/en/stable/api.html)
- **Starter Code**: `toyAgent.py`

### 👥 Agent Types

#### Operators (Sellers)
- Publish products to the market
- Manage price decrements
- Handle sale confirmations

#### Merchants (Buyers)
- Have individual budgets
- Maintain fish preferences (randomly assigned)
- Goal: Obtain at least one fish of each type + maximize preferred fish
- Cannot have negative balances

### ⚙️ Configurable Parameters
- **Number of operators**: Seller count
- **Number of merchants**: Buyer count  
- **Fish per operator**: Quantity offered (can be randomized within range)

### 📡 Communication Protocol

#### Product Broadcast
```json
{
  "operator_id": Integer,
  "product_id": Integer,
  "product_type": "H" | "S" | "T",
  "price": Integer
}
```

#### Purchase Request
```json
{
  "operator_id": Integer,
  "product_id": Integer,
  "msg": "BUY"
}
```

#### Sale Confirmation
```json
{
  "operator_id": Integer,
  "product_id": Integer,
  "merchant_id": Integer,
  "price": Integer,
  "msg": "SOLD"
}
```

### 📊 Output Requirements

Generate two CSV files with timestamps:

**setup_[date].csv**
```
Merchant,Preference,Budget
1,H,100
2,S,100
3,T,100
```

**log_[date].csv**
```
Product,Type,Sale Price,Merchant
1,T,20,2
2,H,0,
3,H,30,1
```

## Part 2: LLM-Augmented Agent Reasoning

### 🤖 Architecture

Each merchant now has two layers:

1. **Reactive Layer** (Python + osBrain)
   - Receives product messages
   - Maintains local state (budget, preferences, stock)
   - Sends "BUY" messages

2. **Cognitive Layer** (LLM)
   - Assists in decision-making
   - Provides reasoning for actions

### 🔑 LLM Setup

**Recommended Model**: Polaris Alpha via OpenRouter (free tier)

1. Create [OpenRouter](https://openrouter.ai/) account
2. Obtain free API key
3. Use Python's `requests` module for API calls
4. **Starter Code**: `toyLLMAgent.py`

### 💭 LLM Integration Flow

When receiving a product message, merchants:
1. Formulate a prompt including:
   - Current budget
   - Fish type
   - Personal preference
   - Auction price
   - Additional context

2. Receive structured JSON response:
```json
{
  "action": "BUY" | "WAIT",
  "reason": "Short explanation of the decision"
}
```

3. Act based on LLM guidance

### 🎭 Merchant Personalities

Implement distinct strategies via system prompts:
- **Cautious**: Buys only at low prices
- **Greedy**: Buys quickly to beat competitors
- **Preference-driven**: Prioritizes favorite fish

## Part 3: LLM-Powered Agent Orchestration

### 🌐 Technology: LangGraph

Migrate the auction system to a graph-based architecture where:
- **Nodes** represent agent behaviors
- **Edges** define message-passing/control flow

### 🏗️ Implementation Structure

1. **State Object**: Shared auction data (product, price, round, messages)

2. **Operator Node**:
   - Broadcasts products and prices
   - Updates prices on no purchase
   - Finalizes sales

3. **Merchant Nodes**:
   - Receive broadcasts
   - Invoke LLM reasoning
   - Return structured messages

4. **Evaluator Node**:
   - Supervises auction rounds
   - Updates prices
   - Controls auction flow

5. **Communication Edges**: Operator → Merchants → Operator

**Starter Code**: `toyLanggraphSystem.py`  
**Documentation**: [LangGraph Docs](https://langchain-ai.github.io/langgraph/)

## 📈 Analysis Requirements

- Test with various parameter configurations
- Compare performance across all three implementations
- Use Python notebooks for CSV log analysis
- Evaluate merchant strategies and outcomes

## 📦 Deliverables

### Files to Submit

1. **Python Files** (2 files):
   - Classical + LLM-augmented system implementation
   - LangGraph implementation
   - Both must generate `setup_[date].csv` and `log_[date].csv` when executed

2. **Report**:
   - Implementation comments
   - Challenges encountered
   - Test analysis and results

### Evaluation Criteria

- **Code Quality & Originality**: 60%
- **Analysis & Experiments**: 40%

## ⏰ Deadline

**Monday, December 15 at 18:59**

## 🚀 Getting Started

1. Review the provided starter files:
   - `toyAgent.py` - osBrain basics
   - `toyLLMAgent.py` - LLM integration example
   - `toyLanggraphSystem.py` - LangGraph structure

2. Install required dependencies:
```bash
pip install osbrain requests langgraph langchain
```

3. Set up your OpenRouter API key for Part 2

4. Begin with the classical implementation before adding LLM capabilities

## 📚 Resources

- [osBrain Documentation](https://osbrain.readthedocs.io/)
- [OpenRouter API](https://openrouter.ai/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

---

*This assignment bridges classical distributed AI techniques with modern LLM-powered approaches, providing hands-on experience with the evolution of multi-agent systems.*
