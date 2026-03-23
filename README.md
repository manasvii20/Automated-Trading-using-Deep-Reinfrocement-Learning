# Automated Trading Using Deep Reinforcement Learning

## Overview
This project focuses on building an AI-driven automated trading system using Deep Reinforcement Learning (DRL). Leveraging the FinRL framework, the system trains an intelligent agent to make trading decisions such as Buy, Sell, or Hold based on historical market data.

The objective is to design a system that learns optimal trading strategies by maximizing cumulative rewards while handling market volatility and risk.

---

## Key Features
- Deep Reinforcement Learning-based trading agent  
- Training on real-world historical financial datasets  
- End-to-end FinRL pipeline (Data → Environment → Agent → Evaluation)  
- Performance evaluation using financial KPIs (Sharpe Ratio, cumulative returns)  
- Modular and extensible architecture  

---

## Tech Stack
- Languages & Libraries: Python, NumPy, Pandas, Matplotlib  
- Frameworks: FinRL, OpenAI Gym  
- Machine Learning: Deep Reinforcement Learning (DRL)  
- Models: PPO (Proximal Policy Optimization), A2C (Advantage Actor-Critic) *(update if needed)*  
- Data Handling: Time-series financial data processing  

---

## Dataset
The model is trained and evaluated on historical stock market data, including:

- OHLCV data (Open, High, Low, Close, Volume)  
- Time-series data across multiple trading days  
- Technical indicators (Moving Averages, RSI, etc.)

### Data Sources
- Yahoo Finance API  
- FinRL preprocessed datasets  

### Preprocessing
- Handling missing values and anomalies  
- Feature engineering for technical indicators  
- Feature normalization and scaling  
- Time-series alignment for stable training  

---

## System Architecture
The project follows the FinRL layered architecture:

1. Data Layer – Data collection, cleaning, feature engineering  
2. Environment Layer – Gym-based trading simulation  
3. Agent Layer – DRL model training and policy learning  
4. Application Layer – Backtesting and evaluation  

---

## Reinforcement Learning Details
- State Space: Market indicators, stock prices, portfolio state  
- Action Space: Buy / Sell / Hold  
- Reward Function: Portfolio return-based reward  
- Policy Learning: Neural network-based optimization  
- Training Strategy: Episodic learning over historical data  

---

## Results & Insights
- Learned trading strategies outperforming baseline approaches (e.g., buy-and-hold)  
- Captured short-term market trends through sequential decision-making  
- Observed strong dependency on reward design and hyperparameter tuning  
- Achieved stable learning after multiple training iterations  

---

## Challenges & Learnings
- Handling non-stationary financial data where market patterns change over time  
- Designing a balanced reward function to avoid overly aggressive strategies  
- Managing training instability and high variance in RL models  
- Dealing with delayed rewards in sequential decision-making  
- Preventing overfitting to historical data  
- Ensuring stable convergence through proper scaling and tuning  

---

## My Contributions
- Trained and fine-tuned DRL models (PPO/A2C) using the FinRL framework  
- Designed and implemented the agent training pipeline (state and reward design)  
- Improved training stability and convergence through experimentation  
- Evaluated model performance using financial metrics  
- Applied reinforcement learning techniques to real-world financial datasets  

---

## Team
Developed as a team project (3 members).

---

## How to Run
```bash
git clone https://github.com/manasvii20/Automated-Trading-using-Deep-Reinfrocement-Learning.git
cd Automated-Trading-using-Deep-Reinfrocement-Learning

pip install -r requirements.txt

python train.py
python test.py
