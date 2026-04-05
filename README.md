# NRSim

Simulation framework for studying **multi-dimensional filter bubbles** in news recommendation systems.



## Paper

**Unveiling the Dynamics of Multi-Dimensional Filter Bubbles in News Recommendation**



## Overview

News recommendation systems personalize content based on user preferences, but repeated personalization can reduce content diversity and create *filter bubbles*.

**NRSim** models this process by simulating interactions between users and recommendation systems across three dimensions:

- Topic  
- Sentiment  
- Political leaning  

It enables controlled experiments to study how recommendation feedback loops affect diversity over time.



## Key Features

- Multi-dimensional filter bubble analysis  
- Simulation of user–recommender interaction loops  
- Support for multiple recommendation models (NRMS, NPA, NAML, NCF)  
- Extended MIND dataset with sentiment and political annotations  



## Framework

NRSim operates in repeated cycles:

1. Train / retrain recommendation model  
2. Generate recommendations  
3. Simulate user interactions  
4. Update user preferences  
5. Repeat  



## Metrics

- **TopicEnt** – topic diversity  
- **SentBias** – sentiment balance  
- **PolBias** – political diversity  
- **CTR** – user engagement  



## Dataset

This project is based on the **MIND dataset**, extended with:

- sentiment labels (negative / neutral / positive)  
- political leaning labels (left / center / right)  

Annotation resources are available in:




## Implementation of Unveiling the Dynamics of Multi-Dimensional Filter Bubbles in News Recommendation

- **Checkpoints**: [Download](https://drive.google.com/drive/u/2/folders/1zSDlJ95lAeDtvTP_txGxfYD29L8Crqqw)
- **Dataset**: [Download](https://drive.google.com/drive/u/2/folders/1Sa9l3cjvpZShTeYqPCLKyAG7Lo7JHn0C)
