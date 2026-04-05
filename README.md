# NRSim

Simulation framework for studying multi-dimensional filter bubbles in news recommendation systems.

**Paper:** *[Unveiling the Dynamics of Multi-Dimensional Filter Bubbles in News Recommendation](https://ieeexplore.ieee.org/abstract/document/11400919)*


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

This project is based on the [MIND dataset](https://aclanthology.org/2020.acl-main.331/), extended with:

- sentiment labels (negative / neutral / positive)  
- political leaning labels (left / center / right)  

#### Downloads

- **Dataset**: [Download](https://drive.google.com/drive/u/2/folders/1Sa9l3cjvpZShTeYqPCLKyAG7Lo7JHn0C)  
- **Checkpoints**: [Download](https://drive.google.com/drive/u/2/folders/1zSDlJ95lAeDtvTP_txGxfYD29L8Crqqw)

## Setup

```bash
git clone https://github.com/malshehriai11/NRSim.git
cd NRSim
pip install -r requirements.txt
```

### Run
```bash
python src/run_simulation.py
```

## Citation
If you use this work, please cite:
```bibtex
@inproceedings{alshehri2025unveiling,
  title={Multi-Dimensional Filter Bubbles in News Recommendation},
  author={Alshehri, Manal A. and Zhang, Xiangliang},
  booktitle={IEEE BigData},
  year={2025}
}
```

## License

This project is released under the MIT License.
The MIND dataset is subject to its original license and must be obtained separately.

