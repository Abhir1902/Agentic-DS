# 🧠 Agentic-DS: Agentic System for Data Science Teams

Agentic-DS is a modular, agent-based system that simulates the behavior of a data science team. It uses autonomous agents—each with a specific role like EDA, feature engineering, modeling, or monitoring—to collaboratively tackle machine learning tasks.

> 🚀 Tested on the [Kaggle Playground Series - Season 5 Episode 4](https://www.kaggle.com/competitions/playground-series-s5e4/discussion?sort=recent-comments)  
> 📉 Achieved a **Root Mean Squared Error (RMSE) of 13.5**, demonstrating promising early performance for an automated agentic setup.

---

## 📁 Project Structure

```
Agentic-DS/
├── agents/             # Specialized agents for tasks (EDA, Modeling, etc.)
├── config/             # YAML configuration files for agents and teams
├── data/               # Input datasets (placeholder)
├── models/             # Trained model artifacts
├── outputs/            # Output logs, predictions, and results
├── tools/              # File tools, shell tools used by agents
├── main.py             # Main script to launch the agentic workflow
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Abhir1902/Agentic-DS.git
cd Agentic-DS
```

### 2️⃣ (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Agentic System

```bash
python main.py
```

This will execute the entire pipeline as configured in `config/team.yaml`.

---

## 🧠 What is Agentic-DS?

Agentic-DS is an **agent-based orchestration system** for automating data science workflows. Each agent:

- Has a **dedicated role** (EDA, Feature Engineering, Modeling, Monitoring)
- Uses tools like shell commands and file operations
- Can **communicate** with other agents
- Makes decisions based on defined goals

The system is inspired by how real-world data science teams collaborate on complex problems.

---

## 📊 Experiment Details

- **Dataset**: [Kaggle Playground Series - S5E4](https://www.kaggle.com/competitions/playground-series-s5e4)
- **Task**: Predict podcast popularity score
- **Approach**: Autonomous agents handling data cleaning, feature engineering, model selection, and evaluation
- **Model Used**: Gradient Boosting Regressor
- **Result**: Achieved **13.5 RMSE** on the validation set

---

## 🧰 Tech Stack

- Python 3.10+
- [Agno](https://pypi.org/project/agno/) (Agent Framework)
- scikit-learn
- pandas, numpy, matplotlib
- YAML configuration for teams and roles

---

## 🛣️ Roadmap

- [ ] Add logging and dashboards for agent activity
- [ ] Integrate with LLMs (e.g., OpenAI, LangChain)
- [ ] Support classification/regression/task auto-selection
- [ ] GUI for configuring agent teams and workflows
- [ ] Benchmark across multiple datasets

---

## 🤝 Contributing

Contributions welcome!  

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📬 Contact

- **Twitter**: [@AbhirMirikar](https://x.com/mirikar)
- **LinkedIn**: [Abhir Mirikar](https://www.linkedin.com/in/abhir-m-mirikar-3398b71b9/)
- **GitHub**: [Abhir1902](https://github.com/Abhir1902)

---

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

---

⭐️ If you like this project, please consider starring it on GitHub!