# 🧠 Agentic-DS: Agentic System for Data Science Teams

Agentic-DS is a modular, agent-based system designed to simulate a collaborative data science team. It uses autonomous agents with specialized roles (EDA, Feature Engineering, Modeling, etc.) to automate the end-to-end data science lifecycle.

> 🚀 **Tested on the [Kaggle Podcast Popularity Prediction Challenge](https://www.kaggle.com/competitions/llm-science-podcast/)**  
> Achieved a **Root Mean Squared Error (RMSE) of 13.5**, demonstrating solid initial performance for an automated system.

---

## 📂 Repository Structure

Agentic-DS/ ├── agents/             # Specialized agent definitions (EDA, Modeling, etc.) ├── config/             # Team structure and agent role configs in YAML ├── data/               # Input data or datasets (placeholder) ├── models/             # Trained models and pipelines ├── outputs/            # Output logs, predictions, and artifacts ├── tools/              # Utilities used by agents (shell, file tools, etc.) ├── main.py             # Entry point to run the agentic pipeline ├── requirements.txt    # Python dependencies └── README.md           # Project documentation

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Abhir1902/Agentic-DS.git
cd Agentic-DS

2. Install Dependencies

Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

Then install required packages:

pip install -r requirements.txt

3. Run the Agentic System

python main.py

The agents will sequentially perform EDA, feature engineering, modeling, and evaluation as defined in the config/team.yaml.


---

🔍 Features

✅ Modular agent structure

🧠 Autonomous agents with defined goals and backstories

🔄 Sequential task orchestration: EDA → Feature Engineering → Modeling → Monitoring

🛠️ Tools for shell, file, and dataset manipulation

📈 Built-in ML support (e.g., sklearn regressors)



---

🧪 Use Case: Kaggle LLM Podcast Challenge

Goal: Predict podcast popularity based on textual and structured metadata

Model: Gradient Boosting Regressor

Metric: RMSE

Result: 🥈 Achieved 13.5 RMSE



---

🧰 Tech Stack

Python 3.10+

Agno Agent Framework

scikit-learn

pandas, numpy, matplotlib

YAML configuration



---

🛣️ Roadmap

[ ] Add logging and visual dashboards for agent activity

[ ] Integrate LLMs with external tools (e.g., LangChain, OpenAI APIs)

[ ] Add support for classification tasks

[ ] GUI for customizing pipelines and agent team setup

[ ] Evaluation benchmarking across multiple datasets



---

🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository


2. Create a new branch (git checkout -b feature-name)


3. Commit your changes


4. Push to the branch (git push origin feature-name)


5. Open a Pull Request 🚀




---

📬 Contact

Twitter: @AbhirMirikar

LinkedIn: Abhir Mirikar

GitHub: Abhir1902



---

📄 License

This project is licensed under the MIT License.
# Agentic-DS