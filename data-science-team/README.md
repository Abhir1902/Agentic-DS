# Agentic Data Science Team

An autonomous team of AI agents designed to solve complex data science problems using the Agno framework.

## 🎯 Overview

This project implements a team of specialized AI agents that work together to solve data science problems:

1. **EDA Specialist** - Performs exploratory data analysis and data cleaning
2. **Feature Engineering Specialist** - Creates advanced features and performs feature selection
3. **Model Selector Agent** - Researches and selects optimal models with hyperparameter tuning
4. **Evaluator Agent** - Performs comprehensive model evaluation and provides recommendations
5. **Monitoring Agent** - Coordinates the team and makes strategic decisions
6. **Coding Agent** - Writes production-grade code and handles technical implementation

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **Ollama** running locally with llama3.1 model
3. **Git** for version control

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd data-science-team
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Ollama (if not already running):**
   ```bash
   ollama serve
   ```

4. **Pull the required model:**
   ```bash
   ollama pull llama3.1:latest
   ```

### Usage

1. **Prepare your data:**
   - Place training data in `./data/train/`
   - Place test data in `./data/test/` (optional - will be created if missing)
   - Update `./data/data.md` with your problem description

2. **Run the team:**
   ```bash
   python agents/agents.py
   ```

3. **Monitor progress:**
   - Check `./logs/` for detailed execution logs
   - View `./solution/solution.ipynb` for the complete solution
   - Find processed data in `./data/train/` and `./data/test/`
   - Locate trained models in `./model/`

## 📁 Project Structure

```
data-science-team/
├── agents/
│   └── agents.py              # Main agent definitions and team coordination
├── data/
│   ├── train/                 # Training data directory
│   ├── test/                  # Test data directory
│   └── data.md               # Problem description
├── logs/                     # Execution logs for each epoch
├── model/                    # Trained models (model.pkl/model.h5)
├── solution/
│   └── solution.ipynb        # Complete solution notebook
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## 🤖 Agent Capabilities

### EDA Specialist
- **Data Analysis**: Missing values, outliers, distributions, correlations
- **Data Cleaning**: Handle missing data, outliers, data type conversions
- **Visualization**: Generate comprehensive data quality reports
- **Output**: Cleaned data with naming convention `EDA_agent_<filename>.<ext>`

### Feature Engineering Specialist
- **Advanced Techniques**: Scaling, normalization, polynomial features, interactions
- **Feature Selection**: Statistical methods, correlation analysis, domain knowledge
- **Cross-validation**: Validate feature engineering decisions
- **Output**: Engineered data with naming convention `FE_agent_<filename>.<ext>`

### Model Selector Agent
- **Research**: Latest ML/DL models and techniques
- **Selection**: Choose optimal models based on data characteristics
- **Hyperparameter Tuning**: Grid search, random search, Bayesian optimization
- **Output**: Trained model saved as `model.pkl` or `model.h5`

### Evaluator Agent
- **Comprehensive Evaluation**: Multiple metrics, cross-validation, statistical testing
- **Error Analysis**: Identify patterns in mispredictions
- **Recommendations**: Suggest improvements or deployment readiness
- **Output**: Detailed evaluation reports and recommendations

### Monitoring Agent
- **Coordination**: Manage workflow and agent sequence
- **Decision Making**: Strategic decisions about optimization and deployment
- **Data Management**: Handle data splitting and file organization
- **Quality Control**: Ensure project goals and standards are met

### Coding Agent
- **Production Code**: Write clean, efficient, well-documented code
- **Best Practices**: Error handling, logging, optimization
- **Implementation**: Execute advanced algorithms and techniques
- **Collaboration**: Work with other agents to implement solutions

## 🔄 Workflow

The team operates in epochs with the following workflow:

1. **Epoch 1**: EDA → Feature Engineering → Model Selection → Evaluation
2. **Subsequent Epochs**: Monitoring Agent decides next steps based on evaluation results
3. **Optimization**: Iterative improvements until stopping criteria are met
4. **Completion**: Final model and comprehensive solution delivered

## 📊 Output Files

### Data Files
- `./data/train/EDA_agent_*.csv` - EDA processed training data
- `./data/test/EDA_agent_*.csv` - EDA processed test data
- `./data/train/FE_agent_*.csv` - Feature engineered training data
- `./data/test/FE_agent_*.csv` - Feature engineered test data

### Model Files
- `./model/model.pkl` or `./model/model.h5` - Trained model

### Documentation
- `./solution/solution.ipynb` - Complete solution with all code and analysis
- `./logs/epoch_*.txt` - Detailed logs for each epoch

## ⚙️ Configuration

### Model Configuration
The team uses Ollama with llama3.1 model. You can modify the model configuration in `agents/agents.py`:

```python
ds_model = Ollama(
    id="llama3.1:latest",
    name="DataScienceModel",
    host="http://localhost:11434",
    timeout=300,
    options={"temperature": 0.1, "num_predict": 4096}
)
```

### Optimization Settings
- **Max Optimization Epochs**: 5 (configurable in `run_data_science_project()`)
- **Stopping Criteria**: Determined by Monitoring Agent based on evaluation results

## 🛠️ Customization

### Adding New Agents
1. Define the agent using the `Agent` class
2. Add to the team members list
3. Update the workflow logic in the main function

### Modifying Agent Instructions
Each agent has detailed instructions that can be customized for specific use cases or requirements.

### Extending Capabilities
The framework supports adding new tools and capabilities to agents through the Agno tools system.

## 🔒 Security and Safety

- **Data Privacy**: No data is shared externally
- **Code Safety**: All code is executed locally
- **Error Handling**: Comprehensive error handling and logging
- **Validation**: Input validation and sanitization

## 📝 Logging

Detailed logs are maintained for:
- Each epoch execution
- Agent decisions and actions
- Error messages and resolutions
- Performance metrics and results

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Ollama not running**: Start Ollama with `ollama serve`
2. **Model not found**: Pull the model with `ollama pull llama3.1:latest`
3. **Dependencies missing**: Install with `pip install -r requirements.txt`
4. **Permission errors**: Ensure write permissions for data and logs directories

### Getting Help

- Check the logs in `./logs/` for detailed error information
- Review the agent outputs for specific issues
- Ensure all prerequisites are properly installed

## 🎉 Success Stories

The Agentic Data Science Team has been successfully used for:
- Regression problems with complex feature engineering
- Classification tasks with multiple algorithms
- Time series forecasting with advanced models
- Anomaly detection with unsupervised learning

---

**Happy Data Science! 🚀** 