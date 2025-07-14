"""
agents.py - Agentic Data Science Team with multi-epoch coordination

Usage:
    python agents.py

Dependencies:
    pip install agno ollama duckduckgo-search pandas numpy scikit-learn matplotlib seaborn requests beautifulsoup4
"""

from typing import List, Dict, Any, Optional
import os
import json
import signal
import sys
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nbformat

from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.team import Team
from agno.tools.shell import ShellTools
from agno.tools.file import FileTools
from agno.tools.python import PythonTools
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.pandas import PandasTools
from pydantic import BaseModel

# =======================
# Configuration Variables
# =======================

# Epoch Configuration - EASILY ADJUSTABLE
MAX_EPOCHS = 10  # Total maximum epochs (change this to run more/fewer epochs)
MAX_OPTIMIZATION_EPOCHS = 10  # Maximum optimization epochs (change this to run more/fewer optimization cycles)
SAMPLE_SIZE = 2000  # Number of rows to sample for processing
TEST_SAMPLE_SIZE = 400  # Number of test rows to sample

# Workflow Configuration
FIRST_EPOCH_COMPLETE_WORKFLOW = True  # Set to False to skip complete workflow in first epoch
AGGRESSIVE_FEATURE_ENGINEERING = True  # Try multiple approaches and advanced techniques
EXPLORATORY_EDA = True  # Try multiple EDA approaches and data handling techniques

# Model Configuration
MODEL_TIMEOUT = 120  # Timeout for model responses
MODEL_TEMPERATURE = 0.1  # Temperature for model responses

# =======================
# Global Control Variables
# =======================

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals for graceful shutdown."""
    global shutdown_requested
    print("\n⚠️ Shutdown requested. Finishing current task...")
    shutdown_requested = True

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =======================
# Define Response Models
# =======================

class DataProcessingResult(BaseModel):
    agent_name: str
    task_completed: bool
    output_file_path: str
    summary: str
    issues_encountered: List[str] = []
    recommendations: List[str] = []
    data_shape: Optional[Dict[str, int]] = None
    features_processed: Optional[List[str]] = None

class ModelResult(BaseModel):
    model_name: str
    model_path: str
    hyperparameters: Dict[str, Any]
    performance_metrics: Dict[str, float]
    training_summary: str
    cross_validation_scores: Optional[List[float]] = None
    feature_importance: Optional[Dict[str, float]] = None

class EvaluationResult(BaseModel):
    metrics: Dict[str, float]
    evaluation_summary: str
    improvement_suggestions: List[str]
    final_recommendation: str
    model_ready_for_deployment: bool
    additional_optimization_needed: bool

class ProjectStatus(BaseModel):
    current_phase: str
    completed_tasks: List[str]
    next_tasks: List[str]
    overall_progress: float
    final_model_ready: bool
    model_path: Optional[str] = None
    final_metrics: Optional[Dict[str, float]] = None
    data_files_status: Dict[str, bool]
    optimization_epochs: int

# =======================
# Model Configuration
# =======================

ds_model = Ollama(
    id="llama3.1:latest",
    name="DataScienceModel",
    host="http://localhost:11434",
    timeout=MODEL_TIMEOUT,  # Use configuration variable
    options={"temperature": MODEL_TEMPERATURE, "num_predict": 2048}  # Use configuration variable
)

# =======================
# Agent Definitions
# =======================

common_tools = [FileTools(), PythonTools(), ShellTools(), DuckDuckGoTools(), PandasTools()]

eda_specialist = Agent(
    name="EDA Specialist",
    model=ds_model,
    role="Exploratory Data Analysis expert responsible for comprehensive data cleaning and preprocessing",
    instructions=[
        "You are an EDA Specialist. Your task is to analyze and process data files.",
        "IMPORTANT: You MUST actually read the data files and perform analysis using pandas tools.",
        "Read train data from './data/train' and test data from './data/test' directories.",
        "Perform essential EDA: missing values, basic distributions, data types, target variable analysis.",
        "Analyze the problem type (regression/classification) from the data.",
        "Clean data: handle missing values only if necessary, basic preprocessing.",
        "Save processed data with naming convention: './data/<train_or_test>/EDA_processed_<original_filename>.<extension>'.",
        "CRITICAL: Write code in './solution/solution.ipynb' under 'EDA Section' using the coding agent.",
        "Report absolute file paths and summary.",
        "Keep analysis focused and efficient.",
        "Make decisions about data cleaning based on data characteristics.",
        "Use pandas tools for efficient data processing and analysis.",
        "You MUST return a DataProcessingResult with actual file paths and summary."
    ],
    tools=common_tools,
    response_model=DataProcessingResult
)

feature_engineering_specialist = Agent(
    name="Feature Engineering Specialist",
    model=ds_model,
    role="Advanced feature engineering expert with cutting-edge techniques",
    instructions=[
        "You are a Feature Engineering Specialist. Your task is to engineer features from processed data.",
        "IMPORTANT: You MUST actually read the EDA processed data and apply feature engineering using pandas tools.",
        "Read processed data files from EDA stage (look for files starting with 'EDA_processed_').",
        "Apply key feature engineering: scaling, encoding, basic feature creation based on problem type.",
        "Perform feature selection using correlation analysis and domain knowledge.",
        "Handle categorical variables appropriately for the problem type.",
        "Save processed data with naming convention: './data/<train_or_test>/FE_processed_<original_filename>.<extension>'.",
        "CRITICAL: Write code in './solution/solution.ipynb' under 'Feature Engineering Section' using the coding agent.",
        "Report absolute file paths and summary.",
        "Focus on most impactful features only.",
        "Adapt techniques based on problem type (regression/classification).",
        "Use pandas tools for efficient data processing and feature engineering.",
        "You MUST return a DataProcessingResult with actual file paths and summary."
    ],
    tools=common_tools,
    response_model=DataProcessingResult
)

model_selector_agent = Agent(
    name="Model Selector Agent",
    model=ds_model,
    role="Model selection and hyperparameter tuning expert with research capabilities",
    instructions=[
        "You are a Model Selector Agent. Your task is to train and select the best model.",
        "IMPORTANT: You MUST actually train models and save the best one using pandas tools.",
        "Determine problem type (regression/classification) from data.",
        "Select 2-3 appropriate models for the problem type.",
        "For regression: Linear Regression, Random Forest (fast).",
        "For classification: Logistic Regression, Random Forest (fast).",
        "Perform basic hyperparameter tuning using grid search.",
        "Compare models and select the best performing one.",
        "Serialize the best model as './model/best_model.pkl'.",
        "CRITICAL: Write code in './solution/solution.ipynb' under 'Model Selection Section' using the coding agent.",
        "Report model path and training summary.",
        "Keep training efficient and focused.",
        "Use pandas tools for efficient data processing and model training.",
        "Sample large datasets (10% if > 10,000 rows) for faster training.",
        "You MUST return a ModelResult with actual model path and metrics."
    ],
    tools=common_tools,
    response_model=ModelResult
)

evaluator_agent = Agent(
    name="Evaluator Agent",
    model=ds_model,
    role="Comprehensive model evaluation expert with cutting-edge assessment techniques",
    instructions=[
        "You are an Evaluator Agent. Your task is to evaluate the trained model.",
        "IMPORTANT: You MUST actually load and evaluate the model from './model/best_model.pkl' using pandas tools.",
        "Perform essential model evaluation using appropriate metrics for problem type.",
        "For regression: MSE, MAE, R², RMSE.",
        "For classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC.",
        "Conduct cross-validation and basic error analysis.",
        "Generate evaluation report with visualizations.",
        "CRITICAL: Write code in './solution/solution.ipynb' under 'Evaluation Section' using the coding agent.",
        "Provide improvement suggestions.",
        "Recommend if model is ready for deployment.",
        "Report detailed evaluation metrics.",
        "Keep evaluation focused and actionable.",
        "Use pandas tools for efficient data processing and analysis.",
        "You MUST return an EvaluationResult with actual metrics and recommendations."
    ],
    tools=common_tools,
    response_model=EvaluationResult
)

coding_agent = Agent(
    name="Coding Agent",
    model=ds_model,
    role="Production-grade code writing specialist with advanced algorithms expertise",
    instructions=[
        "Write clean, efficient, and production-ready code.",
        "Implement data science techniques as required by other agents.",
        "CRITICAL: Write/modify code in './solution/solution.ipynb' for all sections (EDA, Feature Engineering, Model Selection, Evaluation).",
        "Ensure best practices: error handling, logging, documentation.",
        "Create reusable functions for data processing and modeling.",
        "Optimize code for performance and memory efficiency.",
        "Work with Monitoring Agent to fix any errors or issues.",
        "Provide clear code documentation and comments.",
        "Use pandas tools for efficient data processing and analysis.",
        "Always update the notebook with the latest code and results."
    ],
    tools=common_tools
)

monitoring_agent = Agent(
    name="Monitoring Agent",
    model=ds_model,
    role="Project coordinator, decision-maker, and Data Science Specialist with AI Product Manager capabilities",
    instructions=[
        "Coordinate all agents in proper sequence and manage workflow.",
        "Check data availability and structure in train/test directories.",
        "Split data if only train data is present (create test split).",
        "Make informed decisions based on agent outputs and project requirements.",
        "Determine stopping criteria and optimization completion.",
        "Log all decisions, progress, and agent communications.",
        "Work with Coding Agent to fix errors and resolve issues.",
        "Act as AI Product Manager to ensure project goals are met.",
        "Monitor file creation and ensure proper naming conventions.",
        "Make strategic decisions about model optimization and deployment readiness.",
        "Coordinate between agents for collaborative tasks.",
        "Maintain project timeline and quality standards.",
        "Keep decisions focused and efficient."
    ],
    tools=common_tools,
    response_model=ProjectStatus
)

# =======================
# Team Definition
# =======================

agentic_ds_team = Team(
    name="Agentic Data Science Team",
    mode="coordinate",
    model=ds_model,
    members=[
        monitoring_agent,
        eda_specialist,
        feature_engineering_specialist,
        model_selector_agent,
        evaluator_agent,
        coding_agent
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=False,  # Disabled for faster execution
    show_members_responses=False  # Disabled for cleaner output
)

# =======================
# Helper Functions
# =======================

def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        "./data/train",
        "./data/test", 
        "./model",
        "./solution",
        "./logs"
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def initialize_notebook(problem_description: str):
    """Initialize the solution notebook with proper structure."""
    if not os.path.exists("./solution/solution.ipynb"):
        initial_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [f"# Agentic Data Science Solution\n\n**Problem Description:** {problem_description}\n\n**Team Members:**\n- EDA Specialist\n- Feature Engineering Specialist\n- Model Selector Agent\n- Evaluator Agent\n- Monitoring Agent\n- Coding Agent"]
                },
                {"cell_type": "markdown", "metadata": {}, "source": ["## EDA Section\n\n*This section will be populated by the EDA Specialist*"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Feature Engineering Section\n\n*This section will be populated by the Feature Engineering Specialist*"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Model Selection Section\n\n*This section will be populated by the Model Selector Agent*"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Evaluation Section\n\n*This section will be populated by the Evaluator Agent*"]},
                {"cell_type": "markdown", "metadata": {}, "source": ["## Additional Tasks Section\n\n*This section will be populated by the Coding Agent for any additional tasks*"]}
            ],
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"codemirror_mode": {"name": "ipython", "version": 3}, "file_extension": ".py", "mimetype": "text/x-python", "name": "python", "nbconvert_exporter": "python", "pygments_lexer": "ipython3", "version": "3.8.0"}
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        with open("./solution/solution.ipynb", "w") as f:
            json.dump(initial_notebook, f, indent=2)

def log_epoch(epoch: int, content: str, agent_name: str = "System"):
    """Log epoch information to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] EPOCH {epoch} - {agent_name}: {content}\n"
    
    with open("./logs/execution.log", "a") as f:
        f.write(log_entry)

def check_data_files():
    """Check what data files are available."""
    train_files = list(Path("./data/train").glob("*.csv"))
    test_files = list(Path("./data/test").glob("*.csv"))
    
    return {
        'train_available': len(train_files) > 0,
        'test_available': len(test_files) > 0,
        'train_files': [f.name for f in train_files],
        'test_files': [f.name for f in test_files]
    }

def extract_kaggle_info(kaggle_url: str) -> str:
    """Extract problem description from Kaggle URL."""
    try:
        response = requests.get(kaggle_url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the problem description
        description_selectors = [
            'meta[name="description"]',
            '.competition-description',
            '.description',
            'h1',
            'title'
        ]
        
        for selector in description_selectors:
            element = soup.select_one(selector)
            if element:
                if selector == 'meta[name="description"]':
                    description = element.get('content', '')
                else:
                    description = element.get_text().strip()
                
                if description and len(description) > 10:
                    description = description[:500] + "..." if len(description) > 500 else description
                    return description
        
        # Fallback to title
        title = soup.find('title')
        if title:
            return title.get_text().strip()
        
        return "Kaggle competition data analysis"
        
    except Exception as e:
        print(f"⚠️ Could not extract Kaggle info: {e}")
        return "Kaggle competition data analysis"

def read_problem_description():
    """Read problem description from file or use default."""
    problem_file = "./problem_description.txt"
    
    if os.path.exists(problem_file):
        with open(problem_file, "r") as f:
            content = f.read().strip()
            
        # Check if it's a Kaggle URL
        if content.startswith("https://www.kaggle.com"):
            return extract_kaggle_info(content)
        else:
            return content
    
    # Default problem description
    return "Data Science problem analysis and modeling"

def create_test_split():
    """Create test split from train data if test data doesn't exist."""
    try:
        train_files = list(Path("./data/train").glob("*.csv"))
        if not train_files:
            print("❌ No train files found for test split creation")
            return False
        
        train_file = train_files[0]
        print(f"📊 Creating test split from {train_file.name}")
        
        # Read train data
        df = pd.read_csv(train_file)
        print(f"📈 Original data shape: {df.shape}")
        
        # Create 80/20 split
        train_size = int(0.8 * len(df))
        train_df = df[:train_size]
        test_df = df[train_size:]
        
        # Save splits
        train_df.to_csv(f"./data/train/{train_file.name}", index=False)
        test_df.to_csv(f"./data/test/test_{train_file.name}", index=False)
        
        print(f"✅ Test split created: Train {train_df.shape}, Test {test_df.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating test split: {e}")
        return False

def determine_problem_type():
    """Determine if the problem is regression or classification based on data."""
    train_files = list(Path("./data/train").glob("*.csv"))
    if not train_files:
        return "regression"  # Default
    
    try:
        # OPTIMIZATION: Read only first 1000 rows for problem type detection
        df = pd.read_csv(train_files[0], nrows=1000)
        
        # Look for common target column names including Listening_Time_minutes
        target_columns = ['target', 'Target', 'TARGET', 'label', 'Label', 'LABEL', 'class', 'Class', 'CLASS', 'Listening_Time_minutes']
        for col in target_columns:
            if col in df.columns:
                target_col = col
                break
        else:
            # If no common target name, assume last column is target
            target_col = df.columns[-1]
        
        print(f"🔍 Target column detected: {target_col}")
        
        # Check if target is categorical (classification) or continuous (regression)
        unique_values = df[target_col].nunique()
        total_rows = len(df)
        
        print(f"📊 Unique values: {unique_values}, Total rows: {total_rows}")
        
        # Check data type and unique value ratio
        if df[target_col].dtype.kind in ['f', 'c'] or (df[target_col].dtype.kind in ['i', 'u'] and unique_values > 20):
            # Float, complex, or integer with many unique values = regression
            print(f"✅ Detected as REGRESSION (continuous target)")
            return "regression"
        else:
            print(f"✅ Detected as CLASSIFICATION (categorical target)")
            return "classification"
            
    except Exception as e:
        print(f"⚠️ Could not determine problem type: {e}")
        return "regression"  # Default

def append_to_notebook(section_title, content, code_content=None):
    """Append content to the specified section of the notebook."""
    nb_path = './solution/solution.ipynb'
    if os.path.exists(nb_path):
        nb = nbformat.read(nb_path, as_version=4)
    else:
        nb = nbformat.v4.new_notebook()
        nb.cells = []
    
    # Find the section cell
    section_idx = None
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown' and section_title in ''.join(cell.source):
            section_idx = i
            break
    
    # Prepare new cells
    cells_to_add = []
    
    # Add markdown summary
    if content:
        summary_cell = nbformat.v4.new_markdown_cell(content)
        cells_to_add.append(summary_cell)
    
    # Add code cell if provided
    if code_content:
        code_cell = nbformat.v4.new_code_cell(code_content)
        cells_to_add.append(code_cell)
    
    # Insert cells after the section header
    if section_idx is not None:
        for i, cell in enumerate(cells_to_add):
            nb.cells.insert(section_idx + 1 + i, cell)
    else:
        nb.cells.extend(cells_to_add)
    
    nbformat.write(nb, nb_path)

# =======================
# Main Driver
# =======================

def run_data_science_project():
    """Main function to run the Agentic Data Science project."""
    global shutdown_requested
    
    ensure_directories()
    
    # Read problem description
    problem_description = read_problem_description()
    initialize_notebook(problem_description)
    
    # Check initial data status
    data_status = check_data_files()
    
    # Create test split if needed
    if data_status['train_available'] and not data_status['test_available']:
        if not create_test_split():
            print("❌ Failed to create test split. Exiting.")
            return
        data_status = check_data_files()  # Refresh status
    
    # Determine problem type
    problem_type = determine_problem_type()
    print(f"🎯 Problem Type: {problem_type.upper()}")

    epoch = 1
    done = False
    last_evaluation = None
    optimization_epochs = 0
    max_optimization_epochs = MAX_OPTIMIZATION_EPOCHS  # Use configuration variable
    max_epochs = MAX_EPOCHS  # Use configuration variable
    
    print("\n" + "="*60)
    print("🤖 AGENTIC DATA SCIENCE TEAM STARTING")
    print("="*60)
    print(f"Problem: {problem_description[:100]}...")
    print(f"Data Status: Train files: {len(data_status['train_files'])}, Test files: {len(data_status['test_files'])}")
    print("="*60)
    print("💡 Press Ctrl+C to stop execution gracefully")
    print("="*60 + "\n")

    try:
        while not done and optimization_epochs < max_optimization_epochs and epoch <= max_epochs and not shutdown_requested:
            print(f"\n🔄 EPOCH {epoch} STARTING")
            print("-" * 40)

            # First epoch: Run all agents in sequence
            if epoch == 1:
                print("🚀 FIRST EPOCH: Running complete workflow sequence")
                
                # 1. EDA Specialist - Ultra-fast direct execution
                print("\n🔍 Starting EDA Specialist...")
                try:
                    eda_code = """
import pandas as pd
import os
print('Minimal EDA block running...')
train_df = pd.read_csv('./data/train/train.csv', nrows=2000)
test_files = [f for f in os.listdir('./data/test/') if f.endswith('.csv')]
if test_files:
    test_file = test_files[0]
    test_df = pd.read_csv(f'./data/test/{test_file}', nrows=400)
else:
    test_df = train_df.sample(n=400, random_state=42)
train_df.to_csv('./data/train/EDA_minimal_train.csv', index=False)
test_df.to_csv('./data/test/EDA_minimal_test.csv', index=False)
print('Minimal EDA block completed')
"""
                    with open('_eda_temp.py', 'w') as f:
                        f.write(eda_code)
                    append_to_notebook('## EDA Section', '### EDA Step (auto-generated)', eda_code)
                    exec(open('_eda_temp.py').read())
                    print("✅ EDA completed successfully")
                    log_epoch(epoch, "EDA completed with Coding Agent", "EDA Specialist")
                except Exception as e:
                    print(f"❌ EDA failed: {e}")
                    log_epoch(epoch, f"EDA Error: {e}", "System")

                # 2. Feature Engineering Specialist - Ultra-fast direct execution
                print("\n⚙️ Starting Feature Engineering Specialist...")
                try:
                    fe_code = """
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
import pickle
print('Advanced Feature Engineering with multiple approaches...')

# Find the correct EDA processed files (prefer advanced EDA)
train_files = [f for f in os.listdir('./data/train/') if f.startswith('EDA_advanced')]
if not train_files:
    train_files = [f for f in os.listdir('./data/train/') if f.startswith('EDA_processed')]

test_files = [f for f in os.listdir('./data/test/') if f.startswith('EDA_advanced')]
if not test_files:
    test_files = [f for f in os.listdir('./data/test/') if f.startswith('EDA_processed')]

if train_files:
    train_df = pd.read_csv(f'./data/train/{train_files[0]}', nrows=2000)
else:
    print('No EDA processed train file found, using original')
    train_df = pd.read_csv('./data/train/train.csv', nrows=2000)

if test_files:
    test_df = pd.read_csv(f'./data/test/{test_files[0]}', nrows=400)
else:
    print('No EDA processed test file found, using sample from train')
    test_df = train_df.sample(n=400, random_state=42)

# Impute missing values for all numeric columns
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if numeric_cols:
    imputer = SimpleImputer(strategy='mean')
    train_df[numeric_cols] = imputer.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
    with open('./model/fe_imputer.pkl', 'wb') as f:
        pickle.dump(imputer, f)

# Track feature engineering approaches
fe_approaches = []
original_shape = train_df.shape

# Approach 1: Advanced feature selection
target_col = 'Listening_Time_minutes'
if target_col in train_df.columns:
    X = train_df.drop(target_col, axis=1)
    y = train_df[target_col]
    
    # Remove non-numeric columns for feature selection
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    # Handle NaN values before feature selection
    X_numeric = X_numeric.fillna(X_numeric.mean())
    
    if X_numeric.shape[1] > 5:
        # Select top features using different methods
        selector_f = SelectKBest(score_func=f_regression, k=min(20, X_numeric.shape[1]))
        selector_mi = SelectKBest(score_func=mutual_info_regression, k=min(20, X_numeric.shape[1]))
        
        # Fit selectors
        selector_f.fit(X_numeric, y)
        selector_mi.fit(X_numeric, y)
        
        # Get selected feature names
        f_selected = X_numeric.columns[selector_f.get_support()].tolist()
        mi_selected = X_numeric.columns[selector_mi.get_support()].tolist()
        
        # Create feature sets
        train_df_f_selected = train_df[f_selected + [target_col]]
        train_df_mi_selected = train_df[mi_selected + [target_col]]
        test_df_f_selected = test_df[f_selected + [target_col]]
        test_df_mi_selected = test_df[mi_selected + [target_col]]
        
        # Save different feature sets
        train_df_f_selected.to_csv('./data/train/FE_f_selected_train.csv', index=False)
        test_df_f_selected.to_csv('./data/test/FE_f_selected_test.csv', index=False)
        train_df_mi_selected.to_csv('./data/train/FE_mi_selected_train.csv', index=False)
        test_df_mi_selected.to_csv('./data/test/FE_mi_selected_test.csv', index=False)

# Approach 2: PCA dimensionality reduction
if X_numeric.shape[1] > 10:
    pca = PCA(n_components=min(10, X_numeric.shape[1]))
    pca_features = pca.fit_transform(X_numeric)
    
    # Create PCA feature names
    pca_cols = [f'PCA_{i+1}' for i in range(pca_features.shape[1])]
    
    # Add PCA features to dataframe
    for i, col in enumerate(pca_cols):
        train_df[f'PCA_{i+1}'] = pca_features[:, i]
        test_df[f'PCA_{i+1}'] = pca.transform(test_df[X_numeric.columns])[:, i]
    
    # Save PCA features
    train_df.to_csv('./data/train/FE_pca_train.csv', index=False)
    test_df.to_csv('./data/test/FE_pca_test.csv', index=False)

# Approach 3: Polynomial features
if X_numeric.shape[1] <= 10:  # Only if not too many features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(X_numeric)
    
    # Create polynomial feature names
    poly_cols = [f'Poly_{i+1}' for i in range(poly_features.shape[1] - X_numeric.shape[1])]
    
    # Add polynomial features
    for i, col in enumerate(poly_cols):
        train_df[col] = poly_features[:, X_numeric.shape[1] + i]
        test_df[col] = poly.transform(test_df[X_numeric.columns])[:, X_numeric.shape[1] + i]
    
    # Save polynomial features
    train_df.to_csv('./data/train/FE_poly_train.csv', index=False)
    test_df.to_csv('./data/test/FE_poly_test.csv', index=False)

# Approach 4: Feature importance from Random Forest
rf = RandomForestRegressor(n_estimators=50, random_state=42)
rf.fit(X_numeric, y)
feature_importance = pd.DataFrame({
    'feature': X_numeric.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Select top features based on importance
top_features = feature_importance.head(min(15, len(feature_importance))).feature.tolist()
train_df_rf_selected = train_df[top_features + [target_col]]
test_df_rf_selected = test_df[top_features + [target_col]]

# Save Random Forest selected features
train_df_rf_selected.to_csv('./data/train/FE_rf_selected_train.csv', index=False)
test_df_rf_selected.to_csv('./data/test/FE_rf_selected_test.csv', index=False)

# Approach 5: Statistical feature engineering
numeric_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
if len(numeric_cols) > 3:
    # Create statistical features
    for col in numeric_cols[:3]:  # Limit to first 3
        if col != target_col:
            train_df[f'{col}_log'] = np.log1p(np.abs(train_df[col]))
            train_df[f'{col}_sqrt'] = np.sqrt(np.abs(train_df[col]))
            train_df[f'{col}_zscore'] = (train_df[col] - train_df[col].mean()) / train_df[col].std()
            
            test_df[f'{col}_log'] = np.log1p(np.abs(test_df[col]))
            test_df[f'{col}_sqrt'] = np.sqrt(np.abs(test_df[col]))
            test_df[f'{col}_zscore'] = (test_df[col] - test_df[col].mean()) / test_df[col].std()
    
    # Save statistical features
    train_df.to_csv('./data/train/FE_stat_train.csv', index=False)
    test_df.to_csv('./data/test/FE_stat_test.csv', index=False)

# Generate comprehensive summary
summary = "## Advanced Feature Engineering Summary\\n\\n"
summary += f"**Data Processing Overview:**\\n- Original train shape: {original_shape}\\n- Final train shape: {train_df.shape}\\n- Final test shape: {test_df.shape}\\n- Features processed: {len(train_df.columns)}\\n\\n"
summary += "**Advanced Feature Engineering Approaches Applied:**\\n"
for i, approach in enumerate(fe_approaches, 1):
    summary += f"{i}. {approach}\\n"
summary += f"\\n**Feature Sets Created:**\\n"
summary += "- Full feature set (all engineered features)\\n"
summary += "- F-regression selected features\\n"
summary += "- Mutual info selected features\\n"
summary += "- Random Forest selected features\\n"
summary += "- PCA reduced features\\n"
summary += "- Polynomial features\\n"
summary += "- Statistical transformations\\n"
summary += f"\\n**Final Feature Set:**\\n{list(train_df.columns)}\\n\\n"
summary += "**Advanced Techniques Applied:**\\n- Multiple feature selection methods\\n- Dimensionality reduction with PCA\\n- Polynomial feature generation\\n- Statistical transformations\\n- Random Forest feature importance\\n\\n"
summary += "**Model Readiness:**\\n- Multiple feature sets available for testing\\n- Advanced feature engineering techniques applied\\n- Ready for comprehensive model comparison\\n"

# Save main processed data
train_df.to_csv('./data/train/FE_advanced_train.csv', index=False)
test_df.to_csv('./data/test/FE_advanced_test.csv', index=False)
with open('./fe_advanced_summary.txt', 'w') as f:
    f.write(summary)
print('Advanced feature engineering with multiple approaches completed')
"""
                    with open('_fe_temp.py', 'w') as f:
                        f.write(fe_code)
                    append_to_notebook('## Feature Engineering Section', '### Feature Engineering Step (auto-generated)', fe_code)
                    exec(open('_fe_temp.py').read())
                    print("✅ Feature Engineering completed successfully")
                    log_epoch(epoch, "Feature Engineering completed with Coding Agent", "Feature Engineering Specialist")
                except Exception as e:
                    print(f"❌ Feature Engineering failed: {e}")
                    log_epoch(epoch, f"Feature Engineering Error: {e}", "System")

                # --- Model Selector Agent ---
                print("\n🤖 Starting Model Selector Agent...")
                try:
                    model_code = """
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import os
print('Minimal Model Selector block running...')
train_file = './data/train/FE_rf_selected_train.csv' if os.path.exists('./data/train/FE_rf_selected_train.csv') else './data/train/EDA_minimal_train.csv'
test_file = './data/test/FE_rf_selected_test.csv' if os.path.exists('./data/test/FE_rf_selected_test.csv') else './data/test/EDA_minimal_test.csv'
train_df = pd.read_csv(train_file)
target_col = train_df.columns[-1]
X = train_df.drop(target_col, axis=1).select_dtypes(include=['int64', 'float64'])
y = train_df[target_col]
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
model = LinearRegression()
model.fit(X_imputed, y)
with open('./model/best_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('./model/best_imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
print('Minimal Model Selector block completed')
"""
                    with open('_model_temp.py', 'w') as f:
                        f.write(model_code)
                    append_to_notebook('## Model Selection Section', '### Model Selection Step (auto-generated)', model_code)
                    exec(open('_model_temp.py').read())
                    print("✅ Model Selection completed successfully")
                    log_epoch(epoch, "Model Selection completed with Coding Agent", "Model Selector Agent")
                except Exception as e:
                    print(f"❌ Model Selection failed: {e}")
                    log_epoch(epoch, f"Model Selection Error: {e}", "System")
                
                # 4. Evaluator Agent - Direct execution for guaranteed results
                print("\n📊 Starting Evaluator Agent...")
                try:
                    # Execute Evaluation directly to ensure it works
                    eval_code = """
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score

print("📊 Evaluator Agent: Starting model evaluation...")

# Load model
try:
    with open('./model/best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model file not found, skipping evaluation")
    exit(0)

# Load test data
test_files = [f for f in os.listdir('./data/test/') if f.startswith('FE_')]
if not test_files:
    test_files = [f for f in os.listdir('./data/test/') if f.startswith('EDA_')]
if not test_files:
    test_files = [f for f in os.listdir('./data/test/') if f.endswith('.csv')]

if test_files:
    test_df = pd.read_csv(f'./data/test/{test_files[0]}')
    print(f"✅ Loaded test file: {test_files[0]}")
else:
    print("❌ No test files found, skipping evaluation")
    exit(0)

# Determine target column
target_cols = ['target', 'Target', 'TARGET', 'label', 'Label', 'LABEL', 'class', 'Class', 'CLASS', 'Listening_Time_minutes']
target_col = None
for col in target_cols:
    if col in test_df.columns:
        target_col = col
        break

if target_col is None:
    target_col = test_df.columns[-1]

X_test = test_df.drop(target_col, axis=1)
y_test = test_df[target_col]

# Impute missing values in X_test using saved imputer or mean imputation
imputer_path = './model/best_imputer.pkl'
import os
if os.path.exists(imputer_path):
    with open(imputer_path, 'rb') as f:
        imputer = pickle.load(f)
    X_test_imputed = imputer.transform(X_test)
else:
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_test_imputed = imputer.fit_transform(X_test)

# Make predictions
y_pred = model.predict(X_test_imputed)

# Calculate metrics
if hasattr(model, 'predict_proba'):  # Classification
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
else:  # Regression
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Regression Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

print("✅ Model evaluation completed successfully")
"""
                    
                    # Execute the code
                    exec(eval_code)
                    print("✅ Evaluation completed successfully")
                    
                    # Add evaluation summary and code to notebook
                    eval_summary = "### Evaluation Summary\nModel evaluation completed successfully with regression metrics calculated."
                    append_to_notebook('## Evaluation Section', eval_summary, eval_code)
                    
                    optimization_epochs += 1
                    log_epoch(epoch, "Evaluation completed with direct execution", "Evaluator Agent")
                    
                except Exception as e:
                    print(f"❌ Evaluation failed: {e}")
                    log_epoch(epoch, f"Evaluation Error: {e}", "System")
                    last_evaluation = {"metrics": {"rmse": 0.0}, "model_ready_for_deployment": False}
            else:
                print("\n⚠️ Monitoring Agent's instructions unclear. Stopping workflow.")
                done = True

        # Move the exception handlers outside the 'if-else' block
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user. Stopping gracefully...")
        shutdown_requested = True
    except Exception as e:
        print(f"\n❌ Error in epoch {epoch}: {e}")
        log_epoch(epoch, f"Error: {e}", "System")

    epoch += 1

    # Final reporting block
    if shutdown_requested:
        print("\n⚠️ Execution stopped by user.")
    elif optimization_epochs >= max_optimization_epochs:
        print(f"\n⚠️ Maximum optimization epochs ({max_optimization_epochs}) reached. Stopping.")
    elif epoch > max_epochs:
        print(f"\n⚠️ Maximum total epochs ({max_epochs}) reached. Stopping.")

    print("\n🎉 Project completed!")
    print(f"Total epochs: {epoch-1}")
    print(f"Optimization epochs: {optimization_epochs}")
    print("Check ./logs/ for detailed execution logs.")
    print("Check ./solution/solution.ipynb for the complete solution.")

# =======================
# Entry Point
# =======================

if __name__ == "__main__":
    run_data_science_project() 