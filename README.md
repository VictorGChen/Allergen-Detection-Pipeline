# Allergy Detection Pipeline

A comprehensive machine learning pipeline for allergen detection and IgE level prediction using NHANES and ImmPort datasets. This pipeline implements Random Forest regression/classification models with SHAP-based interpretability analysis.

## Features

- **Multi-source Data Integration**: Combines NHANES clinical data with ImmPort immunological datasets
- **Comprehensive Feature Engineering**: 100+ engineered features including sensitization patterns, demographic factors, and seasonal indicators
- **Multiple Model Types**: Random Forest, Gradient Boosting, Logistic Regression, and Neural Networks
- **SHAP Analysis**: Model interpretability with feature importance and interaction analysis
- **Production Ready**: Robust error handling, logging, and configuration management
- **Mock Data Generation**: Built-in synthetic data generation for testing and demonstration

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/allergy-detection-pipeline.git
cd allergy-detection-pipeline

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Run complete pipeline with mock data (no credentials needed)
python main_pipeline.py

# Run with custom configuration
python main_pipeline.py --config my_config.json

# Run in demo mode (smaller dataset, faster execution)
python main_pipeline.py --demo

# Skip certain steps (useful for development)
python main_pipeline.py --skip ingestion engineering
```

### Python API Usage

```python
from main_pipeline import AllergyDetectionPipeline
from config import PipelineConfig

# Initialize pipeline
config = PipelineConfig()
pipeline = AllergyDetectionPipeline(config)

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Or run individual components
nhanes_data, immport_data = pipeline.data_ingestion.ingest_all_data()
engineered_features = pipeline.feature_engineer.engineer_features(nhanes_data)
training_results = pipeline.model_trainer.train_all_models(engineered_features)
```

## Project Structure

```
allergy-detection-pipeline/
├── config.py                 # Configuration management
├── data_ingestion.py         # NHANES/ImmPort data downloading
├── feature_engineering.py    # Feature creation and transformation
├── model_training.py         # Model training and evaluation
├── shap_analysis.py          # SHAP interpretability analysis
├── main_pipeline.py          # Main orchestrator
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
├── data/                     # Data directory (auto-created)
│   ├── raw/                  # Original downloaded data
│   ├── interim/              # Intermediate processing
│   ├── processed/            # Final processed features
│   └── cache/                # Cached downloads
│
├── outputs/                  # Output directory (auto-created)
│   ├── models/               # Trained model files
│   ├── results/              # Training results JSON
│   ├── shap/                 # SHAP visualizations
│   ├── plots/                # General visualizations
│   └── reports/              # Final reports
│
└── logs/                     # Log files (auto-created)
```

## Pipeline Components

### 1. Data Ingestion (`data_ingestion.py`)

- Downloads NHANES XPT files programmatically
- Authenticates with ImmPort API for immunological data
- Falls back to realistic mock data generation when APIs unavailable
- Handles data caching for efficient re-runs

**Key NHANES Datasets:**
- AL_IGE_D: Allergen-specific and total IgE measurements
- DEMO_D: Demographics
- BMX_D: Body measurements
- DR1TOT_D: Dietary intake
- AGQ_D: Allergy questionnaire

### 2. Feature Engineering (`feature_engineering.py`)

**Features Created:**
- **Sensitization Indicators**: Binary flags for 19 allergen sensitizations
- **Aggregated Measures**: Total sensitizations, poly-sensitization status
- **Allergen Categories**: Indoor, outdoor, and food allergen groups
- **Demographic Features**: Age groups, income categories, education levels
- **IgE Statistics**: Mean, median, max specific IgE levels
- **Seasonal Features**: Interview month, pollen season indicators
- **Body Measurements**: BMI categories, height-weight ratios
- **Dietary Patterns**: Macronutrient percentages, caloric intake categories

**Advanced Techniques:**
- Log transformation for IgE values
- Missing value imputation (median for numeric, mode for categorical)
- SMOTE-like augmentation for imbalanced classes
- Polynomial feature generation
- Feature interactions (age-IgE, season-allergen)

### 3. Model Training (`model_training.py`)

**Classification Tasks:**
- Poly-sensitization prediction (≥3 allergen sensitizations)
- IgE class prediction (Normal/Borderline/Elevated/Very High)
- Binary elevated IgE detection

**Regression Tasks:**
- Log-transformed total IgE level prediction
- Individual allergen-specific IgE prediction

**Models Implemented:**
- Random Forest (primary model)
- Gradient Boosting
- Logistic/Ridge Regression
- Neural Networks (if TensorFlow installed)

**Evaluation Metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression: MSE, RMSE, MAE, R²
- Cross-validation scores

### 4. SHAP Analysis (`shap_analysis.py`)

**Interpretability Features:**
- Global feature importance rankings
- Individual prediction explanations
- Feature interaction analysis
- Allergen-specific pattern recognition

**Visualizations Generated:**
- Summary plots showing feature impact
- Bar plots of top important features
- Waterfall plots for individual predictions
- Force plots for multiple predictions (HTML)
- Model comparison plots

## Configuration

The pipeline uses a comprehensive configuration system (`config.py`):

```python
# Example configuration usage
from config import PipelineConfig

config = PipelineConfig()

# Adjust data settings
config.data.ige_threshold = 0.35  # kU/L for sensitization
config.data.missing_threshold = 0.3  # Max 30% missing values

# Adjust model settings
config.model.n_estimators = 200  # Random Forest trees
config.model.test_size = 0.2  # 80-20 train-test split

# Adjust analysis settings
config.analysis.shap_background_samples = 100
config.analysis.max_display_features = 20

# Save configuration
config.save_to_file(Path("my_config.json"))
```

## Data Sources

### NHANES (National Health and Nutrition Examination Survey)
- **Source**: CDC (https://wwwn.cdc.gov/nchs/nhanes/)
- **Cycle**: 2005-2006 (primary for IgE data)
- **Access**: Direct download, no authentication required

### ImmPort (Immunology Database and Analysis Portal)
- **Source**: NIAID (https://www.immport.org)
- **Access**: Requires free registration
- **Data Types**: ELISA, Flow Cytometry, Gene Expression

## Environment Variables

For real data access (optional):
```bash
export IMMPORT_USERNAME="your_username"
export IMMPORT_PASSWORD="your_password"
```

## Performance Considerations

- **Memory Usage**: ~2-4 GB for full pipeline with 10,000 samples
- **Execution Time**: 
  - Mock data: 5-10 minutes
  - Real data download: 15-30 minutes (depends on connection)
  - Full pipeline: 20-45 minutes
- **Disk Space**: ~500 MB for cached data and outputs

## Testing

Run the test suite:
```bash
pytest tests/ -v --cov=.
```

## Known Limitations

1. **Data Availability**: NHANES IgE data primarily from 2005-2006 cycle
2. **ImmPort Integration**: Limited to publicly available studies
3. **SHAP Analysis**: Computationally intensive for large datasets
4. **Neural Networks**: Requires TensorFlow installation

## Troubleshooting

### Common Issues

1. **ImportError for pyreadstat**:
   ```bash
   pip install pyreadstat --no-binary :all:
   ```

2. **SHAP installation issues**:
   ```bash
   pip install shap --no-binary shap
   ```

3. **Memory errors with large datasets**:
   - Reduce `shap_test_samples` in configuration
   - Use `--demo` mode for testing

4. **No data downloaded**:
   - Pipeline automatically falls back to mock data
   - Check internet connection for real data access

## Citation

If you use this pipeline in research, please cite:

```
@software{allergy_detection_pipeline,
  title = {Allergy Detection Pipeline: ML-based IgE Analysis},
  author = {Victor Chen},
  year = {2024},
  url = {https://github.com/yourusername/allergy-detection-pipeline}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request
