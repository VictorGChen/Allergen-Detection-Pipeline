"""
Configuration management for the allergy detection pipeline.
This module handles all configuration settings and constants.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import json

@dataclass
class DataConfig:
    """Configuration for data sources and paths"""
    # NHANES settings
    nhanes_base_url: str = "https://wwwn.cdc.gov/Nchs/Nhanes"
    nhanes_cycle: str = "2005-2006"
    nhanes_files: Dict[str, str] = field(default_factory=lambda: {
        'ige_data': 'AL_IGE_D.XPT',
        'demographics': 'DEMO_D.XPT', 
        'questionnaire': 'AGQ_D.XPT',
        'dietary': 'DR1TOT_D.XPT',
        'exam': 'BMX_D.XPT'
    })
    
    # ImmPort settings
    immport_base_url: str = "https://www.immport.org"
    immport_username: Optional[str] = None
    immport_password: Optional[str] = None
    
    # Local paths
    raw_data_dir: Path = Path("data/raw")
    interim_data_dir: Path = Path("data/interim")
    processed_data_dir: Path = Path("data/processed")
    cache_dir: Path = Path("data/cache")
    
    # Data processing settings
    ige_threshold: float = 0.35  # kU/L for allergen sensitization
    total_ige_threshold: float = 100.0  # kU/L for elevated total IgE
    missing_threshold: float = 0.3  # Maximum proportion of missing values
    
    # Feature engineering thresholds
    poly_sensitization_threshold: int = 3  # Number of sensitizations for poly-sensitization
    severe_imbalance_ratio: float = 3.0  # Ratio threshold for class imbalance
    low_income_threshold: float = 1.3  # PIR threshold for low income
    middle_income_threshold: float = 3.5  # PIR threshold for middle income
    high_school_education: int = 3  # Education level code for high school
    
    def __post_init__(self):
        """Create directories if they don't exist"""
        for dir_path in [self.raw_data_dir, self.interim_data_dir, 
                         self.processed_data_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation"""
    # Model settings
    model_type: str = "random_forest"
    test_size: float = 0.2
    random_state: int = 42
    cross_validation_folds: int = 5
    
    # Random Forest specific
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    
    # Neural Network specific (for regression)
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 50
    early_stopping_patience: int = 5
    
    # Feature engineering
    create_interaction_features: bool = True
    polynomial_features_degree: int = 2
    scale_features: bool = True
    
    # Output paths
    model_dir: Path = Path("outputs/models")
    results_dir: Path = Path("outputs/results")
    
    def __post_init__(self):
        """Create output directories"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

@dataclass
class AnalysisConfig:
    """Configuration for analysis and visualization"""
    # SHAP settings
    shap_background_samples: int = 100
    shap_test_samples: int = 500
    max_display_features: int = 20
    
    # Visualization settings
    figure_size: tuple = (10, 6)
    dpi: int = 300
    color_palette: str = "viridis"
    
    # Reporting
    generate_html_report: bool = True
    save_intermediate_plots: bool = True
    
    # Output paths
    shap_dir: Path = Path("outputs/shap")
    plots_dir: Path = Path("outputs/plots")
    reports_dir: Path = Path("outputs/reports")
    
    def __post_init__(self):
        """Create output directories"""
        for dir_path in [self.shap_dir, self.plots_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass 
class LoggingConfig:
    """Configuration for logging"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: Path = Path("logs")
    log_file: str = "allergy_pipeline.log"
    
    def __post_init__(self):
        """Create log directory"""
        self.log_dir.mkdir(parents=True, exist_ok=True)

class PipelineConfig:
    """Main configuration class that combines all config components"""
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize pipeline configuration.
        
        Args:
            config_file: Optional path to JSON configuration file
        """
        self.data = DataConfig()
        self.model = ModelConfig()
        self.analysis = AnalysisConfig()
        self.logging = LoggingConfig()
        
        if config_file and config_file.exists():
            self.load_from_file(config_file)
        
        # Load environment variables for sensitive data
        self._load_env_vars()
    
    def _load_env_vars(self):
        """Load sensitive configuration from environment variables"""
        self.data.immport_username = os.environ.get("IMMPORT_USERNAME")
        self.data.immport_password = os.environ.get("IMMPORT_PASSWORD")
    
    def load_from_file(self, config_file: Path):
        """Load configuration from JSON file with validation"""
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        if not config_file.suffix.lower() == '.json':
            raise ValueError(f"Configuration file must be JSON format: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                config_dict = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_file}: {e}")
        
        if not isinstance(config_dict, dict):
            raise ValueError("Configuration file must contain a JSON object")
        
        # Update data config
        if 'data' in config_dict:
            for key, value in config_dict['data'].items():
                if hasattr(self.data, key):
                    setattr(self.data, key, value)
        
        # Update model config
        if 'model' in config_dict:
            for key, value in config_dict['model'].items():
                if hasattr(self.model, key):
                    setattr(self.model, key, value)
        
        # Update analysis config
        if 'analysis' in config_dict:
            for key, value in config_dict['analysis'].items():
                if hasattr(self.analysis, key):
                    setattr(self.analysis, key, value)
    
    def save_to_file(self, config_file: Path):
        """Save current configuration to JSON file"""
        config_dict = {
            'data': {
                'nhanes_cycle': self.data.nhanes_cycle,
                'ige_threshold': self.data.ige_threshold,
                'total_ige_threshold': self.data.total_ige_threshold,
                'missing_threshold': self.data.missing_threshold
            },
            'model': {
                'model_type': self.model.model_type,
                'test_size': self.model.test_size,
                'n_estimators': self.model.n_estimators,
                'random_state': self.model.random_state
            },
            'analysis': {
                'shap_background_samples': self.analysis.shap_background_samples,
                'max_display_features': self.analysis.max_display_features
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        errors = []
        
        # Validate data config
        if self.data.ige_threshold <= 0:
            errors.append("IgE threshold must be positive")
        if self.data.missing_threshold < 0 or self.data.missing_threshold > 1:
            errors.append("Missing threshold must be between 0 and 1")
        
        # Validate model config
        if self.model.test_size <= 0 or self.model.test_size >= 1:
            errors.append("Test size must be between 0 and 1")
        if self.model.n_estimators <= 0:
            errors.append("Number of estimators must be positive")
        
        # Validate analysis config
        if self.analysis.shap_background_samples <= 0:
            errors.append("SHAP background samples must be positive")
        
        if errors:
            for error in errors:
                print(f"Configuration error: {error}")
            return False
        
        return True

# Feature name mappings for NHANES codes
NHANES_FEATURE_MAPPINGS = {
    # IgE measurements
    "LBXIGE": "Total_IgE",
    "LBXD1": "Dermatophagoides_farinae",
    "LBXD2": "Dermatophagoides_pteronyssinus", 
    "LBXE1": "Cat_epithelium",
    "LBXE5": "Dog_epithelium",
    "LBXI6": "German_cockroach",
    "LBXM1": "Alternaria_alternata",
    "LBXW1": "Ragweed",
    "LBXG5": "Rye_grass",
    "LBXG8": "Bermuda_grass",
    "LBXT7": "Oak",
    "LBXT3": "Birch",
    "LBXF13": "Peanut",
    "LBXF1": "Egg_white",
    "LBXF2": "Milk",
    "LBXF24": "Shrimp",
    
    # Demographics
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age_years",
    "RIDRETH1": "Race_ethnicity",
    "DMDBORN4": "Country_birth",
    "DMDEDUC2": "Education_level",
    "INDFMPIR": "Poverty_income_ratio",
    
    # Body measurements
    "BMXBMI": "BMI",
    "BMXHT": "Height_cm",
    "BMXWT": "Weight_kg",
    
    # Dietary
    "DR1TKCAL": "Calorie_intake",
    "DR1TPROT": "Protein_intake",
    "DR1TFAT": "Fat_intake",
    "DR1TCARB": "Carb_intake"
}

# IgE classification thresholds
IGE_CLASSES = {
    "Normal": (0, 87),
    "Borderline": (87, 180),
    "Elevated": (180, 400),
    "Very_High": (400, float('inf'))
}

# Default column types for data validation
EXPECTED_COLUMN_TYPES = {
    "SEQN": "int64",
    "LBXIGE": "float64",
    "RIDAGEYR": "float64",
    "RIAGENDR": "int64",
    "BMXBMI": "float64"
}