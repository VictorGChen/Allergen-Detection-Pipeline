"""
Data ingestion module for the allergy detection pipeline.
Modified to work with your local database_files directory structure.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

from config import PipelineConfig, NHANES_FEATURE_MAPPINGS

logger = logging.getLogger(__name__)

class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass

class LocalDataIngestion:
    """Handle data ingestion from your local dataset structure"""
    
    def __init__(self, config: PipelineConfig, data_root: str = "database_files/Final_Dataset_With_Protocols"):
        self.config = config
        self.data_root = Path(data_root)
        
        # Define the study folders as they appear in your repo
        self.study_folders = {
            'observational_food': 'AnObservationalStudyofFoodAllergy',
            'peanut_epicutaneous': 'PeanutEpicutaneousImmunotherapy', 
            'peanut_sublingual': 'PeanutSublingualImmunotherapy'
        }
        
        # NHANES folder path
        self.nhanes_folder = self.data_root / 'NHANES' / 'NHANES'
        
        # Check if data exists
        if not self.data_root.exists():
            logger.warning(f"Data root {self.data_root} not found. Will use mock data.")
            self.use_mock = True
        else:
            self.use_mock = False
            logger.info(f"Found local data at {self.data_root}")
    
    def read_tab_file(self, filepath: Path, delimiter: str = '\t') -> pd.DataFrame:
        """Read tab-delimited files with proper error handling"""
        try:
            # Try different delimiters
            delimiters = ['\t', '~@@~', '~@~', ',']
            
            for delim in delimiters:
                try:
                    df = pd.read_csv(filepath, sep=delim, engine='python', 
                                   on_bad_lines='skip', low_memory=False)
                    if len(df.columns) > 1:  # Successfully parsed
                        return df
                except:
                    continue
            
            # If all fail, try with pandas default
            return pd.read_csv(filepath, sep=None, engine='python')
            
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            return pd.DataFrame()
    
    def load_immport_study(self, study_name: str) -> Dict[str, pd.DataFrame]:
        """Load data from a single ImmPort study folder"""
        study_path = self.data_root / self.study_folders.get(study_name, study_name)
        data_dict = {}
        
        if not study_path.exists():
            logger.warning(f"Study path {study_path} not found")
            return data_dict
        
        # Look for Tab folders
        tab_folders = list(study_path.glob("*Tab/Tab"))
        mysql_folders = list(study_path.glob("*MySQL/MySQL"))
        
        # Use Tab folders if available, otherwise MySQL
        data_folders = tab_folders if tab_folders else mysql_folders
        
        for folder in data_folders:
            # Read the main data files
            files_to_load = [
                'subject.txt',
                'immune_exposure.txt', 
                'planned_visit.txt',
                'arm_2_subject.txt'
            ]
            
            for filename in files_to_load:
                filepath = folder / filename
                if filepath.exists():
                    df = self.read_tab_file(filepath)
                    if not df.empty:
                        data_dict[filename.replace('.txt', '')] = df
                        logger.info(f"Loaded {filename} from {study_name}: shape {df.shape}")
        
        # Also check for IgE data in StudyFiles
        study_files = list(study_path.glob("**/StudyFiles/*.txt"))
        for filepath in study_files:
            if 'ige' in filepath.name.lower():
                df = self.read_tab_file(filepath)
                if not df.empty:
                    data_dict['ige_data'] = df
                    logger.info(f"Loaded IgE data from {filepath.name}: shape {df.shape}")
        
        # Also check EMP-123 folder if it's the observational study
        if study_name == 'observational_food':
            emp_path = self.data_root / 'EMP-123'
            if emp_path.exists():
                emp_files = list(emp_path.glob("*.txt"))
                for filepath in emp_files:
                    df = self.read_tab_file(filepath)
                    if not df.empty:
                        data_dict[f'emp_{filepath.stem}'] = df
                        logger.info(f"Loaded EMP data from {filepath.name}: shape {df.shape}")
        
        return data_dict
    
    def load_nhanes_data(self) -> pd.DataFrame:
        """Load NHANES XPT files from local directory"""
        if not self.nhanes_folder.exists():
            logger.warning(f"NHANES folder {self.nhanes_folder} not found")
            return pd.DataFrame()
        
        nhanes_data = {}
        
        # List of NHANES files to load (prioritize IgE and demographics)
        xpt_files = [
            'AL_IGE_D.xpt',  # IgE data - most important
            'DEMO_D.xpt',    # Demographics D cycle
            'DEMO_E.xpt',    # Demographics E cycle
            'DEMO_L.xpt',    # Demographics L cycle
            'DBQ_D.xpt',     # Dietary questionnaire D
            'DBQ_F.xpt',     # Dietary questionnaire F
            'DR1TOT_D.xpt',  # Dietary recall D
            'DR1TOT_E.xpt',  # Dietary recall E
            'DR1TOT_L.xpt',  # Dietary recall L
            'MCQ_D.xpt',     # Medical conditions D
            'MCQ_E.xpt',     # Medical conditions E
            'MCQ_L.xpt'      # Medical conditions L
        ]
        
        for filename in xpt_files:
            filepath = self.nhanes_folder / filename
            if filepath.exists():
                try:
                    # Try to read XPT file
                    try:
                        import pyreadstat
                        df, meta = pyreadstat.read_xport(str(filepath))
                        nhanes_data[filename.replace('.xpt', '')] = df
                        logger.info(f"Loaded NHANES {filename}: shape {df.shape}")
                    except ImportError:
                        # Fallback to pandas
                        df = pd.read_sas(filepath, format='xport', encoding='latin1')
                        nhanes_data[filename.replace('.xpt', '')] = df
                        logger.info(f"Loaded NHANES {filename} with pandas: shape {df.shape}")
                except Exception as e:
                    logger.warning(f"Could not load {filename}: {e}")
        
        # Merge NHANES datasets intelligently
        if nhanes_data:
            # Start with IgE data if available (most important)
            if 'AL_IGE_D' in nhanes_data:
                base_df = nhanes_data['AL_IGE_D']
                logger.info(f"Using AL_IGE_D as base with {len(base_df)} samples")
            else:
                # Otherwise start with demographics
                demo_dfs = [df for key, df in nhanes_data.items() if 'DEMO' in key]
                if demo_dfs:
                    base_df = demo_dfs[0]
                else:
                    base_df = list(nhanes_data.values())[0]
            
            # Merge other datasets
            for name, df in nhanes_data.items():
                if name != 'AL_IGE_D' and not df.equals(base_df):
                    # Merge on SEQN if it exists
                    if 'SEQN' in df.columns and 'SEQN' in base_df.columns:
                        # Avoid duplicate columns
                        common_cols = set(base_df.columns) & set(df.columns)
                        common_cols.discard('SEQN')
                        if common_cols:
                            df = df.drop(columns=list(common_cols), errors='ignore')
                        base_df = base_df.merge(df, on='SEQN', how='left')
            
            return base_df
        
        return pd.DataFrame()
    
    def combine_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and combine all available data sources"""
        
        all_immport_data = []
        
        # Load each ImmPort study
        for study_key, study_name in self.study_folders.items():
            logger.info(f"Loading {study_key} study...")
            study_data = self.load_immport_study(study_key)
            
            if study_data:
                # Combine subject-level data
                if 'subject' in study_data:
                    subject_df = study_data['subject'].copy()
                    subject_df['study'] = study_key
                    
                    # Merge with immune exposure if available
                    if 'immune_exposure' in study_data:
                        immune_df = study_data['immune_exposure']
                        merge_cols = list(set(subject_df.columns) & set(immune_df.columns))
                        if merge_cols:
                            subject_df = subject_df.merge(immune_df, on=merge_cols[0], how='left')
                    
                    # Merge with IgE data if available
                    if 'ige_data' in study_data:
                        ige_df = study_data['ige_data']
                        # Try to merge on common column
                        merge_cols = list(set(subject_df.columns) & set(ige_df.columns))
                        if merge_cols:
                            subject_df = subject_df.merge(ige_df, on=merge_cols[0], how='left')
                    
                    all_immport_data.append(subject_df)
        
        # Combine all ImmPort studies
        if all_immport_data:
            combined_immport = pd.concat(all_immport_data, ignore_index=True, sort=False)
            logger.info(f"Combined ImmPort data shape: {combined_immport.shape}")
        else:
            logger.warning("No ImmPort data loaded, using mock data")
            combined_immport = self.generate_mock_immport_data()
        
        # Load NHANES data
        nhanes_data = self.load_nhanes_data()
        if nhanes_data.empty:
            logger.warning("No NHANES data loaded, using mock data")
            nhanes_data = self.generate_mock_nhanes_data()
        else:
            logger.info(f"NHANES data shape: {nhanes_data.shape}")
        
        return nhanes_data, combined_immport
    
    def generate_mock_nhanes_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic mock NHANES data for testing"""
        np.random.seed(42)
        
        logger.info(f"Generating mock NHANES data with {n_samples} samples")
        
        # Basic demographics
        data = {
            'SEQN': range(1, n_samples + 1),
            'RIAGENDR': np.random.choice([1, 2], n_samples),  # 1=Male, 2=Female
            'RIDAGEYR': np.random.normal(45, 20, n_samples).clip(2, 85),
            'RIDRETH1': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'DMDEDUC2': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'INDFMPIR': np.random.exponential(2, n_samples).clip(0, 5),
        }
        
        # Generate correlated IgE measurements
        log_total_ige = np.random.normal(3.5, 1.2, n_samples)
        data['LBXIGE'] = np.exp(log_total_ige).clip(2, 5000)
        
        # Allergen-specific IgE
        allergen_codes = ['LBXD1', 'LBXD2', 'LBXE1', 'LBXE5', 'LBXI6', 'LBXM1',
                         'LBXW1', 'LBXG5', 'LBXG8', 'LBXT7', 'LBXT3',
                         'LBXF13', 'LBXF1', 'LBXF2', 'LBXF24']
        
        for allergen in allergen_codes:
            # Probability of sensitization increases with total IgE
            sensitization_prob = 1 / (1 + np.exp(-(log_total_ige - 3.5)))
            sensitized = np.random.binomial(1, sensitization_prob * 0.3)  # 30% base rate
            
            # Generate values
            base_values = np.random.exponential(0.1, n_samples)
            sensitized_values = np.where(
                sensitized,
                np.random.exponential(2, n_samples) + 0.35,
                base_values
            )
            data[allergen] = sensitized_values.clip(0.01, 100)
        
        return pd.DataFrame(data)
    
    def generate_mock_immport_data(self, n_samples: int = 500) -> pd.DataFrame:
        """Generate realistic mock ImmPort data for testing"""
        np.random.seed(43)
        
        logger.info(f"Generating mock ImmPort data with {n_samples} samples")
        
        data = {
            'subject_id': [f"SUB{i:04d}" for i in range(1, n_samples + 1)],
            'study': np.random.choice(['observational_food', 'peanut_sublingual', 
                                     'peanut_epicutaneous'], n_samples),
            'age': np.random.normal(30, 15, n_samples).clip(5, 75),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'baseline_ige': np.exp(np.random.normal(4, 1, n_samples)),
            'peanut_ige': np.random.exponential(2, n_samples),
            'milk_ige': np.random.exponential(1.5, n_samples),
            'egg_ige': np.random.exponential(1.8, n_samples),
            'treatment_response': np.random.choice(['responder', 'non_responder', 'partial'], n_samples)
        }
        
        return pd.DataFrame(data)

class AllergyDataIngestion:
    """Main class to orchestrate data ingestion from multiple sources"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Check for local data in various locations
        possible_paths = [
            "database_files/Final_Dataset_With_Protocols",
            "Final_Dataset_With_Protocols",
            "../database_files/Final_Dataset_With_Protocols"
        ]
        
        self.local_ingestion = None
        for path in possible_paths:
            if Path(path).exists():
                self.local_ingestion = LocalDataIngestion(config, path)
                logger.info(f"Found local data at {path}")
                break
        
        if not self.local_ingestion:
            logger.info("No local data found, will use mock data")
            self.local_ingestion = LocalDataIngestion(config)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging"""
        log_file = self.config.logging.log_dir / self.config.logging.log_file
        
        logging.basicConfig(
            level=getattr(logging, self.config.logging.log_level),
            format=self.config.logging.log_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    def ingest_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Ingest data from all sources.
        
        Returns:
            Tuple of (NHANES DataFrame, ImmPort DataFrame)
        """
        logger.info("Starting data ingestion pipeline")
        
        # Load from local files or generate mock
        nhanes_data, immport_data = self.local_ingestion.combine_all_data()
        
        # Apply feature name mappings to NHANES
        nhanes_data = self.apply_feature_mappings(nhanes_data)
        
        # Save raw data
        nhanes_path = self.config.data.raw_data_dir / "nhanes_raw.csv"
        nhanes_data.to_csv(nhanes_path, index=False)
        logger.info(f"Saved raw NHANES data to {nhanes_path}")
        
        immport_path = self.config.data.raw_data_dir / "immport_raw.csv"
        immport_data.to_csv(immport_path, index=False)
        logger.info(f"Saved raw ImmPort data to {immport_path}")
        
        # Basic data quality checks
        self.validate_data(nhanes_data, "NHANES")
        self.validate_data(immport_data, "ImmPort")
        
        logger.info("Data ingestion completed successfully")
        return nhanes_data, immport_data
    
    def apply_feature_mappings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply human-readable names to NHANES features"""
        rename_dict = {}
        for old_name, new_name in NHANES_FEATURE_MAPPINGS.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logger.info(f"Renamed {len(rename_dict)} columns to human-readable names")
        
        return df
    
    def validate_data(self, df: pd.DataFrame, source: str):
        """Perform basic data validation"""
        logger.info(f"Validating {source} data...")
        
        # Check for empty dataframe
        if df.empty:
            raise DataIngestionError(f"{source} data is empty")
        
        # Check dimensions
        logger.info(f"{source} data shape: {df.shape}")
        
        # Check missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        high_missing = missing_pct[missing_pct > self.config.data.missing_threshold * 100]
        
        if not high_missing.empty:
            logger.warning(f"Columns with >{self.config.data.missing_threshold*100:.0f}% missing values:")
            for col, pct in high_missing.items():
                logger.warning(f"  - {col}: {pct:.1f}%")
        
        # Check data types
        logger.info(f"{source} data types summary: {df.dtypes.value_counts().to_dict()}")