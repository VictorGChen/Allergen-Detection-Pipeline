"""
Data ingestion module for the allergy detection pipeline.
Handles downloading and processing NHANES and ImmPort data with mock data fallback.
"""

import os
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import logging
import json
import hashlib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from config import PipelineConfig, NHANES_FEATURE_MAPPINGS

logger = logging.getLogger(__name__)

class DataIngestionError(Exception):
    """Custom exception for data ingestion errors"""
    pass

class NHANESDataIngestion:
    """Handle NHANES data downloading and processing"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.base_url = f"{config.data.nhanes_base_url}/{config.data.nhanes_cycle}"
        self.cache_dir = config.data.cache_dir
        self.use_mock_data = False
        
    def download_file(self, filename: str, force_download: bool = False) -> Optional[Path]:
        """
        Download NHANES XPT file with caching.
        
        Args:
            filename: Name of the XPT file to download
            force_download: Force re-download even if cached
            
        Returns:
            Path to the downloaded file or None if failed
        """
        cache_path = self.cache_dir / filename
        
        # Check cache first
        if cache_path.exists() and not force_download:
            logger.info(f"Using cached file: {filename}")
            return cache_path
        
        # Try to download
        url = f"{self.base_url}/{filename}"
        try:
            logger.info(f"Downloading {filename} from {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save to cache
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Successfully downloaded {filename}")
            return cache_path
            
        except (requests.RequestException, IOError) as e:
            logger.warning(f"Failed to download {filename}: {e}")
            logger.info("Will use mock data instead")
            self.use_mock_data = True
            return None
    
    def read_xpt_file(self, filepath: Path) -> pd.DataFrame:
        """Read NHANES XPT file"""
        try:
            import pyreadstat
            df, meta = pyreadstat.read_xport(str(filepath))
            return df
        except ImportError:
            # Fallback to pandas if pyreadstat not available
            try:
                df = pd.read_sas(filepath, format='xport')
                return df
            except Exception as e:
                logger.error(f"Failed to read XPT file {filepath}: {e}")
                return pd.DataFrame()
    
    def generate_mock_nhanes_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate realistic mock NHANES data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with mock NHANES data
        """
        np.random.seed(self.config.model.random_state)
        
        logger.info(f"Generating mock NHANES data with {n_samples} samples")
        
        # Basic demographics
        data = {
            'SEQN': range(1, n_samples + 1),
            'RIAGENDR': np.random.choice([1, 2], n_samples),  # 1=Male, 2=Female
            'RIDAGEYR': np.random.normal(45, 20, n_samples).clip(2, 85),
            'RIDRETH1': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.15, 0.15, 0.4, 0.2, 0.1]),
            'DMDEDUC2': np.random.choice([1, 2, 3, 4, 5], n_samples),
            'INDFMPIR': np.random.exponential(2, n_samples).clip(0, 5),
        }
        
        # Body measurements
        data['BMXHT'] = np.random.normal(170, 10, n_samples).clip(140, 200)
        data['BMXWT'] = np.random.normal(75, 15, n_samples).clip(40, 150)
        data['BMXBMI'] = data['BMXWT'] / (data['BMXHT']/100)**2
        
        # Generate correlated IgE measurements
        # Total IgE (log-normal distribution)
        log_total_ige = np.random.normal(3.5, 1.2, n_samples)
        data['LBXIGE'] = np.exp(log_total_ige).clip(2, 5000)
        
        # Allergen-specific IgE (correlated with total IgE)
        allergen_codes = ['LBXD1', 'LBXD2', 'LBXE1', 'LBXE5', 'LBXI6', 'LBXM1',
                         'LBXW1', 'LBXG5', 'LBXG8', 'LBXT7', 'LBXT3',
                         'LBXF13', 'LBXF1', 'LBXF2', 'LBXF24']
        
        for allergen in allergen_codes:
            # Probability of sensitization increases with total IgE
            sensitization_prob = 1 / (1 + np.exp(-(log_total_ige - 3.5)))
            sensitized = np.random.binomial(1, sensitization_prob)
            
            # Generate values: mostly below threshold, some above for sensitized
            base_values = np.random.exponential(0.1, n_samples)
            sensitized_values = np.where(
                sensitized,
                np.random.exponential(2, n_samples) + 0.35,
                base_values
            )
            data[allergen] = sensitized_values.clip(0.01, 100)
        
        # Dietary data
        data['DR1TKCAL'] = np.random.normal(2000, 500, n_samples).clip(500, 4000)
        data['DR1TPROT'] = np.random.normal(80, 20, n_samples).clip(20, 200)
        data['DR1TFAT'] = np.random.normal(70, 20, n_samples).clip(10, 150)
        data['DR1TCARB'] = np.random.normal(250, 50, n_samples).clip(50, 500)
        
        # Add some missing values realistically
        missing_cols = ['INDFMPIR', 'BMXBMI', 'DR1TKCAL', 'DR1TPROT']
        for col in missing_cols:
            missing_idx = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
            data[col][missing_idx] = np.nan
        
        df = pd.DataFrame(data)
        
        # Add interview date information
        base_date = datetime(2005, 1, 1)
        interview_dates = [base_date + timedelta(days=int(x)) for x in np.random.uniform(0, 730, n_samples)]
        df['interview_month'] = [d.month for d in interview_dates]
        df['interview_year'] = [d.year for d in interview_dates]
        
        return df
    
    def ingest_nhanes_data(self) -> pd.DataFrame:
        """
        Main method to ingest NHANES data.
        
        Returns:
            Merged NHANES DataFrame
        """
        dataframes = {}
        
        # Try to download and read each file
        for data_type, filename in self.config.data.nhanes_files.items():
            file_path = self.download_file(filename)
            
            if file_path and file_path.exists():
                df = self.read_xpt_file(file_path)
                if not df.empty:
                    dataframes[data_type] = df
                    logger.info(f"Loaded {data_type}: {df.shape}")
        
        # If no real data was loaded, use mock data
        if not dataframes or self.use_mock_data:
            logger.info("Using mock NHANES data")
            return self.generate_mock_nhanes_data()
        
        # Merge dataframes on SEQN
        merged_df = dataframes.get('demographics', pd.DataFrame())
        for name, df in dataframes.items():
            if name != 'demographics' and not df.empty:
                if 'SEQN' in df.columns and 'SEQN' in merged_df.columns:
                    merged_df = merged_df.merge(df, on='SEQN', how='inner')
        
        logger.info(f"Merged NHANES data shape: {merged_df.shape}")
        return merged_df

class ImmPortDataIngestion:
    """Handle ImmPort data downloading and processing"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.base_url = config.data.immport_base_url
        self.token = None
        
    def authenticate(self) -> bool:
        """Authenticate with ImmPort API"""
        if not self.config.data.immport_username or not self.config.data.immport_password:
            logger.warning("ImmPort credentials not provided, will use mock data")
            return False
        
        try:
            response = requests.post(
                f"{self.base_url}/auth/token",
                data={
                    'username': self.config.data.immport_username,
                    'password': self.config.data.immport_password
                },
                timeout=10
            )
            
            if response.status_code == 200:
                self.token = response.json().get('access_token')
                logger.info("Successfully authenticated with ImmPort")
                return True
            else:
                logger.warning("Failed to authenticate with ImmPort")
                return False
                
        except Exception as e:
            logger.warning(f"ImmPort authentication failed: {e}")
            return False
    
    def generate_mock_immport_data(self, n_samples: int = 500) -> pd.DataFrame:
        """
        Generate realistic mock ImmPort immunological data.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with mock ImmPort data
        """
        np.random.seed(self.config.model.random_state + 1)
        
        logger.info(f"Generating mock ImmPort data with {n_samples} samples")
        
        data = {
            'subject_id': [f"SUB{i:04d}" for i in range(1, n_samples + 1)],
            'study_id': np.random.choice(['SDY1', 'SDY2', 'SDY3'], n_samples),
            'age': np.random.normal(40, 15, n_samples).clip(18, 75),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
        }
        
        # Cytokine measurements (pg/mL)
        cytokines = ['IL4', 'IL5', 'IL13', 'IFN_gamma', 'IL10', 'TNF_alpha']
        for cytokine in cytokines:
            # Log-normal distribution for cytokine levels
            log_values = np.random.normal(2, 1, n_samples)
            data[f'{cytokine}_baseline'] = np.exp(log_values).clip(0.1, 1000)
            
            # Stimulated values (usually higher)
            data[f'{cytokine}_stimulated'] = data[f'{cytokine}_baseline'] * np.random.lognormal(1, 0.5, n_samples)
        
        # T-cell populations (%)
        data['CD4_percent'] = np.random.normal(45, 10, n_samples).clip(20, 70)
        data['CD8_percent'] = np.random.normal(25, 8, n_samples).clip(10, 45)
        data['Treg_percent'] = np.random.normal(5, 2, n_samples).clip(1, 15)
        data['Th2_percent'] = np.random.normal(8, 3, n_samples).clip(2, 20)
        
        # Gene expression (normalized counts)
        genes = ['GATA3', 'TBX21', 'FOXP3', 'IL4R', 'FCER1A']
        for gene in genes:
            data[f'{gene}_expression'] = np.random.lognormal(3, 1, n_samples).clip(1, 10000)
        
        df = pd.DataFrame(data)
        
        # Add some correlations
        # Higher Th2 percentage correlates with higher IL4, IL5, IL13
        th2_effect = (df['Th2_percent'] - 8) / 3
        df['IL4_baseline'] *= (1 + 0.3 * th2_effect).clip(0.5, 2)
        df['IL5_baseline'] *= (1 + 0.3 * th2_effect).clip(0.5, 2)
        df['IL13_baseline'] *= (1 + 0.3 * th2_effect).clip(0.5, 2)
        
        return df
    
    def ingest_immport_data(self) -> pd.DataFrame:
        """
        Main method to ingest ImmPort data.
        
        Returns:
            ImmPort DataFrame
        """
        # Try to authenticate
        if not self.authenticate():
            logger.info("Using mock ImmPort data")
            return self.generate_mock_immport_data()
        
        # Try to fetch real data
        try:
            headers = {
                'Authorization': f"bearer {self.token}",
                'Content-Type': "application/json"
            }
            
            # Example endpoint for immunological data
            response = requests.get(
                f"{self.base_url}/data/query/result/elisa",
                headers=headers,
                params={'studyAccession': 'SDY1'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                df = pd.DataFrame(data)
                logger.info(f"Loaded ImmPort data: {df.shape}")
                return df
            else:
                logger.warning("Failed to fetch ImmPort data, using mock data")
                return self.generate_mock_immport_data()
                
        except Exception as e:
            logger.warning(f"ImmPort data fetch failed: {e}, using mock data")
            return self.generate_mock_immport_data()

class AllergyDataIngestion:
    """Main class to orchestrate data ingestion from multiple sources"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.nhanes_ingestion = NHANESDataIngestion(config)
        self.immport_ingestion = ImmPortDataIngestion(config)
        
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
        
        # Ingest NHANES data
        logger.info("Ingesting NHANES data...")
        nhanes_data = self.nhanes_ingestion.ingest_nhanes_data()
        
        # Apply feature name mappings
        nhanes_data = self.apply_feature_mappings(nhanes_data)
        
        # Save raw data
        nhanes_path = self.config.data.raw_data_dir / "nhanes_raw.csv"
        nhanes_data.to_csv(nhanes_path, index=False)
        logger.info(f"Saved raw NHANES data to {nhanes_path}")
        
        # Ingest ImmPort data
        logger.info("Ingesting ImmPort data...")
        immport_data = self.immport_ingestion.ingest_immport_data()
        
        # Save raw data
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
        logger.info(f"{source} data) data types:")