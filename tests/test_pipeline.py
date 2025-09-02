"""
Unit tests for the Allergy Detection Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PipelineConfig
from data_ingestion import LocalDataIngestion, AllergyDataIngestion
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer

class TestDataIngestion:
    """Test data ingestion functionality"""
    
    def test_config_initialization(self):
        """Test configuration initialization"""
        config = PipelineConfig()
        assert config is not None
        assert config.validate() == True
    
    def test_local_data_detection(self):
        """Test that local data is detected properly"""
        config = PipelineConfig()
        ingestion = LocalDataIngestion(config, "database_files")
        
        # Check if data root is set correctly
        assert ingestion.data_root == Path("database_files")
    
    def test_mock_data_generation(self):
        """Test mock data generation"""
        config = PipelineConfig()
        ingestion = LocalDataIngestion(config)
        
        # Generate mock NHANES data
        nhanes_mock = ingestion.generate_mock_nhanes_data(100)
        assert len(nhanes_mock) == 100
        assert 'SEQN' in nhanes_mock.columns
        assert 'LBXIGE' in nhanes_mock.columns
        
        # Generate mock ImmPort data
        immport_mock = ingestion.generate_mock_immport_data(50)
        assert len(immport_mock) == 50
        assert 'subject_id' in immport_mock.columns
    
    def test_tab_file_reading(self):
        """Test tab-delimited file reading"""
        config = PipelineConfig()
        ingestion = LocalDataIngestion(config)
        
        # Create a temporary tab file
        temp_file = Path("temp_test.txt")
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        df.to_csv(temp_file, sep='\t', index=False)
        
        # Read it back
        result = ingestion.read_tab_file(temp_file)
        assert not result.empty
        assert len(result) == 3
        
        # Clean up
        temp_file.unlink()

class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization"""
        config = PipelineConfig()
        engineer = FeatureEngineer(config)
        assert engineer is not None
    
    def test_sensitization_features(self):
        """Test sensitization feature creation"""
        config = PipelineConfig()
        engineer = FeatureEngineer(config)
        
        # Create test data
        test_data = pd.DataFrame({
            'LBXD1': [0.1, 0.5, 1.0],  # Dust mite
            'LBXE1': [0.2, 0.4, 0.8],  # Cat
            'LBXF13': [0.05, 0.35, 2.0]  # Peanut
        })
        
        # Apply sensitization features
        result = engineer.create_sensitization_features(test_data)
        
        # Check that sensitization columns were created
        assert 'LBXD1_sensitized' in result.columns
        assert 'LBXE1_sensitized' in result.columns
        assert 'LBXF13_sensitized' in result.columns
    
    def test_demographic_features(self):
        """Test demographic feature creation"""
        config = PipelineConfig()
        engineer = FeatureEngineer(config)
        
        # Create test data
        test_data = pd.DataFrame({
            'RIDAGEYR': [5, 25, 70],
            'RIAGENDR': [1, 2, 1],
            'INDFMPIR': [0.5, 2.0, 4.0]
        })
        
        # Apply demographic features
        result = engineer.create_demographic_features(test_data)
        
        # Check that features were created
        assert 'is_child' in result.columns
        assert 'is_elderly' in result.columns
        assert 'age_squared' in result.columns

class TestModelTraining:
    """Test model training functionality"""
    
    def test_model_trainer_initialization(self):
        """Test model trainer initialization"""
        config = PipelineConfig()
        trainer = ModelTrainer(config)
        assert trainer is not None
    
    def test_data_preparation(self):
        """Test data preparation for model training"""
        config = PipelineConfig()
        trainer = ModelTrainer(config)
        
        # Create test data
        test_data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Prepare data
        X_train, X_test, y_train, y_test, features = trainer.prepare_data(
            test_data, 'target', 'classification'
        )
        
        assert X_train.shape[0] == 80  # 80% train
        assert X_test.shape[0] == 20   # 20% test
        assert len(features) == 3

class TestPipeline:
    """Test complete pipeline integration"""
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        from main_pipeline import AllergyDetectionPipeline
        
        pipeline = AllergyDetectionPipeline()
        assert pipeline is not None
        assert pipeline.config is not None
    
    def test_pipeline_with_mock_data(self):
        """Test pipeline with mock data"""
        from main_pipeline import AllergyDetectionPipeline
        
        # Initialize pipeline
        pipeline = AllergyDetectionPipeline()
        
        # Use mock data
        pipeline.config.data.sample_size = 50  # Small sample for testing
        
        # Test data ingestion step
        pipeline.nhanes_data = pipeline.data_ingestion.local_ingestion.generate_mock_nhanes_data(50)
        pipeline.immport_data = pipeline.data_ingestion.local_ingestion.generate_mock_immport_data(25)
        
        assert len(pipeline.nhanes_data) == 50
        assert len(pipeline.immport_data) == 25

@pytest.fixture
def sample_config():
    """Fixture for configuration"""
    return PipelineConfig()

@pytest.fixture
def sample_data():
    """Fixture for sample data"""
    return pd.DataFrame({
        'LBXIGE': np.random.lognormal(3.5, 1.2, 100),
        'RIDAGEYR': np.random.normal(45, 20, 100),
        'RIAGENDR': np.random.choice([1, 2], 100)
    })

def test_config_validation(sample_config):
    """Test configuration validation"""
    assert sample_config.validate() == True
    
    # Test invalid configuration
    sample_config.model.test_size = 1.5  # Invalid value
    assert sample_config.validate() == False