"""
Main pipeline orchestrator for the allergy detection system.
Coordinates all components and provides a complete end-to-end workflow.
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import all pipeline components
from config import PipelineConfig
from data_ingestion import AllergyDataIngestion
from feature_engineering import FeatureEngineer
from model_training import ModelTrainer
from shap_analysis import SHAPAnalyzer, AllergyFeatureAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AllergyDetectionPipeline:
    """Main orchestrator for the complete allergy detection pipeline"""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = PipelineConfig(config_path)
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid configuration. Please check settings.")
        
        # Initialize components
        self.data_ingestion = AllergyDataIngestion(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.shap_analyzer = SHAPAnalyzer(self.config)
        self.allergy_analyzer = AllergyFeatureAnalyzer(self.config)
        
        # Pipeline state
        self.nhanes_data = None
        self.immport_data = None
        self.engineered_features = None
        self.models = {}
        self.results = {}
        
        logger.info("Allergy Detection Pipeline initialized successfully")
    
    def run_complete_pipeline(self, skip_steps: Optional[list] = None) -> Dict:
        """
        Execute the complete pipeline from data ingestion to analysis.
        
        Args:
            skip_steps: Optional list of steps to skip ['ingestion', 'engineering', 'training', 'analysis']
            
        Returns:
            Dictionary containing all pipeline results
        """
        logger.info("="*50)
        logger.info("Starting Allergy Detection Pipeline")
        logger.info("="*50)
        
        skip_steps = skip_steps or []
        pipeline_results = {}
        
        try:
            # Step 1: Data Ingestion
            if 'ingestion' not in skip_steps:
                logger.info("\n" + "="*30)
                logger.info("STEP 1: DATA INGESTION")
                logger.info("="*30)
                
                self.nhanes_data, self.immport_data = self.data_ingestion.ingest_all_data()
                
                pipeline_results['data_ingestion'] = {
                    'nhanes_shape': self.nhanes_data.shape,
                    'immport_shape': self.immport_data.shape,
                    'nhanes_columns': self.nhanes_data.columns.tolist()[:20],  # First 20 columns
                    'status': 'completed'
                }
            else:
                # Load from saved files
                self._load_saved_data()
            
            # Step 2: Feature Engineering
            if 'engineering' not in skip_steps:
                logger.info("\n" + "="*30)
                logger.info("STEP 2: FEATURE ENGINEERING")
                logger.info("="*30)
                
                self.engineered_features = self.feature_engineer.engineer_features(
                    self.nhanes_data, 
                    self.immport_data
                )
                
                pipeline_results['feature_engineering'] = {
                    'final_shape': self.engineered_features.shape,
                    'n_features': len([col for col in self.engineered_features.columns 
                                     if col not in ['SEQN', 'subject_id']]),
                    'target_columns': [col for col in self.engineered_features.columns 
                                      if col in ['poly_sensitized', 'ige_class', 'log_total_ige']],
                    'status': 'completed'
                }
            else:
                # Load engineered features
                self._load_engineered_features()
            
            # Step 3: Model Training
            if 'training' not in skip_steps:
                logger.info("\n" + "="*30)
                logger.info("STEP 3: MODEL TRAINING")
                logger.info("="*30)
                
                training_results = self.model_trainer.train_all_models(self.engineered_features)
                self.models = self.model_trainer.models
                
                # Summarize best models
                best_models = {}
                for task, models_results in training_results.items():
                    if isinstance(models_results, dict):
                        # Find best model for this task
                        best_score = -np.inf
                        best_model = None
                        
                        for model_name, metrics in models_results.items():
                            if isinstance(metrics, dict):
                                # Use F1 score for classification, R2 for regression
                                score = metrics.get('f1_score', metrics.get('r2_score', -np.inf))
                                if score > best_score:
                                    best_score = score
                                    best_model = model_name
                        
                        if best_model:
                            best_models[task] = {
                                'model': best_model,
                                'score': best_score
                            }
                
                pipeline_results['model_training'] = {
                    'tasks_completed': list(training_results.keys()),
                    'n_models_trained': len(self.models),
                    'best_models': best_models,
                    'detailed_results': training_results,
                    'status': 'completed'
                }
            else:
                # Load saved models
                self._load_saved_models()
            
            # Step 4: SHAP Analysis
            if 'analysis' not in skip_steps:
                logger.info("\n" + "="*30)
                logger.info("STEP 4: SHAP ANALYSIS")
                logger.info("="*30)
                
                shap_results = self._run_shap_analysis()
                
                pipeline_results['shap_analysis'] = {
                    'models_analyzed': list(shap_results.keys()),
                    'analysis_results': shap_results,
                    'status': 'completed'
                }
            
            # Step 5: Generate Final Report
            logger.info("\n" + "="*30)
            logger.info("STEP 5: GENERATING REPORT")
            logger.info("="*30)
            
            self._generate_final_report(pipeline_results)
            
            logger.info("\n" + "="*50)
            logger.info("Pipeline completed successfully!")
            logger.info("="*50)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            pipeline_results['error'] = str(e)
            pipeline_results['status'] = 'failed'
            return pipeline_results
    
    def _run_shap_analysis(self) -> Dict:
        """Run SHAP analysis on trained models"""
        shap_results = {}
        
        # Select key models for SHAP analysis
        key_models = [
            ('poly_random_forest', 'poly_sensitized'),
            ('ige_class_random_forest', 'ige_class'),
            ('ige_reg_rf_regressor', 'log_total_ige')
        ]
        
        for model_name, target in key_models:
            if model_name in self.models:
                logger.info(f"Running SHAP analysis for {model_name}")
                
                # Prepare test data
                feature_cols = [col for col in self.engineered_features.columns 
                              if col not in ['SEQN', 'subject_id', 'poly_sensitized', 
                                           'ige_class', 'log_total_ige', 'elevated_ige',
                                           'sensitization_severity', 'ige_class']]
                
                X_test = self.engineered_features[feature_cols].head(500)  # Use subset for speed
                
                # Run SHAP analysis
                try:
                    results = self.shap_analyzer.analyze_model(
                        self.models[model_name],
                        X_test,
                        model_name
                    )
                    shap_results[model_name] = results
                except Exception as e:
                    logger.warning(f"SHAP analysis failed for {model_name}: {str(e)}")
                    shap_results[model_name] = {'error': str(e)}
        
        # Run allergen-specific analysis
        if self.engineered_features is not None:
            allergen_cols = [col for col in self.engineered_features.columns 
                           if any(allergen in col for allergen in 
                                 ['Dermatophagoides', 'Cat_', 'Dog_', 'Ragweed', 
                                  'Peanut', 'Milk', 'Egg_', 'Shrimp'])]
            
            if allergen_cols and 'poly_random_forest' in self.models:
                allergen_patterns = self.allergy_analyzer.analyze_allergen_patterns(
                    self.models['poly_random_forest'],
                    self.engineered_features[allergen_cols],
                    allergen_cols
                )
                shap_results['allergen_patterns'] = allergen_patterns
        
        return shap_results
    
    def _generate_final_report(self, pipeline_results: Dict):
        """Generate comprehensive final report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'test_size': self.config.model.test_size,
                'n_estimators': self.config.model.n_estimators,
                'ige_threshold': self.config.data.ige_threshold
            },
            'pipeline_results': pipeline_results,
            'summary': self._generate_summary(pipeline_results)
        }
        
        # Save JSON report
        report_path = self.config.analysis.reports_dir / f"pipeline_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict('records')
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_report = convert_to_serializable(report)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_report, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {report_path}")
        
        # Generate executive summary
        self._print_executive_summary(pipeline_results)
    
    def _generate_summary(self, pipeline_results: Dict) -> Dict:
        """Generate executive summary of results"""
        summary = {
            'data_statistics': {},
            'model_performance': {},
            'key_findings': []
        }
        
        # Data statistics
        if 'data_ingestion' in pipeline_results:
            summary['data_statistics'] = {
                'total_samples': pipeline_results['data_ingestion']['nhanes_shape'][0],
                'total_features': pipeline_results['data_ingestion']['nhanes_shape'][1]
            }
        
        # Model performance
        if 'model_training' in pipeline_results:
            training = pipeline_results['model_training']
            if 'best_models' in training:
                summary['model_performance'] = training['best_models']
        
        # Key findings from SHAP
        if 'shap_analysis' in pipeline_results:
            shap = pipeline_results['shap_analysis']
            if 'analysis_results' in shap:
                for model_name, results in shap['analysis_results'].items():
                    if 'global_importance' in results and 'top_features' in results['global_importance']:
                        top_3 = list(results['global_importance']['top_features'].keys())[:3]
                        summary['key_findings'].append(f"Top features for {model_name}: {', '.join(top_3)}")
        
        return summary
    
    def _print_executive_summary(self, pipeline_results: Dict):
        """Print executive summary to console"""
        print("\n" + "="*60)
        print("EXECUTIVE SUMMARY - ALLERGY DETECTION PIPELINE")
        print("="*60)
        
        # Data summary
        if 'data_ingestion' in pipeline_results:
            data = pipeline_results['data_ingestion']
            print(f"\nData Processed:")
            print(f"  - NHANES samples: {data['nhanes_shape'][0]:,}")
            print(f"  - ImmPort samples: {data['immport_shape'][0]:,}")
        
        # Feature engineering summary
        if 'feature_engineering' in pipeline_results:
            features = pipeline_results['feature_engineering']
            print(f"\nFeatures Engineered:")
            print(f"  - Total features: {features['n_features']}")
            print(f"  - Target variables: {', '.join(features['target_columns'])}")
        
        # Model performance summary
        if 'model_training' in pipeline_results:
            training = pipeline_results['model_training']
            if 'best_models' in training:
                print(f"\nBest Models:")
                for task, info in training['best_models'].items():
                    print(f"  - {task}: {info['model']} (score: {info['score']:.3f})")
        
        # Top features summary
        if 'shap_analysis' in pipeline_results:
            shap = pipeline_results['shap_analysis']
            if 'analysis_results' in shap:
                print(f"\nTop Predictive Features:")
                for model_name, results in list(shap['analysis_results'].items())[:1]:  # Show first model only
                    if 'global_importance' in results and 'top_features' in results['global_importance']:
                        for i, (feature, importance) in enumerate(list(results['global_importance']['top_features'].items())[:5], 1):
                            print(f"  {i}. {feature}: {importance:.4f}")
        
        print("\n" + "="*60)
    
    def _load_saved_data(self):
        """Load previously saved data"""
        nhanes_path = self.config.data.raw_data_dir / "nhanes_raw.csv"
        immport_path = self.config.data.raw_data_dir / "immport_raw.csv"
        
        if nhanes_path.exists():
            self.nhanes_data = pd.read_csv(nhanes_path)
            logger.info(f"Loaded saved NHANES data: {self.nhanes_data.shape}")
        else:
            raise FileNotFoundError(f"NHANES data not found at {nhanes_path}")
        
        if immport_path.exists():
            self.immport_data = pd.read_csv(immport_path)
            logger.info(f"Loaded saved ImmPort data: {self.immport_data.shape}")
        else:
            logger.warning("ImmPort data not found, using empty DataFrame")
            self.immport_data = pd.DataFrame()
    
    def _load_engineered_features(self):
        """Load previously engineered features"""
        features_path = self.config.data.processed_data_dir / "engineered_features.csv"
        
        if features_path.exists():
            self.engineered_features = pd.read_csv(features_path)
            logger.info(f"Loaded engineered features: {self.engineered_features.shape}")
        else:
            raise FileNotFoundError(f"Engineered features not found at {features_path}")
    
    def _load_saved_models(self):
        """Load previously trained models"""
        import joblib
        
        model_files = list(self.config.model.model_dir.glob("*.pkl"))
        
        for model_file in model_files:
            model_name = model_file.stem
            self.models[model_name] = joblib.load(model_file)
            logger.info(f"Loaded model: {model_name}")
        
        if not self.models:
            logger.warning("No saved models found")

def main():
    """Main entry point for the pipeline"""
    parser = argparse.ArgumentParser(description='Allergy Detection Pipeline')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--skip', nargs='*', 
                       choices=['ingestion', 'engineering', 'training', 'analysis'],
                       help='Steps to skip')
    parser.add_argument('--demo', action='store_true', 
                       help='Run in demo mode with small dataset')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    config_path = Path(args.config) if args.config else None
    pipeline = AllergyDetectionPipeline(config_path)
    
    # Set demo mode if requested
    if args.demo:
        logger.info("Running in DEMO mode with reduced dataset size")
        pipeline.config.data.sample_size = 100
        pipeline.config.model.epochs = 10
        pipeline.config.analysis.shap_test_samples = 50
    
    # Run pipeline
    try:
        results = pipeline.run_complete_pipeline(skip_steps=args.skip)
        
        # Exit with success
        if results.get('status') == 'failed':
            sys.exit(1)
        else:
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()