"""
SHAP analysis module for model interpretability in the allergy detection pipeline.
Provides comprehensive feature importance and interaction analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path
import joblib
import json

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available. Model interpretability analysis will be limited.")

from config import PipelineConfig, NHANES_FEATURE_MAPPINGS

logger = logging.getLogger(__name__)

class SHAPAnalyzer:
    """Main class for SHAP-based model interpretability"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.explainers = {}
        self.shap_values = {}
        self.feature_importance = {}
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP is not installed. Run 'pip install shap' for full functionality.")
    
    def analyze_model(self, model, X_data: pd.DataFrame, model_name: str, 
                     background_samples: Optional[int] = None) -> Dict:
        """
        Perform SHAP analysis on a trained model.
        
        Args:
            model: Trained model to analyze
            X_data: Feature data (test set)
            model_name: Name identifier for the model
            background_samples: Number of background samples for explainer
            
        Returns:
            Dictionary containing SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Returning basic feature importance if possible.")
            return self._get_basic_feature_importance(model, X_data)
        
        logger.info(f"Starting SHAP analysis for {model_name}")
        
        # Determine background samples
        if background_samples is None:
            background_samples = min(self.config.analysis.shap_background_samples, len(X_data))
        
        # Create appropriate explainer
        explainer = self._create_explainer(model, X_data, background_samples)
        self.explainers[model_name] = explainer
        
        # Calculate SHAP values
        logger.info("Calculating SHAP values...")
        test_samples = min(self.config.analysis.shap_test_samples, len(X_data))
        X_test_sample = X_data.iloc[:test_samples]
        
        shap_values = explainer(X_test_sample)
        self.shap_values[model_name] = shap_values
        
        # Extract analysis results
        results = {
            'global_importance': self._get_global_importance(shap_values, X_test_sample),
            'feature_interactions': self._get_feature_interactions(shap_values),
            'summary_stats': self._get_shap_summary_stats(shap_values),
        }
        
        # Generate visualizations
        self._generate_shap_plots(shap_values, X_test_sample, model_name)
        
        return results
    
    def _create_explainer(self, model, X_data: pd.DataFrame, background_samples: int):
        """Create appropriate SHAP explainer based on model type"""
        
        # Sample background data
        if len(X_data) > background_samples:
            background = shap.sample(X_data, background_samples, random_state=self.config.model.random_state)
        else:
            background = X_data
        
        # Determine model type and create explainer
        model_type = type(model).__name__
        
        if 'RandomForest' in model_type or 'GradientBoosting' in model_type:
            # Tree-based models
            logger.info(f"Using TreeExplainer for {model_type}")
            return shap.TreeExplainer(model)
        elif 'Linear' in model_type or 'Logistic' in model_type or 'Ridge' in model_type:
            # Linear models
            logger.info(f"Using LinearExplainer for {model_type}")
            return shap.LinearExplainer(model, background)
        else:
            # Default to KernelExplainer for other models
            logger.info(f"Using KernelExplainer for {model_type}")
            
            # Handle different prediction methods
            if hasattr(model, 'predict_proba'):
                predict_fn = lambda x: model.predict_proba(x)[:, 1] if len(model.classes_) == 2 else model.predict_proba(x)
            else:
                predict_fn = model.predict
            
            return shap.KernelExplainer(predict_fn, background)
    
    def _get_global_importance(self, shap_values, X_data: pd.DataFrame) -> Dict:
        """Calculate global feature importance from SHAP values"""
        
        # Get feature names
        if hasattr(X_data, 'columns'):
            feature_names = X_data.columns.tolist()
        else:
            feature_names = [f'feature_{i}' for i in range(X_data.shape[1])]
        
        # Calculate mean absolute SHAP values
        if hasattr(shap_values, 'values'):
            if len(shap_values.values.shape) == 3:  # Multi-class
                mean_shap = np.mean(np.abs(shap_values.values), axis=(0, 1))
            else:
                mean_shap = np.mean(np.abs(shap_values.values), axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Create importance dictionary with human-readable names
        importance_dict = {}
        for idx, feature in enumerate(feature_names):
            # Map to human-readable name if available
            readable_name = self._get_readable_feature_name(feature)
            importance_dict[readable_name] = float(mean_shap[idx])
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        # Get top features
        top_features = dict(list(importance_dict.items())[:self.config.analysis.max_display_features])
        
        return {
            'all_features': importance_dict,
            'top_features': top_features
        }
    
    def _get_feature_interactions(self, shap_values) -> Dict:
        """Analyze feature interactions from SHAP values"""
        interactions = {}
        
        if hasattr(shap_values, 'values'):
            values = shap_values.values
            if len(values.shape) == 2:  # Binary or regression
                # Calculate correlation between SHAP values
                shap_df = pd.DataFrame(values)
                correlation = shap_df.corr()
                
                # Find top interactions (high correlation in SHAP values)
                top_interactions = []
                for i in range(len(correlation)):
                    for j in range(i+1, len(correlation)):
                        if abs(correlation.iloc[i, j]) > 0.5:  # Threshold for strong interaction
                            top_interactions.append({
                                'feature1': i,
                                'feature2': j,
                                'correlation': float(correlation.iloc[i, j])
                            })
                
                interactions['top_pairs'] = sorted(top_interactions, 
                                                 key=lambda x: abs(x['correlation']), 
                                                 reverse=True)[:10]
        
        return interactions
    
    def _get_shap_summary_stats(self, shap_values) -> Dict:
        """Calculate summary statistics for SHAP values"""
        stats = {}
        
        if hasattr(shap_values, 'values'):
            values = shap_values.values
            
            # Flatten for multi-class
            if len(values.shape) == 3:
                values = values.reshape(-1, values.shape[-1])
            
            stats = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'q25': float(np.percentile(values, 25)),
                'q75': float(np.percentile(values, 75))
            }
        
        return stats
    
    def _get_readable_feature_name(self, feature_name: str) -> str:
        """Convert feature name to human-readable format"""
        
        # Check if it's in NHANES mappings
        for code, readable in NHANES_FEATURE_MAPPINGS.items():
            if code.lower() in feature_name.lower():
                return readable
        
        # Handle engineered feature patterns
        if 'sensitized' in feature_name:
            base_name = feature_name.replace('_sensitized', '')
            return f"{base_name} Sensitization"
        elif 'log_' in feature_name:
            base_name = feature_name.replace('log_', '')
            return f"Log({base_name})"
        elif feature_name.startswith('poly_'):
            return f"Polynomial {feature_name}"
        elif '_interaction' in feature_name:
            return feature_name.replace('_', ' ').title()
        
        # Default: capitalize and replace underscores
        return feature_name.replace('_', ' ').title()
    
    def _generate_shap_plots(self, shap_values, X_data: pd.DataFrame, model_name: str):
        """Generate and save SHAP visualizations"""
        
        if not SHAP_AVAILABLE:
            logger.warning("Cannot generate SHAP plots without SHAP library")
            return
        
        output_dir = self.config.analysis.shap_dir / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Generating SHAP plots for {model_name}")
        
        # 1. Summary plot
        plt.figure(figsize=self.config.analysis.figure_size)
        shap.summary_plot(shap_values, X_data, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / 'summary_plot.png', dpi=self.config.analysis.dpi, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot of feature importance
        plt.figure(figsize=self.config.analysis.figure_size)
        shap.plots.bar(shap_values, max_display=self.config.analysis.max_display_features, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / 'importance_bar_plot.png', dpi=self.config.analysis.dpi, bbox_inches='tight')
        plt.close()
        
        # 3. Waterfall plot for first prediction
        if hasattr(shap_values, 'values') and len(shap_values.values) > 0:
            plt.figure(figsize=self.config.analysis.figure_size)
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            plt.savefig(output_dir / 'waterfall_plot_first.png', dpi=self.config.analysis.dpi, bbox_inches='tight')
            plt.close()
        
        # 4. Force plot for multiple predictions (save as HTML)
        if self.config.analysis.generate_html_report:
            force_plot = shap.force_plot(
                explainer.expected_value if hasattr(self.explainers[model_name], 'expected_value') else 0,
                shap_values.values[:100] if hasattr(shap_values, 'values') else shap_values[:100],
                X_data.iloc[:100]
            )
            shap.save_html(str(output_dir / 'force_plot.html'), force_plot)
        
        logger.info(f"SHAP plots saved to {output_dir}")
    
    def _get_basic_feature_importance(self, model, X_data: pd.DataFrame) -> Dict:
        """Get basic feature importance without SHAP (fallback method)"""
        
        importance_dict = {}
        
        # Check if model has feature_importances_ attribute (tree-based models)
        if hasattr(model, 'feature_importances_'):
            feature_names = X_data.columns.tolist() if hasattr(X_data, 'columns') else [f'feature_{i}' for i in range(X_data.shape[1])]
            
            for idx, importance in enumerate(model.feature_importances_):
                readable_name = self._get_readable_feature_name(feature_names[idx])
                importance_dict[readable_name] = float(importance)
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'global_importance': {
                    'all_features': importance_dict,
                    'top_features': dict(list(importance_dict.items())[:self.config.analysis.max_display_features])
                },
                'note': 'Basic feature importance (install SHAP for detailed analysis)'
            }
        
        # Check for coefficient-based importance (linear models)
        elif hasattr(model, 'coef_'):
            feature_names = X_data.columns.tolist() if hasattr(X_data, 'columns') else [f'feature_{i}' for i in range(X_data.shape[1])]
            
            coef = model.coef_.flatten() if len(model.coef_.shape) > 1 else model.coef_
            
            for idx, coef_val in enumerate(coef):
                readable_name = self._get_readable_feature_name(feature_names[idx])
                importance_dict[readable_name] = abs(float(coef_val))
            
            # Sort by absolute coefficient value
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return {
                'global_importance': {
                    'all_features': importance_dict,
                    'top_features': dict(list(importance_dict.items())[:self.config.analysis.max_display_features])
                },
                'note': 'Coefficient-based importance (install SHAP for detailed analysis)'
            }
        
        return {
            'error': 'Model does not support feature importance extraction',
            'note': 'Install SHAP for model-agnostic feature importance'
        }
    
    def generate_report(self, all_results: Dict):
        """Generate comprehensive SHAP analysis report"""
        
        report_path = self.config.analysis.reports_dir / f"shap_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_results = convert_to_serializable(all_results)
        
        with open(report_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"SHAP analysis report saved to {report_path}")
        
        # Generate summary visualization if multiple models analyzed
        if len(all_results) > 1:
            self._generate_comparison_plot(all_results)
    
    def _generate_comparison_plot(self, all_results: Dict):
        """Generate comparison plot across multiple models"""
        
        # Extract top features from each model
        comparison_data = {}
        
        for model_name, results in all_results.items():
            if 'global_importance' in results and 'top_features' in results['global_importance']:
                top_features = results['global_importance']['top_features']
                for feature, importance in list(top_features.items())[:10]:
                    if feature not in comparison_data:
                        comparison_data[feature] = {}
                    comparison_data[feature][model_name] = importance
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df = comparison_df.fillna(0)
        
        # Plot
        plt.figure(figsize=(12, 8))
        comparison_df.plot(kind='barh')
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Comparison Across Models')
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        output_path = self.config.analysis.shap_dir / 'model_comparison.png'
        plt.savefig(output_path, dpi=self.config.analysis.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {output_path}")

class AllergyFeatureAnalyzer:
    """Specialized analyzer for allergy-specific feature patterns"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.shap_analyzer = SHAPAnalyzer(config)
    
    def analyze_allergen_patterns(self, model, X_data: pd.DataFrame, 
                                 allergen_columns: List[str]) -> Dict:
        """Analyze patterns specific to allergen sensitization"""
        
        results = {}
        
        # Group allergens by type
        indoor_allergens = ['Dermatophagoides_farinae', 'Dermatophagoides_pteronyssinus',
                           'Cat_epithelium', 'Dog_epithelium', 'German_cockroach']
        outdoor_allergens = ['Ragweed', 'Rye_grass', 'Bermuda_grass', 'Oak', 'Birch']
        food_allergens = ['Peanut', 'Egg_white', 'Milk', 'Shrimp']
        
        # Analyze each group
        for group_name, allergen_list in [('indoor', indoor_allergens),
                                          ('outdoor', outdoor_allergens),
                                          ('food', food_allergens)]:
            group_cols = [col for col in allergen_columns 
                         if any(allergen in col for allergen in allergen_list)]
            
            if group_cols and SHAP_AVAILABLE:
                # Calculate SHAP values for this group
                group_data = X_data[group_cols] if all(col in X_data.columns for col in group_cols) else None
                
                if group_data is not None:
                    results[f'{group_name}_allergens'] = {
                        'columns': group_cols,
                        'mean_importance': float(np.mean([
                            self.shap_analyzer.feature_importance.get(col, 0) 
                            for col in group_cols
                        ])),
                        'max_importance': float(np.max([
                            self.shap_analyzer.feature_importance.get(col, 0) 
                            for col in group_cols
                        ]))
                    }
        
        return results
    
    def identify_key_interactions(self, shap_values, feature_names: List[str]) -> Dict:
        """Identify key feature interactions relevant to allergy detection"""
        
        interactions = {}
        
        # Look for specific interaction patterns
        age_features = [f for f in feature_names if 'age' in f.lower()]
        ige_features = [f for f in feature_names if 'ige' in f.lower()]
        season_features = [f for f in feature_names if 'season' in f.lower() or 'month' in f.lower()]
        
        # Age-IgE interactions
        if age_features and ige_features:
            interactions['age_ige'] = {
                'features': {'age': age_features, 'ige': ige_features},
                'description': 'Age and IgE level interactions'
            }
        
        # Season-allergen interactions
        if season_features:
            outdoor_features = [f for f in feature_names if any(
                allergen in f for allergen in ['Ragweed', 'grass', 'Oak', 'Birch']
            )]
            if outdoor_features:
                interactions['season_outdoor'] = {
                    'features': {'season': season_features, 'outdoor': outdoor_features},
                    'description': 'Seasonal patterns with outdoor allergen sensitization'
                }
        
        return interactions