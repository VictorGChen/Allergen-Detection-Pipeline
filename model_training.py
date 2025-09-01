"""
Model training module for the allergy detection pipeline.
Handles both classification and regression models with comprehensive evaluation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging
from pathlib import Path
import joblib
import json
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    classification_report, confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow for neural network models (with graceful fallback)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logging.warning("TensorFlow not available. Neural network models will be skipped.")

from config import PipelineConfig

logger = logging.getLogger(__name__)

class ModelTrainer:
    """Main class for training and evaluating models"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.results = {}
        self.feature_importance = {}
        
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    task_type: str = 'classification') -> Tuple:
        """
        Prepare data for model training.
        
        Args:
            df: Feature DataFrame
            target_column: Name of target column
            task_type: 'classification' or 'regression'
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        logger.info(f"Preparing data for {task_type} task with target: {target_column}")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Separate features and target
        exclude_cols = [target_column, 'SEQN', 'subject_id', 'ige_class', 
                       'elevated_ige', 'poly_sensitized', 'sensitization_severity']
        exclude_cols = [col for col in exclude_cols if col in df.columns]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Remove any non-numeric columns for now
        numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[numeric_cols].copy()
        y = df[target_column].copy()
        
        # Handle missing values in target
        valid_indices = ~y.isna()
        X = X[valid_indices]
        y = y[valid_indices]
        
        # Encode target for classification
        if task_type == 'classification':
            if y.dtype == 'object' or y.dtype == 'category':
                self.encoders[target_column] = LabelEncoder()
                y = self.encoders[target_column].fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.model.test_size,
            random_state=self.config.model.random_state,
            stratify=y if task_type == 'classification' else None
        )
        
        # Scale features if configured
        if self.config.model.scale_features:
            scaler_key = f"{target_column}_{task_type}"
            self.scalers[scaler_key] = StandardScaler()
            X_train = self.scalers[scaler_key].fit_transform(X_train)
            X_test = self.scalers[scaler_key].transform(X_test)
        
        logger.info(f"Data prepared: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        return X_train, X_test, y_train, y_test, numeric_cols
    
    def train_classification_models(self, X_train, X_test, y_train, y_test, 
                                   feature_names: List[str], model_name_prefix: str = "") -> Dict:
        """Train multiple classification models and compare performance"""
        logger.info("Training classification models...")
        
        results = {}
        
        # Calculate class weights for imbalanced data
        classes = np.unique(y_train)
        if len(classes) > 1:
            class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
            class_weight_dict = dict(zip(classes, class_weights))
        else:
            class_weight_dict = None
        
        # 1. Random Forest
        logger.info("Training Random Forest Classifier...")
        rf_model = RandomForestClassifier(
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            min_samples_split=self.config.model.min_samples_split,
            min_samples_leaf=self.config.model.min_samples_leaf,
            class_weight=class_weight_dict,
            random_state=self.config.model.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_pred_proba = rf_model.predict_proba(X_test)
        
        results['random_forest'] = self._evaluate_classification(y_test, rf_pred, rf_pred_proba)
        results['random_forest']['feature_importance'] = dict(zip(feature_names, rf_model.feature_importances_))
        self.models[f'{model_name_prefix}random_forest'] = rf_model
        
        # 2. Gradient Boosting
        logger.info("Training Gradient Boosting Classifier...")
        gb_model = GradientBoostingClassifier(
            n_estimators=self.config.model.n_estimators,
            max_depth=min(self.config.model.max_depth or 5, 5),
            random_state=self.config.model.random_state
        )
        gb_model.fit(X_train, y_train)
        gb_pred = gb_model.predict(X_test)
        gb_pred_proba = gb_model.predict_proba(X_test)
        
        results['gradient_boosting'] = self._evaluate_classification(y_test, gb_pred, gb_pred_proba)
        results['gradient_boosting']['feature_importance'] = dict(zip(feature_names, gb_model.feature_importances_))
        self.models[f'{model_name_prefix}gradient_boosting'] = gb_model
        
        # 3. Logistic Regression
        logger.info("Training Logistic Regression...")
        lr_model = LogisticRegression(
            class_weight=class_weight_dict,
            random_state=self.config.model.random_state,
            max_iter=1000
        )
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        lr_pred_proba = lr_model.predict_proba(X_test)
        
        results['logistic_regression'] = self._evaluate_classification(y_test, lr_pred, lr_pred_proba)
        self.models[f'{model_name_prefix}logistic_regression'] = lr_model
        
        # 4. Neural Network (if TensorFlow available)
        if TF_AVAILABLE and len(classes) > 1:
            logger.info("Training Neural Network...")
            nn_model = self._build_nn_classifier(X_train.shape[1], len(classes))
            
            # Prepare data for neural network
            y_train_cat = to_categorical(y_train, num_classes=len(classes))
            y_test_cat = to_categorical(y_test, num_classes=len(classes))
            
            # Train
            history = nn_model.fit(
                X_train, y_train_cat,
                validation_split=0.2,
                epochs=self.config.model.epochs,
                batch_size=self.config.model.batch_size,
                callbacks=self._get_nn_callbacks(f'{model_name_prefix}nn_classifier'),
                verbose=0
            )
            
            nn_pred_proba = nn_model.predict(X_test)
            nn_pred = np.argmax(nn_pred_proba, axis=1)
            
            results['neural_network'] = self._evaluate_classification(y_test, nn_pred, nn_pred_proba)
            results['neural_network']['history'] = history.history
            self.models[f'{model_name_prefix}neural_network'] = nn_model
        
        # Cross-validation for best model
        best_model_name = max(results.keys(), key=lambda x: results[x].get('f1_score', 0))
        best_model = self.models[f'{model_name_prefix}{best_model_name}']
        
        if hasattr(best_model, 'predict'):
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=StratifiedKFold(n_splits=self.config.model.cross_validation_folds),
                scoring='f1_weighted'
            )
            results[best_model_name]['cv_scores'] = cv_scores.tolist()
            results[best_model_name]['cv_mean'] = cv_scores.mean()
            results[best_model_name]['cv_std'] = cv_scores.std()
        
        logger.info(f"Best model: {best_model_name}")
        return results
    
    def train_regression_models(self, X_train, X_test, y_train, y_test, 
                               feature_names: List[str], model_name_prefix: str = "") -> Dict:
        """Train multiple regression models and compare performance"""
        logger.info("Training regression models...")
        
        results = {}
        
        # 1. Random Forest Regressor
        logger.info("Training Random Forest Regressor...")
        rf_model = RandomForestRegressor(
            n_estimators=self.config.model.n_estimators,
            max_depth=self.config.model.max_depth,
            min_samples_split=self.config.model.min_samples_split,
            min_samples_leaf=self.config.model.min_samples_leaf,
            random_state=self.config.model.random_state,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        results['random_forest'] = self._evaluate_regression(y_test, rf_pred)
        results['random_forest']['feature_importance'] = dict(zip(feature_names, rf_model.feature_importances_))
        self.models[f'{model_name_prefix}rf_regressor'] = rf_model
        
        # 2. Ridge Regression
        logger.info("Training Ridge Regression...")
        ridge_model = Ridge(alpha=1.0, random_state=self.config.model.random_state)
        ridge_model.fit(X_train, y_train)
        ridge_pred = ridge_model.predict(X_test)
        
        results['ridge'] = self._evaluate_regression(y_test, ridge_pred)
        self.models[f'{model_name_prefix}ridge'] = ridge_model
        
        # 3. Neural Network (if TensorFlow available)
        if TF_AVAILABLE:
            logger.info("Training Neural Network Regressor...")
            nn_model = self._build_nn_regressor(X_train.shape[1])
            
            history = nn_model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=self.config.model.epochs,
                batch_size=self.config.model.batch_size,
                callbacks=self._get_nn_callbacks(f'{model_name_prefix}nn_regressor'),
                verbose=0
            )
            
            nn_pred = nn_model.predict(X_test).flatten()
            
            results['neural_network'] = self._evaluate_regression(y_test, nn_pred)
            results['neural_network']['history'] = history.history
            self.models[f'{model_name_prefix}nn_regressor'] = nn_model
        
        # Cross-validation for best model
        best_model_name = max(results.keys(), key=lambda x: results[x].get('r2_score', -np.inf))
        best_model = self.models[f'{model_name_prefix}{("rf_regressor" if best_model_name == "random_forest" else best_model_name)}']
        
        if hasattr(best_model, 'predict'):
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=self.config.model.cross_validation_folds,
                scoring='r2'
            )
            results[best_model_name]['cv_scores'] = cv_scores.tolist()
            results[best_model_name]['cv_mean'] = cv_scores.mean()
            results[best_model_name]['cv_std'] = cv_scores.std()
        
        logger.info(f"Best model: {best_model_name}")
        return results
    
    def _evaluate_classification(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """Evaluate classification model performance"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
            'classification_report': classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        
        # Add AUC if binary classification
        if y_pred_proba is not None and len(np.unique(y_true)) == 2:
            if len(y_pred_proba.shape) > 1:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def _evaluate_regression(self, y_true, y_pred) -> Dict:
        """Evaluate regression model performance"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2_score': r2_score(y_true, y_pred),
            'explained_variance': 1 - np.var(y_true - y_pred) / np.var(y_true)
        }
    
    def _build_nn_classifier(self, input_dim: int, num_classes: int) -> Model:
        """Build neural network for classification"""
        model = Sequential([
            Dense(self.config.model.hidden_layers[0], activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(self.config.model.dropout_rate),
            Dense(self.config.model.hidden_layers[1], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.model.dropout_rate),
            Dense(self.config.model.hidden_layers[2], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.model.dropout_rate),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.model.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _build_nn_regressor(self, input_dim: int) -> Model:
        """Build neural network for regression"""
        model = Sequential([
            Dense(self.config.model.hidden_layers[0], activation='relu', input_dim=input_dim),
            BatchNormalization(),
            Dropout(self.config.model.dropout_rate),
            Dense(self.config.model.hidden_layers[1], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.model.dropout_rate),
            Dense(self.config.model.hidden_layers[2], activation='relu'),
            BatchNormalization(),
            Dropout(self.config.model.dropout_rate),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.config.model.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _get_nn_callbacks(self, model_name: str) -> list:
        """Get callbacks for neural network training"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.model.early_stopping_patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        if TF_AVAILABLE:
            model_path = self.config.model.model_dir / f"{model_name}.h5"
            callbacks.append(
                ModelCheckpoint(
                    str(model_path),
                    monitor='val_loss',
                    save_best_only=True
                )
            )
        
        return callbacks
    
    def train_all_models(self, df: pd.DataFrame) -> Dict:
        """Train all configured models on different targets"""
        logger.info("Starting comprehensive model training...")
        
        all_results = {}
        
        # 1. Classification: Poly-sensitization prediction
        if 'poly_sensitized' in df.columns:
            logger.info("Training models for poly-sensitization prediction...")
            X_train, X_test, y_train, y_test, features = self.prepare_data(
                df, 'poly_sensitized', 'classification'
            )
            all_results['poly_sensitization'] = self.train_classification_models(
                X_train, X_test, y_train, y_test, features, 'poly_'
            )
        
        # 2. Classification: IgE class prediction
        if 'ige_class' in df.columns:
            logger.info("Training models for IgE class prediction...")
            X_train, X_test, y_train, y_test, features = self.prepare_data(
                df, 'ige_class', 'classification'
            )
            all_results['ige_class'] = self.train_classification_models(
                X_train, X_test, y_train, y_test, features, 'ige_class_'
            )
        
        # 3. Regression: Total IgE level prediction
        if 'log_total_ige' in df.columns:
            logger.info("Training models for IgE level regression...")
            X_train, X_test, y_train, y_test, features = self.prepare_data(
                df, 'log_total_ige', 'regression'
            )
            all_results['ige_regression'] = self.train_regression_models(
                X_train, X_test, y_train, y_test, features, 'ige_reg_'
            )
        
        # Save results
        self.save_results(all_results)
        self.save_models()
        
        return all_results
    
    def save_results(self, results: Dict):
        """Save training results to JSON"""
        results_path = self.config.model.results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy arrays to lists for JSON serialization
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
        
        serializable_results = convert_to_serializable(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")
    
    def save_models(self):
        """Save trained models"""
        for model_name, model in self.models.items():
            if TF_AVAILABLE and hasattr(model, 'save'):
                # Save Keras model
                model_path = self.config.model.model_dir / f"{model_name}.h5"
                model.save(str(model_path))
            else:
                # Save sklearn model
                model_path = self.config.model.model_dir / f"{model_name}.pkl"
                joblib.dump(model, model_path)
            
            logger.info(f"Saved model {model_name} to {model_path}")
        
        # Save scalers and encoders
        if self.scalers:
            joblib.dump(self.scalers, self.config.model.model_dir / "scalers.pkl")
        if self.encoders:
            joblib.dump(self.encoders, self.config.model.model_dir / "encoders.pkl")