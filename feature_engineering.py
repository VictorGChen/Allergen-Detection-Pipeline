"""
Feature engineering module for the allergy detection pipeline.
Handles feature creation, transformation, and augmentation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
import logging
from pathlib import Path

from config import PipelineConfig, IGE_CLASSES

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Main class for feature engineering operations"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = []
        self.ige_columns = []
        
    def engineer_features(self, nhanes_data: pd.DataFrame, 
                         immport_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Main feature engineering pipeline.
        
        Args:
            nhanes_data: NHANES DataFrame
            immport_data: Optional ImmPort DataFrame
            
        Returns:
            Engineered feature DataFrame
        """
        logger.info("Starting feature engineering...")
        
        # Create a copy to avoid modifying original data
        df = nhanes_data.copy()
        
        # Identify IgE columns
        self.ige_columns = [col for col in df.columns 
                           if col.startswith('LBX') or 'IgE' in col or col in [
                               'Dermatophagoides_farinae', 'Dermatophagoides_pteronyssinus',
                               'Cat_epithelium', 'Dog_epithelium', 'German_cockroach',
                               'Alternaria_alternata', 'Ragweed', 'Rye_grass', 
                               'Bermuda_grass', 'Oak', 'Birch', 'Peanut', 
                               'Egg_white', 'Milk', 'Shrimp'
                           ]]
        
        # 1. Create sensitization features
        df = self.create_sensitization_features(df)
        
        # 2. Create demographic features
        df = self.create_demographic_features(df)
        
        # 3. Create IgE summary features
        df = self.create_ige_summary_features(df)
        
        # 4. Create seasonal features
        df = self.create_seasonal_features(df)
        
        # 5. Create body measurement features
        df = self.create_body_measurement_features(df)
        
        # 6. Create dietary features (if available)
        df = self.create_dietary_features(df)
        
        # 7. Merge ImmPort features if available
        if immport_data is not None and not immport_data.empty:
            df = self.merge_immport_features(df, immport_data)
        
        # 8. Create interaction features if configured
        if self.config.model.create_interaction_features:
            df = self.create_interaction_features(df)
        
        # 9. Handle missing values
        df = self.handle_missing_values(df)
        
        # 10. Apply data augmentation if needed
        df = self.augment_data_if_needed(df)
        
        # 11. Log transformation for IgE values
        df = self.apply_log_transformation(df)
        
        # 12. Create target variables
        df = self.create_target_variables(df)
        
        # Save processed features
        output_path = self.config.data.processed_data_dir / "engineered_features.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved engineered features to {output_path}")
        
        logger.info(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def create_sensitization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create allergen sensitization indicators"""
        logger.info("Creating sensitization features...")
        
        # Identify allergen-specific IgE columns (excluding total IgE)
        allergen_cols = [col for col in self.ige_columns 
                        if col not in ['Total_IgE', 'LBXIGE']]
        
        # Create binary sensitization indicators
        for col in allergen_cols:
            if col in df.columns:
                sens_col = f"{col}_sensitized"
                df[sens_col] = (df[col] >= self.config.data.ige_threshold).astype(int)
                
                # Handle missing values
                df.loc[df[col].isna(), sens_col] = np.nan
        
        # Count total sensitizations
        sens_cols = [col for col in df.columns if col.endswith('_sensitized')]
        if sens_cols:
            df['total_sensitizations'] = df[sens_cols].sum(axis=1, skipna=True)
            df['any_sensitization'] = (df['total_sensitizations'] > 0).astype(int)
            df['poly_sensitized'] = (df['total_sensitizations'] >= 3).astype(int)
            
            # Categorize by allergen type
            indoor_allergens = ['Dermatophagoides_farinae', 'Dermatophagoides_pteronyssinus',
                              'Cat_epithelium', 'Dog_epithelium', 'German_cockroach', 
                              'Alternaria_alternata']
            outdoor_allergens = ['Ragweed', 'Rye_grass', 'Bermuda_grass', 'Oak', 'Birch']
            food_allergens = ['Peanut', 'Egg_white', 'Milk', 'Shrimp']
            
            for allergen_type, allergen_list in [('indoor', indoor_allergens), 
                                                 ('outdoor', outdoor_allergens),
                                                 ('food', food_allergens)]:
                type_cols = [f"{a}_sensitized" for a in allergen_list 
                           if f"{a}_sensitized" in df.columns]
                if type_cols:
                    df[f'{allergen_type}_sensitizations'] = df[type_cols].sum(axis=1, skipna=True)
                    df[f'any_{allergen_type}_sensitization'] = (df[f'{allergen_type}_sensitizations'] > 0).astype(int)
        
        logger.info(f"Created {len(sens_cols)} sensitization features")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic-based features"""
        logger.info("Creating demographic features...")
        
        # Age groups
        if 'Age_years' in df.columns or 'RIDAGEYR' in df.columns:
            age_col = 'Age_years' if 'Age_years' in df.columns else 'RIDAGEYR'
            df['age_group'] = pd.cut(df[age_col], 
                                    bins=[0, 6, 12, 18, 40, 60, 100],
                                    labels=['preschool', 'school_age', 'adolescent', 
                                           'young_adult', 'middle_age', 'elderly'])
            df['is_child'] = (df[age_col] < 18).astype(int)
            df['is_elderly'] = (df[age_col] >= 65).astype(int)
            df['age_squared'] = df[age_col] ** 2
            df['age_cubed'] = df[age_col] ** 3
        
        # Gender features
        if 'Gender' in df.columns or 'RIAGENDR' in df.columns:
            gender_col = 'Gender' if 'Gender' in df.columns else 'RIAGENDR'
            df['is_male'] = (df[gender_col] == 1).astype(int)
            df['is_female'] = (df[gender_col] == 2).astype(int)
        
        # Poverty income ratio categories
        if 'Poverty_income_ratio' in df.columns or 'INDFMPIR' in df.columns:
            pir_col = 'Poverty_income_ratio' if 'Poverty_income_ratio' in df.columns else 'INDFMPIR'
            df['low_income'] = (df[pir_col] < 1.3).astype(int)
            df['middle_income'] = ((df[pir_col] >= 1.3) & (df[pir_col] < 3.5)).astype(int)
            df['high_income'] = (df[pir_col] >= 3.5).astype(int)
        
        # Education level (for adults)
        if 'Education_level' in df.columns or 'DMDEDUC2' in df.columns:
            edu_col = 'Education_level' if 'Education_level' in df.columns else 'DMDEDUC2'
            df['less_than_high_school'] = (df[edu_col] < 3).astype(int)
            df['high_school'] = (df[edu_col] == 3).astype(int)
            df['some_college'] = (df[edu_col] == 4).astype(int)
            df['college_graduate'] = (df[edu_col] == 5).astype(int)
        
        return df
    
    def create_ige_summary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create IgE summary statistics"""
        logger.info("Creating IgE summary features...")
        
        allergen_cols = [col for col in self.ige_columns 
                        if col in df.columns and col not in ['Total_IgE', 'LBXIGE']]
        
        if allergen_cols:
            # Statistical summaries
            df['mean_specific_ige'] = df[allergen_cols].mean(axis=1, skipna=True)
            df['median_specific_ige'] = df[allergen_cols].median(axis=1, skipna=True)
            df['max_specific_ige'] = df[allergen_cols].max(axis=1, skipna=True)
            df['min_specific_ige'] = df[allergen_cols].min(axis=1, skipna=True)
            df['std_specific_ige'] = df[allergen_cols].std(axis=1, skipna=True)
            
            # Percentiles
            df['ige_25percentile'] = df[allergen_cols].quantile(0.25, axis=1)
            df['ige_75percentile'] = df[allergen_cols].quantile(0.75, axis=1)
            df['ige_iqr'] = df['ige_75percentile'] - df['ige_25percentile']
            
            # Ratio features
            total_ige_col = 'Total_IgE' if 'Total_IgE' in df.columns else 'LBXIGE'
            if total_ige_col in df.columns:
                df['specific_to_total_ratio'] = df['mean_specific_ige'] / (df[total_ige_col] + 1)
                df['max_specific_to_total_ratio'] = df['max_specific_ige'] / (df[total_ige_col] + 1)
        
        return df
    
    def create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create seasonal and temporal features"""
        logger.info("Creating seasonal features...")
        
        if 'interview_month' in df.columns:
            # Season mapping
            season_map = {
                1: 'winter', 2: 'winter', 3: 'spring',
                4: 'spring', 5: 'spring', 6: 'summer',
                7: 'summer', 8: 'summer', 9: 'fall',
                10: 'fall', 11: 'fall', 12: 'winter'
            }
            df['season'] = df['interview_month'].map(season_map)
            
            # One-hot encode seasons
            season_dummies = pd.get_dummies(df['season'], prefix='season')
            df = pd.concat([df, season_dummies], axis=1)
            
            # High pollen season (spring and early fall)
            df['high_pollen_season'] = df['interview_month'].isin([3, 4, 5, 9]).astype(int)
            
            # Cyclical encoding for month
            df['month_sin'] = np.sin(2 * np.pi * df['interview_month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['interview_month'] / 12)
        
        return df
    
    def create_body_measurement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create body measurement features"""
        logger.info("Creating body measurement features...")
        
        # BMI categories
        if 'BMI' in df.columns or 'BMXBMI' in df.columns:
            bmi_col = 'BMI' if 'BMI' in df.columns else 'BMXBMI'
            df['underweight'] = (df[bmi_col] < 18.5).astype(int)
            df['normal_weight'] = ((df[bmi_col] >= 18.5) & (df[bmi_col] < 25)).astype(int)
            df['overweight'] = ((df[bmi_col] >= 25) & (df[bmi_col] < 30)).astype(int)
            df['obese'] = (df[bmi_col] >= 30).astype(int)
            df['bmi_squared'] = df[bmi_col] ** 2
        
        # Height and weight ratios
        if 'Height_cm' in df.columns and 'Weight_kg' in df.columns:
            df['height_weight_ratio'] = df['Height_cm'] / (df['Weight_kg'] + 1)
        elif 'BMXHT' in df.columns and 'BMXWT' in df.columns:
            df['height_weight_ratio'] = df['BMXHT'] / (df['BMXWT'] + 1)
        
        return df
    
    def create_dietary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create dietary-related features"""
        logger.info("Creating dietary features...")
        
        dietary_cols = ['Calorie_intake', 'Protein_intake', 'Fat_intake', 'Carb_intake',
                       'DR1TKCAL', 'DR1TPROT', 'DR1TFAT', 'DR1TCARB']
        
        available_cols = [col for col in dietary_cols if col in df.columns]
        
        if available_cols:
            # Macronutrient ratios
            if 'Calorie_intake' in df.columns or 'DR1TKCAL' in df.columns:
                cal_col = 'Calorie_intake' if 'Calorie_intake' in df.columns else 'DR1TKCAL'
                
                if 'Protein_intake' in df.columns or 'DR1TPROT' in df.columns:
                    prot_col = 'Protein_intake' if 'Protein_intake' in df.columns else 'DR1TPROT'
                    df['protein_pct_calories'] = (df[prot_col] * 4) / (df[cal_col] + 1) * 100
                
                if 'Fat_intake' in df.columns or 'DR1TFAT' in df.columns:
                    fat_col = 'Fat_intake' if 'Fat_intake' in df.columns else 'DR1TFAT'
                    df['fat_pct_calories'] = (df[fat_col] * 9) / (df[cal_col] + 1) * 100
                
                if 'Carb_intake' in df.columns or 'DR1TCARB' in df.columns:
                    carb_col = 'Carb_intake' if 'Carb_intake' in df.columns else 'DR1TCARB'
                    df['carb_pct_calories'] = (df[carb_col] * 4) / (df[cal_col] + 1) * 100
                
                # Calorie categories
                df['low_calorie'] = (df[cal_col] < 1500).astype(int)
                df['normal_calorie'] = ((df[cal_col] >= 1500) & (df[cal_col] < 2500)).astype(int)
                df['high_calorie'] = (df[cal_col] >= 2500).astype(int)
        
        return df
    
    def merge_immport_features(self, nhanes_df: pd.DataFrame, 
                              immport_df: pd.DataFrame) -> pd.DataFrame:
        """Merge ImmPort immunological features"""
        logger.info("Merging ImmPort features...")
        
        # For mock data integration, we'll create a mapping based on age and gender
        # In real scenario, you'd have proper patient IDs to match
        
        # Select key immunological features from ImmPort
        immport_features = ['IL4_baseline', 'IL5_baseline', 'IL13_baseline',
                           'IFN_gamma_baseline', 'Th2_percent', 'CD4_percent',
                           'GATA3_expression', 'IL4R_expression']
        
        available_features = [f for f in immport_features if f in immport_df.columns]
        
        if available_features:
            # Create synthetic matching (in production, use real patient IDs)
            # For demo, randomly sample ImmPort features for NHANES subjects
            n_nhanes = len(nhanes_df)
            n_immport = len(immport_df)
            
            if n_immport > 0:
                # Randomly assign ImmPort features to NHANES subjects
                indices = np.random.choice(n_immport, n_nhanes, replace=True)
                
                for feature in available_features:
                    nhanes_df[f'immport_{feature}'] = immport_df[feature].iloc[indices].values
                
                logger.info(f"Added {len(available_features)} ImmPort features")
        
        return nhanes_df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        logger.info("Creating interaction features...")
        
        # Age-IgE interactions
        if 'Age_years' in df.columns or 'RIDAGEYR' in df.columns:
            age_col = 'Age_years' if 'Age_years' in df.columns else 'RIDAGEYR'
            
            if 'Total_IgE' in df.columns or 'LBXIGE' in df.columns:
                ige_col = 'Total_IgE' if 'Total_IgE' in df.columns else 'LBXIGE'
                df['age_ige_interaction'] = df[age_col] * np.log1p(df[ige_col])
            
            if 'total_sensitizations' in df.columns:
                df['age_sensitization_interaction'] = df[age_col] * df['total_sensitizations']
        
        # BMI-IgE interactions
        if 'BMI' in df.columns or 'BMXBMI' in df.columns:
            bmi_col = 'BMI' if 'BMI' in df.columns else 'BMXBMI'
            
            if 'Total_IgE' in df.columns or 'LBXIGE' in df.columns:
                ige_col = 'Total_IgE' if 'Total_IgE' in df.columns else 'LBXIGE'
                df['bmi_ige_interaction'] = df[bmi_col] * np.log1p(df[ige_col])
        
        # Season-sensitization interactions
        if 'high_pollen_season' in df.columns and 'outdoor_sensitizations' in df.columns:
            df['season_outdoor_interaction'] = df['high_pollen_season'] * df['outdoor_sensitizations']
        
        # Polynomial features for selected columns
        if self.config.model.polynomial_features_degree > 1:
            key_features = ['age_squared', 'bmi_squared', 'mean_specific_ige']
            available_key = [f for f in key_features if f in df.columns]
            
            if available_key:
                poly = PolynomialFeatures(degree=self.config.model.polynomial_features_degree,
                                         include_bias=False)
                poly_features = poly.fit_transform(df[available_key])
                poly_names = [f'poly_{i}' for i in range(poly_features.shape[1])]
                
                for i, name in enumerate(poly_names[:10]):  # Limit to 10 polynomial features
                    df[name] = poly_features[:, i]
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with sophisticated imputation"""
        logger.info("Handling missing values...")
        
        # Identify numeric and categorical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # For numeric columns, use median imputation
        if numeric_cols:
            df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        
        # For categorical columns, use mode or 'unknown'
        for col in categorical_cols:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col].fillna(mode_val[0], inplace=True)
            else:
                df[col].fillna('unknown', inplace=True)
        
        # Special handling for sensitization features (should be 0 if IgE is below threshold)
        sens_cols = [col for col in df.columns if col.endswith('_sensitized')]
        for col in sens_cols:
            df[col].fillna(0, inplace=True)
        
        logger.info(f"Imputed missing values. Remaining nulls: {df.isnull().sum().sum()}")
        return df
    
    def augment_data_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply data augmentation for imbalanced classes"""
        logger.info("Checking if data augmentation is needed...")
        
        # Check if we have the poly_sensitized target
        if 'poly_sensitized' not in df.columns:
            return df
        
        # Check class balance
        class_counts = df['poly_sensitized'].value_counts()
        if len(class_counts) < 2:
            logger.warning("Only one class present, skipping augmentation")
            return df
        
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        if imbalance_ratio > 3:  # If imbalance is severe
            logger.info(f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). Applying augmentation...")
            
            # Use SMOTE-like augmentation
            minority_data = df[df['poly_sensitized'] == minority_class]
            n_synthetic = int(len(minority_data) * 0.5)  # Generate 50% more minority samples
            
            if n_synthetic > 0 and len(minority_data) > 5:
                synthetic_samples = self.generate_synthetic_samples(minority_data, n_synthetic)
                df = pd.concat([df, synthetic_samples], ignore_index=True)
                logger.info(f"Added {len(synthetic_samples)} synthetic samples")
        
        return df
    
    def generate_synthetic_samples(self, minority_data: pd.DataFrame, 
                                  n_samples: int) -> pd.DataFrame:
        """Generate synthetic samples using nearest neighbors interpolation"""
        
        # Select numeric features only
        numeric_cols = minority_data.select_dtypes(include=[np.number]).columns.tolist()
        X = minority_data[numeric_cols].values
        
        # Fit nearest neighbors
        k = min(5, len(minority_data) - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        
        synthetic_samples = []
        
        for _ in range(n_samples):
            # Random sample from minority class
            idx = np.random.randint(0, len(X))
            sample = X[idx]
            
            # Find k nearest neighbors
            distances, indices = nn.kneighbors(sample.reshape(1, -1), n_neighbors=k+1)
            indices = indices.flatten()[1:]  # Exclude the sample itself
            
            # Random neighbor
            neighbor_idx = np.random.choice(indices)
            neighbor = X[neighbor_idx]
            
            # Generate synthetic sample by interpolation
            gap = np.random.random()
            synthetic = sample + gap * (neighbor - sample)
            synthetic_samples.append(synthetic)
        
        # Create DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples, columns=numeric_cols)
        
        # For non-numeric columns, randomly sample from minority class
        non_numeric_cols = [col for col in minority_data.columns if col not in numeric_cols]
        for col in non_numeric_cols:
            synthetic_df[col] = np.random.choice(minority_data[col].values, n_samples)
        
        return synthetic_df
    
    def apply_log_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to IgE values"""
        logger.info("Applying log transformation to IgE values...")
        
        # Transform total IgE
        if 'Total_IgE' in df.columns:
            df['log_total_ige'] = np.log1p(df['Total_IgE'])
        elif 'LBXIGE' in df.columns:
            df['log_total_ige'] = np.log1p(df['LBXIGE'])
        
        # Transform specific IgE values
        allergen_cols = [col for col in self.ige_columns 
                        if col in df.columns and col not in ['Total_IgE', 'LBXIGE']]
        
        for col in allergen_cols:
            df[f'log_{col}'] = np.log1p(df[col])
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for modeling"""
        logger.info("Creating target variables...")
        
        # Classification target: IgE class
        if 'Total_IgE' in df.columns or 'LBXIGE' in df.columns:
            ige_col = 'Total_IgE' if 'Total_IgE' in df.columns else 'LBXIGE'
            
            def classify_ige(value):
                if pd.isna(value):
                    return 'Unknown'
                for class_name, (low, high) in IGE_CLASSES.items():
                    if low <= value < high:
                        return class_name
                return 'Very_High'
            
            df['ige_class'] = df[ige_col].apply(classify_ige)
            
            # Binary classification
            df['elevated_ige'] = (df[ige_col] > self.config.data.total_ige_threshold).astype(int)
        
        # Regression target: log-transformed total IgE (already created)
        
        # Multi-class targets
        if 'total_sensitizations' in df.columns:
            df['sensitization_severity'] = pd.cut(df['total_sensitizations'],
                                                 bins=[-0.1, 0, 2, 5, 100],
                                                 labels=['none', 'mild', 'moderate', 'severe'])
        
        return df