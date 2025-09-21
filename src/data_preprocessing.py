import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class HousePricesPreprocessor:
    """
    Clase para preprocesar los datos del dataset House Prices
    basado en el análisis exploratorio realizado.
    """
    
    def __init__(self):
        # Definir variables basadas en el análisis EDA
        self.numeric_features = [
            'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
            'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
            'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'Fireplaces',
            'BsmtFinSF1', 'LotFrontage', 'WoodDeckSF', '2ndFlrSF',
            'OpenPorchSF'
        ]
        
        self.categorical_features = [
            'MSZoning', 'Neighborhood', 'Condition1', 'HouseStyle',
            'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
            'Foundation', 'BsmtQual', 'BsmtExposure', 'KitchenQual',
            'GarageType', 'GarageFinish', 'SaleType', 'SaleCondition'
        ]
        
        # Variables a eliminar (alta correlación o muchos missing values)
        self.columns_to_drop = [
            'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
            'LotFrontage', 'GarageYrBlt', 'GarageCond', 'GarageQual'
        ]
        
        # Inicializar transformers
        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        self.preprocessor = None
        
    def fit_transform(self, train_df, test_df=None):
        """
        Ajustar y transformar los datos de entrenamiento y prueba.
        
        Parameters:
        train_df: DataFrame de entrenamiento
        test_df: DataFrame de prueba (opcional)
        
        Returns:
        X_train_processed, y_train, X_test_processed (si test_df provided)
        """
        # Crear copias para no modificar los originales
        train_data = train_df.copy()
        
        # 1. Manejar valores atípicos identificados en el EDA
        train_data = self._handle_outliers(train_data)
        
        # 2. Transformar variable objetivo (log transformation para normalizar)
        y_train = np.log1p(train_data['SalePrice'])
        
        # 3. Separar features
        X_train = train_data.drop('SalePrice', axis=1)
        
        # 4. Configurar preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Eliminar columnas no especificadas
        )
        
        # 5. Ajustar y transformar datos de entrenamiento
        X_train_processed = self.preprocessor.fit_transform(X_train)
        
        # 6. Procesar datos de prueba si se proporcionan
        if test_df is not None:
            test_data = test_df.copy()
            X_test_processed = self.preprocessor.transform(test_data)
            return X_train_processed, y_train, X_test_processed
        
        return X_train_processed, y_train
    
    def transform(self, df):
        """Transformar nuevos datos usando el preprocessor ajustado"""
        if self.preprocessor is None:
            raise ValueError("Debe llamar a fit_transform primero")
        return self.preprocessor.transform(df)
    
    def _handle_outliers(self, df):
        """Manejar valores atípicos identificados en el EDA"""
        # Eliminar observaciones problemáticas identificadas en el EDA
        # Casas con área habitable grande pero precio bajo
        outlier_indices = df[(df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)].index
        df = df.drop(outlier_indices)
        
        return df
    
    def get_feature_names(self):
        """Obtener nombres de las features después del preprocessing"""
        if self.preprocessor is None:
            raise ValueError("Debe llamar a fit_transform primero")
        
        numeric_features = self.numeric_features
        categorical_features = self.preprocessor.named_transformers_['cat']\
            .named_steps['onehot'].get_feature_names_out(self.categorical_features)
        
        return list(numeric_features) + list(categorical_features)

def load_and_preprocess_data():
    """
    Función principal para cargar y preprocesar los datos.
    """
    print("Cargando datos...")
    train_df = pd.read_csv('/home/mz8k/house-prices-ml-pipeline/data/raw/train.csv')
    test_df = pd.read_csv('/home/mz8k/house-prices-ml-pipeline/data/raw/test.csv')
    
    print("Datos cargados:")
    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    
    # Inicializar preprocessor
    preprocessor = HousePricesPreprocessor()
    
    print("Iniciando preprocesamiento...")
    X_train_processed, y_train, X_test_processed = preprocessor.fit_transform(train_df, test_df)
    
    print("Preprocesamiento completado:")
    print(f"X_train procesado: {X_train_processed.shape}")
    print(f"X_test procesado: {X_test_processed.shape}")
    
    # Obtener nombres de features
    feature_names = preprocessor.get_feature_names()
    print(f"Número de features finales: {len(feature_names)}")
    
    return X_train_processed, y_train, X_test_processed, feature_names, test_df['Id']

if __name__ == "__main__":
    X_train, y_train, X_test, feature_names, test_ids = load_and_preprocess_data()
    
    # Guardar datos procesados
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/X_test.npy', X_test)
    
    # Guardar feature names y test ids
    pd.Series(feature_names).to_csv('data/processed/feature_names.csv', index=False)
    test_ids.to_csv('data/processed/test_ids.csv', index=False)
    
    print("Datos procesados guardados en data/processed/")