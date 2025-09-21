import joblib
from data_preprocessing import HousePricesPreprocessor
import pandas as pd

def save_preprocessor():
    """Guardar el preprocessor entrenado para uso futuro"""
    print("Entrenando y guardando el preprocessor...")
    
    # Cargar datos
    train_df = pd.read_csv('data/raw/train.csv')
    
    # Entrenar preprocessor
    preprocessor = HousePricesPreprocessor()
    X_train_processed, y_train = preprocessor.fit_transform(train_df)
    
    # Guardar preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    print("Preprocessor guardado en models/preprocessor.joblib")
    
    # Guardar feature names
    feature_names = preprocessor.get_feature_names()
    pd.Series(feature_names).to_csv('models/feature_names.csv', index=False)
    print("Feature names guardados en models/feature_names.csv")

if __name__ == "__main__":
    # Crear directorio models si no existe
    import os
    os.makedirs('models', exist_ok=True)
    
    save_preprocessor()