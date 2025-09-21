import numpy as np
import pandas as pd
import joblib
import os

def generate_predictions():
    """
    Generar predicciones para el set de prueba de Kaggle
    """
    print("Generando predicciones para el set de prueba...")
    
    try:
        # Cargar el modelo entrenado
        model_files = [f for f in os.listdir('models') if f.endswith('.joblib')]
        if not model_files:
            print("No se encontró ningún modelo entrenado")
            return
        
        best_model_path = f'models/{model_files[0]}'
        best_model = joblib.load(best_model_path)
        print(f"Modelo cargado: {model_files[0]}")
        
        # Cargar datos de prueba procesados
        X_test = np.load('data/processed/X_test.npy')
        test_ids = pd.read_csv('data/processed/test_ids.csv')
        
        print(f"Datos de prueba: {X_test.shape}")
        print(f"IDs de prueba: {test_ids.shape}")
        
        # Hacer predicciones
        print("Realizando predicciones...")
        y_pred_log = best_model.predict(X_test)
        
        # Convertir a escala original (dólares)
        y_pred_dollars = np.expm1(y_pred_log)
        
        # Crear submission file
        submission = pd.DataFrame({
            'Id': test_ids['Id'],
            'SalePrice': y_pred_dollars
        })
        
        # Guardar predicciones
        os.makedirs('submissions', exist_ok=True)
        submission_file = 'submissions/submission.csv'
        submission.to_csv(submission_file, index=False)
        
        print(f"Predicciones guardadas en: {submission_file}")
        print(f"Rango de predicciones: ${submission['SalePrice'].min():,.0f} - ${submission['SalePrice'].max():,.0f}")
        print(f"Precio promedio predicho: ${submission['SalePrice'].mean():,.0f}")
        
        # Mostrar primeras filas
        print("\n📋 Primeras 5 predicciones:")
        print(submission.head())
        
        return submission
        
    except Exception as e:
        print(f"Error generando predicciones: {e}")
        return None

if __name__ == "__main__":
    generate_predictions()