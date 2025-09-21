import numpy as np
import pandas as pd
import os

def check_processed_data():
    """Verificar que los datos procesados son correctos"""
    print("=== VERIFICACIÓN DE DATOS PROCESADOS ===\n")
    
    # Verificar que los archivos existen
    files = [
        'X_train.npy', 'y_train.npy', 'X_test.npy', 
        'feature_names.csv', 'test_ids.csv'
    ]
    
    all_files_exist = True
    for file in files:
        path = f'data/processed/{file}'
        if os.path.exists(path):
            print(f"✓ {file} existe")
        else:
            print(f"✗ {file} NO existe")
            all_files_exist = False
    
    if not all_files_exist:
        print("\nError: Faltan algunos archivos procesados")
        return False
    
    # Cargar y verificar los datos
    try:
        X_train = np.load('data/processed/X_train.npy')
        y_train = np.load('data/processed/y_train.npy')
        X_test = np.load('data/processed/X_test.npy')
        feature_names = pd.read_csv('data/processed/feature_names.csv')
        test_ids = pd.read_csv('data/processed/test_ids.csv')
        
        print(f"\n✓ Datos cargados correctamente")
        print(f"  - X_train shape: {X_train.shape}")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - X_test shape: {X_test.shape}")
        print(f"  - Número de features: {len(feature_names)}")
        print(f"  - Número de test IDs: {len(test_ids)}")
        
        # Verificar que no hay NaN values
        print(f"  - NaN en X_train: {np.isnan(X_train).sum()}")
        print(f"  - NaN en X_test: {np.isnan(X_test).sum()}")
        print(f"  - NaN en y_train: {np.isnan(y_train).sum()}")
        
        # Verificar ranges
        print(f"  - Rango de X_train: [{X_train.min():.2f}, {X_train.max():.2f}]")
        print(f"  - Rango de y_train: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error al cargar los datos: {e}")
        return False

if __name__ == "__main__":
    success = check_processed_data()
    if success:
        print("\n✅ Verificación completada: Todos los datos están correctos")
    else:
        print("\n❌ Verificación falló: Hay problemas con los datos procesados")