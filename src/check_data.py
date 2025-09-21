import pandas as pd
import os

def check_data():
    """Verificar que los datos se descargaron correctamente"""
    print("Verificando datos descargados...")
    
    files = os.listdir('data/raw')
    print("Archivos en data/raw:", files)
    
    if 'train.csv' in files:
        train_df = pd.read_csv('data/raw/train.csv')
        print(f"\nTrain dataset: {train_df.shape[0]} filas, {train_df.shape[1]} columnas")
        print("\nPrimeras 5 filas:")
        print(train_df.head())
        
        print("\nInformación del dataset:")
        print(train_df.info())
        
        print("\nColumnas disponibles:")
        for col in train_df.columns:
            print(f"  - {col}")
    
    return True

if __name__ == "__main__":
    check_data()