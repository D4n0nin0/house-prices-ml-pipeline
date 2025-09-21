import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

class HousePricesModelTrainer:
    """
    Clase para entrenar y evaluar múltiples modelos de machine learning
    en el dataset de House Prices.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(random_state=42),
            'Lasso Regression': Lasso(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
            'LightGBM': lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        }
        
        self.best_model = None
        self.best_model_name = None
        self.cv_results = {}
        self.final_results = {}
    
    def load_processed_data(self):
        """
        Cargar los datos procesados del preprocesamiento.
        """
        print("Cargando datos procesados...")
        try:
            X_train = np.load('data/processed/X_train.npy')
            y_train = np.load('data/processed/y_train.npy')
            X_test = np.load('data/processed/X_test.npy')
            
            print(f"Datos cargados:")
            print(f"X_train: {X_train.shape}")
            print(f"y_train: {y_train.shape}")
            print(f"X_test: {X_test.shape}")
            
            return X_train, y_train, X_test
            
        except Exception as e:
            print(f"Error al cargar datos: {e}")
            return None, None, None
    
    def train_models(self, X_train, y_train, cv_folds=5):
        """
        Entrenar y evaluar múltiples modelos usando cross-validation.
        """
        print("\n" + "="*60)
        print("ENTRENAMIENTO Y EVALUACIÓN DE MODELOS")
        print("="*60)
        
        results = []
        
        for name, model in self.models.items():
            print(f"\nEvaluando {name}...")
            
            start_time = time.time()
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=cv_folds, scoring='neg_root_mean_squared_error',
                n_jobs=-1
            )
            
            # Convertir a RMSE positivo
            cv_rmse = -cv_scores
            cv_mean_rmse = cv_rmse.mean()
            cv_std_rmse = cv_rmse.std()
            
            # Entrenar modelo en todo el dataset
            model.fit(X_train, y_train)
            
            # Predecir en training (para R²)
            y_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_pred)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # Guardar resultados
            result = {
                'model': name,
                'cv_mean_rmse': cv_mean_rmse,
                'cv_std_rmse': cv_std_rmse,
                'train_r2': train_r2,
                'training_time': training_time
            }
            
            results.append(result)
            self.cv_results[name] = model
            
            print(f"   RMSE CV: {cv_mean_rmse:.4f} (±{cv_std_rmse:.4f})")
            print(f"   R² Train: {train_r2:.4f}")
            print(f"   Tiempo: {training_time:.2f} segundos")
        
        # Crear DataFrame con resultados
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('cv_mean_rmse')
        
        # Seleccionar el mejor modelo
        self.best_model_name = results_df.iloc[0]['model']
        self.best_model = self.cv_results[self.best_model_name]
        
        print("\n" + "="*60)
        print("MEJOR MODELO SELECCIONADO:")
        print(f"   {self.best_model_name}")
        print(f"   RMSE CV: {results_df.iloc[0]['cv_mean_rmse']:.4f}")
        print("="*60)
        
        return results_df
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Optimización de hiperparámetros para el mejor modelo.
        """
        print("\n" + "="*60)
        print("OPTIMIZACIÓN DE HIPERPARÁMETROS")
        print("="*60)
        
        if self.best_model_name == 'Random Forest':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42, n_jobs=-1)
            
        elif self.best_model_name == 'Gradient Boosting':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
            
        elif self.best_model_name == 'XGBoost':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
            
        elif self.best_model_name == 'LightGBM':
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'num_leaves': [31, 50, 70],
                'subsample': [0.8, 0.9, 1.0]
            }
            model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
            
        else:
            print("El mejor modelo no requiere optimización de hiperparámetros.")
            return self.best_model
        
        # Búsqueda de grid con cross-validation
        print(f"Optimizando {self.best_model_name}...")
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Mejores parámetros: {grid_search.best_params_}")
        print(f"Mejor RMSE: {-grid_search.best_score_:.4f}")
        
        # Actualizar el mejor modelo
        self.best_model = grid_search.best_estimator_
        
        return self.best_model
    
    def evaluate_model(self, X_train, y_train):
        """
        Evaluación final del mejor modelo.
        """
        print("\n" + "="*60)
        print("EVALUACIÓN FINAL DEL MEJOR MODELO")
        print("="*60)
        
        # Predecir en training
        y_pred = self.best_model.predict(X_train)
        
        # Calcular métricas
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        
        # Guardar resultados finales
        self.final_results = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"Modelo: {self.best_model_name}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Visualizar predicciones vs valores reales
        plt.figure(figsize=(10, 6))
        plt.scatter(y_train, y_pred, alpha=0.5)
        plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
        plt.xlabel('Valores Reales (log scale)')
        plt.ylabel('Predicciones (log scale)')
        plt.title(f'Predicciones vs Reales - {self.best_model_name}')
        plt.show()
        
        # Residual plot
        residuals = y_train - y_pred
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicciones')
        plt.ylabel('Residuales')
        plt.title(f'Análisis de Residuales - {self.best_model_name}')
        plt.show()
        
        return self.final_results
    
    def save_model(self):
        """
        Guardar el mejor modelo entrenado.
        """
        if self.best_model is not None:
            # Crear directorio si no existe
            import os
            os.makedirs('models', exist_ok=True)
            
            # Guardar modelo
            model_path = f'models/best_model_{self.best_model_name.lower().replace(" ", "_")}.joblib'
            joblib.dump(self.best_model, model_path)
            
            # Guardar resultados
            results_df = pd.DataFrame([self.final_results])
            results_df['model'] = self.best_model_name
            results_df.to_csv('models/model_performance.csv', index=False)
            
            print(f"\nModelo guardado en: {model_path}")
            print(f"Resultados guardados en: models/model_performance.csv")
            
            return model_path
        else:
            print("No hay modelo para guardar.")
            return None

def main():
    """
    Función principal para ejecutar el entrenamiento completo.
    """
    print("HOUSE PRICES - ENTRENAMIENTO DE MODELOS")
    print("="*50)
    
    # Inicializar trainer
    trainer = HousePricesModelTrainer()
    
    # Cargar datos
    X_train, y_train, X_test = trainer.load_processed_data()
    
    if X_train is None:
        print("Error: No se pudieron cargar los datos.")
        return
    
    # Entrenar y evaluar modelos
    results_df = trainer.train_models(X_train, y_train)
    
    # Optimizar hiperparámetros
    trainer.hyperparameter_tuning(X_train, y_train)
    
    # Evaluación final
    trainer.evaluate_model(X_train, y_train)
    
    # Guardar modelo
    trainer.save_model()
    
    # Mostrar resultados comparativos
    print("\n" + "="*60)
    print("COMPARATIVA DE MODELOS")
    print("="*60)
    print(results_df.to_string(index=False))
    
    return trainer

if __name__ == "__main__":
    trainer = main()