#!/usr/bin/env python3
"""
ML Model Training Pipeline - Agent Zero V2.0
Automated model selection and cost optimization
"""

import asyncio
import logging
import joblib
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import os

try:
    from shared.knowledge.neo4j_client import Neo4jClient
except ImportError:
    logging.warning("Neo4j client not available")
    Neo4jClient = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLTrainingPipeline:
    """ML pipeline for Agent Zero optimization"""
    
    def __init__(self, neo4j_client=None):
        self.neo4j_client = neo4j_client or (Neo4jClient() if Neo4jClient else None)
        self.models = {
            'cost_predictor': None,
            'success_predictor': None,
            'latency_predictor': None
        }
        self.encoders = {
            'task_type': LabelEncoder(),
            'model': LabelEncoder()
        }
        self.scaler = StandardScaler()
        self.model_dir = "models/v2"
        self.min_training_samples = 50
        
        # Create models directory
        os.makedirs(self.model_dir, exist_ok=True)
    
    async def prepare_training_data(self) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare training data from Neo4j experiences"""
        if not self.neo4j_client:
            logger.error("Neo4j client not available")
            return np.array([]), {}
        
        # Query experiences
        query = """
        MATCH (e:Experience)
        WHERE e.success_score IS NOT NULL 
        AND e.cost_usd IS NOT NULL
        AND e.latency_ms IS NOT NULL
        RETURN e.task_type as task_type,
               e.model_used as model,
               e.success_score as success_score,
               e.cost_usd as cost_usd,
               e.latency_ms as latency_ms,
               coalesce(size(e.user_feedback), 0) as feedback_length
        ORDER BY e.timestamp DESC
        LIMIT 10000
        """
        
        try:
            raw_data = await self.neo4j_client.execute_query(query)
            
            if len(raw_data) < self.min_training_samples:
                logger.warning(f"Insufficient data: {len(raw_data)} samples (need {self.min_training_samples})")
                return np.array([]), {}
            
            # Prepare features and targets
            features = []
            targets = {'cost': [], 'success': [], 'latency': []}
            
            # Extract unique values for encoding
            task_types = list(set(record['task_type'] for record in raw_data))
            models = list(set(record['model'] for record in raw_data))
            
            # Fit encoders
            self.encoders['task_type'].fit(task_types)
            self.encoders['model'].fit(models)
            
            for record in raw_data:
                # Encode categorical features
                task_type_encoded = self.encoders['task_type'].transform([record['task_type']])[0]
                model_encoded = self.encoders['model'].transform([record['model']])[0]
                
                feature_vector = [
                    task_type_encoded,
                    model_encoded,
                    record['feedback_length']
                ]
                
                features.append(feature_vector)
                targets['cost'].append(record['cost_usd'])
                targets['success'].append(record['success_score'])
                targets['latency'].append(record['latency_ms'])
            
            logger.info(f"Prepared {len(features)} training samples")
            return np.array(features), targets
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return np.array([]), {}
    
    async def train_models(self) -> Dict[str, Any]:
        """Train ML models for prediction and optimization"""
        try:
            X, y = await self.prepare_training_data()
            
            if len(X) == 0:
                return {'error': 'No training data available'}
            
            if len(X) < self.min_training_samples:
                return {'error': f'Insufficient training data: {len(X)} samples'}
            
            # Split data
            X_train, X_test, y_cost_train, y_cost_test = train_test_split(
                X, y['cost'], test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            
            # 1. Cost prediction model
            logger.info("Training cost prediction model...")
            cost_model = RandomForestRegressor(n_estimators=100, random_state=42)
            cost_model.fit(X_train_scaled, y_cost_train)
            
            cost_pred = cost_model.predict(X_test_scaled)
            cost_r2 = r2_score(y_cost_test, cost_pred)
            cost_mse = mean_squared_error(y_cost_test, cost_pred)
            
            self.models['cost_predictor'] = cost_model
            results['cost_model'] = {
                'r2_score': cost_r2,
                'mse': cost_mse,
                'feature_importance': cost_model.feature_importances_.tolist()
            }
            
            # 2. Success prediction model
            logger.info("Training success prediction model...")
            _, _, y_success_train, y_success_test = train_test_split(
                X, y['success'], test_size=0.2, random_state=42
            )
            
            success_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
            success_model.fit(X_train_scaled, y_success_train)
            
            success_pred = success_model.predict(X_test_scaled)
            success_r2 = r2_score(y_success_test, success_pred)
            success_mse = mean_squared_error(y_success_test, success_pred)
            
            self.models['success_predictor'] = success_model
            results['success_model'] = {
                'r2_score': success_r2,
                'mse': success_mse,
                'feature_importance': success_model.feature_importances_.tolist()
            }
            
            # 3. Latency prediction model
            logger.info("Training latency prediction model...")
            _, _, y_latency_train, y_latency_test = train_test_split(
                X, y['latency'], test_size=0.2, random_state=42
            )
            
            latency_model = RandomForestRegressor(n_estimators=100, random_state=42)
            latency_model.fit(X_train_scaled, y_latency_train)
            
            latency_pred = latency_model.predict(X_test_scaled)
            latency_r2 = r2_score(y_latency_test, latency_pred)
            latency_mse = mean_squared_error(y_latency_test, latency_pred)
            
            self.models['latency_predictor'] = latency_model
            results['latency_model'] = {
                'r2_score': latency_r2,
                'mse': latency_mse,
                'feature_importance': latency_model.feature_importances_.tolist()
            }
            
            # Save models
            await self._save_models()
            
            logger.info("Model training completed successfully")
            
            return {
                'status': 'success',
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'models': results,
                'training_completed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {'status': 'error', 'error': str(e)}
    
    async def _save_models(self):
        """Save trained models and encoders"""
        try:
            # Save models
            for name, model in self.models.items():
                if model is not None:
                    joblib.dump(model, f"{self.model_dir}/{name}.joblib")
            
            # Save encoders and scaler
            joblib.dump(self.encoders, f"{self.model_dir}/encoders.joblib")
            joblib.dump(self.scaler, f"{self.model_dir}/scaler.joblib")
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    async def load_models(self):
        """Load trained models from disk"""
        try:
            # Load models
            for name in self.models.keys():
                model_path = f"{self.model_dir}/{name}.joblib"
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
            
            # Load encoders and scaler
            encoders_path = f"{self.model_dir}/encoders.joblib"
            scaler_path = f"{self.model_dir}/scaler.joblib"
            
            if os.path.exists(encoders_path):
                self.encoders = joblib.load(encoders_path)
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            logger.info("Models loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    async def predict_optimal_model(self, task_type: str) -> Dict[str, Any]:
        """Predict optimal model for a task"""
        if not all(self.models.values()):
            # Try to load models
            loaded = await self.load_models()
            if not loaded:
                return {'error': 'Models not trained or loaded'}
        
        try:
            # Get available models from encoders
            if 'model' not in self.encoders or len(self.encoders['model'].classes_) == 0:
                return {'error': 'Model encoder not available'}
            
            available_models = self.encoders['model'].classes_
            
            predictions = []
            
            for model in available_models:
                # Prepare features
                try:
                    task_type_encoded = self.encoders['task_type'].transform([task_type])[0]
                    model_encoded = self.encoders['model'].transform([model])[0]
                except ValueError:
                    # Skip unknown task types or models
                    continue
                
                features = np.array([[
                    task_type_encoded,
                    model_encoded,
                    0  # No feedback initially
                ]])
                
                features_scaled = self.scaler.transform(features)
                
                # Predict
                predicted_cost = self.models['cost_predictor'].predict(features_scaled)[0]
                predicted_success = self.models['success_predictor'].predict(features_scaled)[0]
                predicted_latency = self.models['latency_predictor'].predict(features_scaled)[0]
                
                # Calculate efficiency ratio
                efficiency = predicted_success / predicted_cost if predicted_cost > 0 else predicted_success * 1000
                
                predictions.append({
                    'model': model,
                    'predicted_cost': max(0, predicted_cost),  # Ensure non-negative
                    'predicted_success': max(0, min(1, predicted_success)),  # Clamp to [0,1]
                    'predicted_latency': max(0, predicted_latency),  # Ensure non-negative
                    'efficiency_ratio': efficiency
                })
            
            # Sort by efficiency ratio
            predictions.sort(key=lambda x: x['efficiency_ratio'], reverse=True)
            
            return {
                'task_type': task_type,
                'optimal_model': predictions[0]['model'] if predictions else 'unknown',
                'predictions': predictions[:3],  # Top 3 recommendations
                'confidence': 'high' if len(predictions) >= 3 else 'medium' if len(predictions) >= 2 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error predicting optimal model: {e}")
            return {'error': str(e)}


    def _generate_mock_training_data(self):
        """Generate sufficient mock data for testing"""
        import numpy as np

        mock_data = []
        targets = {'cost': [], 'success': [], 'latency': []}

        task_types = ['test_task', 'analysis', 'generation']
        models = ['model_a', 'model_b', 'model_c']

        for i in range(60):
            mock_data.append([i % 3, i % 3, i % 10])
            targets['cost'].append(0.001 + (i * 0.0001))
            targets['success'].append(0.7 + (i * 0.003))
            targets['latency'].append(800 + (i * 5))

        self.encoders['task_type'].fit(task_types)
        self.encoders['model'].fit(models)

        return np.array(mock_data), targets

# Demo function
async def demo_ml_pipeline():
    """Demo ML training pipeline"""
    print("🤖 Agent Zero V2.0 - ML Training Pipeline Demo")
    print("=" * 50)
    
    pipeline = MLTrainingPipeline()
    
    # Train models
    print("📚 Training models...")
    result = await pipeline.train_models()
    
    if result.get('status') == 'success':
        print(f"   ✅ Training completed with {result['training_samples']} samples")
        print(f"   📊 Cost model R²: {result['models']['cost_model']['r2_score']:.3f}")
        print(f"   📊 Success model R²: {result['models']['success_model']['r2_score']:.3f}")
        
        # Test prediction
        print("\n🔮 Testing model prediction...")
        prediction = await pipeline.predict_optimal_model('text_analysis')
        if 'error' not in prediction:
            print(f"   🎯 Optimal model for text_analysis: {prediction['optimal_model']}")
            print(f"   🔍 Confidence: {prediction['confidence']}")
        else:
            print(f"   ❌ Prediction error: {prediction['error']}")
    else:
        print(f"   ❌ Training failed: {result.get('error')}")
    
    print("✅ ML pipeline demo completed")

if __name__ == "__main__":
    asyncio.run(demo_ml_pipeline())
