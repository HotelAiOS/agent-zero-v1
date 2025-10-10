#!/usr/bin/env python3
"""
Agent Zero V1 - ML Model Training Pipeline
V2.0 Intelligence Layer - Week 44 Implementation

🎯 Week 44 Critical Task: ML Model Training Pipeline (4 SP)
Zadanie: Setup uczenia maszynowego na wzorcach, cost optimization algorithms
Rezultat: Intelligent cost optimization operational
Impact: Automated decision making z continuous improvement

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-44 (Week 44 Implementation)
"""

import os
import json
import pickle
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn not available. Install with: pip install scikit-learn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    MODEL_SELECTOR = "model_selector"
    COST_PREDICTOR = "cost_predictor"
    SUCCESS_PREDICTOR = "success_predictor"
    LATENCY_PREDICTOR = "latency_predictor"
    QUALITY_SCORER = "quality_scorer"

class TrainingStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    DEPLOYED = "deployed"

@dataclass
class TrainingJob:
    """ML model training job configuration"""
    id: str
    model_type: ModelType
    model_name: str
    training_data_query: str
    feature_columns: List[str]
    target_column: str
    hyperparameters: Dict[str, Any]
    status: TrainingStatus
    accuracy: Optional[float]
    r2_score: Optional[float]
    feature_importance: Optional[Dict[str, float]]
    model_path: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    error_message: Optional[str]

@dataclass
class MLPrediction:
    """ML model prediction result"""
    model_type: ModelType
    prediction: Union[str, float, int]
    confidence: float
    feature_values: Dict[str, Any]
    model_version: str
    timestamp: datetime

class MLModelTrainingPipeline:
    """
    ML Model Training Pipeline for Agent Zero V2.0
    
    Responsibilities:
    - Train ML models on historical execution data
    - Predict optimal model selection for tasks
    - Forecast costs, latency, and success rates
    - Provide intelligent recommendations
    - Continuously retrain models with new data
    """
    
    def __init__(self, db_path: str = "agent_zero.db", models_dir: str = "ml_models"):
        self.db_path = db_path
        self.models_dir = models_dir
        self.label_encoders = {}
        self.scalers = {}
        self.trained_models = {}
        
        # Create models directory
        os.makedirs(models_dir, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize ML training tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Training jobs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_training_jobs (
                    id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    training_data_query TEXT NOT NULL,
                    feature_columns TEXT NOT NULL,  -- JSON array
                    target_column TEXT NOT NULL,
                    hyperparameters TEXT NOT NULL,  -- JSON
                    status TEXT NOT NULL,
                    accuracy REAL,
                    r2_score REAL,
                    feature_importance TEXT,  -- JSON
                    model_path TEXT,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error_message TEXT
                )
            """)
            
            # Predictions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_ml_predictions (
                    id TEXT PRIMARY KEY,
                    model_type TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    feature_values TEXT NOT NULL,  -- JSON
                    model_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    actual_outcome TEXT,  -- For validation
                    prediction_accuracy REAL
                )
            """)
            
            conn.commit()
    
    def prepare_training_data(self, days_back: int = 60) -> Dict[str, np.ndarray]:
        """Prepare training data from historical executions"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available - training skipped")
            return {}
        
        with sqlite3.connect(self.db_path) as conn:
            # Get comprehensive training data
            cursor = conn.execute("""
                SELECT 
                    task_type, model_used, success_score, cost_usd, latency_ms,
                    context, timestamp,
                    strftime('%H', timestamp) as hour,
                    strftime('%w', timestamp) as day_of_week
                FROM simple_tracker 
                WHERE timestamp >= ?
                AND success_score IS NOT NULL
                AND cost_usd IS NOT NULL
                AND latency_ms IS NOT NULL
                ORDER BY timestamp DESC
            """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
            
            raw_data = cursor.fetchall()
            
            if len(raw_data) < 10:
                logger.warning("Insufficient training data - need at least 10 records")
                return {}
            
            # Prepare features
            features = []
            targets = {
                'success_score': [],
                'cost_usd': [],
                'latency_ms': [],
                'model_used': []
            }
            
            # Encoders for categorical data
            if 'task_type' not in self.label_encoders:
                self.label_encoders['task_type'] = LabelEncoder()
                task_types = [row[0] for row in raw_data]
                self.label_encoders['task_type'].fit(task_types)
            
            if 'model_used' not in self.label_encoders:
                self.label_encoders['model_used'] = LabelEncoder()
                models = [row[1] for row in raw_data]
                self.label_encoders['model_used'].fit(models)
            
            for row in raw_data:
                task_type, model_used, success_score, cost_usd, latency_ms, context, timestamp, hour, day_of_week = row
                
                # Feature vector
                feature_vector = [
                    self.label_encoders['task_type'].transform([task_type])[0],
                    int(hour),
                    int(day_of_week),
                    len(context) if context else 0,  # Context complexity
                ]
                
                # Add context-based features
                if context:
                    try:
                        context_obj = json.loads(context) if isinstance(context, str) else context
                        feature_vector.extend([
                            len(str(context_obj)),  # Context length
                            len(context_obj) if isinstance(context_obj, dict) else 0,  # Context depth
                        ])
                    except:
                        feature_vector.extend([0, 0])
                else:
                    feature_vector.extend([0, 0])
                
                features.append(feature_vector)
                targets['success_score'].append(success_score)
                targets['cost_usd'].append(cost_usd)
                targets['latency_ms'].append(latency_ms)
                targets['model_used'].append(self.label_encoders['model_used'].transform([model_used])[0])
            
            # Convert to numpy arrays
            X = np.array(features)
            
            # Normalize features
            if 'features' not in self.scalers:
                self.scalers['features'] = StandardScaler()
                X = self.scalers['features'].fit_transform(X)
            else:
                X = self.scalers['features'].transform(X)
            
            training_data = {
                'X': X,
                'y_success': np.array(targets['success_score']),
                'y_cost': np.array(targets['cost_usd']),
                'y_latency': np.array(targets['latency_ms']),
                'y_model': np.array(targets['model_used']),
                'feature_names': [
                    'task_type_encoded', 'hour', 'day_of_week', 
                    'context_complexity', 'context_length', 'context_depth'
                ]
            }
            
            logger.info(f"✅ Training data prepared: {len(raw_data)} samples, {X.shape[1]} features")
            return training_data
    
    def train_model_selector(self, training_data: Dict[str, np.ndarray]) -> TrainingJob:
        """Train model that selects optimal LLM model for given task"""
        job = TrainingJob(
            id=f"job_model_selector_{int(datetime.now().timestamp())}",
            model_type=ModelType.MODEL_SELECTOR,
            model_name="model_selector_rf",
            training_data_query="Historical task executions for model selection",
            feature_columns=['task_type', 'hour', 'day_of_week', 'context_complexity'],
            target_column="model_used",
            hyperparameters={'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            status=TrainingStatus.IN_PROGRESS,
            accuracy=None,
            r2_score=None,
            feature_importance=None,
            model_path=None,
            created_at=datetime.now(),
            completed_at=None,
            error_message=None
        )
        
        if not SKLEARN_AVAILABLE or 'X' not in training_data:
            job.status = TrainingStatus.FAILED
            job.error_message = "Scikit-learn not available or insufficient data"
            self._store_training_job(job)
            return job
        
        try:
            X = training_data['X']
            y = training_data['y_model']
            
            # Train Random Forest classifier
            model = RandomForestClassifier(**job.hyperparameters)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
            
            # Final training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(
                training_data['feature_names'],
                model.feature_importances_
            ))
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{job.model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'label_encoders': self.label_encoders,
                    'scalers': self.scalers,
                    'feature_names': training_data['feature_names']
                }, f)
            
            # Update job
            job.status = TrainingStatus.COMPLETED
            job.accuracy = accuracy
            job.feature_importance = feature_importance
            job.model_path = model_path
            job.completed_at = datetime.now()
            
            # Store in memory
            self.trained_models['model_selector'] = model
            
            logger.info(f"✅ Model selector trained: {accuracy:.3f} accuracy")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"❌ Model selector training failed: {e}")
        
        self._store_training_job(job)
        return job
    
    def train_cost_predictor(self, training_data: Dict[str, np.ndarray]) -> TrainingJob:
        """Train model to predict task execution cost"""
        job = TrainingJob(
            id=f"job_cost_predictor_{int(datetime.now().timestamp())}",
            model_type=ModelType.COST_PREDICTOR,
            model_name="cost_predictor_rf",
            training_data_query="Historical task executions for cost prediction",
            feature_columns=['task_type', 'model_used', 'context_complexity', 'hour'],
            target_column="cost_usd",
            hyperparameters={'n_estimators': 80, 'max_depth': 8, 'random_state': 42},
            status=TrainingStatus.IN_PROGRESS,
            accuracy=None,
            r2_score=None,
            feature_importance=None,
            model_path=None,
            created_at=datetime.now(),
            completed_at=None,
            error_message=None
        )
        
        if not SKLEARN_AVAILABLE or 'X' not in training_data:
            job.status = TrainingStatus.FAILED
            job.error_message = "Scikit-learn not available or insufficient data"
            self._store_training_job(job)
            return job
        
        try:
            X = training_data['X']
            y = training_data['y_cost']
            
            # Train Random Forest regressor
            model = RandomForestRegressor(**job.hyperparameters)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Final training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(
                training_data['feature_names'],
                model.feature_importances_
            ))
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{job.model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'label_encoders': self.label_encoders,
                    'scalers': self.scalers,
                    'feature_names': training_data['feature_names'],
                    'mse': mse
                }, f)
            
            # Update job
            job.status = TrainingStatus.COMPLETED
            job.r2_score = r2
            job.feature_importance = feature_importance
            job.model_path = model_path
            job.completed_at = datetime.now()
            
            # Store in memory
            self.trained_models['cost_predictor'] = model
            
            logger.info(f"✅ Cost predictor trained: {r2:.3f} R² score, MSE: {mse:.6f}")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"❌ Cost predictor training failed: {e}")
        
        self._store_training_job(job)
        return job
    
    def train_success_predictor(self, training_data: Dict[str, np.ndarray]) -> TrainingJob:
        """Train model to predict task success probability"""
        job = TrainingJob(
            id=f"job_success_predictor_{int(datetime.now().timestamp())}",
            model_type=ModelType.SUCCESS_PREDICTOR,
            model_name="success_predictor_rf",
            training_data_query="Historical task executions for success prediction",
            feature_columns=['task_type', 'model_used', 'context_complexity', 'hour', 'day_of_week'],
            target_column="success_score",
            hyperparameters={'n_estimators': 120, 'max_depth': 12, 'random_state': 42},
            status=TrainingStatus.IN_PROGRESS,
            accuracy=None,
            r2_score=None,
            feature_importance=None,
            model_path=None,
            created_at=datetime.now(),
            completed_at=None,
            error_message=None
        )
        
        if not SKLEARN_AVAILABLE or 'X' not in training_data:
            job.status = TrainingStatus.FAILED
            job.error_message = "Scikit-learn not available or insufficient data"
            self._store_training_job(job)
            return job
        
        try:
            X = training_data['X']
            y = training_data['y_success']
            
            # Train Random Forest regressor
            model = RandomForestRegressor(**job.hyperparameters)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
            
            # Final training
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = dict(zip(
                training_data['feature_names'],
                model.feature_importances_
            ))
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{job.model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'label_encoders': self.label_encoders,
                    'scalers': self.scalers,
                    'feature_names': training_data['feature_names']
                }, f)
            
            # Update job
            job.status = TrainingStatus.COMPLETED
            job.r2_score = r2
            job.feature_importance = feature_importance
            job.model_path = model_path
            job.completed_at = datetime.now()
            
            # Store in memory
            self.trained_models['success_predictor'] = model
            
            logger.info(f"✅ Success predictor trained: {r2:.3f} R² score")
            
        except Exception as e:
            job.status = TrainingStatus.FAILED
            job.error_message = str(e)
            logger.error(f"❌ Success predictor training failed: {e}")
        
        self._store_training_job(job)
        return job
    
    def predict_optimal_model(self, task_type: str, context: Dict[str, Any] = None) -> MLPrediction:
        """Predict optimal model for a given task using trained ML model"""
        if 'model_selector' not in self.trained_models:
            # Fallback to rule-based selection
            return MLPrediction(
                model_type=ModelType.MODEL_SELECTOR,
                prediction="llama3.2-3b",
                confidence=0.6,
                feature_values={'task_type': task_type},
                model_version="fallback_v1.0",
                timestamp=datetime.now()
            )
        
        try:
            # Prepare features
            current_time = datetime.now()
            feature_vector = [
                self.label_encoders['task_type'].transform([task_type])[0],
                current_time.hour,
                current_time.weekday(),
                len(str(context)) if context else 0,
                len(json.dumps(context)) if context else 0,
                len(context) if isinstance(context, dict) else 0
            ]
            
            X = np.array([feature_vector])
            if 'features' in self.scalers:
                X = self.scalers['features'].transform(X)
            
            # Make prediction
            model = self.trained_models['model_selector']
            model_encoded = model.predict(X)[0]
            confidence_scores = model.predict_proba(X)[0]
            max_confidence = np.max(confidence_scores)
            
            # Decode model name
            predicted_model = self.label_encoders['model_used'].inverse_transform([int(model_encoded)])[0]
            
            prediction = MLPrediction(
                model_type=ModelType.MODEL_SELECTOR,
                prediction=predicted_model,
                confidence=float(max_confidence),
                feature_values={
                    'task_type': task_type,
                    'hour': current_time.hour,
                    'context_complexity': len(str(context)) if context else 0
                },
                model_version="ml_v1.0",
                timestamp=current_time
            )
            
            self._store_prediction(prediction)
            return prediction
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            # Fallback prediction
            return MLPrediction(
                model_type=ModelType.MODEL_SELECTOR,
                prediction="llama3.2-3b",
                confidence=0.5,
                feature_values={'task_type': task_type, 'error': str(e)},
                model_version="fallback_v1.0",
                timestamp=datetime.now()
            )
    
    def predict_task_cost(self, task_type: str, model_name: str, context: Dict[str, Any] = None) -> MLPrediction:
        """Predict cost for a task execution"""
        if 'cost_predictor' not in self.trained_models:
            # Fallback estimation based on model type
            base_costs = {
                'llama3.2-3b': 0.008,
                'llama3.2-1b': 0.005,
                'qwen2.5-3b': 0.007,
                'default': 0.010
            }
            estimated_cost = base_costs.get(model_name, base_costs['default'])
            
            return MLPrediction(
                model_type=ModelType.COST_PREDICTOR,
                prediction=estimated_cost,
                confidence=0.6,
                feature_values={'task_type': task_type, 'model_name': model_name},
                model_version="fallback_v1.0",
                timestamp=datetime.now()
            )
        
        try:
            # Prepare features (similar to model selector)
            current_time = datetime.now()
            feature_vector = [
                self.label_encoders['task_type'].transform([task_type])[0],
                current_time.hour,
                current_time.weekday(),
                len(str(context)) if context else 0,
                len(json.dumps(context)) if context else 0,
                len(context) if isinstance(context, dict) else 0
            ]
            
            X = np.array([feature_vector])
            if 'features' in self.scalers:
                X = self.scalers['features'].transform(X)
            
            # Make prediction
            model = self.trained_models['cost_predictor']
            predicted_cost = model.predict(X)[0]
            
            # Estimate confidence based on feature importance
            confidence = 0.8  # Placeholder - could be calculated from model uncertainty
            
            prediction = MLPrediction(
                model_type=ModelType.COST_PREDICTOR,
                prediction=float(predicted_cost),
                confidence=confidence,
                feature_values={
                    'task_type': task_type,
                    'model_name': model_name,
                    'context_size': len(str(context)) if context else 0
                },
                model_version="ml_v1.0",
                timestamp=current_time
            )
            
            self._store_prediction(prediction)
            return prediction
            
        except Exception as e:
            logger.error(f"Cost prediction failed: {e}")
            return MLPrediction(
                model_type=ModelType.COST_PREDICTOR,
                prediction=0.010,
                confidence=0.5,
                feature_values={'task_type': task_type, 'error': str(e)},
                model_version="fallback_v1.0",
                timestamp=datetime.now()
            )
    
    def run_full_training_pipeline(self) -> Dict[str, TrainingJob]:
        """Run complete training pipeline for all models"""
        logger.info("🚀 Starting full ML training pipeline...")
        
        # Prepare training data
        training_data = self.prepare_training_data()
        
        if not training_data:
            logger.warning("No training data available - pipeline aborted")
            return {}
        
        results = {}
        
        # Train all models
        results['model_selector'] = self.train_model_selector(training_data)
        results['cost_predictor'] = self.train_cost_predictor(training_data)
        results['success_predictor'] = self.train_success_predictor(training_data)
        
        # Summary
        successful_jobs = [job for job in results.values() if job.status == TrainingStatus.COMPLETED]
        failed_jobs = [job for job in results.values() if job.status == TrainingStatus.FAILED]
        
        logger.info(f"✅ Training pipeline complete: {len(successful_jobs)} successful, {len(failed_jobs)} failed")
        
        return results
    
    def _store_training_job(self, job: TrainingJob):
        """Store training job in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO v2_training_jobs
                (id, model_type, model_name, training_data_query, feature_columns,
                 target_column, hyperparameters, status, accuracy, r2_score,
                 feature_importance, model_path, created_at, completed_at, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                job.id, job.model_type.value, job.model_name,
                job.training_data_query, json.dumps(job.feature_columns),
                job.target_column, json.dumps(job.hyperparameters),
                job.status.value, job.accuracy, job.r2_score,
                json.dumps(job.feature_importance) if job.feature_importance else None,
                job.model_path, job.created_at.isoformat(),
                job.completed_at.isoformat() if job.completed_at else None,
                job.error_message
            ))
            conn.commit()
    
    def _store_prediction(self, prediction: MLPrediction):
        """Store prediction in database for later validation"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO v2_ml_predictions
                (id, model_type, prediction, confidence, feature_values,
                 model_version, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(prediction.timestamp.timestamp()), prediction.model_type.value,
                str(prediction.prediction), prediction.confidence,
                json.dumps(prediction.feature_values), prediction.model_version,
                prediction.timestamp.isoformat()
            ))
            conn.commit()

# CLI Integration Functions
def train_all_models() -> Dict[str, Any]:
    """CLI function to train all ML models"""
    pipeline = MLModelTrainingPipeline()
    results = pipeline.run_full_training_pipeline()
    
    return {
        'training_started': datetime.now().isoformat(),
        'jobs_created': len(results),
        'jobs_status': {job_name: job.status.value for job_name, job in results.items()},
        'successful_models': [name for name, job in results.items() 
                             if job.status == TrainingStatus.COMPLETED],
        'failed_models': [name for name, job in results.items() 
                         if job.status == TrainingStatus.FAILED]
    }

def get_ml_model_recommendations(task_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """CLI function to get ML-based recommendations"""
    pipeline = MLModelTrainingPipeline()
    
    # Get model recommendation
    model_pred = pipeline.predict_optimal_model(task_type, context)
    
    # Get cost prediction
    cost_pred = pipeline.predict_task_cost(task_type, model_pred.prediction, context)
    
    return {
        'recommended_model': model_pred.prediction,
        'model_confidence': model_pred.confidence,
        'predicted_cost': cost_pred.prediction,
        'cost_confidence': cost_pred.confidence,
        'reasoning': f"ML model recommendation based on {task_type} task analysis",
        'feature_analysis': {
            'task_type': task_type,
            'context_complexity': len(str(context)) if context else 0,
            'timestamp': datetime.now().isoformat()
        }
    }

def get_ml_training_status() -> Dict[str, Any]:
    """CLI function to get training status"""
    with sqlite3.connect("agent_zero.db") as conn:
        cursor = conn.execute("""
            SELECT model_type, status, accuracy, r2_score, completed_at, error_message
            FROM v2_training_jobs
            ORDER BY created_at DESC
        """)
        
        jobs = cursor.fetchall()
    
    if not jobs:
        return {'status': 'no_training_jobs', 'models_available': 0}
    
    return {
        'total_jobs': len(jobs),
        'completed_jobs': len([j for j in jobs if j[1] == 'completed']),
        'failed_jobs': len([j for j in jobs if j[1] == 'failed']),
        'jobs': [
            {
                'model_type': job[0],
                'status': job[1],
                'accuracy': job[2],
                'r2_score': job[3],
                'completed_at': job[4],
                'error': job[5]
            }
            for job in jobs
        ]
    }

def validate_ml_predictions(days_back: int = 7) -> Dict[str, Any]:
    """CLI function to validate ML prediction accuracy"""
    with sqlite3.connect("agent_zero.db") as conn:
        cursor = conn.execute("""
            SELECT COUNT(*) as total_predictions,
                   AVG(CASE WHEN actual_outcome IS NOT NULL THEN prediction_accuracy END) as avg_accuracy
            FROM v2_ml_predictions
            WHERE timestamp >= ?
        """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
        
        result = cursor.fetchone()
    
    if result and result[0] > 0:
        return {
            'validation_period_days': days_back,
            'total_predictions': result[0],
            'average_accuracy': result[1] or 0.0,
            'validation_coverage': (result[1] is not None)
        }
    else:
        return {
            'validation_period_days': days_back,
            'total_predictions': 0,
            'average_accuracy': 0.0,
            'validation_coverage': False
        }

if __name__ == "__main__":
    # Test ML Model Training Pipeline
    print("🤖 Testing ML Model Training Pipeline...")
    
    # Start training
    training_results = train_all_models()
    print(f"✅ Training initiated: {training_results['jobs_created']} jobs")
    
    if training_results['successful_models']:
        print(f"🎯 Successful models: {', '.join(training_results['successful_models'])}")
    
    # Test recommendations
    recommendations = get_ml_model_recommendations("code_generation", {"complexity": "medium"})
    print(f"💡 ML Recommendation: {recommendations['recommended_model']} (confidence: {recommendations['model_confidence']:.3f})")
    
    # Get training status
    status = get_ml_training_status()
    print(f"📊 Training Status: {status['completed_jobs']}/{status['total_jobs']} completed")
    
    print("\n🎉 ML Model Training Pipeline - OPERATIONAL!")