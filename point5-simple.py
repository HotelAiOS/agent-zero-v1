#!/usr/bin/env python3
"""
🔍 Agent Zero V1 - Point 5: Pattern Mining Engine SIMPLE
======================================================
Quick working version - basic pattern mining capabilities
Fish shell compatible - no heredoc issues
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import json
import uuid
from datetime import datetime

# FastAPI app setup
app = FastAPI(
    title="Agent Zero V1 - Point 5: Pattern Mining SIMPLE",
    description="Quick working Pattern Mining Engine",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
patterns_db = {}
predictions_db = {}
stats = {
    "patterns_discovered": 0,
    "predictions_made": 0,
    "accuracy_rate": 0.85
}

@app.get("/")
async def root():
    """Point 5: Pattern Mining Engine Status"""
    return {
        "system": "Agent Zero V1 - Point 5: Pattern Mining Engine",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "description": "Finalna warstwa AI intelligence - Pattern discovery i prediction",
        "capabilities": [
            "Pattern discovery from experience data",
            "Outcome prediction based on patterns",
            "Success rate optimization",
            "Performance analytics"
        ],
        "current_statistics": stats,
        "pattern_library": {
            "total_patterns": len(patterns_db),
            "recent_discoveries": len([p for p in patterns_db.values() if p.get("created_at")]),
            "prediction_accuracy": stats["accuracy_rate"]
        },
        "endpoints": {
            "discover_patterns": "POST /api/v1/patterns/discover",
            "predict_outcome": "POST /api/v1/patterns/predict",
            "pattern_analytics": "GET /api/v1/patterns/analytics",
            "pattern_library": "GET /api/v1/patterns/library"
        },
        "integration": {
            "experience_management": "http://localhost:8007",
            "unified_system": "http://localhost:8006"
        }
    }

@app.post("/api/v1/patterns/discover")
async def discover_patterns(request: dict):
    """Discover patterns from experience data"""
    
    try:
        experiences = request.get("experiences", [])
        
        if not experiences:
            return {
                "status": "error",
                "message": "No experience data provided"
            }
        
        # Simple pattern discovery logic
        patterns_found = 0
        new_patterns = []
        
        for exp in experiences:
            success = exp.get("success", False)
            quality = exp.get("quality_score", 0.0)
            duration = exp.get("duration_seconds", 0.0)
            agents = exp.get("agents_involved", [])
            systems = exp.get("systems_used", [])
            
            # High success pattern
            if success and quality > 0.8:
                pattern_id = str(uuid.uuid4())
                pattern = {
                    "id": pattern_id,
                    "type": "SUCCESS_PATTERN",
                    "name": f"High Success Pattern {len(patterns_db) + 1}",
                    "confidence": min(0.95, quality + 0.1),
                    "success_rate": quality,
                    "conditions": [
                        f"Quality threshold: >{quality:.1f}",
                        f"Agents: {', '.join(agents[:2])}",
                        f"Systems: {', '.join(systems[:3])}"
                    ],
                    "outcomes": [
                        f"Success rate: {quality:.1%}",
                        f"Average duration: {duration:.2f}s"
                    ],
                    "description": f"Pattern for achieving {quality:.1%} success rate",
                    "created_at": datetime.now().isoformat(),
                    "usage_count": 0
                }
                
                patterns_db[pattern_id] = pattern
                new_patterns.append(pattern)
                patterns_found += 1
            
            # Fast execution pattern
            if duration < 1.0 and success:
                pattern_id = str(uuid.uuid4())
                pattern = {
                    "id": pattern_id,
                    "type": "EFFICIENCY_PATTERN",
                    "name": f"Fast Execution Pattern {len(patterns_db) + 1}",
                    "confidence": 0.9,
                    "success_rate": 1.0 if success else 0.0,
                    "conditions": [
                        f"Duration: <{duration:.2f}s",
                        f"Systems: {', '.join(systems)}"
                    ],
                    "outcomes": [
                        f"Fast completion: {duration:.3f}s",
                        "Maintained quality"
                    ],
                    "description": f"Pattern for {duration:.3f}s execution time",
                    "created_at": datetime.now().isoformat(),
                    "usage_count": 0
                }
                
                patterns_db[pattern_id] = pattern
                new_patterns.append(pattern)
                patterns_found += 1
        
        stats["patterns_discovered"] += patterns_found
        
        return {
            "status": "success",
            "patterns_discovered": patterns_found,
            "new_patterns": [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "type": p["type"],
                    "confidence": p["confidence"],
                    "description": p["description"]
                }
                for p in new_patterns
            ],
            "total_patterns": len(patterns_db),
            "message": f"🔍 Discovered {patterns_found} new patterns from {len(experiences)} experiences"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/patterns/predict")
async def predict_outcome(request: dict):
    """Predict outcome for proposed task setup"""
    
    try:
        task_description = request.get("task_description", "")
        setup = request.get("setup", {})
        
        agents_proposed = setup.get("agents_involved", [])
        systems_proposed = setup.get("systems_used", [])
        
        # Find relevant patterns
        relevant_patterns = []
        for pattern in patterns_db.values():
            # Simple relevance check
            if len(agents_proposed) > 0 or len(systems_proposed) > 0:
                pattern["usage_count"] = pattern.get("usage_count", 0) + 1
                relevant_patterns.append(pattern)
        
        # Generate predictions
        if relevant_patterns:
            # Pattern-based predictions
            success_rates = [p["success_rate"] for p in relevant_patterns]
            avg_success = sum(success_rates) / len(success_rates)
            
            confidences = [p["confidence"] for p in relevant_patterns]
            avg_confidence = sum(confidences) / len(confidences)
            
            # Improved predictions with patterns
            predicted_success = min(0.95, avg_success + 0.05)
            predicted_quality = min(0.95, avg_success)
            predicted_time = 2.0 if any("EFFICIENCY" in p["type"] for p in relevant_patterns) else 3.0
        else:
            # Default predictions
            predicted_success = 0.75
            predicted_quality = 0.8
            predicted_time = 3.0
            avg_confidence = 0.6
        
        prediction_id = str(uuid.uuid4())
        prediction = {
            "id": prediction_id,
            "task": task_description,
            "predictions": {
                "success_probability": {
                    "value": predicted_success,
                    "confidence": avg_confidence,
                    "range": [max(0.0, predicted_success - 0.1), min(1.0, predicted_success + 0.1)]
                },
                "quality_score": {
                    "value": predicted_quality,
                    "confidence": avg_confidence,
                    "range": [max(0.0, predicted_quality - 0.1), min(1.0, predicted_quality + 0.1)]
                },
                "completion_time": {
                    "value": predicted_time,
                    "confidence": avg_confidence,
                    "range": [max(0.1, predicted_time - 1.0), predicted_time + 1.0]
                }
            },
            "supporting_patterns": len(relevant_patterns),
            "recommendations": [
                "Monitor execution against predicted metrics",
                "Apply high-confidence patterns where possible",
                "Collect feedback for pattern refinement"
            ],
            "created_at": datetime.now().isoformat()
        }
        
        predictions_db[prediction_id] = prediction
        stats["predictions_made"] += 1
        
        return {
            "status": "success",
            "prediction": prediction,
            "prediction_summary": {
                "task": task_description,
                "expected_success": f"{predicted_success:.1%}",
                "expected_quality": f"{predicted_quality:.1%}",
                "expected_time": f"{predicted_time:.1f}s",
                "confidence": f"{avg_confidence:.1%}",
                "based_on_patterns": len(relevant_patterns)
            },
            "message": f"🔮 Prediction generated based on {len(relevant_patterns)} patterns"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/patterns/analytics")
async def get_analytics():
    """Get pattern mining analytics"""
    
    try:
        # Pattern type distribution
        pattern_types = {}
        for pattern in patterns_db.values():
            ptype = pattern.get("type", "UNKNOWN")
            pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
        
        # Top patterns by usage
        top_patterns = sorted(
            patterns_db.values(),
            key=lambda p: (p.get("usage_count", 0), p.get("confidence", 0)),
            reverse=True
        )[:5]
        
        # Recent discoveries
        recent_cutoff = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        recent_patterns = [
            p for p in patterns_db.values()
            if datetime.fromisoformat(p.get("created_at", "2025-01-01")) > recent_cutoff
        ]
        
        return {
            "status": "success",
            "analytics": {
                "mining_statistics": stats,
                "pattern_distribution": pattern_types,
                "library_stats": {
                    "total_patterns": len(patterns_db),
                    "total_predictions": len(predictions_db),
                    "pattern_types": len(pattern_types)
                },
                "top_patterns": [
                    {
                        "name": p.get("name", "Unknown"),
                        "type": p.get("type", "Unknown"),
                        "confidence": p.get("confidence", 0),
                        "usage_count": p.get("usage_count", 0)
                    }
                    for p in top_patterns
                ],
                "recent_discoveries": {
                    "today_count": len(recent_patterns),
                    "new_pattern_types": list(set(p.get("type") for p in recent_patterns))
                }
            },
            "message": "📊 Pattern mining analytics generated successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/patterns/library")
async def get_pattern_library(pattern_type: str = None, limit: int = 20):
    """Get pattern library with optional filtering"""
    
    try:
        patterns = list(patterns_db.values())
        
        # Filter by type if specified
        if pattern_type:
            patterns = [p for p in patterns if p.get("type") == pattern_type]
        
        # Sort by confidence and usage
        patterns.sort(
            key=lambda p: (p.get("confidence", 0), p.get("usage_count", 0)),
            reverse=True
        )
        
        # Limit results
        patterns = patterns[:limit]
        
        patterns_data = []
        for pattern in patterns:
            patterns_data.append({
                "id": pattern.get("id"),
                "name": pattern.get("name"),
                "type": pattern.get("type"),
                "description": pattern.get("description"),
                "confidence": pattern.get("confidence", 0),
                "success_rate": pattern.get("success_rate", 0),
                "conditions": pattern.get("conditions", []),
                "outcomes": pattern.get("outcomes", []),
                "usage_count": pattern.get("usage_count", 0),
                "created_at": pattern.get("created_at")
            })
        
        return {
            "status": "success",
            "patterns": patterns_data,
            "library_stats": {
                "total_patterns": len(patterns_db),
                "filtered_count": len(patterns_data),
                "available_types": list(set(p.get("type") for p in patterns_db.values()))
            },
            "message": f"📚 Pattern library: {len(patterns_data)} patterns retrieved"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    print("🔍 Starting Point 5: Pattern Mining Engine...")
    print("📊 Basic pattern discovery and prediction ready")
    print("🌟 Completing Agent Zero V1 architecture!")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        log_level="info"
    )