#!/usr/bin/env python3
"""
32GB RAM Resource Monitor and Optimization
Direct implementation for Agent Zero Phase 4
"""

import psutil
import json
from datetime import datetime
from typing import Dict, Any

# Resource monitoring for 32GB RAM system
def analyze_current_memory():
    """Analyze current memory usage for 32GB system"""
    
    memory = psutil.virtual_memory()
    
    analysis = {
        "timestamp": datetime.now().isoformat(),
        "total_ram_gb": round(memory.total / (1024**3), 2),
        "available_gb": round(memory.available / (1024**3), 2),
        "used_gb": round(memory.used / (1024**3), 2),
        "free_gb": round(memory.free / (1024**3), 2),
        "percent_used": memory.percent,
        "free_for_models_gb": round((memory.available - 2*(1024**3)) / (1024**3), 2)  # Reserve 2GB safety
    }
    
    return analysis

# Model recommendations based on available memory
def get_model_recommendations(available_gb: float):
    """Get model recommendations for available memory"""
    
    models = {
        "llama3.2:1b": {"size_gb": 4, "speed": "very_fast", "accuracy": "good"},
        "llama3.2:3b": {"size_gb": 8, "speed": "fast", "accuracy": "very_good"}, 
        "codellama:7b": {"size_gb": 16, "speed": "medium", "accuracy": "excellent"},
        "mistral:7b": {"size_gb": 16, "speed": "medium", "accuracy": "excellent"}
    }
    
    recommendations = {}
    feasible_models = []
    
    for model, specs in models.items():
        can_load = available_gb >= specs["size_gb"]
        safety_margin = available_gb - specs["size_gb"]
        
        recommendations[model] = {
            **specs,
            "can_load": can_load,
            "safety_margin_gb": safety_margin,
            "status": "safe" if safety_margin > 4 else "risky" if can_load else "insufficient"
        }
        
        if can_load:
            feasible_models.append(model)
    
    # Determine strategy
    if len(feasible_models) >= 3:
        strategy = "multi_model_concurrent"
    elif len(feasible_models) >= 2:
        strategy = "dual_model_switching"  
    elif len(feasible_models) >= 1:
        strategy = "single_model_focused"
    else:
        strategy = "memory_optimization_required"
    
    return {
        "models": recommendations,
        "feasible_models": feasible_models,
        "recommended_primary": feasible_models[0] if feasible_models else None,
        "strategy": strategy
    }

# Generate optimization suggestions
def generate_optimization_suggestions(memory_analysis: Dict[str, Any], recommendations: Dict[str, Any]):
    """Generate practical optimization suggestions"""
    
    suggestions = []
    available = memory_analysis["free_for_models_gb"]
    used_percent = memory_analysis["percent_used"]
    
    print(f"\n💡 32GB RAM OPTIMIZATION SUGGESTIONS:")
    print(f"=" * 50)
    
    if used_percent > 85:
        suggestions.append("⚠️ HIGH MEMORY USAGE: Free memory before loading AI models")
        print("🚨 Current memory usage is high - consider closing other applications")
    
    if available < 8:
        suggestions.append("💡 LIMITED MEMORY: Use llama3.2:1b (4GB) or implement model switching")
        print("🔧 Recommended: Start with lightweight llama3.2:1b model")
    elif available < 16:
        suggestions.append("🎯 MODERATE MEMORY: llama3.2:3b recommended, avoid 7b models")
        print("✅ Recommended: llama3.2:3b is ideal for current memory availability")
    else:
        suggestions.append("🚀 SUFFICIENT MEMORY: Can handle larger models with switching")
        print("🎉 Good news: You can handle multiple models with proper switching")
    
    strategy = recommendations["strategy"]
    
    if strategy == "single_model_focused":
        print("🎯 STRATEGY: Single Model Focus")
        print("   • Use llama3.2:3b as primary model")
        print("   • Implement robust fallback mechanisms") 
        print("   • Focus on quality over quantity")
        
    elif strategy == "dual_model_switching":
        print("⚡ STRATEGY: Dual Model Switching")
        print("   • Primary: llama3.2:3b for standard tasks")
        print("   • Secondary: llama3.2:1b for quick tasks")
        print("   • Implement dynamic loading/unloading")
        
    elif strategy == "memory_optimization_required":
        print("🚨 STRATEGY: Memory Optimization Critical")
        print("   • Close other applications")
        print("   • Consider system cleanup")
        print("   • Start with minimal setup")
    
    return suggestions

# Main analysis execution
print("🔍 AGENT ZERO PHASE 4 - 32GB RAM ANALYSIS")
print("=" * 50)

# Get current memory status
memory_analysis = analyze_current_memory()

print(f"\n📊 CURRENT MEMORY STATUS:")
print(f"   Total RAM: {memory_analysis['total_ram_gb']}GB")
print(f"   Used: {memory_analysis['used_gb']}GB ({memory_analysis['percent_used']:.1f}%)")
print(f"   Available: {memory_analysis['available_gb']}GB")
print(f"   Free for AI models: {memory_analysis['free_for_models_gb']}GB")

# Get model recommendations
recommendations = get_model_recommendations(memory_analysis["free_for_models_gb"])

print(f"\n🤖 AI MODEL RECOMMENDATIONS:")
print(f"   Strategy: {recommendations['strategy']}")
print(f"   Primary model: {recommendations['recommended_primary']}")
print(f"   Feasible models: {len(recommendations['feasible_models'])}")

print(f"\n📋 MODEL COMPATIBILITY:")
for model, specs in recommendations["models"].items():
    status_icon = "✅" if specs["can_load"] else "❌"
    print(f"   {status_icon} {model}: {specs['size_gb']}GB - {specs['status']}")

# Generate optimization suggestions
suggestions = generate_optimization_suggestions(memory_analysis, recommendations)

print(f"\n🚀 IMMEDIATE ACTIONS FOR PHASE 4:")
print(f"   1. Continue with working llama3.2:3b model")
print(f"   2. Implement single-model reasoning first")
print(f"   3. Add memory monitoring to development")
print(f"   4. Build quality foundation before adding complexity")

print(f"\n📅 OPTIMIZED WEEK 44 GOALS:")
print(f"   • Single model AI reasoning (4 SP)")
print(f"   • Memory-optimized implementation (2 SP)")
print(f"   • Basic security audit trail (2 SP)")  
print(f"   • Testing and documentation (2 SP)")
print(f"   📊 Total: 10 SP (realistic and quality-focused)")

print(f"\n✅ ANALYSIS COMPLETE - READY FOR OPTIMIZED DEVELOPMENT!")

# Save analysis results
results = {
    "memory_analysis": memory_analysis,
    "model_recommendations": recommendations,
    "optimization_suggestions": suggestions,
    "recommended_approach": "single_model_quality_focused"
}

print(f"\n📁 Saving analysis to phase4_32gb_analysis.json...")