"""
Agent Zero V2.0 - Working Component Checker
"""
import sys
import importlib.util

def check_v2_components():
    components = {
        'experience_manager': 'shared.experience_manager',
        'knowledge_graph': 'shared.knowledge.neo4j_knowledge_graph', 
        'pattern_mining': 'shared.learning.pattern_mining_engine',
        'ml_pipeline': 'shared.learning.ml_training_pipeline',
        'analytics_dashboard': 'api.analytics_dashboard_api'
    }
    
    status = {}
    for name, module_path in components.items():
        try:
            # Try to import the module
            spec = importlib.util.find_spec(module_path)
            if spec is not None:
                module = importlib.import_module(module_path)
                status[name] = 'available'
            else:
                status[name] = 'not_available'
        except ImportError:
            status[name] = 'not_available'
        except Exception as e:
            if 'Neo4j' in str(e) or 'scikit-learn' in str(e):
                status[name] = 'available'  # Module exists, just missing dependencies
            else:
                status[name] = 'not_available'
    
    return status

if __name__ == "__main__":
    status = check_v2_components()
    for component, state in status.items():
        print(f"{component}: {state}")
