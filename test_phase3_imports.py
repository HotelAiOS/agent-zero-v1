#!/usr/bin/env python3
"""Test Phase 3 imports"""
import sys
sys.path.insert(0, '/home/ianua/projects/agent-zero-v1')

def test_imports():
    try:
        from shared.agents.code_reviewer import CodeReviewerAgent, ReviewResult
        print("✅ CodeReviewerAgent import OK")
        
        from shared.learning.experience_recorder import ExperienceRecorder, ProjectExperience
        print("✅ ExperienceRecorder import OK")
        
        from shared.collaboration.collaboration_manager import CollaborationManager, CollaborationTask
        print("✅ CollaborationManager import OK")
        
        # Test basic instantiation
        reviewer = CodeReviewerAgent()
        recorder = ExperienceRecorder()
        manager = CollaborationManager()
        
        print("✅ All Phase 3 components can be instantiated")
        print("🎉 Phase 3 stubs fully functional!")
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    return True

if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)
