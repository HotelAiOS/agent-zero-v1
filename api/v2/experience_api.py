# Mock Experience API for testing without FastAPI dependency
class MockExperienceAPI:
    def __init__(self):
        self.name = "Experience Management API V2.0"
        self.version = "2.0.0"
        self.status = "operational"
    
    def get_status(self):
        return {"status": "OK", "api": self.name}

# For compatibility
app = MockExperienceAPI()
