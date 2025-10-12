# Mock Analytics API for testing without FastAPI dependency
class MockAnalyticsAPI:
    def __init__(self):
        self.name = "Analytics Dashboard API V2.0" 
        self.version = "2.0.0"
        self.status = "operational"
    
    def get_dashboard(self):
        return {"status": "OK", "dashboard": "available"}

# For compatibility
app = MockAnalyticsAPI()
