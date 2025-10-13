"""Neo4j Database Connector"""
class Neo4jConnector:
    def __init__(self, uri="bolt://localhost:7687"):
        self.uri = uri
        self.connected = False
        
    def connect(self):
        self.connected = True
        return True
        
    def execute_query(self, query):
        return {"result": "query executed"}
