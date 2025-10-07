class Task:
    """
    Prosta reprezentacja zadania.
    Pola: id, type, description, opcjonalne metadata.
    """
    def __init__(self, id: str, type: str, description: str, metadata: dict = None):
        self.id = id
        self.type = type
        self.description = description
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Task(id={self.id!r}, type={self.type!r}, description={self.description!r})"
