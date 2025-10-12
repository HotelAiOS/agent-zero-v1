# V2.0 API Module
try:
    from .experience_api import app as experience_app
    from .analytics_api import app as analytics_app
    APIS_AVAILABLE = True
except ImportError:
    APIS_AVAILABLE = False
