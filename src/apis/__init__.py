"""MapAgent API module"""

# Re-export
def __getattr__(name):
    if name == "MapAPI":
        from .map_api import MapAPI
        return MapAPI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MapAPI"]
