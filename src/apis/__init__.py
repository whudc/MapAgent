"""MapAgent API 模块"""

# 延迟导入以避免循环依赖
def __getattr__(name):
    if name == "MapAPI":
        from .map_api import MapAPI
        return MapAPI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["MapAPI"]