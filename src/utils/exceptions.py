"""基础异常类（不含内置名称冲突）"""


class EdgeIDSException(Exception):
    """Edge-IDS 基础异常"""
    def __init__(self, message: str = "", error_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"[{self.error_code}] {self.message} - {self.details}"
        return f"[{self.error_code}] {self.message}"


class ConfigError(EdgeIDSException):
    def __init__(self, message: str = "配置错误", details: dict = None):
        super().__init__(message, error_code=400, details=details)


class ModelError(EdgeIDSException):
    def __init__(self, message: str = "模型错误", details: dict = None):
        super().__init__(message, error_code=500, details=details)


class ModelNotFoundError(ModelError):
    def __init__(self, model_path: str):
        super().__init__(f"模型文件不存在: {model_path}", details={'model_path': model_path})


class CaptureError(EdgeIDSException):
    def __init__(self, message: str = "数据包捕获错误", details: dict = None):
        super().__init__(message, error_code=500, details=details)


class InferenceError(EdgeIDSException):
    def __init__(self, message: str = "推理错误", details: dict = None):
        super().__init__(message, error_code=500, details=details)


class FeatureError(EdgeIDSException):
    def __init__(self, message: str = "特征提取错误", details: dict = None):
        super().__init__(message, error_code=500, details=details)
