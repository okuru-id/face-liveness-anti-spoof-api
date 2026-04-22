from enum import Enum


class ErrorCode(str, Enum):
    INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    UNAUTHORIZED = "UNAUTHORIZED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"


class AppError(Exception):
    def __init__(self, code: ErrorCode, message: str, status_code: int):
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class InvalidImageFormatError(AppError):
    def __init__(self, message: str = "Image must be JPEG or PNG, base64 encoded"):
        super().__init__(ErrorCode.INVALID_IMAGE_FORMAT, message, 400)


class ImageTooLargeError(AppError):
    def __init__(self, message: str = "Image size exceeds 5MB limit"):
        super().__init__(ErrorCode.IMAGE_TOO_LARGE, message, 400)


class UnauthorizedError(AppError):
    def __init__(self, message: str = "Invalid or missing API Key"):
        super().__init__(ErrorCode.UNAUTHORIZED, message, 401)


class RateLimitExceededError(AppError):
    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(ErrorCode.RATE_LIMIT_EXCEEDED, message, 429)


class ModelUnavailableError(AppError):
    def __init__(self, message: str = "Model is currently unavailable"):
        super().__init__(ErrorCode.MODEL_UNAVAILABLE, message, 503)


class InternalError(AppError):
    def __init__(self, message: str = "Internal server error"):
        super().__init__(ErrorCode.INTERNAL_ERROR, message, 500)
