class WhiteboxError(Exception):
    """Base exception for all WhiteBox API errors."""

    def __init__(self, message, status_code=None, response=None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class AuthenticationError(WhiteboxError):
    """Raised when the API key is invalid or missing."""
    pass


class RateLimitError(WhiteboxError):
    """Raised when the API rate limit is exceeded."""

    def __init__(self, message, retry_after=None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class InsufficientCreditsError(WhiteboxError):
    """Raised when the account has insufficient credits."""
    pass
