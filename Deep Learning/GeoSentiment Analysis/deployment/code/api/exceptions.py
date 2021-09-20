class ForbiddenInput(Exception):
    def __init__(self, code = 2, message = "Forbidden Input"):
        self.code = code
        self.message = message

class TweepyException(Exception):

    """Exception raised for errors in Tweepy module.

    Attributes:
        code -- the code error
        message -- explanation of the error
    """

    def __init__(self, code, message, status_code):
        self.code = code
        self.message = message
        self.status_code = status_code
