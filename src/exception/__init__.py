import sys
import logging

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    This function extracts traceback information from the provided `sys` module to 
    identify where the exception occurred. It formats a descriptive error message 
    containing:
        - The Python script file in which the error originated
        - The exact line number of the failure
        - The original exception message

    The formatted message is also logged using the global logger for consistent 
    error tracking across the application.

    Parameters
    ----------
    error : Exception
        The exception instance raised during execution.
    error_detail : sys
        The `sys` module, required to extract traceback information using `sys.exc_info()`.

    Returns
    -------
    str
        A formatted error message containing file name, line number, and the exception text.
    """
    
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename

    line_number = exc_tb.tb_lineno

    error_message = f"Error occured in python script: [{file_name}] at line number [{line_number}]: {str(error)}"

    logging.error(error_message)

    return error_message

class CustomException(Exception):
    """
    A standardized application-level exception that captures detailed context 
    including file name, line number, and the original error message.
    """

    def __init__(self, error_message: str, error_detail: sys):
        """
        Initialize the custom exception with detailed error context.

        This constructor accepts the original error message and the `sys` module
        to extract traceback details (file name and line number). It then delegates
        the construction of a fully formatted diagnostic message to
        `error_message_detail`, ensuring consistent and descriptive error reporting
        across the application.

        Parameters
        ----------
        error_message : str
            The raw exception message or description of the error.
        error_detail : sys
            The `sys` module used to retrieve traceback information for the error.
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        """
        Return the detailed error message for display or logging.

        Returns
        -------
        str
            A formatted string containing file name, line number, and the original
            exception message.
        """
        return self.error_message