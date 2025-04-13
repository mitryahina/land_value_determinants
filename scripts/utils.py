import logging
import functools
import sys
import pandas as pd

# Configure the logger
logger = logging.getLogger(__name__)
logger.handlers.clear()
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def log_dataframe_shape(func):
    """Decorator to log function calls and DataFrame shape before and after."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get the DataFrame from the arguments (assuming it's always the first argument)
        df = args[0] if isinstance(args[0], pd.DataFrame) else kwargs.get('df')

        # Log the function name and the shape of the DataFrame before execution
        logger.info(f"Running function: {func.__name__}")
        logger.info(f"Shape before: {df.shape}")

        # Run the function
        result = func(*args, **kwargs)

        # Log the shape of the DataFrame after execution
        df_after = result if isinstance(result, pd.DataFrame) else args[0]
        logger.info(f"Shape after: {df_after.shape}")

        return result

    return wrapper
