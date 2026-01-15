import logging
import sys

def setup_logging():
    """Configures logging for the entire project to log to both a file and Jupyter Notebook."""
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,  # Use logging.INFO or higher for production
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=[
                logging.FileHandler("capybara.log", mode="w"),  # Save logs to a file
                logging.StreamHandler(sys.stdout)  # Display logs in Jupyter Notebook
            ]
        )
    # Set Matplotlib's logging level to WARNING to suppress debug messages
    logging.getLogger("matplotlib").setLevel(logging.WARNING)