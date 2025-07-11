import logging
import sys

from pathlib import Path


def init_logger(file_name: str) -> None:
    """
    Initialize logging, creating necessary folder and file if it doesn't already exist

    Parameters
    ______________
    file_name: str
        the name of the file for logging

    """
    # Assume script is called from top-level directory
    log_dir = Path("logs")
    if not log_dir.exists():
        log_dir.mkdir(parents=True)

    # Configue handlers to print logs to file and std out
    file_handler = logging.FileHandler(filename=f"logs/{file_name}")
    stdout_handler = logging.StreamHandler(sys.stdout)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[file_handler, stdout_handler],
    )


# Todo: Calculate the conditional probabilities
# Todo: Calculate the conditional expected number of rolls
# Todo: calculate the payout for odds bets
# Todo: Calculate the payout of place bets for each
# Todo: Calculate the risk to return ratio for multiple place bets
# Calculate the risk to return ratio of a place bet , pay line, and odds bet