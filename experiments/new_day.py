from pathlib import Path
from datetime import datetime
import dad


def create_folder_structure():
    dad.logger.info("New day, have fun!")

    # Get the current date
    current_date = datetime.now()

    # Calculate the current week number (1-52)
    week_number = current_date.isocalendar()[1]

    # Get the current day name (e.g., "Monday", "Tuesday", etc.)
    day_name = current_date.strftime("%A")

    # Create the folder structure using pathlib
    folder_path = Path(__file__).parent / "experiments" / f"w{week_number:02d}" / day_name.lower()

    # Create the folders if they don't exist
    folder_path.mkdir(parents=True, exist_ok=True)

    dad.logger.info(f"Folder structure created: {folder_path}")


if __name__ == "__main__":
    create_folder_structure()
