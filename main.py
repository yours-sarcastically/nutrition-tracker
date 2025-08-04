"""Main entry point for the Personalized Evidence-Based Nutrition Tracker."""

import logging
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.ui.main import NutritionTrackerUI
from src.config import NutritionConfig

def setup_logging():
    """Setup logging configuration."""
    config = NutritionConfig()
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('nutrition_tracker.log') if not config.DEBUG else logging.NullHandler()
        ]
    )

def main():
    """Main function to run the nutrition tracker application."""
    try:
        # Setup logging
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Validate configuration
        config = NutritionConfig()
        if not config.validate_file_paths():
            logger.error("Required data files not found. Please ensure nutrition_database_final.csv exists in the data directory.")
            sys.exit(1)
        
        logger.info("Starting Personalized Evidence-Based Nutrition Tracker")
        
        # Initialize and run the application
        app = NutritionTrackerUI()
        app.run()
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
