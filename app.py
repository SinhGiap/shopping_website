"""
 Flask Shopping Website
    Main Application Entry Point
"""

import os
import sys

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.app_factory import create_app
from backend.config.settings import Config

def main():
    """Main application entry point"""
    print("Starting Enhanced Shopping Website...")
    print("Initializing ML models and search engine...")
    
    # Create the Flask application
    app = create_app()
    
    print("Application ready!")
    print(f"Access the website at: http://{Config.HOST}:{Config.PORT}")
    
    # Run the application
    app.run(
        debug=Config.DEBUG, 
        port=Config.PORT, 
        host=Config.HOST
    )

if __name__ == '__main__':
    main()
