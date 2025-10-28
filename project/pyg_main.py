import sys
import os
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from project.pyg_neural_system import PyGNeuralSystem
from project.vision import ThreadedScreenCapture
from project.utils.error_handler import ErrorHandler
from project.utils.config_manager import ConfigManager
from project.system.state_manager import StateManager
from project.ui.main_window import MainWindow
import logging
import contextlib
from typing import Tuple, Any, Generator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pyg_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def managed_resources() -> Generator[list[Any], None, None]:
    """Context manager for system resources"""
    resources: list[Any] = []
    try:
        yield resources
    finally:
        for resource in reversed(resources):
            try:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'stop'):
                    resource.stop()
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")

def initialize_system(config_manager: ConfigManager) -> Tuple[PyGNeuralSystem, ThreadedScreenCapture]:
    """Initialize the neural system and screen capture"""
    try:
        # Get configurations
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')

        # Check for None configs
        if sensory_config is None:
            raise ValueError("Sensory configuration not found")
        if workspace_config is None:
            raise ValueError("Workspace configuration not found")
        if system_config is None:
            raise ValueError("System configuration not found")

        # Calculate dynamic nodes
        n_dynamic = sensory_config['width'] * sensory_config['height'] * 5

        # Initialize neural system
        system = PyGNeuralSystem(
            sensory_config['width'],
            sensory_config['height'],
            n_dynamic,
            workspace_size=(workspace_config['width'], workspace_config['height'])
        )

        # Initialize screen capture
        capture = ThreadedScreenCapture(
            sensory_config['width'],
            sensory_config['height']
        )

        return system, capture
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def main() -> None:
    parser = argparse.ArgumentParser(description='PyG Neural System')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        default='INFO', help='Set the logging level')
    args = parser.parse_args()

    # Set logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize managers
        config_manager = ConfigManager()
        state_manager = StateManager()

        with managed_resources() as resources:  # type: ignore[assignment]
            # Initialize system components
            system, capture = initialize_system(config_manager)
            resources.extend([system, capture])

            # Start connection worker
            system.start_connection_worker(batch_size=25)

            # Start screen capture
            capture.start()

            # Create main window
            main_window = MainWindow(config_manager, state_manager)
            resources.append(main_window)

            # Start the system
            main_window.start_system(system, capture)
            main_window.run()

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        ErrorHandler.show_error("Fatal Error", f"System failed to start: {str(e)}")
        raise

if __name__ == '__main__':
    main()
