from dgl_neural_system import DGLNeuralSystem
from vision import ThreadedScreenCapture
from utils.error_handler import ErrorHandler
from utils.config_manager import ConfigManager
from system.state_manager import StateManager
from ui.main_window import MainWindow
import logging
import contextlib
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dgl_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@contextlib.contextmanager
def managed_resources():
    """Context manager for system resources"""
    resources = []
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

def initialize_system(config_manager: ConfigManager) -> Tuple[DGLNeuralSystem, ThreadedScreenCapture]:
    """Initialize the neural system and screen capture"""
    try:
        # Get configurations
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')

        # Calculate dynamic nodes
        n_dynamic = sensory_config['width'] * sensory_config['height'] * 5

        # Initialize neural system
        system = DGLNeuralSystem(
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

def main():
    try:
        # Initialize managers
        config_manager = ConfigManager()
        state_manager = StateManager()

        with managed_resources() as resources:
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
