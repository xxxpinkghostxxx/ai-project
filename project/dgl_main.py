from dgl_neural_system import DGLNeuralSystem
from vision import ThreadedScreenCapture
from utils.error_handler import ErrorHandler
from utils.config_manager import ConfigManager
from system.state_manager import StateManager
from ui.main_window import MainWindow
import logging
import contextlib
import sys
from typing import Tuple, Optional

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
    """Context manager for system resources with enhanced error handling"""
    resources = []
    try:
        yield resources
    finally:
        cleanup_errors = []
        for resource in reversed(resources):
            try:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'stop'):
                    resource.stop()
                else:
                    logger.warning(f"Resource {type(resource).__name__} has no cleanup method")
            except Exception as e:
                error_msg = f"Error cleaning up resource {type(resource).__name__}: {e}"
                logger.error(error_msg)
                cleanup_errors.append(error_msg)
        
        if cleanup_errors:
            logger.warning(f"Cleanup completed with {len(cleanup_errors)} errors")

def validate_config(config_manager: ConfigManager) -> bool:
    """Validate system configuration before initialization"""
    try:
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')
        
        # Validate sensory config
        if not isinstance(sensory_config.get('width'), int) or sensory_config['width'] <= 0:
            logger.error("Invalid sensory width configuration")
            return False
        if not isinstance(sensory_config.get('height'), int) or sensory_config['height'] <= 0:
            logger.error("Invalid sensory height configuration")
            return False
            
        # Validate workspace config
        if not isinstance(workspace_config.get('width'), int) or workspace_config['width'] <= 0:
            logger.error("Invalid workspace width configuration")
            return False
        if not isinstance(workspace_config.get('height'), int) or workspace_config['height'] <= 0:
            logger.error("Invalid workspace height configuration")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False

def initialize_system(config_manager: ConfigManager) -> Tuple[DGLNeuralSystem, ThreadedScreenCapture]:
    """Initialize the neural system and screen capture with validation"""
    try:
        # Validate configuration first
        if not validate_config(config_manager):
            raise ValueError("Invalid system configuration")
        
        # Get configurations
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')

        # Calculate dynamic nodes with bounds checking
        sensory_area = sensory_config['width'] * sensory_config['height']
        n_dynamic = sensory_area * 5
        
        # Reasonable bounds for node count
        max_nodes = 1000000  # 1M nodes max
        if n_dynamic > max_nodes:
            logger.warning(f"Dynamic node count {n_dynamic} exceeds maximum {max_nodes}, capping")
            n_dynamic = max_nodes

        logger.info(f"Initializing system: {sensory_area} sensory nodes, {n_dynamic} dynamic nodes")

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

        logger.info("System initialization completed successfully")
        return system, capture
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

def main():
    """Main entry point with comprehensive error handling"""
    try:
        logger.info("Starting neural system...")
        
        # Initialize managers
        config_manager = ConfigManager()
        state_manager = StateManager()

        with managed_resources() as resources:
            # Initialize system components
            system, capture = initialize_system(config_manager)
            resources.extend([system, capture])

            # Start connection worker with error handling
            try:
                system.start_connection_worker(batch_size=25)
                logger.info("Connection worker started")
            except Exception as e:
                logger.error(f"Failed to start connection worker: {e}")
                raise

            # Start screen capture with error handling
            try:
                capture.start()
                logger.info("Screen capture started")
            except Exception as e:
                logger.error(f"Failed to start screen capture: {e}")
                raise

            # Create main window
            try:
                main_window = MainWindow(config_manager, state_manager)
                resources.append(main_window)
                logger.info("Main window created")
            except Exception as e:
                logger.error(f"Failed to create main window: {e}")
                raise

            # Start the system
            try:
                main_window.start_system(system, capture)
                logger.info("System started, entering main loop")
                main_window.run()
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down gracefully")
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                raise

    except Exception as e:
        logger.error(f"Fatal error: {e}")
        ErrorHandler.show_error("Fatal Error", f"System failed to start: {str(e)}")
        sys.exit(1)
    finally:
        logger.info("Neural system shutdown complete")

if __name__ == '__main__':
    main()
