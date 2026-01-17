"""Main module for the PyTorch Geometric Neural System.

This module contains the main entry point and system initialization logic
for the PyG neural system application.

Key Features:
- System initialization and configuration management
- Qt application setup and lifecycle management
- Resource management with context-based cleanup
- Comprehensive error handling and logging
- Neural system and UI integration

Usage:
    python project/pyg_main.py [--log-level LEVEL]

    Where LEVEL can be: DEBUG, INFO, WARNING, ERROR, CRITICAL
"""

import sys
import os

# Add parent directory to path so we can import 'project' as a module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
import logging
import contextlib
import argparse
from typing import Any
from collections.abc import Generator
import torch

# Import Qt application components
# pylint: disable=wrong-import-position
from PyQt6.QtWidgets import QApplication  # type: ignore[import-untyped]  # pylint: disable=no-name-in-module

from project.pyg_neural_system import PyGNeuralSystem  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.vision import ThreadedScreenCapture  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.utils.error_handler import ErrorHandler  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.utils.config_manager import ConfigManager  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.system.state_manager import StateManager  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.ui.modern_main_window import ModernMainWindow  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.workspace.workspace_system import WorkspaceNodeSystem  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error
from project.workspace.config import EnergyReadingConfig  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-error

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
    """Enhanced context manager for system resources with lifecycle management.

    Provides lifecycle management and recovery mechanisms for system resources.
    """
    resources: list[Any] = []
    # Register shutdown cleanup for resource manager
    from project.utils.shutdown_utils import ShutdownDetector  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel,import-error
    ShutdownDetector.register_resource_manager_cleanup()
    try:
        yield resources
    finally:
        logger.info("Starting resource cleanup via managed_resources")
        cleanup_errors: list[str] = []

        for resource in reversed(resources):
            try:
                if hasattr(resource, 'shutdown'):
                    resource.shutdown()
                elif hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'stop'):
                    resource.stop()
                logger.debug("Successfully cleaned up resource: %s", type(resource).__name__)
            except Exception as e:  # pylint: disable=broad-exception-caught
                error_info = f"Error cleaning up resource {type(resource).__name__}: {str(e)}"
                logger.error(error_info)
                cleanup_errors.append(error_info)

        # Force cleanup of resource manager if still available
        try:
            from project.system.global_storage import GlobalStorage  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel,import-error
            resource_manager = GlobalStorage.retrieve('ui_resource_manager')
            if resource_manager and hasattr(resource_manager, 'force_cleanup'):
                resource_manager.force_cleanup()
        except Exception as e:  # pylint: disable=broad-exception-caught
            error_info = f"Error during final resource manager cleanup: {str(e)}"
            logger.warning(error_info)
            cleanup_errors.append(error_info)

        # Report cleanup summary
        if cleanup_errors:
            logger.warning(
                "Resource cleanup completed with %d errors: %s",
                len(cleanup_errors),
                ', '.join(cleanup_errors)
            )
        else:
            logger.info("Resource cleanup completed successfully")

def initialize_system(
    config_manager: ConfigManager
) -> tuple[PyGNeuralSystem, ThreadedScreenCapture, WorkspaceNodeSystem | None]:
    """Initialize the neural system, screen capture, and workspace system

    Args:
        config_manager: Configuration manager instance

    Returns:
        Tuple containing initialized neural system, screen capture, and workspace system

    Raises:
        ValueError: If required configurations are not found
        Exception: If system initialization fails
    """
    try:
        # Get configurations with proper type validation
        sensory_config = config_manager.get_config('sensory')
        workspace_config = config_manager.get_config('workspace')
        system_config = config_manager.get_config('system')

        # Check for None configs with more descriptive error messages
        if sensory_config is None:
            raise ValueError("Sensory configuration not found - check pyg_config.json")
        if workspace_config is None:
            raise ValueError("Workspace configuration not found - check pyg_config.json")
        if system_config is None:
            raise ValueError("System configuration not found - check pyg_config.json")

        # Validate configuration types
        if not isinstance(sensory_config, dict):
            raise ValueError(f"Invalid sensory config type: {type(sensory_config)}")
        if not isinstance(workspace_config, dict):
            raise ValueError(f"Invalid workspace config type: {type(workspace_config)}")
        if not isinstance(system_config, dict):
            raise ValueError(f"Invalid system config type: {type(system_config)}")

        # Calculate dynamic nodes with validation
        try:
            width = int(sensory_config['width'])  # type: ignore
            height = int(sensory_config['height'])  # type: ignore
            n_dynamic: int = width * height * 5
            if n_dynamic <= 0:
                raise ValueError("Invalid dynamic node count calculated")
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Missing or invalid sensory configuration: {e}") from e

        # Resolve device preference (cpu/cuda/auto)
        device_pref = system_config.get('device', 'auto')  # type: ignore
        cuda_available = torch.cuda.is_available()
        if isinstance(device_pref, str):
            device_pref = device_pref.lower()
        if device_pref in ('auto', None):
            device = 'cuda' if cuda_available else 'cpu'
        elif device_pref == 'cuda' and cuda_available:
            device = 'cuda'
        else:
            device = 'cpu'

        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:  # pylint: disable=broad-exception-caught
                gpu_name = "CUDA device"
            logger.info("CUDA available, using device=%s (%s)", device, gpu_name)
        else:
            logger.info("CUDA not available, using CPU")

        # Initialize neural system with error handling
        try:
            workspace_width = int(workspace_config['width'])  # type: ignore
            workspace_height = int(workspace_config['height'])  # type: ignore
            system = PyGNeuralSystem(
                width,  # type: ignore
                height,  # type: ignore
                n_dynamic,
                workspace_size=(workspace_width, workspace_height),
                device=device
            )
            logger.info("Neural system initialized on device: %s", device)
        except Exception as e:
            logger.error("Failed to initialize neural system: %s", str(e))
            raise RuntimeError(f"Neural system initialization failed: {str(e)}") from e

        # Initialize screen capture with error handling
        try:
            capture = ThreadedScreenCapture(
                width,
                height
            )
        except Exception as e:
            logger.error("Failed to initialize screen capture: %s", str(e))
            raise RuntimeError(f"Screen capture initialization failed: {str(e)}") from e

        # Initialize workspace system with error handling
        try:
            workspace_enabled = workspace_config.get('enabled', True)  # type: ignore
            if workspace_enabled:
                workspace_system = WorkspaceNodeSystem(system, EnergyReadingConfig())
                logger.info("Workspace system initialized successfully")
            else:
                workspace_system = None
                logger.info("Workspace system disabled in configuration")
        except Exception as e:
            logger.error("Failed to initialize workspace system: %s", str(e))
            raise RuntimeError(f"Workspace system initialization failed: {str(e)}") from e

        logger.info("System initialization completed successfully")
        return system, capture, workspace_system
    except Exception as e:
        logger.error("Failed to initialize system: %s", str(e))
        logger.debug("System initialization error occurred during configuration processing")
        # Enhanced error handling with recovery attempt
        error_msg = f"System initialization failed: {str(e)}"
        logger.error(error_msg)

        # Attempt recovery by validating system state
        try:
            recovery_success = _attempt_system_recovery()
            if recovery_success:
                logger.info("System recovery successful, retrying initialization...")
                return initialize_system(config_manager)  # Retry initialization
            else:
                logger.error("System recovery failed")
        except Exception as recovery_error:  # pylint: disable=broad-exception-caught
            logger.error("Recovery attempt failed: %s", str(recovery_error))

        raise RuntimeError(error_msg) from e

def _attempt_system_recovery() -> bool:
    """
    Attempt to recover the system from initialization failures.

    This function tries to clean up any partially initialized resources
    and reset the system to a clean state for retry.

    Returns:
        True if recovery was successful, False otherwise
    """
    try:
        logger.info("Attempting system recovery...")

        # Clean up global storage
        try:
            from project.system.global_storage import GlobalStorage  # type: ignore[import-untyped,import-not-found]  # pylint: disable=import-outside-toplevel,import-error
            GlobalStorage.clear()
            logger.info("Global storage cleared")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error clearing global storage: %s", str(e))

        # Force garbage collection
        import gc  # pylint: disable=import-outside-toplevel
        gc.collect()
        logger.info("Garbage collection completed")

        # Reset any cached configurations
        try:
            # Clear configuration cache by reinitializing
            _ = ConfigManager()  # Reinitialize to clear any cached state
            logger.info("Configuration cache reset")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error resetting configuration cache: %s", str(e))

        logger.info("System recovery completed successfully")
        return True

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("System recovery failed: %s", str(e))
        return False

def main() -> None:
    """Main entry point for the PyG Neural System application.

    Handles command line arguments, initializes system components,
    and starts the main application window.

    This function performs the following steps:
    1. Parses command line arguments for log level configuration
    2. Initializes the Qt application (required for UI components)
    3. Sets up configuration and state management
    4. Initializes the neural system and screen capture
    5. Creates and runs the main application window
    6. Manages resource cleanup on exit

    The application follows a structured initialization pattern with
    comprehensive error handling at each stage to ensure robustness.

    Raises:
        RuntimeError: If critical system initialization fails
        Exception: For other unexpected errors during execution

    Note:
        This function should be called as the entry point when running
        the application directly. It handles all Qt application setup
        and teardown automatically.
    """
    parser = argparse.ArgumentParser(
        description='PyG Neural System'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set the logging level'
    )
    args = parser.parse_args()

    # Set logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize Qt application first
        try:
            # Check if QApplication already exists
            app: QApplication | None = QApplication.instance()  # type: ignore[assignment]
            if app is None:
                app = QApplication(sys.argv)  # type: ignore[call-overload]
                logger.info("Qt application initialized successfully")
            else:
                logger.info("Using existing Qt application instance")
        except Exception as qt_error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize Qt application: %s", str(qt_error))
            qt_error_msg = f"Failed to initialize Qt application: {str(qt_error)}"
            ErrorHandler.show_error("Qt Error", qt_error_msg)  # type: ignore[arg-type]
            raise RuntimeError(f"Qt initialization failed: {str(qt_error)}") from qt_error

        # Initialize managers with error handling
        try:
            config_manager = ConfigManager()
            state_manager = StateManager()
            # Honor config flag for detailed logging at startup
            try:
                detailed_logging = config_manager.get_config('system', 'detailed_logging')
                if isinstance(detailed_logging, bool) and detailed_logging:
                    logging.getLogger().setLevel(logging.DEBUG)
                    logger.info("Detailed logging enabled via config")
            except Exception as log_cfg_error:  # pylint: disable=broad-exception-caught
                logger.warning("Could not apply detailed logging setting: %s", log_cfg_error)
        except Exception as init_error:  # pylint: disable=broad-exception-caught
            logger.error("Failed to initialize managers: %s", str(init_error))
            init_error_msg = f"Failed to initialize system managers: {str(init_error)}"
            ErrorHandler.show_error("Initialization Error", init_error_msg)  # type: ignore
            raise RuntimeError(f"Manager initialization failed: {str(init_error)}") from init_error

        try:
            with managed_resources() as resources:  # type: ignore[assignment]
                # Initialize system components
                try:
                    system, capture, workspace_system = initialize_system(config_manager)
                    resources.extend([system, capture])
                    if workspace_system:
                        resources.append(workspace_system)
                except Exception as sys_init_error:  # pylint: disable=broad-exception-caught
                    logger.error("System initialization failed: %s", str(sys_init_error))
                    sys_error_msg = f"Failed to initialize system components: {str(sys_init_error)}"
                    ErrorHandler.show_error("System Error", sys_error_msg)  # type: ignore[call-arg]
                    raise

                # Start connection worker with error handling
                try:
                    system.start_connection_worker(batch_size=25)
                    logger.info("Connection worker started successfully")
                except Exception as worker_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to start connection worker: %s", str(worker_error))
                    worker_error_msg = f"Failed to start connection worker: {str(worker_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "Worker Error", worker_error_msg
                    )
                    raise

                # Start screen capture with error handling
                try:
                    capture.start()
                    logger.info("Screen capture started successfully")
                except Exception as capture_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to start screen capture: %s", str(capture_error))
                    capture_error_msg = f"Failed to start screen capture: {str(capture_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "Capture Error", capture_error_msg
                    )
                    raise

                # Create main window with error handling
                try:
                    main_window = ModernMainWindow(config_manager, state_manager)
                    resources.append(main_window)
                except Exception as window_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to create main window: %s", str(window_error))
                    window_error_msg = f"Failed to create main window: {str(window_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "UI Error", window_error_msg
                    )
                    raise

                # Provide components to UI but let user explicitly start via button
                try:
                    main_window.set_components(system, capture, workspace_system)
                    main_window.show()  # Show the main window
                    logger.info("Entering Qt event loop - application will stay open")
                    app.exec()  # type: ignore[union-attr]
                    logger.info("Qt event loop exited - application closing")
                except Exception as run_error:  # pylint: disable=broad-exception-caught
                    logger.error("Failed to run main window: %s", str(run_error))
                    run_error_msg = f"Failed to run application: {str(run_error)}"
                    ErrorHandler.show_error(  # type: ignore[union-attr]
                        "Runtime Error", run_error_msg
                    )
                    raise

        except Exception as resource_error:  # pylint: disable=broad-exception-caught
            logger.error("Resource management error: %s", str(resource_error))
            resource_error_msg = f"Resource management failed: {str(resource_error)}"
            ErrorHandler.show_error(  # type: ignore[union-attr]
                "Resource Error", resource_error_msg
            )
            raise

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.error("Fatal error (%s): %s", type(e).__name__, str(e))
        fatal_error_msg = f"System failed to start ({type(e).__name__}): {str(e)}"
        ErrorHandler.show_error("Fatal Error", fatal_error_msg)  # type: ignore[union-attr]

        # Clean up Qt application if it was initialized
        try:
            qt_app: QApplication | None = QApplication.instance()  # type: ignore[assignment]
            if qt_app is not None:
                logger.info("Cleaning up Qt application")
                qt_app.quit()  # type: ignore[union-attr]
                qt_app.deleteLater()  # type: ignore[union-attr]
        except Exception as cleanup_error:  # pylint: disable=broad-exception-caught
            logger.warning("Error during Qt cleanup: %s", str(cleanup_error))

        # Add system exit for fatal errors
        sys.exit(1)

if __name__ == '__main__':
    main()
