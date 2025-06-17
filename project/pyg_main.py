#!/usr/bin/env python3
"""
PyTorch Geometric Neural System Main Entry Point

This module provides the main entry point for running the PyTorch Geometric-based
neural system. It handles configuration loading, system initialization, and
provides both interactive and batch execution modes.
"""

import sys
import os
import json
import logging
import argparse
import time
import signal
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from pyg_neural_system import PyGNeuralSystem
    from vision import VisionSystem
    from config import Config
    from utils.error_handler import ErrorHandler
    from utils.config_manager import ConfigManager
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all dependencies are installed and the project is properly configured.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pyg_neural_system.log')
    ]
)

logger = logging.getLogger(__name__)

class PyGNeuralSystemRunner:
    """Main runner class for the PyG Neural System"""
    
    def __init__(self, config_path: str = None):
        """Initialize the system runner"""
        self.config_path = config_path or os.path.join(project_root.parent, 'pyg_config.json')
        self.config = None
        self.neural_system = None
        self.vision_system = None
        self.running = False
        self.error_handler = ErrorHandler()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.stop()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            logger.info(f"Loading configuration from: {self.config_path}")
            
            if not os.path.exists(self.config_path):
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r') as f:
                config_data = json.load(f)
            
            # Validate configuration
            required_sections = ['sensory', 'workspace', 'system', 'neural']
            for section in required_sections:
                if section not in config_data:
                    raise ValueError(f"Missing required configuration section: {section}")
            
            self.config = config_data
            logger.info("Configuration loaded successfully")
            return config_data
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def initialize_systems(self):
        """Initialize neural and vision systems"""
        try:
            if not self.config:
                self.load_config()
            
            logger.info("Initializing PyG Neural System...")
            self.neural_system = PyGNeuralSystem(self.config)
            
            # Initialize vision system if enabled
            if self.config.get('sensory', {}).get('enabled', False):
                logger.info("Initializing Vision System...")
                self.vision_system = VisionSystem(
                    self.config['sensory']['canvas_width'],
                    self.config['sensory']['canvas_height']
                )
                self.vision_system.start()
            
            logger.info("Systems initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing systems: {e}")
            self.error_handler.handle_error(e)
            raise
    
    def run(self, max_iterations: Optional[int] = None, interactive: bool = False):
        """Run the neural system"""
        try:
            if not self.neural_system:
                self.initialize_systems()
            
            self.running = True
            iteration = 0
            update_interval = self.config['system']['update_interval'] / 1000.0  # Convert to seconds
            
            logger.info("Starting neural system execution...")
            logger.info(f"Update interval: {update_interval:.3f}s")
            if max_iterations:
                logger.info(f"Max iterations: {max_iterations}")
            
            start_time = time.time()
            last_metrics_time = start_time
            
            while self.running:
                iteration += 1
                iteration_start = time.time()
                
                try:
                    # Update sensory input if vision system is available
                    if self.vision_system and self.vision_system.is_running():
                        frame = self.vision_system.get_current_frame()
                        if frame is not None:
                            # Resize frame to match sensory dimensions
                            import cv2
                            sensory_frame = cv2.resize(
                                frame, 
                                (self.config['sensory']['width'], self.config['sensory']['height'])
                            )
                            # Convert to grayscale and normalize
                            if len(sensory_frame.shape) == 3:
                                sensory_frame = cv2.cvtColor(sensory_frame, cv2.COLOR_BGR2GRAY)
                            sensory_frame = sensory_frame.astype(float) / 255.0
                            
                            # Update neural system with sensory input
                            self.neural_system.update_sensory_nodes(sensory_frame)
                    
                    # Update neural system
                    self.neural_system.update()
                    
                    # Log metrics periodically
                    current_time = time.time()
                    if current_time - last_metrics_time >= 10.0:  # Every 10 seconds
                        metrics = self.neural_system.get_metrics()
                        logger.info(f"Iteration {iteration}: {metrics}")
                        last_metrics_time = current_time
                    
                    # Check for max iterations
                    if max_iterations and iteration >= max_iterations:
                        logger.info(f"Reached maximum iterations ({max_iterations})")
                        break
                    
                    # Interactive mode check
                    if interactive and iteration % 100 == 0:
                        response = input(f"Iteration {iteration}. Continue? (y/n/q): ").strip().lower()
                        if response in ['n', 'q', 'quit', 'exit']:
                            break
                    
                    # Sleep to maintain update interval
                    iteration_time = time.time() - iteration_start
                    sleep_time = max(0, update_interval - iteration_time)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    logger.info("Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {e}")
                    self.error_handler.handle_error(e)
                    if not self.error_handler.should_continue():
                        break
            
            total_time = time.time() - start_time
            logger.info(f"Execution completed. Total iterations: {iteration}, Total time: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Critical error during execution: {e}")
            self.error_handler.handle_error(e)
            raise
        finally:
            self.stop()
    
    def stop(self):
        """Stop the system gracefully"""
        try:
            logger.info("Stopping systems...")
            self.running = False
            
            if self.vision_system:
                self.vision_system.stop()
                logger.info("Vision system stopped")
            
            if self.neural_system:
                self.neural_system.cleanup()
                logger.info("Neural system stopped")
            
            logger.info("All systems stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        status = {
            'running': self.running,
            'config_loaded': self.config is not None,
            'neural_system_initialized': self.neural_system is not None,
            'vision_system_initialized': self.vision_system is not None,
        }
        
        if self.neural_system:
            status['neural_metrics'] = self.neural_system.get_metrics()
        
        if self.vision_system:
            status['vision_running'] = self.vision_system.is_running()
        
        return status

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='PyTorch Geometric Neural System')
    parser.add_argument('--config', '-c', help='Configuration file path', default=None)
    parser.add_argument('--max-iterations', '-m', type=int, help='Maximum number of iterations', default=None)
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    parser.add_argument('--status', '-s', action='store_true', help='Show system status and exit')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        runner = PyGNeuralSystemRunner(args.config)
        
        if args.status:
            # Show status and exit
            runner.initialize_systems()
            status = runner.get_status()
            print(json.dumps(status, indent=2))
            return
        
        # Run the system
        runner.run(max_iterations=args.max_iterations, interactive=args.interactive)
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()