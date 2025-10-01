"""
Constants and shared imports for the UI engine.

This module contains all constants, shared imports, and global variables
used across the UI components.
"""

import os
import sys

from src.core.interfaces.real_time_visualization import IRealTimeVisualization
from src.core.interfaces.service_registry import IServiceRegistry
from src.ui.ui_state_manager import get_ui_state_manager
from src.utils.event_bus import get_event_bus

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize core services
ui_state = get_ui_state_manager()
event_bus = get_event_bus()

_service_registry: IServiceRegistry = None
_visualization_service: IRealTimeVisualization = None

# Define constants inline
CONSTANTS = {
    'NEURAL_SIMULATION_TITLE': 'Neural Simulation System',
    'MAIN_WINDOW_TAG': 'main_window',
    'STATUS_TEXT_TAG': 'status_text',
    'NODES_TEXT_TAG': 'nodes_text',
    'EDGES_TEXT_TAG': 'edges_text',
    'ENERGY_TEXT_TAG': 'energy_text',
    'CONNECTIONS_TEXT_TAG': 'connections_text',
    'SIMULATION_STATUS_RUNNING': 'Running',
    'SIMULATION_STATUS_STOPPED': 'Stopped',
    'LOGS_MODAL_TAG': 'logs_modal',
    'LOGS_TEXT_TAG': 'logs_text'
}
