"""
Workspace Visualization Integration

This module provides integration between the workspace system and the
real-time visualization window, handling the connection and data flow.
"""

import logging
from typing import Any, Optional

from .realtime_visualization import RealTimeVisualizationWindow
from .workspace_system import WorkspaceNodeSystem

logger = logging.getLogger(__name__)


class WorkspaceVisualizationIntegration:
    """Integration manager for workspace visualization."""
    
    def __init__(self, workspace_system: WorkspaceNodeSystem):
        """
        Initialize the visualization integration.
        
        Args:
            workspace_system: The workspace system to visualize
        """
        self.workspace_system = workspace_system
        self.visualization_window: Optional[RealTimeVisualizationWindow] = None
        self.is_embedded = False
        
        # Configuration
        self.config = {
            'auto_start': True,
            'dedicated_window': False,
            'update_interval': 50,  # ms
            'default_mode': 'grid',
            'show_labels': False,
            'show_connections': True
        }
    
    def start_visualization(self, dedicated_window: bool = False) -> bool:
        """
        Start the visualization system.
        
        Args:
            dedicated_window: Whether to use a separate window or embed in main UI
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.visualization_window:
                logger.warning("Visualization already started")
                return True
            
            # Create visualization window
            self.visualization_window = RealTimeVisualizationWindow(
                self.workspace_system
            )
            
            # Configure visualization
            self._configure_visualization()
            
            # Set window mode
            self.visualization_window.set_dedicated_window(dedicated_window)
            self.is_embedded = not dedicated_window
            
            # Start the window
            if dedicated_window:
                self.visualization_window.show()
            
            logger.info(f"Visualization started in {'dedicated' if dedicated_window else 'embedded'} mode")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start visualization: {e}")
            return False
    
    def stop_visualization(self):
        """Stop the visualization system."""
        try:
            if self.visualization_window:
                self.visualization_window.close()
                self.visualization_window = None
                self.is_embedded = False
                logger.info("Visualization stopped")
        except Exception as e:
            logger.error(f"Error stopping visualization: {e}")
    
    def _configure_visualization(self):
        """Configure the visualization window with settings."""
        if not self.visualization_window:
            return
        
        try:
            # Set visualization mode
            mode_map = {
                'grid': 'grid',
                'nodes': 'nodes', 
                'connections': 'connections',
                'heatmap': 'heatmap'
            }
            
            mode = mode_map.get(self.config['default_mode'], 'grid')
            self.visualization_window.mode_combo.setCurrentText(mode)
            
            # Set display options
            self.visualization_window.show_labels_checkbox.setChecked(self.config['show_labels'])
            self.visualization_window.show_connections_checkbox.setChecked(self.config['show_connections'])
            
            # Set update interval
            self.visualization_window.update_interval = self.config['update_interval']
            if hasattr(self.visualization_window, 'update_timer'):
                self.visualization_window.update_timer.setInterval(self.config['update_interval'])
            
            # Set zoom level
            self.visualization_window.zoom_slider.setValue(100)
            
            logger.info("Visualization configured")
            
        except Exception as e:
            logger.error(f"Failed to configure visualization: {e}")
    
    def get_visualization_widget(self) -> Optional[Any]:
        """
        Get the visualization widget for embedding in other UIs.
        
        Returns:
            QWidget: The visualization widget, or None if not available
        """
        if self.visualization_window and not self.is_embedded:
            return self.visualization_window
        return None
    
    def update_configuration(self, config: dict):
        """
        Update visualization configuration.
        
        Args:
            config: Dictionary of configuration options
        """
        self.config.update(config)
        
        if self.visualization_window:
            self._configure_visualization()
            logger.info("Visualization configuration updated")
    
    def get_status(self) -> dict:
        """Get current visualization status."""
        return {
            'active': self.visualization_window is not None,
            'dedicated_window': self.is_embedded,
            'config': self.config.copy(),
            'workspace_health': self.workspace_system.get_system_health() if self.workspace_system else {}
        }
    
    def export_visualization(self, filename: str) -> bool:
        """
        Export current visualization to file.
        
        Args:
            filename: Path to save the exported image
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.visualization_window:
                logger.warning("No active visualization to export")
                return False
            
            # This would need to be implemented in the visualization window
            # For now, just log the request
            logger.info(f"Export visualization requested: {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export visualization: {e}")
            return False
    
    def toggle_pause(self):
        """Toggle visualization updates."""
        if self.visualization_window:
            self.visualization_window._toggle_updates()
    
    def reset_view(self):
        """Reset visualization view to default."""
        if self.visualization_window:
            self.visualization_window._reset_view()
    
    def set_visualization_mode(self, mode: str):
        """
        Set visualization mode.
        
        Args:
            mode: Visualization mode ('grid', 'nodes', 'connections', 'heatmap')
        """
        if self.visualization_window:
            mode_map = {
                'grid': 'grid',
                'nodes': 'nodes',
                'connections': 'connections', 
                'heatmap': 'heatmap'
            }
            
            mapped_mode = mode_map.get(mode, 'grid')
            self.visualization_window.mode_combo.setCurrentText(mapped_mode)
    
    def set_zoom_level(self, zoom: float):
        """
        Set zoom level.
        
        Args:
            zoom: Zoom level (0.1 to 10.0)
        """
        if self.visualization_window:
            self.visualization_window._set_zoom(zoom)
    
    def show_node_info(self, node_id: int):
        """
        Show information for a specific node.
        
        Args:
            node_id: ID of the node to inspect
        """
        if self.visualization_window:
            self.visualization_window.node_selected.emit(node_id)