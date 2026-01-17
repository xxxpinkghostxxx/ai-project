"""
Workspace Visualization Integration

This module integrates the workspace system with the existing
ModernMainWindow for seamless visualization, including the new
real-time visualization window.
"""

from typing import Any, List, Optional
import logging

from PyQt6.QtWidgets import QFrame, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtCore import Qt

from .renderer import WorkspaceRenderer
from .config import EnergyReadingConfig
from .realtime_visualization import RealTimeVisualizationWindow, VisualizationMode
from .visualization_integration import WorkspaceVisualizationIntegration

logger = logging.getLogger(__name__)


class WorkspaceVisualization:
    """Integrates workspace system with ModernMainWindow."""
    
    def __init__(self, main_window: Any, workspace_system: Any):
        """
        Initialize workspace visualization.
        
        Args:
            main_window: Reference to ModernMainWindow
            workspace_system: Reference to WorkspaceNodeSystem
        """
        self.main_window = main_window
        self.workspace_system = workspace_system
        self.visualization_integration: Optional[WorkspaceVisualizationIntegration] = None
        self.realtime_window: Optional[RealTimeVisualizationWindow] = None
        
        # Create workspace panel
        self._create_workspace_panel()
        
        # Connect to workspace system updates
        self.workspace_system.add_observer(self)
        
        # Initialize integration
        self._initialize_integration()
        
        logger.info("Workspace visualization initialized")
    
    def _initialize_integration(self):
        """Initialize the visualization integration system."""
        try:
            self.visualization_integration = WorkspaceVisualizationIntegration(
                self.workspace_system
            )
            logger.info("Visualization integration initialized")
        except Exception as e:
            logger.error(f"Failed to initialize visualization integration: {e}")
    
    def _create_workspace_panel(self):
        """Create dedicated workspace visualization panel."""
        # Create new panel in the left visualization area
        workspace_panel = QFrame()
        workspace_panel.setFrameShape(QFrame.Shape.StyledPanel)
        workspace_panel.setStyleSheet("background-color: #181818; border-radius: 8px;")
        
        # Layout for workspace panel
        layout = QVBoxLayout(workspace_panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        
        # Title and controls
        header_layout = QHBoxLayout()
        title_label = QLabel("Workspace Energy Grid")
        title_label.setStyleSheet("color: #e0e0e0; font-weight: bold; font-size: 14px;")
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # Control buttons
        self.realtime_button = QPushButton("Open Real-time View")
        self.realtime_button.setStyleSheet("""
            QPushButton {
                background-color: #225577;
                color: #e0e0e0;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #3377aa;
            }
        """)
        self.realtime_button.clicked.connect(self._open_realtime_view)
        header_layout.addWidget(self.realtime_button)
        
        layout.addLayout(header_layout)
        
        # Create renderer
        config = EnergyReadingConfig()
        self.renderer = WorkspaceRenderer(
            grid_size=config.grid_size,
            pixel_size=config.pixel_size
        )
        
        # Add renderer view to layout
        layout.addWidget(self.renderer.get_view(), stretch=1)
        
        # Add to main window (replace or augment existing left panel)
        # This would need to be integrated with the existing UI structure
        
        logger.info("Workspace panel created")
    
    def _open_realtime_view(self):
        """Open the real-time visualization window."""
        try:
            if self.visualization_integration:
                success = self.visualization_integration.start_visualization(
                    dedicated_window=True
                )
                if success:
                    self.realtime_window = self.visualization_integration.visualization_window
                    self.realtime_button.setText("Close Real-time View")
                    self.realtime_button.clicked.disconnect()
                    self.realtime_button.clicked.connect(self._close_realtime_view)
                    logger.info("Real-time visualization window opened")
                else:
                    logger.error("Failed to open real-time visualization window")
        except Exception as e:
            logger.error(f"Error opening real-time view: {e}")
    
    def _close_realtime_view(self):
        """Close the real-time visualization window."""
        try:
            if self.visualization_integration:
                self.visualization_integration.stop_visualization()
                self.realtime_window = None
                self.realtime_button.setText("Open Real-time View")
                self.realtime_button.clicked.disconnect()
                self.realtime_button.clicked.connect(self._open_realtime_view)
                logger.info("Real-time visualization window closed")
        except Exception as e:
            logger.error(f"Error closing real-time view: {e}")
    
    def on_workspace_update(self, energy_grid: List[List[float]]):
        """
        Handle workspace system updates.
        
        Args:
            energy_grid: 2D list of energy values from workspace system
        """
        try:
            # Calculate energy trends for visual effects
            energy_trends = self._calculate_energy_trends(energy_grid)
            
            # Render the grid in embedded view
            self.renderer.render_grid(energy_grid, energy_trends)
            
            # Update status bar
            self._update_status_bar(energy_grid)
            
            # Update real-time window if open
            if self.realtime_window:
                self.realtime_window._update_visualization()
            
        except Exception as e:
            logger.error(f"Error updating workspace visualization: {e}")
    
    def _calculate_energy_trends(self, energy_grid: List[List[float]]) -> List[List[str]]:
        """Calculate energy trends for visual effects."""
        trends = []
        
        for y in range(len(energy_grid)):
            row_trends = []
            for x in range(len(energy_grid[y])):
                # Get node data for trend calculation
                node_id = y * len(energy_grid[y]) + x
                node_data = self.workspace_system.get_node_data(node_id)
                trend = node_data.get('energy_trend', 'stable')
                row_trends.append(trend)
            trends.append(row_trends)
        
        return trends
    
    def _update_status_bar(self, energy_grid: List[List[float]]):
        """Update status bar with workspace information."""
        # Calculate statistics
        flat_energies = [energy for row in energy_grid for energy in row]
        if flat_energies:
            avg_energy = sum(flat_energies) / len(flat_energies)
            max_energy = max(flat_energies)
            min_energy = min(flat_energies)
            
            status_text = (
                f"Workspace: Avg={avg_energy:.1f}, "
                f"Max={max_energy:.1f}, Min={min_energy:.1f}"
            )
            
            # Update status bar (this would need to be connected to the main window's status bar)
            # self.main_window.status_bar.showMessage(status_text)
    
    def set_shading_mode(self, mode: str):
        """Set the shading mode for visualization."""
        self.renderer.set_shading_mode(mode)
        if self.realtime_window:
            # Map string mode to enum
            mode_map = {
                'grid': VisualizationMode.GRID_VIEW,
                'nodes': VisualizationMode.NODE_VIEW,
                'connections': VisualizationMode.CONNECTION_VIEW,
                'heatmap': VisualizationMode.HEATMAP_VIEW
            }
            enum_mode = mode_map.get(mode, VisualizationMode.GRID_VIEW)
            self.realtime_window.renderer.set_visualization_mode(enum_mode)
    
    def set_color_scheme(self, scheme: str):
        """Set the color scheme for visualization."""
        self.renderer.set_color_scheme(scheme)
        if self.realtime_window:
            # Apply to real-time window if it supports color schemes
            pass
    
    def resize_grid(self, new_grid_size: tuple[int, int], new_pixel_size: int):
        """Resize the visualization grid."""
        self.renderer.resize_grid(new_grid_size, new_pixel_size)
        if self.realtime_window:
            self.realtime_window.renderer.resize_grid(new_grid_size, new_pixel_size)
    
    def get_visualization_integration(self) -> Optional[WorkspaceVisualizationIntegration]:
        """Get the visualization integration instance."""
        return self.visualization_integration
    
    def get_realtime_window(self) -> Optional[RealTimeVisualizationWindow]:
        """Get the real-time visualization window."""
        return self.realtime_window
    
    def is_realtime_active(self) -> bool:
        """Check if real-time visualization is active."""
        return self.realtime_window is not None and self.realtime_window.isVisible()