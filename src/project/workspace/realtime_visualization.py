"""
Real-time Workspace Visualization Window

This module provides an advanced real-time visualization window for the workspace
node map with interactive features including zoom, pan, node inspection, and
real-time updates of node states and connections.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame,
                            QLabel, QPushButton, QSlider, QCheckBox, QComboBox,
                            QGroupBox, QScrollArea, QGridLayout, QGraphicsView,
                            QGraphicsScene, QGraphicsRectItem, QGraphicsEllipseItem,
                            QGraphicsLineItem, QGraphicsTextItem, QGraphicsItem,
                            QMenu, QMenuBar, QStatusBar, QSplitter, QSizePolicy)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot, QPointF, QRectF, QSize, QEvent
from PyQt6.QtGui import (QColor, QBrush, QPen, QFont, QPainter, QMouseEvent,
                        QWheelEvent, QKeyEvent, QPixmap, QTransform, QAction)

from .renderer import WorkspaceRenderer
from .config import EnergyReadingConfig
from .workspace_system import WorkspaceNodeSystem

logger = logging.getLogger(__name__)


class VisualizationMode(Enum):
    """Visualization modes for different display styles."""
    GRID_VIEW = "grid"
    NODE_VIEW = "nodes"
    CONNECTION_VIEW = "connections"
    HEATMAP_VIEW = "heatmap"


class InteractionMode(Enum):
    """Interaction modes for user input."""
    PAN = "pan"
    ZOOM = "zoom"
    INSPECT = "inspect"
    SELECT = "select"


@dataclass
class NodeInfo:
    """Information about a workspace node."""
    node_id: int
    x: int
    y: int
    energy: float
    energy_trend: str
    connections: int
    is_active: bool
    last_update: float


class EnhancedWorkspaceRenderer(WorkspaceRenderer):
    """Enhanced renderer with interactive features and advanced visualization."""
    
    def __init__(self, grid_size: Tuple[int, int] = (16, 16), pixel_size: int = 20):
        super().__init__(grid_size, pixel_size)
        
        # Enhanced visualization settings
        self.visualization_mode = VisualizationMode.GRID_VIEW
        self.interaction_mode = InteractionMode.PAN
        self.show_node_labels = False
        self.show_connections = True
        self.show_energy_trends = True
        self.zoom_level = 1.0
        self.pan_offset = QPointF(0, 0)
        
        # Interactive elements
        self.node_items = {}  # node_id -> QGraphicsEllipseItem
        self.connection_items = []  # list of QGraphicsLineItem
        self.label_items = {}  # node_id -> QGraphicsTextItem
        
        # Selection and inspection
        self.selected_node = None
        self.hovered_node = None
        
        # Additional attributes for hover effects
        self._hovered_item = None
        
        # Performance optimization
        self.max_visible_nodes = 1000
        self.node_update_threshold = 0.1  # Only update if energy changed significantly
        
    def set_visualization_mode(self, mode: VisualizationMode):
        """Set the visualization mode."""
        self.visualization_mode = mode
        self._update_visualization()
    
    def set_interaction_mode(self, mode: InteractionMode):
        """Set the interaction mode."""
        self.interaction_mode = mode
    
    def toggle_node_labels(self, show: bool):
        """Toggle node labels display."""
        self.show_node_labels = show
        self._update_labels()
    
    def toggle_connections(self, show: bool):
        """Toggle connections display."""
        self.show_connections = show
        self._update_connections()
    
    def _update_labels(self):
        """Update node labels display."""
        # Implementation for updating labels
        pass
    
    def _update_connections(self):
        """Update connections display."""
        # Implementation for updating connections
        pass
    
    def set_zoom_level(self, zoom: float):
        """Set zoom level with bounds checking."""
        self.zoom_level = max(0.1, min(10.0, zoom))
        self._apply_transform()
    
    def set_pan_offset(self, offset: QPointF):
        """Set pan offset."""
        self.pan_offset = offset
        self._apply_transform()
    
    def _update_visualization(self):
        """Update visualization based on current mode."""
        if self.visualization_mode == VisualizationMode.GRID_VIEW:
            self._render_grid_view()
        elif self.visualization_mode == VisualizationMode.NODE_VIEW:
            self._render_node_view()
        elif self.visualization_mode == VisualizationMode.CONNECTION_VIEW:
            self._render_connection_view()
        elif self.visualization_mode == VisualizationMode.HEATMAP_VIEW:
            self._render_heatmap_view()
    
    def _render_grid_view(self):
        """Render traditional grid view."""
        # Use parent implementation for basic grid
        super().render_grid(self._get_energy_data(), self._get_trend_data())
    
    def _render_node_view(self):
        """Render individual node visualization."""
        self.scene.clear()
        self.node_items.clear()
        self.connection_items.clear()
        self.label_items.clear()
        
        # Get node data from workspace system
        node_data = self._get_node_data()
        
        for node_info in node_data:
            if not node_info.is_active and self.visualization_mode != VisualizationMode.CONNECTION_VIEW:
                continue
                
            # Create node visual
            x, y = node_info.x, node_info.y
            screen_x = x * self.pixel_size * self.zoom_level + self.pan_offset.x()
            screen_y = y * self.pixel_size * self.zoom_level + self.pan_offset.y()
            
            # Create node item
            node_item = QGraphicsEllipseItem(screen_x, screen_y, 
                                          self.pixel_size * self.zoom_level,
                                          self.pixel_size * self.zoom_level)
            
            # Set color based on energy
            color = self._get_node_color(node_info.energy, node_info.energy_trend)
            node_item.setBrush(QBrush(color))
            node_item.setPen(QPen(QColor(255, 255, 255, 50)))
            node_item.setZValue(2)
            
            # Store node info in item
            node_item.setData(0, node_info.node_id)
            node_item.setAcceptHoverEvents(True)
            node_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
            
            self.scene.addItem(node_item)
            self.node_items[node_info.node_id] = node_item
            
            # Add label if enabled
            if self.show_node_labels:
                self._add_node_label(node_info, screen_x, screen_y)
    
    def _render_connection_view(self):
        """Render connection visualization."""
        self.scene.clear()
        self.node_items.clear()
        self.connection_items.clear()
        self.label_items.clear()
        
        # Get connection data
        connections = self._get_connections()
        
        for conn in connections:
            source_id, target_id = conn['source'], conn['target']
            source_info = self._get_node_info(source_id)
            target_info = self._get_node_info(target_id)
            
            if source_info and target_info:
                # Draw connection line
                line = QGraphicsLineItem(
                    source_info.x * self.pixel_size * self.zoom_level + self.pan_offset.x(),
                    source_info.y * self.pixel_size * self.zoom_level + self.pan_offset.y(),
                    target_info.x * self.pixel_size * self.zoom_level + self.pan_offset.x(),
                    target_info.y * self.pixel_size * self.zoom_level + self.pan_offset.y()
                )
                
                # Set line properties based on connection strength
                strength = conn.get('strength', 0.5)
                line.setPen(QPen(QColor(100, 200, 255, int(strength * 255)), 1))
                line.setZValue(1)
                
                self.scene.addItem(line)
                self.connection_items.append(line)
    
    def _render_heatmap_view(self):
        """Render heatmap visualization."""
        # Create heatmap based on energy distribution
        energy_data = self._get_energy_data()
        if energy_data:
            # Use parent grid rendering but with heatmap colors
            self._apply_heatmap_colors(energy_data)
    
    def _apply_heatmap_colors(self, energy_data: List[List[float]]):
        """Apply heatmap colors to energy data."""
        # Implementation for heatmap rendering
        pass
    
    def _get_node_color(self, energy: float, trend: str) -> QColor:
        """Get color for node based on energy and trend."""
        # Map energy to color (blue to red)
        normalized_energy = min(1.0, max(0.0, energy / 100.0))
        
        if trend == 'increasing':
            r = int(255 * normalized_energy)
            g = int(100 * normalized_energy)
            b = int(100 * (1 - normalized_energy))
        elif trend == 'decreasing':
            r = int(100 * (1 - normalized_energy))
            g = int(255 * normalized_energy)
            b = int(100 * normalized_energy)
        else:  # stable
            r = int(100 * normalized_energy)
            g = int(100 * normalized_energy)
            b = int(255 * normalized_energy)
        
        return QColor(r, g, b)
    
    def _add_node_label(self, node_info: NodeInfo, x: float, y: float):
        """Add label to node."""
        label = QGraphicsTextItem(f"{node_info.energy:.1f}")
        label.setPos(x + 5, y + 5)
        label.setDefaultTextColor(QColor(255, 255, 255))
        label.setFont(QFont("Arial", 8))
        label.setZValue(3)
        
        self.scene.addItem(label)
        self.label_items[node_info.node_id] = label
    
    def _apply_transform(self):
        """Apply zoom and pan transform to the scene."""
        transform = QTransform()
        transform.scale(self.zoom_level, self.zoom_level)
        self.view.setTransform(transform)
    
    def _get_energy_data(self) -> List[List[float]]:
        """Get current energy data from workspace system."""
        # This would be implemented to get data from the actual workspace system
        return [[0.0] * self.grid_size[0] for _ in range(self.grid_size[1])]
    
    def _get_trend_data(self) -> List[List[str]]:
        """Get energy trend data."""
        return [['stable'] * self.grid_size[0] for _ in range(self.grid_size[1])]
    
    def _get_node_data(self) -> List[NodeInfo]:
        """Get node information from workspace system."""
        return []
    
    def _get_connections(self) -> List[Dict]:
        """Get connection data from workspace system."""
        return []
    
    def _get_node_info(self, node_id: int) -> Optional[NodeInfo]:
        """Get information for specific node."""
        return None


class RealTimeVisualizationWindow(QMainWindow):
    """Advanced real-time visualization window for workspace node map."""
    
    # Signals for communication
    node_selected = pyqtSignal(int)
    visualization_updated = pyqtSignal()
    
    def __init__(self, workspace_system: WorkspaceNodeSystem, parent: Optional[QWidget] = None):
        """
        Initialize the real-time visualization window.
        
        Args:
            workspace_system: The workspace system to visualize
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.workspace_system = workspace_system
        self.setWindowTitle("Workspace Node Map - Real-time Visualization")
        self.setMinimumSize(800, 600)
        
        # Visualization settings
        self.renderer = EnhancedWorkspaceRenderer()
        self.is_dedicated_window = False
        self.auto_update_enabled = True
        self.update_interval = 50  # ms
        
        # Setup UI
        self._setup_ui()
        self._setup_connections()
        
        # Start update timer
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_visualization)
        self.update_timer.start(self.update_interval)
        
        # Register as observer
        if workspace_system:
            workspace_system.add_observer(self)
    
    def _setup_ui(self):
        """Setup the user interface."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Create splitter for visualization and controls
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Visualization
        viz_panel = self._create_visualization_panel()
        splitter.addWidget(viz_panel)
        
        # Right panel: Controls and information
        control_panel = self._create_control_panel()
        splitter.addWidget(control_panel)
        
        splitter.setSizes([600, 300])
        main_layout.addWidget(splitter)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.setCentralWidget(main_widget)
    
    def _create_visualization_panel(self) -> QWidget:
        """Create the visualization panel with enhanced features."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Visualization frame
        viz_frame = QFrame()
        viz_frame.setFrameShape(QFrame.Shape.StyledPanel)
        viz_frame.setStyleSheet("background-color: #121212; border-radius: 6px;")
        viz_layout = QVBoxLayout(viz_frame)
        
        # Graphics view with enhanced features
        self.view = self.renderer.get_view()
        self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        
        # Context menu for view
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_context_menu)
        
        viz_layout.addWidget(self.view)
        
        # Toolbar for visualization controls
        toolbar = self._create_visualization_toolbar()
        viz_layout.addWidget(toolbar)
        
        layout.addWidget(viz_frame)
        
        return panel
    
    def _create_control_panel(self) -> QWidget:
        """Create the control panel with settings and information."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Visualization settings group
        settings_group = QGroupBox("Visualization Settings")
        settings_layout = QGridLayout(settings_group)
        
        # Mode selection
        mode_label = QLabel("View Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([mode.value for mode in VisualizationMode])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        
        # Interaction mode
        interaction_label = QLabel("Interaction:")
        self.interaction_combo = QComboBox()
        self.interaction_combo.addItems([mode.value for mode in InteractionMode])
        self.interaction_combo.currentTextChanged.connect(self._on_interaction_changed)
        
        # Display options
        self.show_labels_checkbox = QCheckBox("Show Node Labels")
        self.show_labels_checkbox.toggled.connect(self.renderer.toggle_node_labels)
        
        self.show_connections_checkbox = QCheckBox("Show Connections")
        self.show_connections_checkbox.toggled.connect(self.renderer.toggle_connections)
        
        self.show_trends_checkbox = QCheckBox("Show Energy Trends")
        self.show_trends_checkbox.toggled.connect(self._on_trends_toggled)
        
        # Zoom controls
        zoom_label = QLabel("Zoom:")
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(10, 1000)
        self.zoom_slider.setValue(100)
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        
        self.zoom_value_label = QLabel("1.0x")
        
        # Layout settings
        settings_layout.addWidget(mode_label, 0, 0)
        settings_layout.addWidget(self.mode_combo, 0, 1)
        settings_layout.addWidget(interaction_label, 1, 0)
        settings_layout.addWidget(self.interaction_combo, 1, 1)
        settings_layout.addWidget(self.show_labels_checkbox, 2, 0, 1, 2)
        settings_layout.addWidget(self.show_connections_checkbox, 3, 0, 1, 2)
        settings_layout.addWidget(self.show_trends_checkbox, 4, 0, 1, 2)
        settings_layout.addWidget(zoom_label, 5, 0)
        settings_layout.addWidget(self.zoom_slider, 5, 1)
        settings_layout.addWidget(self.zoom_value_label, 6, 1)
        
        layout.addWidget(settings_group)
        
        # Node information panel
        info_group = QGroupBox("Selected Node Information")
        info_layout = QVBoxLayout(info_group)
        
        self.node_info_label = QLabel("No node selected")
        self.node_info_label.setWordWrap(True)
        self.node_info_label.setStyleSheet("color: #e0e0e0; font-family: 'Consolas';")
        
        info_layout.addWidget(self.node_info_label)
        layout.addWidget(info_group)
        
        # Statistics panel
        stats_group = QGroupBox("Workspace Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel()
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("color: #e0e0e0; font-family: 'Consolas';")
        
        stats_layout.addWidget(self.stats_label)
        layout.addWidget(stats_group)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.pause_button = QPushButton("Pause Updates")
        self.pause_button.clicked.connect(self._toggle_updates)
        
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self._reset_view)
        
        self.export_button = QPushButton("Export View")
        self.export_button.clicked.connect(self._export_view)
        
        button_layout.addWidget(self.pause_button)
        button_layout.addWidget(self.reset_view_button)
        button_layout.addWidget(self.export_button)
        
        layout.addLayout(button_layout)
        layout.addStretch()
        
        return panel
    
    def _create_visualization_toolbar(self) -> QWidget:
        """Create toolbar with quick controls."""
        toolbar = QWidget()
        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Quick zoom buttons
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("-")
        zoom_reset_btn = QPushButton("100%")
        
        zoom_in_btn.clicked.connect(lambda: self._set_zoom(self.renderer.zoom_level * 1.2))
        zoom_out_btn.clicked.connect(lambda: self._set_zoom(self.renderer.zoom_level / 1.2))
        zoom_reset_btn.clicked.connect(lambda: self._set_zoom(1.0))
        
        layout.addWidget(QLabel("Zoom:"))
        layout.addWidget(zoom_out_btn)
        layout.addWidget(zoom_in_btn)
        layout.addWidget(zoom_reset_btn)
        layout.addStretch()
        
        return toolbar
    
    def _setup_connections(self):
        """Setup signal connections."""
        # Connect renderer signals using event filter
        self.renderer.view.installEventFilter(self)
        
        # Connect selection signal
        self.node_selected.connect(self._on_node_selected)
    
    def eventFilter(self, obj, event):
        """Event filter for handling mouse events."""
        if obj == self.renderer.view:
            if event.type() == event.Type.MouseButtonPress:
                self._on_mouse_press(event)
            elif event.type() == event.Type.MouseMove:
                self._on_mouse_move(event)
            elif event.type() == event.Type.Wheel:
                self._on_wheel_event(event)
        return super().eventFilter(obj, event)
    
    def _on_mode_changed(self, mode_text: str):
        """Handle visualization mode change."""
        try:
            mode = VisualizationMode(mode_text)
            self.renderer.set_visualization_mode(mode)
            self.status_bar.showMessage(f"Switched to {mode_text} mode")
        except ValueError:
            logger.error(f"Invalid visualization mode: {mode_text}")
    
    def _on_interaction_changed(self, interaction_text: str):
        """Handle interaction mode change."""
        try:
            mode = InteractionMode(interaction_text)
            self.renderer.set_interaction_mode(mode)
            self.status_bar.showMessage(f"Interaction mode: {interaction_text}")
        except ValueError:
            logger.error(f"Invalid interaction mode: {interaction_text}")
    
    def _on_trends_toggled(self, checked: bool):
        """Handle energy trends toggle."""
        self.renderer.show_energy_trends = checked
        self.status_bar.showMessage(f"Energy trends {'enabled' if checked else 'disabled'}")
    
    def _on_zoom_changed(self, value: int):
        """Handle zoom slider change."""
        zoom = value / 100.0
        self._set_zoom(zoom)
    
    def _set_zoom(self, zoom: float):
        """Set zoom level."""
        self.renderer.set_zoom_level(zoom)
        self.zoom_slider.setValue(int(zoom * 100))
        self.zoom_value_label.setText(f"{zoom:.1f}x")
        self.status_bar.showMessage(f"Zoom: {zoom:.1f}x")
    
    def _toggle_updates(self):
        """Toggle automatic updates."""
        self.auto_update_enabled = not self.auto_update_enabled
        if self.auto_update_enabled:
            self.update_timer.start(self.update_interval)
            self.pause_button.setText("Pause Updates")
        else:
            self.update_timer.stop()
            self.pause_button.setText("Resume Updates")
        self.status_bar.showMessage(f"Updates {'enabled' if self.auto_update_enabled else 'disabled'}")
    
    def _reset_view(self):
        """Reset view to default zoom and pan."""
        self.renderer.set_zoom_level(1.0)
        self.renderer.set_pan_offset(QPointF(0, 0))
        self.zoom_slider.setValue(100)
        self.zoom_value_label.setText("1.0x")
        self.status_bar.showMessage("View reset to default")
    
    def _export_view(self):
        """Export current view to image."""
        try:
            # Get scene bounding rect
            rect = self.renderer.scene.itemsBoundingRect()
            image = QPixmap(rect.size().toSize())
            image.fill(Qt.GlobalColor.black)
            
            # Render scene to image
            painter = QPainter(image)
            self.renderer.scene.render(painter, QRectF(image.rect()), rect)
            painter.end()
            
            # Save image
            filename = f"workspace_visualization_{int(time.time())}.png"
            image.save(filename)
            self.status_bar.showMessage(f"View exported to {filename}")
            
        except Exception as e:
            logger.error(f"Failed to export view: {e}")
            self.status_bar.showMessage(f"Export failed: {e}")
    
    def _on_mouse_press(self, event: QMouseEvent):
        """Handle mouse press events."""
        # Call original handler
        QGraphicsView.mousePressEvent(self.renderer.view, event)
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.renderer.interaction_mode == InteractionMode.INSPECT:
                self._handle_inspection(event)
    
    def _on_mouse_move(self, event: QMouseEvent):
        """Handle mouse move events."""
        QGraphicsView.mouseMoveEvent(self.renderer.view, event)
        
        if self.renderer.interaction_mode == InteractionMode.INSPECT:
            self._handle_hover_inspection(event)
    
    def _on_wheel_event(self, event: QWheelEvent):
        """Handle wheel events for zoom."""
        if self.renderer.interaction_mode == InteractionMode.ZOOM:
            # Get current zoom
            current_zoom = self.renderer.zoom_level
            
            # Calculate new zoom based on wheel delta
            zoom_factor = 1.0015 ** event.angleDelta().y()
            new_zoom = current_zoom * zoom_factor
            
            self._set_zoom(new_zoom)
        else:
            # Default wheel behavior (scroll)
            QGraphicsView.wheelEvent(self.renderer.view, event)
    
    def _handle_inspection(self, event: QMouseEvent):
        """Handle node inspection on click."""
        pos = self.renderer.view.mapToScene(event.pos())
        
        # Find items at position
        items = self.renderer.scene.items(pos)
        
        for item in items:
            if hasattr(item, 'data') and item.data(0) is not None:
                node_id = item.data(0)
                self.node_selected.emit(node_id)
                break
    
    def _handle_hover_inspection(self, event: QMouseEvent):
        """Handle node hover inspection."""
        pos = self.renderer.view.mapToScene(event.pos())
        items = self.renderer.scene.items(pos)
        
        for item in items:
            if hasattr(item, 'data') and item.data(0) is not None:
                node_id = item.data(0)
                if node_id != self.renderer.hovered_node:
                    self.renderer.hovered_node = node_id
                    if isinstance(item, QGraphicsEllipseItem):
                        self._update_hover_effect(item, True)
                    break
        else:
            if self.renderer.hovered_node is not None:
                # Clear hover effect
                self._clear_hover_effect()
                self.renderer.hovered_node = None
    
    def _update_hover_effect(self, item: QGraphicsEllipseItem, hovered: bool):
        """Update hover effect on item."""
        if hovered:
            item.setPen(QPen(QColor(255, 255, 255), 2))
            item.setZValue(10)  # Bring to front
        else:
            item.setPen(QPen(QColor(255, 255, 255, 50)))
            item.setZValue(2)
    
    def _clear_hover_effect(self):
        """Clear hover effects from all items."""
        for item in self.renderer.scene.items():
            if hasattr(item, 'data') and item.data(0) is not None:
                if isinstance(item, QGraphicsEllipseItem):
                    item.setPen(QPen(QColor(255, 255, 255, 50)))
                    item.setZValue(2)
    
    def _on_node_selected(self, node_id: int):
        """Handle node selection."""
        # Store selected node in the window, not the renderer
        self.selected_node_id = node_id
        
        # Get node information
        node_info = self._get_node_info(node_id)
        if node_info:
            info_text = (
                f"<b>Node ID:</b> {node_info.node_id}<br>"
                f"<b>Position:</b> ({node_info.x}, {node_info.y})<br>"
                f"<b>Energy:</b> {node_info.energy:.2f}<br>"
                f"<b>Trend:</b> {node_info.energy_trend}<br>"
                f"<b>Connections:</b> {node_info.connections}<br>"
                f"<b>Status:</b> {'Active' if node_info.is_active else 'Inactive'}<br>"
                f"<b>Last Update:</b> {node_info.last_update:.2f}s ago"
            )
            self.node_info_label.setText(info_text)
            self.status_bar.showMessage(f"Selected node {node_id}")
        else:
            self.node_info_label.setText("Node information not available")
    
    def _get_node_info(self, node_id: int) -> Optional[NodeInfo]:
        """Get information for a specific node."""
        # This would be implemented to get data from the workspace system
        return None
    
    def _show_context_menu(self, pos):
        """Show context menu for visualization."""
        menu = QMenu(self)
        
        # Zoom actions
        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.triggered.connect(lambda: self._set_zoom(self.renderer.zoom_level * 1.2))
        menu.addAction(zoom_in_action)
        
        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.triggered.connect(lambda: self._set_zoom(self.renderer.zoom_level / 1.2))
        menu.addAction(zoom_out_action)
        
        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.triggered.connect(lambda: self._set_zoom(1.0))
        menu.addAction(reset_zoom_action)
        
        menu.addSeparator()
        
        # Export action
        export_action = QAction("Export View", self)
        export_action.triggered.connect(self._export_view)
        menu.addAction(export_action)
        
        menu.exec(self.view.mapToGlobal(pos))
    
    @pyqtSlot()
    def _update_visualization(self):
        """Update visualization with current workspace data."""
        if not self.auto_update_enabled:
            return
        
        try:
            # Get current workspace data
            energy_grid = self.workspace_system.get_energy_grid() if self.workspace_system else []
            energy_trends = self.workspace_system.get_energy_trends() if self.workspace_system else []
            
            if energy_grid:
                # Update renderer
                self.renderer.render_grid(energy_grid, energy_trends)
                
                # Update statistics
                self._update_statistics(energy_grid)
                
                # Emit update signal
                self.visualization_updated.emit()
                
                # Update status bar with performance info
                self.status_bar.showMessage(
                    f"Updated at {time.strftime('%H:%M:%S')} | "
                    f"Grid: {len(energy_grid)}x{len(energy_grid[0]) if energy_grid else 0}"
                )
                
        except Exception as e:
            logger.error(f"Error updating visualization: {e}")
            self.status_bar.showMessage(f"Visualization error: {str(e)}")
    
    def _update_statistics(self, energy_grid: List[List[float]]):
        """Update workspace statistics display."""
        if not energy_grid:
            return
        
        # Calculate statistics
        flat_energies = [energy for row in energy_grid for energy in row]
        if not flat_energies:
            return
        
        avg_energy = sum(flat_energies) / len(flat_energies)
        max_energy = max(flat_energies)
        min_energy = min(flat_energies)
        total_energy = sum(flat_energies)
        
        # Count active nodes (energy > threshold)
        active_threshold = 0.1
        active_nodes = sum(1 for energy in flat_energies if energy > active_threshold)
        
        # Get connection statistics if available
        conn_count = 0
        if self.workspace_system:
            conn_count = self.workspace_system.get_connection_count()
        
        stats_text = (
            f"<b>Total Energy:</b> {total_energy:.2f}<br>"
            f"<b>Average Energy:</b> {avg_energy:.2f}<br>"
            f"<b>Max Energy:</b> {max_energy:.2f}<br>"
            f"<b>Min Energy:</b> {min_energy:.2f}<br>"
            f"<b>Active Nodes:</b> {active_nodes}/{len(flat_energies)}<br>"
            f"<b>Connections:</b> {conn_count}<br>"
            f"<b>Update Rate:</b> {1000/self.update_interval:.0f} Hz"
        )
        
        self.stats_label.setText(stats_text)
    
    def set_dedicated_window(self, dedicated: bool):
        """Set whether this is a dedicated window or embedded panel."""
        self.is_dedicated_window = dedicated
        if dedicated:
            self.show()
        else:
            self.hide()
    
    def closeEvent(self, event):
        """Handle window close event."""
        # Stop updates
        self.update_timer.stop()
        
        # Remove as observer
        if self.workspace_system:
            self.workspace_system.remove_observer(self)
        
        event.accept()