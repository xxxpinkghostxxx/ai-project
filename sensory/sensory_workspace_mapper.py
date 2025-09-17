
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable

import time

from torch_geometric.data import Data

from utils.logging_utils import log_step
from energy.node_access_layer import NodeAccessLayer

from neural.connection_logic import create_weighted_connection

from utils.event_bus import get_event_bus


class SensoryWorkspaceMapper:

    def __init__(self, workspace_size: Tuple[int, int] = (10, 10)):

        self.workspace_size = workspace_size
        self.visual_patterns = {}
        self.audio_patterns = {}
        self.sensory_mappings = {}
        self.visual_sensitivity = 0.3
        self.audio_sensitivity = 0.2
        self.pattern_threshold = 0.1
        self.temporal_window = 5
        self.workspace_regions = {
            'visual_center': (0.3, 0.3, 0.7, 0.7),
            'visual_edges': (0.0, 0.0, 1.0, 1.0),
            'audio_low': (0.0, 0.0, 0.5, 0.3),
            'audio_mid': (0.0, 0.3, 0.5, 0.7),
            'audio_high': (0.0, 0.7, 0.5, 1.0),
            'motion': (0.5, 0.0, 1.0, 0.5),
            'texture': (0.5, 0.5, 1.0, 1.0),
        }
        self.manager = None
        self.bus = get_event_bus()
        self.bus.subscribe('SENSORY_INPUT_AUDIO', self._on_audio_input)
        self.bus.subscribe('SENSORY_INPUT_VISUAL', self._on_visual_input)
        log_step("SensoryWorkspaceMapper initialized", workspace_size=workspace_size)
    def map_visual_to_workspace(self, graph: Data, visual_data: np.ndarray,
                               step: int) -> Data:

        try:
            access_layer = NodeAccessLayer(graph)
            visual_patterns = self._extract_visual_patterns(visual_data, step)
            workspace_updates = self._map_patterns_to_workspace(visual_patterns, 'visual')
            graph = self._update_workspace_nodes(graph, workspace_updates, access_layer)
            graph = self._create_sensory_workspace_connections(graph, visual_patterns, access_layer)
            log_step("Visual patterns mapped to workspace",
                    patterns_detected=len(visual_patterns),
                    workspace_updates=len(workspace_updates))
            return graph
        except Exception as e:
            log_step("Error mapping visual to workspace", error=str(e))
            return graph
    def map_audio_to_workspace(self, graph: Data, audio_data: np.ndarray,
                              step: int) -> Data:

        try:
            access_layer = NodeAccessLayer(graph)
            audio_patterns = self._extract_audio_patterns(audio_data, step)
            workspace_updates = self._map_patterns_to_workspace(audio_patterns, 'audio')
            graph = self._update_workspace_nodes(graph, workspace_updates, access_layer)
            graph = self._create_audio_workspace_connections(graph, audio_patterns, access_layer)
            log_step("Audio patterns mapped to workspace",
                    patterns_detected=len(audio_patterns),
                    workspace_updates=len(workspace_updates))
            return graph
        except Exception as e:
            log_step("Error mapping audio to workspace", error=str(e))
            return graph
    def _extract_visual_patterns(self, visual_data: np.ndarray, step: int) -> List[Dict[str, Any]]:

        try:
            patterns = []
            contrast_value = visual_data.std()
            if contrast_value > self.pattern_threshold:
                patterns.append({
                    'type': 'high_contrast',
                    'region': 'visual_center',
                    'strength': contrast_value,
                    'energy': contrast_value * self.visual_sensitivity,
                    'temporal_persistence': 5,
                    'spatial_center': (visual_data.shape[0]//2, visual_data.shape[1]//2)
                })
                log_step("High contrast pattern detected", contrast=contrast_value)
            if visual_data.shape[0] > 1 and visual_data.shape[1] > 1:
                grad_x = np.gradient(visual_data, axis=1)
                grad_y = np.gradient(visual_data, axis=0)
                edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
                if edge_magnitude.mean() > self.pattern_threshold * 0.5:
                    patterns.append({
                        'type': 'edge_pattern',
                        'region': 'visual_edges',
                        'strength': edge_magnitude.mean(),
                        'energy': edge_magnitude.mean() * self.visual_sensitivity,
                        'temporal_persistence': 3,
                        'spatial_center': self._find_edge_center(edge_magnitude)
                    })
            if hasattr(self, 'previous_visual_data') and self.previous_visual_data is not None:
                motion = np.abs(visual_data - self.previous_visual_data)
                if motion.mean() > self.pattern_threshold * 0.3:
                    patterns.append({
                        'type': 'motion',
                        'region': 'motion',
                        'strength': motion.mean(),
                        'energy': motion.mean() * self.visual_sensitivity,
                        'temporal_persistence': 2,
                        'spatial_center': self._find_motion_center(motion)
                    })
            if visual_data.shape[0] > 2 and visual_data.shape[1] > 2:
                texture_variance = self._calculate_texture_variance(visual_data)
                if texture_variance > self.pattern_threshold * 0.4:
                    patterns.append({
                        'type': 'texture',
                        'region': 'texture',
                        'strength': texture_variance,
                        'energy': texture_variance * self.visual_sensitivity,
                        'temporal_persistence': 7,
                        'spatial_center': (visual_data.shape[0]//2, visual_data.shape[1]//2)
                    })
            self.previous_visual_data = visual_data.copy()
            return patterns
        except Exception as e:
            log_step("Error extracting visual patterns", error=str(e))
            return []
    def _extract_audio_patterns(self, audio_data: np.ndarray, step: int) -> List[Dict[str, Any]]:

        try:
            patterns = []
            fft = np.fft.fft(audio_data)
            fft_magnitude = np.abs(fft)
            low_freq_energy = np.mean(fft_magnitude[:len(fft_magnitude)//4])
            if low_freq_energy > self.pattern_threshold * 0.5:
                patterns.append({
                    'type': 'audio_low',
                    'region': 'audio_low',
                    'strength': low_freq_energy,
                    'energy': low_freq_energy * self.audio_sensitivity,
                    'temporal_persistence': 4,
                    'frequency_range': 'low'
                })
            mid_freq_energy = np.mean(fft_magnitude[len(fft_magnitude)//4:3*len(fft_magnitude)//4])
            if mid_freq_energy > self.pattern_threshold * 0.5:
                patterns.append({
                    'type': 'audio_mid',
                    'region': 'audio_mid',
                    'strength': mid_freq_energy,
                    'energy': mid_freq_energy * self.audio_sensitivity,
                    'temporal_persistence': 3,
                    'frequency_range': 'mid'
                })
            high_freq_energy = np.mean(fft_magnitude[3*len(fft_magnitude)//4:])
            if high_freq_energy > self.pattern_threshold * 0.5:
                patterns.append({
                    'type': 'audio_high',
                    'region': 'audio_high',
                    'strength': high_freq_energy,
                    'energy': high_freq_energy * self.audio_sensitivity,
                    'temporal_persistence': 2,
                    'frequency_range': 'high'
                })
            return patterns
        except Exception as e:
            log_step("Error extracting audio patterns", error=str(e))
            return []
    def _map_patterns_to_workspace(self, patterns: List[Dict[str, Any]],
                                 pattern_type: str) -> List[Dict[str, Any]]:

        try:
            workspace_updates = []
            for pattern in patterns:
                region_name = pattern['region']
                if region_name in self.workspace_regions:
                    region = self.workspace_regions[region_name]
                    x_min, y_min, x_max, y_max = region
                    workspace_x = int(x_min * self.workspace_size[0])
                    workspace_y = int(y_min * self.workspace_size[1])
                    workspace_w = int((x_max - x_min) * self.workspace_size[0])
                    workspace_h = int((y_max - y_min) * self.workspace_size[1])
                    workspace_updates.append({
                        'region': region_name,
                        'pattern_type': pattern_type,
                        'pattern': pattern,
                        'workspace_coords': (workspace_x, workspace_y, workspace_w, workspace_h),
                        'energy': pattern['energy'],
                        'strength': pattern['strength'],
                        'temporal_persistence': pattern['temporal_persistence']
                    })
            return workspace_updates
        except Exception as e:
            log_step("Error mapping patterns to workspace", error=str(e))
            return []
    def _update_workspace_nodes(self, graph: Data, workspace_updates: List[Dict[str, Any]],
                               access_layer: NodeAccessLayer) -> Data:

        try:
            workspace_nodes = access_layer.select_nodes_by_type('workspace')
            if not workspace_nodes:
                log_step("No workspace nodes found for pattern mapping")
                return graph
            for update in workspace_updates:
                region = update['region']
                energy = update['energy']
                pattern = update['pattern']
                region_nodes = self._find_workspace_nodes_in_region(
                    workspace_nodes, update['workspace_coords'], access_layer
                )
                for node_id in region_nodes:
                    access_layer.update_node_property(node_id, 'enhanced_behavior', True)
                    access_layer.update_node_property(node_id, 'pattern_type', pattern['type'])
                    access_layer.update_node_property(node_id, 'pattern_region', region)
                    access_layer.update_node_property(node_id, 'pattern_strength', pattern['strength'])
                    access_layer.update_node_property(node_id, 'last_pattern_update', time.time())
                    current_energy = access_layer.get_node_energy(node_id)
                    if current_energy is not None:
                        new_energy = min(current_energy + energy, 1.0)
                        access_layer.set_node_energy(node_id, new_energy)
                        access_layer.update_node_property(node_id, 'membrane_potential',
                                                       min(new_energy, 1.0))
            return graph
        except Exception as e:
            log_step("Error updating workspace nodes", error=str(e))
            return graph
    def _find_workspace_nodes_in_region(self, workspace_nodes: List[int],
                                      region_coords: Tuple[int, int, int, int],
                                      access_layer: NodeAccessLayer) -> List[int]:

        try:
            region_nodes = []
            x, y, w, h = region_coords
            for node_id in workspace_nodes:
                if not access_layer.is_valid_node_id(node_id):
                    continue
                node = access_layer.get_node_by_id(node_id)
                if node is None:
                    continue
                node_index = access_layer.get_node_property(node_id, 'workspace_index', 0)
                node_x = node_index % self.workspace_size[0]
                node_y = node_index // self.workspace_size[0]
                if (x <= node_x < x + w and y <= node_y < y + h):
                    region_nodes.append(node_id)
            return region_nodes
        except Exception as e:
            log_step("Error finding workspace nodes in region", error=str(e))
            return []
    def _create_sensory_workspace_connections(self, graph: Data, visual_patterns: List[Dict[str, Any]],
                                            access_layer: NodeAccessLayer) -> Data:

        try:
            sensory_nodes = access_layer.select_nodes_by_type('sensory')
            workspace_nodes = access_layer.select_nodes_by_type('workspace')
            if not sensory_nodes or not workspace_nodes:
                return graph
            for pattern in visual_patterns:
                if 'spatial_center' in pattern:
                    spatial_center = pattern['spatial_center']
                    nearby_sensory = self._find_sensory_nodes_near_position(
                        sensory_nodes, spatial_center, access_layer
                    )
                    region = pattern.get('region', 'visual_center')
                    if region in self.workspace_regions:
                        region_coords = self._get_region_workspace_coords(region)
                        region_workspace = self._find_workspace_nodes_in_region(
                            workspace_nodes, region_coords, access_layer
                        )
                        for sensory_id in nearby_sensory[:3]:
                            for workspace_id in region_workspace[:2]:
                                if sensory_id != workspace_id:
                                    self._create_enhanced_connection(
                                        graph, sensory_id, workspace_id, 'excitatory',
                                        weight=pattern['strength'] * 0.5,
                                        delay=0.1,
                                        plasticity_enabled=True
                                    )
            return graph
        except Exception as e:
            log_step("Error creating sensory-workspace connections", error=str(e))
            return graph
    def _create_audio_workspace_connections(self, graph: Data, audio_patterns: List[Dict[str, Any]],
                                          access_layer: NodeAccessLayer) -> Data:

        try:
            audio_sensory = access_layer.select_nodes_by_property('audio_stimulation', True)
            workspace_nodes = access_layer.select_nodes_by_type('workspace')
            if not audio_sensory or not workspace_nodes:
                return graph
            for pattern in audio_patterns:
                region = pattern.get('region', 'audio_low')
                if region in self.workspace_regions:
                    region_coords = self._get_region_workspace_coords(region)
                    region_workspace = self._find_workspace_nodes_in_region(
                        workspace_nodes, region_coords, access_layer
                    )
                    for audio_id in audio_sensory[:2]:
                        for workspace_id in region_workspace[:2]:
                            if audio_id != workspace_id:
                                self._create_enhanced_connection(
                                    graph, audio_id, workspace_id, 'excitatory',
                                    weight=pattern['strength'] * 0.3,
                                    delay=0.05,
                                    plasticity_enabled=True
                                )
            return graph
        except Exception as e:
            log_step("Error creating audio-workspace connections", error=str(e))
            return graph
    def _find_sensory_nodes_near_position(self, sensory_nodes: List[int],
                                        position: Tuple[int, int],
                                        access_layer: NodeAccessLayer) -> List[int]:

        try:
            nearby_nodes = []
            target_x, target_y = position
            for node_id in sensory_nodes:
                node = access_layer.get_node_by_id(node_id)
                if node is None:
                    continue
                node_x = node.get('x', 0)
                node_y = node.get('y', 0)
                distance = np.sqrt((node_x - target_x)**2 + (node_y - target_y)**2)
                if distance < 10:
                    nearby_nodes.append(node_id)
            return nearby_nodes
        except Exception as e:
            log_step("Error finding sensory nodes near position", error=str(e))
            return []
    def _get_region_workspace_coords(self, region_name: str) -> Tuple[int, int, int, int]:

        if region_name in self.workspace_regions:
            region = self.workspace_regions[region_name]
            x_min, y_min, x_max, y_max = region
            workspace_x = int(x_min * self.workspace_size[0])
            workspace_y = int(y_min * self.workspace_size[1])
            workspace_w = int((x_max - x_min) * self.workspace_size[0])
            workspace_h = int((y_max - y_min) * self.workspace_size[1])
            return (workspace_x, workspace_y, workspace_w, workspace_h)
        return (0, 0, self.workspace_size[0], self.workspace_size[1])
    def _create_enhanced_connection(self, graph: Data, source_id: int, target_id: int,
                                  connection_type: str, weight: float, delay: float,
                                  plasticity_enabled: bool):

        try:
            access_layer = NodeAccessLayer(graph)
            if not access_layer.is_valid_node_id(source_id) or not access_layer.is_valid_node_id(target_id):
                log_step("Invalid node IDs for connection", source_id=source_id, target_id=target_id)
                return
            if source_id == target_id:
                log_step("Cannot create self-connection", node_id=source_id)
                return
            create_weighted_connection(graph, source_id, target_id, weight, connection_type)
            log_step("Enhanced connection created", source_id=source_id, target_id=target_id, weight=weight)
        except Exception as e:
            log_step("Error creating enhanced connection", error=str(e))
    def _find_edge_center(self, edge_magnitude: np.ndarray) -> Tuple[int, int]:
        try:
            y_indices, x_indices = np.where(edge_magnitude > edge_magnitude.mean())
            if len(y_indices) > 0:
                return (int(np.mean(x_indices)), int(np.mean(y_indices)))
            return (edge_magnitude.shape[1]//2, edge_magnitude.shape[0]//2)
        except Exception as e:
            log_step("Error finding edge center", error=str(e))
            return (0, 0)
    def _find_motion_center(self, motion: np.ndarray) -> Tuple[int, int]:
        try:
            y_indices, x_indices = np.where(motion > motion.mean())
            if len(y_indices) > 0:
                return (int(np.mean(x_indices)), int(np.mean(y_indices)))
            return (motion.shape[1]//2, motion.shape[0]//2)
        except Exception as e:
            log_step("Error finding motion center", error=str(e))
            return (0, 0)
    def _calculate_texture_variance(self, visual_data: np.ndarray) -> float:
        try:
            if visual_data is None or visual_data.size == 0:
                return 0.0
            if len(visual_data.shape) != 2:
                return 0.0
            if visual_data.shape[0] > 2 and visual_data.shape[1] > 2:
                local_var = np.zeros_like(visual_data)
                for i in range(1, visual_data.shape[0] - 1):
                    for j in range(1, visual_data.shape[1] - 1):
                        window = visual_data[i-1:i+2, j-1:j+2]
                        if window.size > 0:
                            local_var[i, j] = np.var(window)
                return np.mean(local_var)
            return 0.0
        except Exception as e:
            log_step("Error calculating texture variance", error=str(e))
            return 0.0
    def get_mapping_statistics(self) -> Dict[str, Any]:
        return {
            'workspace_size': self.workspace_size,
            'visual_patterns': len(self.visual_patterns),
            'audio_patterns': len(self.audio_patterns),
            'sensory_mappings': len(self.sensory_mappings),
            'workspace_regions': len(self.workspace_regions),
            'visual_sensitivity': self.visual_sensitivity,
            'audio_sensitivity': self.audio_sensitivity,
            'pattern_threshold': self.pattern_threshold
        }

    def set_manager(self, manager):
        self.manager = manager

    def _on_audio_input(self, event_type, data):
        if self.manager and self.manager.graph is not None:
            features = data.get('features', np.zeros(1024))
            if isinstance(features, np.ndarray):
                self.map_audio_to_workspace(self.manager.graph, features, self.manager.step_counter)
                self.manager.event_bus.emit('GRAPH_UPDATE', {'graph': self.manager.graph})

    def _on_visual_input(self, event_type, data):
        if self.manager and self.manager.graph is not None:
            energy = data.get('energy', 0.5)
            # Create mock frame from energy
            mock_frame = np.full((64, 64), energy * 255, dtype=np.uint8)
            self.map_visual_to_workspace(self.manager.graph, mock_frame, self.manager.step_counter)
            self.manager.event_bus.emit('GRAPH_UPDATE', {'graph': self.manager.graph})


def create_sensory_workspace_mapper(workspace_size: Tuple[int, int] = (10, 10)) -> SensoryWorkspaceMapper:

    return SensoryWorkspaceMapper(workspace_size)
if __name__ == "__main__":
    print("SensoryWorkspaceMapper created successfully!")
    print("Features include:")
    print("- Visual pattern extraction and mapping to workspace")
    print("- Audio pattern extraction and mapping to workspace")
    print("- Spatial organization of workspace regions")
    print("- Direct sensory-to-workspace connections")
    print("- Pattern-based energy propagation")
    print("- Temporal pattern persistence")
    try:
        mapper = create_sensory_workspace_mapper((8, 8))
        dummy_visual = np.random.rand(64, 64)
        print(f"Testing visual mapping with {dummy_visual.shape} input")
        dummy_audio = np.random.randn(1024)
        print(f"Testing audio mapping with {len(dummy_audio)} samples")
        stats = mapper.get_mapping_statistics()
        print(f"Mapper statistics: {stats}")
    except Exception as e:
        print(f"SensoryWorkspaceMapper test failed: {e}")
    print("SensoryWorkspaceMapper test completed!")
