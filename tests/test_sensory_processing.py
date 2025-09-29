"""
Comprehensive tests for sensory processing components.

This module provides extensive testing coverage for all sensory processing
components including unit tests, integration tests, edge cases, simulation
scenarios, error handling, performance benchmarks, and real-world usage patterns.
"""

import os
import sys
import threading
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import torch
from torch_geometric.data import Data

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.interfaces.sensory_processor import SensoryInput
from src.core.services.sensory_processing_service import \
    SensoryProcessingService
from src.energy.node_access_layer import NodeAccessLayer
from src.sensory.audio_to_neural_bridge import (AudioToNeuralBridge,
                                                create_audio_to_neural_bridge)
from src.sensory.sensory_workspace_mapper import (
    SensoryWorkspaceMapper, create_sensory_workspace_mapper)
from src.sensory.visual_energy_bridge import (VisualEnergyBridge,
                                              create_visual_energy_bridge)
from src.utils.event_bus import get_event_bus


class TestAudioToNeuralBridge(unittest.TestCase):
    """Unit tests for AudioToNeuralBridge."""

    def setUp(self):
        """Set up test fixtures."""
        self.bridge = create_audio_to_neural_bridge()

    def test_initialization(self):
        """Test bridge initialization."""
        self.assertIsInstance(self.bridge, AudioToNeuralBridge)
        self.assertIsNotNone(self.bridge.audio_features_cache)
        self.assertIsNotNone(self.bridge.sensory_node_mapping)
        self.assertEqual(self.bridge.sample_rate, 22050)
        self.assertEqual(self.bridge.n_mfcc, 13)

    def test_process_audio_to_sensory_nodes(self):
        """Test audio processing to sensory nodes."""
        # Create test audio data
        audio_data = np.random.randn(44100).astype(np.float32)

        sensory_nodes = self.bridge.process_audio_to_sensory_nodes(audio_data)

        self.assertIsInstance(sensory_nodes, list)
        self.assertGreater(len(sensory_nodes), 0)

        # Check node structure
        node = sensory_nodes[0]
        required_keys = ['id', 'type', 'subtype', 'feature_type', 'energy', 'membrane_potential']
        for key in required_keys:
            self.assertIn(key, node)

        self.assertEqual(node['type'], 'sensory')
        self.assertEqual(node['subtype'], 'audio')

    def test_audio_feature_extraction(self):
        """Test audio feature extraction."""
        audio_data = np.random.randn(22050).astype(np.float32)  # 1 second

        features = self.bridge._extract_audio_features(audio_data)

        self.assertIsInstance(features, dict)
        self.assertIn('mfcc', features)
        self.assertIn('mel_spectrogram', features)
        self.assertIn('spectral_features', features)
        self.assertIn('temporal_features', features)

    def test_audio_normalization(self):
        """Test audio normalization."""
        audio_data = np.array([1.0, -1.0, 0.5, -0.5])

        normalized = self.bridge._normalize_audio(audio_data)

        self.assertAlmostEqual(np.max(np.abs(normalized)), 1.0, places=6)

    def test_empty_audio_handling(self):
        """Test handling of empty audio data."""
        audio_data = np.array([])

        sensory_nodes = self.bridge.process_audio_to_sensory_nodes(audio_data)

        # Should return empty list or handle gracefully
        self.assertIsInstance(sensory_nodes, list)

    def test_large_audio_downsampling(self):
        """Test downsampling of large audio files."""
        large_audio = np.random.randn(100000)  # Larger than max_samples

        features = self.bridge._extract_audio_features(large_audio)

        # Should not crash and return features
        self.assertIsInstance(features, dict)

    @patch('sensory.audio_to_neural_bridge.librosa')
    def test_librosa_unavailable_fallback(self, mock_librosa):
        """Test fallback when librosa is unavailable."""
        mock_librosa.feature.mfcc.side_effect = ImportError("librosa not available")

        # Create bridge with mocked librosa
        bridge = AudioToNeuralBridge()
        bridge.LIBROSA_AVAILABLE = False

        audio_data = np.random.randn(22050).astype(np.float32)
        features = bridge._extract_audio_features(audio_data)

        # Should still return features (fallback zeros)
        self.assertIsInstance(features, dict)
        self.assertIn('mfcc', features)

    def test_graph_integration(self):
        """Test integration with neural graph."""
        graph = Data()
        graph.node_labels = []
        graph.x = torch.empty(0, 5)

        audio_data = np.random.randn(22050).astype(np.float32)

        updated_graph = self.bridge.integrate_audio_nodes_into_graph(graph, audio_data)

        self.assertIsInstance(updated_graph, Data)
        self.assertGreater(len(updated_graph.node_labels), 0)

    def test_statistics(self):
        """Test getting audio feature statistics."""
        stats = self.bridge.get_audio_feature_statistics()

        required_keys = ['cached_features', 'sensory_node_mappings', 'sample_rate']
        for key in required_keys:
            self.assertIn(key, stats)

    def test_cache_operations(self):
        """Test cache clearing operations."""
        # Add some dummy data to cache
        self.bridge.audio_features_cache['test'] = 'data'
        self.bridge.sensory_node_mapping['test'] = 'mapping'

        self.bridge.clear_cache()

        self.assertEqual(len(self.bridge.audio_features_cache), 0)
        self.assertEqual(len(self.bridge.sensory_node_mapping), 0)


class TestVisualEnergyBridge(unittest.TestCase):
    """Unit tests for VisualEnergyBridge."""

    def setUp(self):
        """Set up test fixtures."""
        self.bridge = create_visual_energy_bridge()

    def test_initialization(self):
        """Test bridge initialization."""
        self.assertIsInstance(self.bridge, VisualEnergyBridge)
        self.assertIsNotNone(self.bridge.visual_patterns)
        self.assertIsNotNone(self.bridge.energy_history)
        self.assertEqual(self.bridge.visual_sensitivity, 0.5)

    def test_process_visual_to_enhanced_energy(self):
        """Test visual processing to enhanced energy."""
        graph = Data()
        graph.node_labels = [{'id': 'node_0', 'type': 'sensory', 'energy': 0.0}]
        graph.x = torch.randn(1, 5)

        screen_data = np.random.rand(64, 64).astype(np.float32)

        updated_graph = self.bridge.process_visual_to_enhanced_energy(graph, screen_data, 0)

        self.assertIsInstance(updated_graph, Data)

    def test_visual_feature_extraction(self):
        """Test visual feature extraction."""
        screen_data = np.random.rand(64, 64).astype(np.float32)

        features = self.bridge._extract_visual_features(screen_data)

        required_features = ['mean_intensity', 'std_intensity', 'edge_density']
        for feature in required_features:
            self.assertIn(feature, features)

    def test_visual_pattern_detection(self):
        """Test visual pattern detection."""
        features = {
            'std_intensity': 0.8,  # High contrast
            'edge_density': 0.6,   # Edge rich
            'motion_magnitude': 0.7,  # Motion
            'texture_std': 0.5     # Texture
        }

        patterns = self.bridge._detect_visual_patterns(features, 0)

        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)

        # Should detect high contrast pattern
        pattern_types = [p['type'] for p in patterns]
        self.assertIn('high_contrast', pattern_types)

    def test_visual_energy_conversion(self):
        """Test conversion of visual features to energy."""
        graph = Data()
        graph.node_labels = [{'id': 'node_0', 'type': 'sensory'}]

        features = {'mean_intensity': 0.8, 'std_intensity': 0.6}

        updated_graph = self.bridge._convert_visual_to_enhanced_energy(graph, features, 0)

        self.assertIsInstance(updated_graph, Data)

    def test_statistics(self):
        """Test getting visual statistics."""
        stats = self.bridge.get_visual_statistics()

        required_keys = ['patterns_detected', 'memory_patterns', 'enhanced_integration_available']
        for key in required_keys:
            self.assertIn(key, stats)

    def test_sensitivity_settings(self):
        """Test visual sensitivity settings."""
        self.bridge.set_visual_sensitivity(0.8)
        self.assertEqual(self.bridge.visual_sensitivity, 0.8)

        self.bridge.set_pattern_threshold(0.4)
        self.assertEqual(self.bridge.pattern_threshold, 0.4)

        # Test bounds
        self.bridge.set_visual_sensitivity(1.5)  # Should clamp to 1.0
        self.assertEqual(self.bridge.visual_sensitivity, 1.0)

    @patch('sensory.visual_energy_bridge.cv2')
    def test_opencv_unavailable_fallback(self, mock_cv2):
        """Test fallback when OpenCV is unavailable."""
        mock_cv2.absdiff.side_effect = ImportError("cv2 not available")

        screen_data = np.random.rand(64, 64).astype(np.float32)
        features = self.bridge._extract_visual_features(screen_data)

        # Should still return features
        self.assertIsInstance(features, dict)


class TestSensoryWorkspaceMapper(unittest.TestCase):
    """Unit tests for SensoryWorkspaceMapper."""

    def setUp(self):
        """Set up test fixtures."""
        self.mapper = create_sensory_workspace_mapper((10, 10))

    def test_initialization(self):
        """Test mapper initialization."""
        self.assertIsInstance(self.mapper, SensoryWorkspaceMapper)
        self.assertEqual(self.mapper.workspace_size, (10, 10))
        self.assertIsNotNone(self.mapper.bus)

    def test_visual_mapping(self):
        """Test visual pattern mapping to workspace."""
        graph = Data()
        graph.node_labels = [{'id': 'workspace_0', 'type': 'workspace', 'workspace_index': 0}]

        visual_data = np.random.rand(64, 64).astype(np.float32)

        updated_graph = self.mapper.map_visual_to_workspace(graph, visual_data, 0)

        self.assertIsInstance(updated_graph, Data)

    def test_audio_mapping(self):
        """Test audio pattern mapping to workspace."""
        graph = Data()
        graph.node_labels = [{'id': 'workspace_0', 'type': 'workspace', 'workspace_index': 0}]

        audio_data = np.random.randn(1024).astype(np.float32)

        updated_graph = self.mapper.map_audio_to_workspace(graph, audio_data, 0)

        self.assertIsInstance(updated_graph, Data)

    def test_visual_pattern_extraction(self):
        """Test visual pattern extraction."""
        visual_data = np.ones((64, 64)) * 0.5  # Uniform
        visual_data[:32, :] = 0.0  # Half black
        visual_data[32:, :] = 1.0  # Half white, high contrast

        patterns = self.mapper._extract_visual_patterns(visual_data, 0)

        self.assertIsInstance(patterns, list)
        # Should detect high contrast
        pattern_types = [p['type'] for p in patterns]
        self.assertIn('high_contrast', pattern_types)

    def test_audio_pattern_extraction(self):
        """Test audio pattern extraction."""
        # Create test audio with low frequency content
        t = np.linspace(0, 1, 1024)
        audio_data = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

        patterns = self.mapper._extract_audio_patterns(audio_data, 0)

        self.assertIsInstance(patterns, list)
        # Should detect low frequency pattern
        pattern_types = [p['type'] for p in patterns]
        self.assertIn('audio_low', pattern_types)

    def test_workspace_pattern_mapping(self):
        """Test mapping patterns to workspace coordinates."""
        patterns = [{'type': 'test', 'region': 'visual_center', 'energy': 0.8, 'strength': 0.7, 'temporal_persistence': 5}]

        updates = self.mapper._map_patterns_to_workspace(patterns, 'visual')

        self.assertIsInstance(updates, list)
        self.assertGreater(len(updates), 0)

        update = updates[0]
        self.assertIn('workspace_coords', update)

    def test_statistics(self):
        """Test getting mapping statistics."""
        stats = self.mapper.get_mapping_statistics()

        required_keys = ['workspace_size', 'visual_patterns', 'workspace_regions']
        for key in required_keys:
            self.assertIn(key, stats)


class TestSensoryProcessingService(unittest.TestCase):
    """Unit tests for SensoryProcessingService."""

    def setUp(self):
        """Set up test fixtures."""
        self.energy_manager = Mock()
        self.config_service = Mock()
        self.event_coordinator = Mock()

        self.service = SensoryProcessingService(
            self.energy_manager,
            self.config_service,
            self.event_coordinator
        )

    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.service, SensoryProcessingService)
        self.assertIsNotNone(self.service._sensory_buffer)
        self.assertIsNotNone(self.service._adaptation_levels)

    def test_process_sensory_input(self):
        """Test sensory input processing."""
        graph = Data()
        graph.node_labels = []

        updated_graph = self.service.process_sensory_input(graph)

        self.assertIsInstance(updated_graph, Data)
        self.assertEqual(len(self.service._sensory_buffer), 1)

    def test_initialize_sensory_pathways(self):
        """Test sensory pathway initialization."""
        graph = Data()
        graph.node_labels = []

        success = self.service.initialize_sensory_pathways(graph)

        self.assertTrue(success)
        self.assertGreater(len(graph.node_labels), 0)
        self.assertIn('visual', self.service._sensory_pathways)

    def test_sensory_adaptation(self):
        """Test sensory adaptation application."""
        graph = Data()

        # Initialize adaptation levels
        self.service._adaptation_levels = {'visual': 1.0, 'auditory': 1.0}

        updated_graph = self.service.apply_sensory_adaptation(graph, 0.9)

        self.assertIsInstance(updated_graph, Data)
        self.assertLess(self.service._adaptation_levels['visual'], 1.0)

    def test_visual_activation_pattern(self):
        """Test visual activation pattern generation."""
        sensory_input = SensoryInput("visual", {}, 0.8, spatial_location=(0.5, 0.5))

        # Create sensory nodes
        self.service._sensory_pathways['visual'] = ['node_0', 'node_1', 'node_2']

        adapted_intensity = 0.8
        pattern = self.service._generate_activation_pattern(sensory_input, adapted_intensity)

        self.assertIn('activations', pattern)
        self.assertGreater(len(pattern['activations']), 0)

    def test_auditory_activation_pattern(self):
        """Test auditory activation pattern generation."""
        sensory_input = SensoryInput("auditory", {'frequency': 1000}, 0.7)

        self.service._sensory_pathways['auditory'] = ['node_0', 'node_1']

        adapted_intensity = 0.7
        pattern = self.service._generate_activation_pattern(sensory_input, adapted_intensity)

        self.assertIn('activations', pattern)

    def test_get_sensory_state(self):
        """Test getting sensory state."""
        state = self.service.get_sensory_state()

        required_keys = ['processed_inputs', 'buffer_size', 'adaptation_levels']
        for key in required_keys:
            self.assertIn(key, state)

    def test_cleanup(self):
        """Test service cleanup."""
        # Add some data
        self.service._sensory_buffer.append(SensoryInput("test", {}, 0.5))
        self.service._adaptation_levels['test'] = 1.0
        self.service._sensory_pathways['test'] = ['node_0']

        self.service.cleanup()

        self.assertEqual(len(self.service._sensory_buffer), 0)
        self.assertEqual(len(self.service._adaptation_levels), 0)
        self.assertEqual(len(self.service._sensory_pathways), 0)


class TestSensoryIntegration(unittest.TestCase):
    """Integration tests for sensory components working together."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.audio_bridge = create_audio_to_neural_bridge()
        self.visual_bridge = create_visual_energy_bridge()
        self.workspace_mapper = create_sensory_workspace_mapper((8, 8))

        # Create mock graph with sensory and workspace nodes
        self.graph = Data()
        self.graph.node_labels = [
            {'id': 'sensory_0', 'type': 'sensory', 'energy': 0.0},
            {'id': 'workspace_0', 'type': 'workspace', 'workspace_index': 0, 'energy': 0.0}
        ]
        self.graph.x = torch.randn(2, 5)

    def test_audio_visual_integration(self):
        """Test integration of audio and visual processing."""
        initial_len = len(self.graph.node_labels)

        # Process audio
        audio_data = np.random.randn(22050).astype(np.float32)
        audio_graph = self.audio_bridge.integrate_audio_nodes_into_graph(self.graph, audio_data)

        # Process visual
        visual_data = np.random.rand(64, 64).astype(np.float32)
        visual_graph = self.visual_bridge.process_visual_to_enhanced_energy(audio_graph, visual_data, 0)

        # Map to workspace
        final_graph = self.workspace_mapper.map_visual_to_workspace(visual_graph, visual_data, 0)

        self.assertIsInstance(final_graph, Data)
        self.assertGreater(len(final_graph.node_labels), initial_len)

    def test_event_driven_processing(self):
        """Test event-driven sensory processing."""
        bus = get_event_bus()

        # Subscribe mapper to events
        # Process visual through event
        visual_data = np.random.rand(32, 32).astype(np.float32)

        # Simulate visual input event
        bus.emit('SENSORY_INPUT_VISUAL', {'energy': 0.8, 'timestamp': time.time()})

        # Check that mapper received the event (would need manager setup for full test)
        # This is a basic smoke test

    def test_cross_modal_workspace_mapping(self):
        """Test mapping multiple modalities to workspace."""
        # Audio mapping
        audio_data = np.random.randn(2048).astype(np.float32)
        graph_after_audio = self.workspace_mapper.map_audio_to_workspace(self.graph, audio_data, 0)

        # Visual mapping
        visual_data = np.random.rand(64, 64).astype(np.float32)
        final_graph = self.workspace_mapper.map_visual_to_workspace(graph_after_audio, visual_data, 0)

        self.assertIsInstance(final_graph, Data)


class TestSensoryEdgeCases(unittest.TestCase):
    """Edge case tests for sensory components."""

    def test_empty_inputs(self):
        """Test handling of empty or minimal inputs."""
        audio_bridge = create_audio_to_neural_bridge()
        visual_bridge = create_visual_energy_bridge()
        mapper = create_sensory_workspace_mapper()

        # Empty audio
        empty_audio = np.array([])
        nodes = audio_bridge.process_audio_to_sensory_nodes(empty_audio)
        self.assertIsInstance(nodes, list)

        # Minimal visual
        minimal_visual = np.array([[0.0]])
        features = visual_bridge._extract_visual_features(minimal_visual)
        self.assertIsInstance(features, dict)

        # Empty graph
        empty_graph = Data()
        empty_graph.node_labels = []
        updated = mapper.map_visual_to_workspace(empty_graph, np.array([[0.0]]), 0)
        self.assertIsInstance(updated, Data)

    def test_extreme_values(self):
        """Test handling of extreme input values."""
        bridge = create_visual_energy_bridge()

        # All zeros
        zero_visual = np.zeros((64, 64))
        features = bridge._extract_visual_features(zero_visual)
        self.assertIsInstance(features, dict)

        # All maximum values
        max_visual = np.ones((64, 64)) * 255
        features = bridge._extract_visual_features(max_visual)
        self.assertIsInstance(features, dict)

        # Very large audio
        large_audio = np.random.randn(1000000)  # 1M samples
        audio_bridge = create_audio_to_neural_bridge()
        nodes = audio_bridge.process_audio_to_sensory_nodes(large_audio)
        self.assertIsInstance(nodes, list)

    def test_invalid_graph_structures(self):
        """Test handling of invalid graph structures."""
        bridge = create_audio_to_neural_bridge()

        # Graph without node_labels
        invalid_graph = Data()
        audio_data = np.random.randn(22050)

        # Should handle gracefully
        updated = bridge.integrate_audio_nodes_into_graph(invalid_graph, audio_data)
        self.assertIsInstance(updated, Data)

    def test_concurrent_access(self):
        """Test concurrent access to sensory components."""
        bridge = create_audio_to_neural_bridge()

        results = []

        def process_audio():
            audio = np.random.randn(22050)
            nodes = bridge.process_audio_to_sensory_nodes(audio)
            results.append(len(nodes))

        # Start multiple threads
        threads = []
        for _ in range(5):
            t = threading.Thread(target=process_audio)
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join()

        self.assertEqual(len(results), 5)
        for r in results:
            self.assertGreater(r, 0)


class TestSensoryErrorHandling(unittest.TestCase):
    """Error handling tests for sensory components."""

    def test_missing_dependencies(self):
        """Test graceful handling of missing dependencies."""
        # Test without librosa
        with patch('sensory.audio_to_neural_bridge.librosa', None):
            bridge = AudioToNeuralBridge()
            bridge.LIBROSA_AVAILABLE = False

            audio = np.random.randn(22050)
            features = bridge._extract_audio_features(audio)

            # Should return fallback features
            self.assertIsInstance(features, dict)

    def test_corrupted_data(self):
        """Test handling of corrupted input data."""
        bridge = create_visual_energy_bridge()

        # NaN values
        nan_visual = np.full((64, 64), np.nan)
        features = bridge._extract_visual_features(nan_visual)
        self.assertIsInstance(features, dict)  # Should handle gracefully

        # Infinite values
        inf_visual = np.full((64, 64), np.inf)
        features = bridge._extract_visual_features(inf_visual)
        self.assertIsInstance(features, dict)

    def test_memory_errors(self):
        """Test handling of memory-related errors."""
        bridge = create_audio_to_neural_bridge()

        # Very large array that might cause memory issues
        try:
            huge_audio = np.random.randn(10000000)  # 10M samples
            nodes = bridge.process_audio_to_sensory_nodes(huge_audio)
            self.assertIsInstance(nodes, list)
        except MemoryError:
            # Expected for very large inputs
            pass

    @patch('sensory.visual_energy_bridge.cv2.VideoCapture')
    def test_camera_unavailable(self, mock_capture):
        """Test handling when camera is unavailable."""
        mock_capture.return_value.isOpened.return_value = False

        bridge = create_visual_energy_bridge()

        # Should not crash when trying to start stream
        try:
            bridge.start_stream()
            # Should fall back to simulation
        except Exception as e:
            self.fail(f"Bridge failed to handle unavailable camera: {e}")
        finally:
            bridge.stop_stream()


class TestSensoryPerformance(unittest.TestCase):
    """Performance tests for sensory components."""

    def test_audio_processing_performance(self):
        """Test audio processing performance."""
        bridge = create_audio_to_neural_bridge()

        audio_data = np.random.randn(44100).astype(np.float32)  # 1 second

        start_time = time.time()
        nodes = bridge.process_audio_to_sensory_nodes(audio_data)
        processing_time = time.time() - start_time

        # Should process within reasonable time (less than 1 second)
        self.assertLess(processing_time, 1.0)
        self.assertGreater(len(nodes), 0)

    def test_visual_processing_performance(self):
        """Test visual processing performance."""
        bridge = create_visual_energy_bridge()
        graph = Data()
        graph.node_labels = [{'id': 'node_0', 'type': 'sensory'}]

        visual_data = np.random.rand(128, 128).astype(np.float32)

        start_time = time.time()
        updated_graph = bridge.process_visual_to_enhanced_energy(graph, visual_data, 0)
        processing_time = time.time() - start_time

        # Should process within reasonable time
        self.assertLess(processing_time, 0.5)
        self.assertIsInstance(updated_graph, Data)

    def test_workspace_mapping_performance(self):
        """Test workspace mapping performance."""
        mapper = create_sensory_workspace_mapper((20, 20))
        graph = Data()
        graph.node_labels = [{'id': f'workspace_{i}', 'type': 'workspace', 'workspace_index': i} for i in range(400)]

        visual_data = np.random.rand(256, 256).astype(np.float32)

        start_time = time.time()
        updated_graph = mapper.map_visual_to_workspace(graph, visual_data, 0)
        processing_time = time.time() - start_time

        # Should process within reasonable time
        self.assertLess(processing_time, 1.0)
        self.assertIsInstance(updated_graph, Data)

    def test_memory_usage(self):
        """Test memory usage during processing."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        bridge = create_audio_to_neural_bridge()

        # Process multiple audio chunks
        for _ in range(10):
            audio = np.random.randn(44100)
            nodes = bridge.process_audio_to_sensory_nodes(audio)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB)
        self.assertLess(memory_increase, 100.0)


class TestSensoryRealWorldUsage(unittest.TestCase):
    """Real-world usage tests for sensory components."""

    def test_realistic_audio_processing(self):
        """Test processing realistic audio data."""
        bridge = create_audio_to_neural_bridge()

        # Simulate speech-like audio (modulated noise)
        t = np.linspace(0, 1, 22050)
        speech_like = np.sin(2 * np.pi * 300 * t) * np.exp(-t * 2)  # Decaying tone
        speech_like += 0.1 * np.random.randn(len(speech_like))  # Add noise

        nodes = bridge.process_audio_to_sensory_nodes(speech_like)

        self.assertGreater(len(nodes), 0)
        # Check that nodes have reasonable energy values
        energies = [node['energy'] for node in nodes[:10]]
        self.assertTrue(all(0 <= e <= 1 for e in energies))

    def test_realistic_visual_processing(self):
        """Test processing realistic visual data."""
        bridge = create_visual_energy_bridge()
        graph = Data()
        graph.node_labels = [{'id': 'node_0', 'type': 'sensory'}]

        # Simulate a simple image with gradients
        image = np.zeros((64, 64))
        image[:, :32] = np.linspace(0, 1, 32)[np.newaxis, :]  # Left half gradient
        image[:, 32:] = np.linspace(1, 0, 32)[np.newaxis, :]  # Right half gradient

        updated_graph = bridge.process_visual_to_enhanced_energy(graph, image, 0)

        self.assertIsInstance(updated_graph, Data)

    def test_multi_modal_integration(self):
        """Test integration of multiple sensory modalities."""
        audio_bridge = create_audio_to_neural_bridge()
        visual_bridge = create_visual_energy_bridge()
        mapper = create_sensory_workspace_mapper()

        graph = Data()
        graph.node_labels = [
            {'id': 'sensory_0', 'type': 'sensory'},
            {'id': 'workspace_0', 'type': 'workspace', 'workspace_index': 0}
        ]

        # Process audio and visual simultaneously
        audio_data = np.random.randn(22050)
        visual_data = np.random.rand(64, 64)

        # Audio processing
        graph = audio_bridge.integrate_audio_nodes_into_graph(graph, audio_data)

        # Visual processing
        graph = visual_bridge.process_visual_to_enhanced_energy(graph, visual_data, 0)

        # Workspace mapping
        graph = mapper.map_visual_to_workspace(graph, visual_data, 0)
        graph = mapper.map_audio_to_workspace(graph, audio_data, 0)

        self.assertIsInstance(graph, Data)
        self.assertGreater(len(graph.node_labels), 2)  # Should have added nodes

    def test_adaptive_processing(self):
        """Test adaptive sensory processing over time."""
        service = SensoryProcessingService(Mock(), Mock(), Mock())

        graph = Data()
        graph.node_labels = []

        # Initialize pathways
        service.initialize_sensory_pathways(graph)

        # Process multiple inputs to test adaptation
        for i in range(5):
            updated_graph = service.process_sensory_input(graph)
            graph = updated_graph

        # Check that adaptation levels have changed
        state = service.get_sensory_state()
        self.assertGreater(state['processed_inputs'], 0)

    def test_streaming_simulation(self):
        """Test simulated streaming sensory input."""
        bridge = create_visual_energy_bridge()

        # Start simulated visual stream
        bridge.start_stream()

        # Wait a bit for processing
        time.sleep(0.1)

        # Stop stream
        bridge.stop_stream()

        # Should not have crashed
        self.assertFalse(bridge.running)


if __name__ == '__main__':
    unittest.main()






