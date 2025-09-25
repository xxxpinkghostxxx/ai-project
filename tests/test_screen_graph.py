"""
Comprehensive unit tests for screen_graph.py
Tests all functions, edge cases, error handling, performance, and real-world usage.
"""

import sys
import os
import time
import unittest
from unittest.mock import Mock, patch, MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock dependencies
sys.modules['numpy'] = Mock()
import numpy as np
np.__version__ = "1.24.0"
sys.modules['PIL'] = Mock()
sys.modules['PIL.ImageGrab'] = Mock()
sys.modules['cv2'] = Mock()
sys.modules['torch'] = Mock()
sys.modules['torch.nn'] = Mock()
sys.modules['torch_geometric'] = Mock()
sys.modules['torch_geometric.data'] = Mock()
sys.modules['mss'] = Mock()
sys.modules['numba'] = Mock()
import mss

import numpy as np
import PIL.ImageGrab
import cv2
import torch
torch.nn = Mock()
from torch_geometric.data import Data

from ui.screen_graph import rgb_to_gray, capture_screen, create_pixel_gray_graph


class TestScreenGraph(unittest.TestCase):
    """Unit tests for screen_graph functions."""

    def setUp(self):
        """Set up test environment."""
        # Mock numpy functions
        np.array = Mock(return_value=Mock())
        np.dot = Mock(side_effect=lambda a, b: Mock(shape=a.shape[:2] if len(getattr(a, 'shape', ())) > 2 else (getattr(a, 'shape', ())[0], getattr(a, 'shape', ())[1], 3) if len(getattr(a, 'shape', ())) == 2 else getattr(a, 'shape', ())))
        np.zeros = Mock(side_effect=lambda *args, **kwargs: Mock(shape=args[0] if args else ()))
        np.uint8 = Mock()
        np.float32 = Mock()

        # Mock PIL
        PIL.ImageGrab.grab = Mock(return_value=Mock())
        PIL.ImageGrab.Image = Mock()
        PIL.ImageGrab.Image.fromarray = Mock(return_value=Mock())
        PIL.ImageGrab.Image.fromarray.return_value.convert = Mock(return_value=Mock())
        PIL.ImageGrab.Image.fromarray.return_value.convert.return_value.resize = Mock(return_value=Mock())

        # Mock cv2
        cv2.resize = Mock(return_value=Mock())
        cv2.INTER_AREA = Mock()

        # Mock torch
        torch.tensor = Mock(return_value=Mock())
        torch.empty = Mock(return_value=Mock())
        torch.long = Mock()
        torch.float32 = Mock()
        torch.nn = Mock()

        # Mock torch_geometric
        Data = Mock()

        # Make Mock support __getitem__ for array-like access
        def mock_getitem(self, key):
            if isinstance(key, slice) and hasattr(self, 'shape') and isinstance(self.shape, tuple):
                return self.shape[key]
            elif isinstance(key, tuple) and all(isinstance(k, (int, type(Ellipsis))) for k in key) and any(isinstance(k, int) for k in key):
                return 0.5
            else:
                return Mock(shape=getattr(self, 'shape', ()))

        Mock.__getitem__ = mock_getitem

        # Mock mss
        mss.mss = Mock(return_value=Mock())
        mss.mss.return_value.__enter__ = Mock(return_value=Mock())
        mss.mss.return_value.__exit__ = Mock(return_value=None)
        mss.mss.return_value.__enter__.return_value.monitors = [Mock()]
        mss.mss.return_value.__enter__.return_value.grab = Mock(return_value=Mock())

        # Set shapes on mock return values
        PIL.ImageGrab.grab.return_value.shape = (360, 640, 3)
        mss.mss.return_value.__enter__.return_value.grab.return_value.shape = (360, 640, 3)
        cv2.resize.return_value.shape = (180, 320, 3)
        PIL.ImageGrab.Image.fromarray.return_value.convert.return_value.resize.return_value.shape = (180, 320, 3)

    def test_rgb_to_gray(self):
        """Test RGB to grayscale conversion."""
        # Mock RGB array
        mock_arr = Mock()
        mock_arr.shape = (100, 100, 3)
        mock_arr.__getitem__ = Mock(return_value=Mock())

        result = rgb_to_gray(mock_arr)

        np.dot.assert_called_once()
        self.assertIsNotNone(result)

    def test_rgb_to_gray_invalid_shape(self):
        """Test RGB to grayscale with invalid shape."""
        mock_arr = Mock()
        mock_arr.shape = (100, 100)  # Missing color channel

        with self.assertRaises(AssertionError):
            rgb_to_gray(mock_arr)

    def test_capture_screen_with_mss(self):
        """Test screen capture using mss."""
        # Setup mocks
        mock_sct_img = Mock()
        mock_sct_img.__getitem__ = Mock(return_value=Mock())
        mock_sct_img.__getitem__.return_value.__getitem__ = Mock(return_value=Mock())
        mss.mss.return_value.__enter__.return_value.grab.return_value = mock_sct_img

        result = capture_screen()

        mss.mss.assert_called_once()
        self.assertIsNotNone(result)

    def test_capture_screen_with_pil_fallback(self):
        """Test screen capture with PIL fallback when mss fails."""
        # Mock mss failure
        mss.mss.side_effect = Exception("MSS not available")

        # Mock PIL success
        mock_pil_img = Mock()
        mock_pil_img.convert = Mock(return_value=Mock())
        PIL.ImageGrab.grab.return_value = mock_pil_img

        result = capture_screen()

        PIL.ImageGrab.grab.assert_called_once()
        self.assertIsNotNone(result)

    def test_capture_screen_complete_fallback(self):
        """Test screen capture with complete fallback."""
        # Mock both mss and PIL failure
        mss.mss.side_effect = Exception("MSS failed")
        PIL.ImageGrab.grab.side_effect = Exception("PIL failed")

        result = capture_screen()

        np.zeros.assert_called_once()
        self.assertIsNotNone(result)

    def test_capture_screen_with_scaling(self):
        """Test screen capture with scaling."""
        result = capture_screen(scale=0.5)

        # Should apply scaling
        self.assertIsNotNone(result)

    def test_capture_screen_invalid_scale(self):
        """Test screen capture with invalid scale values."""
        with self.assertRaises(ValueError):
            capture_screen(scale=0)

        with self.assertRaises(ValueError):
            capture_screen(scale=3.0)

        with self.assertRaises(ValueError):
            capture_screen(scale="invalid")

    def test_capture_screen_cv2_resize(self):
        """Test screen capture with cv2 resize."""
        # Mock cv2 available
        cv2.resize.return_value = Mock()

        result = capture_screen(scale=0.5)

        cv2.resize.assert_called_once()
        self.assertIsNotNone(result)

    def test_capture_screen_pil_resize_fallback(self):
        """Test screen capture with PIL resize fallback."""
        # Mock cv2 failure
        cv2.resize.side_effect = Exception("CV2 failed")

        result = capture_screen(scale=0.5)

        PIL.ImageGrab.Image.fromarray.assert_called_once()
        self.assertIsNotNone(result)

    def test_create_pixel_gray_graph(self):
        """Test pixel gray graph creation."""
        # Mock grayscale array
        mock_arr = Mock()
        mock_arr.shape = (10, 10)
        mock_arr.flatten = Mock(return_value=Mock())
        mock_arr.__getitem__ = Mock(return_value=Mock())

        result = create_pixel_gray_graph(mock_arr)

        torch.tensor.assert_called_once()
        torch.empty.assert_called_once()
        Data.assert_called_once()
        self.assertIsNotNone(result)
        self.assertEqual(result.h, 10)
        self.assertEqual(result.w, 10)

    def test_create_pixel_gray_graph_large_array(self):
        """Test pixel gray graph creation with large array."""
        # Mock large array that should be sampled
        mock_arr = Mock()
        mock_arr.shape = (100, 100)  # 10,000 pixels, exceeds max_nodes
        mock_arr.__getitem__ = Mock(return_value=Mock(shape=(31, 31)))
        mock_arr.flatten = Mock(return_value=Mock())

        result = create_pixel_gray_graph(mock_arr)

        # Should sample down
        self.assertLessEqual(result.h * result.w, 1000)
        self.assertIsNotNone(result)

    def test_create_pixel_gray_graph_small_array(self):
        """Test pixel gray graph creation with small array."""
        mock_arr = Mock()
        mock_arr.shape = (5, 5)
        mock_arr.flatten = Mock(return_value=Mock())
        mock_arr.__getitem__ = Mock(return_value=Mock())

        result = create_pixel_gray_graph(mock_arr)

        self.assertEqual(result.h, 5)
        self.assertEqual(result.w, 5)
        self.assertIsNotNone(result)

    # Edge cases and error handling
    def test_capture_screen_resize_failure(self):
        """Test screen capture when resize fails."""
        cv2.resize.side_effect = Exception("Resize failed")
        PIL.ImageGrab.Image.fromarray.side_effect = Exception("PIL resize failed")

        result = capture_screen(scale=0.5)

        # Should return original size
        self.assertIsNotNone(result)

    def test_create_pixel_gray_graph_invalid_shape(self):
        """Test create_pixel_gray_graph with invalid array shape."""
        mock_arr = Mock()
        mock_arr.shape = (10,)  # 1D array

        with self.assertRaises(IndexError):
            create_pixel_gray_graph(mock_arr)

    def test_rgb_to_gray_non_array(self):
        """Test rgb_to_gray with non-array input."""
        with self.assertRaises(AttributeError):
            rgb_to_gray("not an array")

    # Performance tests
    def test_capture_screen_performance(self):
        """Test capture_screen performance."""
        start_time = time.time()
        for _ in range(10):
            capture_screen()
        end_time = time.time()
        duration = end_time - start_time
        # Should complete within reasonable time
        self.assertLess(duration, 2.0, "Screen capture too slow")

    def test_create_pixel_gray_graph_performance(self):
        """Test create_pixel_gray_graph performance with various sizes."""
        sizes = [(10, 10), (50, 50), (100, 100)]

        for h, w in sizes:
            mock_arr = Mock()
            mock_arr.shape = (h, w)
            mock_arr.flatten = Mock(return_value=Mock())
            mock_arr.__getitem__ = Mock(return_value=Mock())

            start_time = time.time()
            result = create_pixel_gray_graph(mock_arr)
            end_time = time.time()
            duration = end_time - start_time

            self.assertLess(duration, 1.0, f"Graph creation too slow for {h}x{w}")
            self.assertIsNotNone(result)

    # Real-world usage scenarios
    def test_typical_screen_capture_workflow(self):
        """Test typical screen capture workflow."""
        # Simulate capturing screen at different scales
        scales = [1.0, 0.75, 0.5, 0.25]

        for scale in scales:
            result = capture_screen(scale=scale)
            self.assertIsNotNone(result)

            # Create graph from captured screen
            graph = create_pixel_gray_graph(result)
            self.assertIsNotNone(graph)
            self.assertGreater(graph.h, 0)
            self.assertGreater(graph.w, 0)

    def test_screen_graph_integration(self):
        """Test integration of screen capture and graph creation."""
        # Capture screen
        screen_data = capture_screen(scale=0.25)
        self.assertIsNotNone(screen_data)

        # Convert to gray if needed
        gray_data = rgb_to_gray(screen_data)
        self.assertIsNotNone(gray_data)

        # Create graph
        graph = create_pixel_gray_graph(gray_data)
        self.assertIsNotNone(graph)
        self.assertIsInstance(graph.node_labels, list)
        self.assertGreater(len(graph.node_labels), 0)

        # Verify node structure
        first_node = graph.node_labels[0]
        expected_keys = ['id', 'type', 'behavior', 'x', 'y', 'energy', 'state',
                        'membrane_potential', 'threshold', 'refractory_timer',
                        'last_activation', 'plasticity_enabled', 'eligibility_trace',
                        'last_update', 'is_excitatory', 'I_syn', 'IEG_flag',
                        'plast_enabled', 'theta_burst_counter', 'v_dend']
        for key in expected_keys:
            self.assertIn(key, first_node)

    def test_different_monitor_configurations(self):
        """Test screen capture with different monitor configurations."""
        # Test with multiple monitors
        mss.mss.return_value.__enter__.return_value.monitors = [Mock(), Mock(), Mock()]

        result = capture_screen()
        self.assertIsNotNone(result)

        # Test with no monitors
        mss.mss.return_value.__enter__.return_value.monitors = []

        result = capture_screen()
        self.assertIsNotNone(result)

    def test_memory_efficient_large_screen_handling(self):
        """Test memory-efficient handling of large screens."""
        # Mock very large screen
        mock_arr = Mock()
        mock_arr.shape = (2160, 3840)  # 4K resolution
        mock_arr.flatten = Mock(return_value=Mock())
        mock_arr.__getitem__ = Mock(return_value=Mock(shape=(31, 31)))

        start_time = time.time()
        graph = create_pixel_gray_graph(mock_arr)
        end_time = time.time()
        duration = end_time - start_time

        # Should complete quickly due to sampling
        self.assertLess(duration, 2.0, "Large screen processing too slow")
        self.assertLessEqual(graph.h * graph.w, 1000, "Should sample down large screens")
        self.assertIsNotNone(graph)

    def test_grayscale_conversion_accuracy(self):
        """Test grayscale conversion accuracy."""
        # Create test RGB values
        test_rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])  # Red, Green, Blue
        test_rgb.shape = (1, 3, 3)

        # Mock the dot product to return expected grayscale values
        expected_gray = np.array([[0.2125*255, 0.7154*255, 0.0721*255]])
        np.dot.return_value = expected_gray

        result = rgb_to_gray(test_rgb)

        np.dot.assert_called_with(test_rgb[..., :3], [0.2125, 0.7154, 0.0721])
        self.assertEqual(result, expected_gray)

    def test_graph_node_properties(self):
        """Test that graph nodes have correct properties."""
        mock_arr = Mock()
        mock_arr.shape = (5, 5)
        mock_arr.flatten = Mock(return_value=np.array([0.1, 0.2, 0.3, 0.4, 0.5,
                                                       0.6, 0.7, 0.8, 0.9, 1.0,
                                                       0.0, 0.1, 0.2, 0.3, 0.4,
                                                       0.5, 0.6, 0.7, 0.8, 0.9,
                                                       1.0, 0.0, 0.1, 0.2, 0.3]))
        mock_arr.__getitem__ = Mock(return_value=Mock())

        graph = create_pixel_gray_graph(mock_arr)

        self.assertEqual(len(graph.node_labels), 25)  # 5x5 grid

        # Check first few nodes
        for i in range(min(5, len(graph.node_labels))):
            node = graph.node_labels[i]
            self.assertEqual(node['type'], 'sensory')
            self.assertEqual(node['behavior'], 'sensory')
            self.assertEqual(node['state'], 'active')
            self.assertFalse(node['plasticity_enabled'])
            self.assertTrue(node['is_excitatory'])
            self.assertGreaterEqual(node['energy'], 0.0)
            self.assertLessEqual(node['energy'], 1.0)

    def test_edge_cases_screen_sizes(self):
        """Test edge cases with unusual screen sizes."""
        test_sizes = [(1, 1), (2, 2), (1000, 1), (1, 1000)]

        for h, w in test_sizes:
            mock_arr = Mock()
            mock_arr.shape = (h, w)
            mock_arr.flatten = Mock(return_value=Mock())
            mock_arr.__getitem__ = Mock(return_value=Mock())

            graph = create_pixel_gray_graph(mock_arr)
            self.assertIsNotNone(graph)
            self.assertEqual(graph.h, min(h, 31))  # Limited by sampling
            self.assertEqual(graph.w, min(w, 31))


if __name__ == "__main__":
    unittest.main()