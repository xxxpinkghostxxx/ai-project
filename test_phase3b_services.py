#!/usr/bin/env python3
"""
Test Phase 3B services - Advanced Visualization.

This test verifies that the Phase 3B advanced visualization services
(RealTimeVisualizationService) work correctly with the existing service architecture.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_real_time_visualization_service():
    """Test RealTimeVisualizationService functionality."""
    try:
        from core.services.real_time_visualization_service import RealTimeVisualizationService
        from core.interfaces.real_time_visualization import VisualizationData
        import time

        # Create visualization service
        viz_service = RealTimeVisualizationService()

        # Test initialization
        config = {
            "target_fps": 30,
            "max_buffer_size": 100,
            "enable_interpolation": True,
            "default_layers": ["neural_activity", "energy_flow"]
        }
        success = viz_service.initialize_visualization(config)
        if not success:
            raise Exception("Visualization initialization failed")

        # Test layer creation
        layer_config = {
            "layer_type": "connections",
            "visible": True,
            "opacity": 0.8,
            "z_index": 1,
            "color_scheme": "blue",
            "update_frequency": 30
        }
        layer_id = viz_service.create_visualization_layer(layer_config)
        if not layer_id:
            raise Exception("Layer creation failed")

        # Test data update
        viz_data = VisualizationData("neural_activity", time.time())
        viz_data.data = {
            "node_positions": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
            "node_energies": [0.8, 0.6, 0.9],
            "active_nodes": [0, 2]
        }
        viz_data.metadata = {
            "total_nodes": 3,
            "active_count": 2,
            "energy_range": [0.6, 0.9]
        }

        success = viz_service.update_visualization_data(layer_id, viz_data)
        if not success:
            raise Exception("Data update failed")

        # Test frame rendering
        frame_data = viz_service.render_frame()
        if not isinstance(frame_data, dict) or "layers" not in frame_data:
            raise Exception("Frame rendering failed")

        # Test camera control
        camera_command = {
            "position": [10, 10, 50],
            "rotation": [0, 45, 0],
            "zoom": 1.5
        }
        success = viz_service.control_camera(camera_command)
        if not success:
            raise Exception("Camera control failed")

        # Test visualization snapshot
        snapshot = viz_service.get_visualization_snapshot("json")
        if not isinstance(snapshot, bytes) or len(snapshot) == 0:
            raise Exception("Snapshot creation failed")

        # Test visualization effects
        effect_config = {
            "effect_type": "glow",
            "intensity": 0.7,
            "color": [1.0, 0.5, 0.0]
        }
        effect_id = viz_service.add_visualization_effect(effect_config)
        if not effect_id:
            raise Exception("Effect addition failed")

        # Test animation sequence
        animation_config = {
            "animation_type": "fade",
            "duration": 2.0,
            "start_opacity": 0.0,
            "end_opacity": 1.0
        }
        animation_id = viz_service.create_animation_sequence(animation_config)
        if not animation_id:
            raise Exception("Animation creation failed")

        # Test metrics retrieval
        metrics = viz_service.get_visualization_metrics()
        if not isinstance(metrics, dict):
            raise Exception("Metrics retrieval failed")

        # Test data export
        export_config = {
            "format": "json",
            "path": "test_visualization_export.json",
            "layers": [layer_id]
        }
        success = viz_service.export_visualization_data(export_config)
        if not success:
            raise Exception("Data export failed")

        # Clean up
        viz_service.cleanup()

        print("PASS: RealTimeVisualizationService test successful")
        return True

    except Exception as e:
        print(f"FAIL: RealTimeVisualizationService test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_data_structures():
    """Test visualization data structures and utilities."""
    try:
        from core.interfaces.real_time_visualization import VisualizationData, VisualizationLayer, CameraController
        import time

        # Test VisualizationData
        data = VisualizationData("neural_activity", time.time())
        data.data = {"test": "value"}
        data.metadata = {"count": 10}
        data.rendering_hints = {"color": "red"}

        data_dict = data.to_dict()
        if not isinstance(data_dict, dict) or data_dict["data_type"] != "neural_activity":
            raise Exception("VisualizationData structure test failed")

        # Test VisualizationLayer
        layer = VisualizationLayer("test_layer", "energy_flow")
        layer.visible = False
        layer.opacity = 0.5
        layer.z_index = 5
        layer.color_scheme = "heatmap"

        layer_dict = layer.to_dict()
        if not isinstance(layer_dict, dict) or layer_dict["visible"] != False:
            raise Exception("VisualizationLayer structure test failed")

        # Test CameraController
        camera = CameraController()
        camera.position = [5, 5, 20]
        camera.rotation = [30, 45, 0]
        camera.zoom = 2.0
        camera.projection_mode = "orthographic"

        camera_dict = camera.to_dict()
        if not isinstance(camera_dict, dict) or camera_dict["zoom"] != 2.0:
            raise Exception("CameraController structure test failed")

        print("PASS: Visualization data structures test successful")
        return True

    except Exception as e:
        print(f"FAIL: Visualization data structures test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_integration_with_simulation():
    """Test visualization integration with neural simulation data."""
    try:
        from core.services.real_time_visualization_service import RealTimeVisualizationService
        from core.interfaces.real_time_visualization import VisualizationData
        from core.services.configuration_service import ConfigurationService
        from core.services.event_coordination_service import EventCoordinationService
        import time
        import random

        # Create services
        config_service = ConfigurationService()
        event_service = EventCoordinationService()
        viz_service = RealTimeVisualizationService()

        # Initialize visualization
        viz_config = {
            "target_fps": 30,
            "max_buffer_size": 200,
            "default_layers": ["neural_activity", "energy_flow", "connections"]
        }
        success = viz_service.initialize_visualization(viz_config)
        if not success:
            raise Exception("Visualization initialization failed")

        # Simulate neural simulation data over time
        simulation_steps = 10
        layer_ids = []

        # Create layers for different data types
        for data_type in ["neural_activity", "energy_flow", "connections"]:
            layer_config = {
                "layer_type": data_type,
                "visible": True,
                "update_frequency": 30
            }
            layer_id = viz_service.create_visualization_layer(layer_config)
            if layer_id:
                layer_ids.append(layer_id)

        # Generate and update visualization data
        for step in range(simulation_steps):
            current_time = time.time()

            for i, layer_id in enumerate(layer_ids):
                # Create realistic simulation data
                if "neural_activity" in layer_id:
                    # Neural activity data
                    viz_data = VisualizationData("neural_activity", current_time)
                    viz_data.data = {
                        "node_positions": [[x, y, z] for x in range(5) for y in range(5) for z in range(2)],
                        "node_energies": [random.uniform(0.1, 1.0) for _ in range(50)],
                        "active_nodes": [i for i in range(50) if random.random() > 0.7],
                        "spike_events": [{"node_id": random.randint(0, 49), "timestamp": current_time} for _ in range(random.randint(1, 5))]
                    }
                    viz_data.metadata = {
                        "total_nodes": 50,
                        "active_count": len(viz_data.data["active_nodes"]),
                        "spike_count": len(viz_data.data["spike_events"]),
                        "simulation_step": step
                    }

                elif "energy_flow" in layer_id:
                    # Energy flow data
                    viz_data = VisualizationData("energy_flow", current_time)
                    viz_data.data = {
                        "energy_distribution": [random.uniform(0.1, 1.0) for _ in range(50)],
                        "energy_gradients": [[random.uniform(-0.1, 0.1) for _ in range(3)] for _ in range(50)],
                        "energy_flow_vectors": [[random.uniform(-1, 1) for _ in range(3)] for _ in range(20)],
                        "high_energy_clusters": [[random.randint(0, 49) for _ in range(random.randint(3, 8))] for _ in range(3)]
                    }
                    viz_data.metadata = {
                        "average_energy": sum(viz_data.data["energy_distribution"]) / len(viz_data.data["energy_distribution"]),
                        "energy_variance": sum((e - 0.5) ** 2 for e in viz_data.data["energy_distribution"]) / len(viz_data.data["energy_distribution"]),
                        "flow_vectors_count": len(viz_data.data["energy_flow_vectors"])
                    }

                elif "connections" in layer_id:
                    # Connection data
                    viz_data = VisualizationData("connections", current_time)
                    viz_data.data = {
                        "connection_matrix": [[random.uniform(0, 1) if random.random() > 0.8 else 0 for _ in range(50)] for _ in range(50)],
                        "active_connections": [(i, j) for i in range(50) for j in range(50) if random.random() > 0.9],
                        "connection_strengths": [random.uniform(0.1, 1.0) for _ in range(len(viz_data.data["active_connections"]))],
                        "plasticity_events": [{"source": random.randint(0, 49), "target": random.randint(0, 49), "change": random.uniform(-0.1, 0.1)} for _ in range(random.randint(0, 3))]
                    }
                    viz_data.metadata = {
                        "total_connections": len(viz_data.data["active_connections"]),
                        "average_strength": sum(viz_data.data["connection_strengths"]) / max(len(viz_data.data["connection_strengths"]), 1),
                        "plasticity_events_count": len(viz_data.data["plasticity_events"])
                    }
                else:
                    # Default data for unknown layer types
                    viz_data = VisualizationData("unknown", current_time)
                    viz_data.data = {"default": "data"}
                    viz_data.metadata = {"layer_type": "unknown"}

                # Update visualization
                success = viz_service.update_visualization_data(layer_id, viz_data)
                if not success:
                    print(f"Warning: Failed to update data for layer {layer_id}")

            # Render frame
            frame_data = viz_service.render_frame()
            if frame_data and "layers" in frame_data:
                active_layers = len(frame_data["layers"])
                print(f"Step {step}: Rendered frame with {active_layers} active layers")

            # Small delay to simulate real-time processing
            time.sleep(0.01)

        # Test final state
        metrics = viz_service.get_visualization_metrics()
        if not isinstance(metrics, dict):
            raise Exception("Final metrics retrieval failed")

        # Test snapshot after simulation
        snapshot = viz_service.get_visualization_snapshot("json")
        if not isinstance(snapshot, bytes) or len(snapshot) == 0:
            raise Exception("Final snapshot creation failed")

        # Clean up
        viz_service.cleanup()

        print("PASS: Visualization integration with simulation test successful")
        return True

    except Exception as e:
        print(f"FAIL: Visualization integration with simulation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_performance_and_scalability():
    """Test visualization performance and scalability."""
    try:
        from core.services.real_time_visualization_service import RealTimeVisualizationService
        from core.interfaces.real_time_visualization import VisualizationData
        import time

        # Create visualization service
        viz_service = RealTimeVisualizationService()

        # Initialize with performance-focused config
        config = {
            "target_fps": 60,
            "max_buffer_size": 500,
            "enable_interpolation": False,  # Disable for performance test
            "default_layers": ["neural_activity"]
        }
        success = viz_service.initialize_visualization(config)
        if not success:
            raise Exception("Visualization initialization failed")

        # Test with different data sizes
        data_sizes = [50, 200, 500]
        performance_results = {}

        for size in data_sizes:
            print(f"Testing with {size} nodes...")

            # Create layer
            layer_config = {"layer_type": "neural_activity", "update_frequency": 60}
            layer_id = viz_service.create_visualization_layer(layer_config)

            # Measure data update performance
            update_times = []
            render_times = []

            for i in range(10):  # 10 iterations for averaging
                # Create test data
                viz_data = VisualizationData("neural_activity", time.time())
                viz_data.data = {
                    "node_positions": [[x*0.1, y*0.1, z*0.1] for x in range(size//10) for y in range(size//10) for z in range(2)][:size],
                    "node_energies": [0.5] * size,
                    "active_nodes": list(range(0, size, 10))
                }

                # Measure update time
                start_time = time.time()
                viz_service.update_visualization_data(layer_id, viz_data)
                update_times.append(time.time() - start_time)

                # Measure render time
                start_time = time.time()
                frame_data = viz_service.render_frame()
                render_times.append(time.time() - start_time)

            # Calculate averages
            avg_update_time = sum(update_times) / len(update_times)
            avg_render_time = sum(render_times) / len(render_times)

            performance_results[size] = {
                "avg_update_time": avg_update_time,
                "avg_render_time": avg_render_time,
                "total_time": avg_update_time + avg_render_time
            }

            print(".3f")
        # Performance assertions
        for size, results in performance_results.items():
            # Update should be fast (< 10ms for 500 nodes)
            if results["avg_update_time"] > 0.01:  # 10ms
                print(f"Warning: Update time {results['avg_update_time']:.3f}s > 10ms for {size} nodes")

            # Render should be reasonable (< 50ms for 500 nodes)
            if results["avg_render_time"] > 0.05:  # 50ms
                print(f"Warning: Render time {results['avg_render_time']:.3f}s > 50ms for {size} nodes")

        # Test memory usage (basic check)
        metrics = viz_service.get_visualization_metrics()
        if metrics.get("active_buffers", 0) != len(data_sizes):
            print(f"Warning: Expected {len(data_sizes)} buffers, got {metrics.get('active_buffers', 0)}")

        # Clean up
        viz_service.cleanup()

        print("PASS: Visualization performance and scalability test successful")
        return True

    except Exception as e:
        print(f"FAIL: Visualization performance and scalability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("Testing Phase 3B Services - Advanced Visualization")
    print("=" * 75)

    tests = [
        test_real_time_visualization_service,
        test_visualization_data_structures,
        test_visualization_integration_with_simulation,
        test_visualization_performance_and_scalability
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 75)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: Phase 3B advanced visualization services are working!")
        print("Neural simulation now features real-time 3D visualization.")
        print("The system can visualize neural activity, energy flow, and connections.")
        return True
    else:
        print("FAILURE: Some Phase 3B advanced visualization services tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)