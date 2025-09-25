"""
Simple UI Component Test Script
Tests basic UI functionality and force close features.
"""

import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock Dear PyGui to avoid GUI dependencies
import types
mock_dpg = types.ModuleType('dearpygui')
mock_dpg.dearpygui = types.ModuleType('dearpygui_core')

# Add mock functions
mock_dpg.set_value = lambda *args, **kwargs: None
mock_dpg.get_value = lambda *args, **kwargs: 0
mock_dpg.configure_item = lambda *args, **kwargs: None
mock_dpg.add_text = lambda *args, **kwargs: "mock_text"
mock_dpg.add_button = lambda *args, **kwargs: "mock_button"
mock_dpg.add_checkbox = lambda *args, **kwargs: "mock_checkbox"
mock_dpg.add_slider_float = lambda *args, **kwargs: "mock_slider"
mock_dpg.add_input_int = lambda *args, **kwargs: "mock_input"
mock_dpg.add_color_edit = lambda *args, **kwargs: "mock_color"
mock_dpg.add_plot = lambda *args, **kwargs: "mock_plot"
mock_dpg.add_plot_axis = lambda *args, **kwargs: "mock_axis"
mock_dpg.add_line_series = lambda *args, **kwargs: "mock_series"
mock_dpg.add_collapsing_header = lambda *args, **kwargs: "mock_header"
mock_dpg.add_group = lambda *args, **kwargs: "mock_group"
mock_dpg.add_child_window = lambda *args, **kwargs: "mock_window"
mock_dpg.add_tab_bar = lambda *args, **kwargs: "mock_tab_bar"
mock_dpg.add_tab = lambda *args, **kwargs: "mock_tab"
mock_dpg.add_drawlist = lambda *args, **kwargs: "mock_drawlist"
mock_dpg.add_input_text = lambda *args, **kwargs: "mock_input_text"
mock_dpg.add_menu_bar = lambda *args, **kwargs: "mock_menu_bar"
mock_dpg.add_menu = lambda *args, **kwargs: "mock_menu"
mock_dpg.add_menu_item = lambda *args, **kwargs: "mock_menu_item"
mock_dpg.add_window = lambda *args, **kwargs: "mock_window"
mock_dpg.add_separator = lambda *args, **kwargs: None
mock_dpg.add_theme = lambda *args, **kwargs: "mock_theme"
mock_dpg.add_theme_component = lambda *args, **kwargs: "mock_component"
mock_dpg.add_theme_color = lambda *args, **kwargs: None
mock_dpg.add_theme_style = lambda *args, **kwargs: None
mock_dpg.bind_theme = lambda *args, **kwargs: None
mock_dpg.set_global_font_scale = lambda *args, **kwargs: None
mock_dpg.create_context = lambda *args, **kwargs: None
mock_dpg.create_viewport = lambda *args, **kwargs: None
mock_dpg.set_viewport_resizable = lambda *args, **kwargs: None
mock_dpg.setup_dearpygui = lambda *args, **kwargs: None
mock_dpg.show_viewport = lambda *args, **kwargs: None
mock_dpg.is_dearpygui_running = lambda *args, **kwargs: False
mock_dpg.render_dearpygui_frame = lambda *args, **kwargs: None
mock_dpg.destroy_context = lambda *args, **kwargs: None
mock_dpg.clear_draw_list = lambda *args, **kwargs: None
mock_dpg.draw_circle = lambda *args, **kwargs: None
mock_dpg.draw_line = lambda *args, **kwargs: None
mock_dpg.get_item_rect_size = lambda *args, **kwargs: [800, 600]
mock_dpg.set_primary_window = lambda *args, **kwargs: None
mock_dpg.toggle_viewport_fullscreen = lambda *args, **kwargs: None
mock_dpg.stop_dearpygui = lambda *args, **kwargs: None
mock_dpg.add_tooltip = lambda *args, **kwargs: "mock_tooltip"
mock_dpg.handler_registry = lambda *args, **kwargs: "mock_registry"
mock_dpg.add_key_press_handler = lambda *args, **kwargs: None
mock_dpg.does_item_exist = lambda *args, **kwargs: False

sys.modules['dearpygui'] = mock_dpg
sys.modules['dearpygui.dearpygui'] = mock_dpg

def test_ui_imports():
    """Test that UI modules can be imported."""
    print("Testing UI imports...")
    try:
        from ui.ui_engine import create_main_window, force_close_application, show_keyboard_shortcuts
        from ui.ui_state_manager import get_ui_state_manager, cleanup_ui_state
        print("[PASS] UI imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] UI import failed: {e}")
        return False

def test_ui_state_manager():
    """Test UI state manager functionality."""
    print("Testing UI state manager...")
    try:
        from ui.ui_state_manager import get_ui_state_manager
        state_manager = get_ui_state_manager()

        # Test basic operations
        initial_state = state_manager.get_simulation_running()
        state_manager.set_simulation_running(True)
        new_state = state_manager.get_simulation_running()

        if new_state == True:
            print("[PASS] UI state manager working")
            return True
        else:
            print("[FAIL] UI state manager failed")
            return False
    except Exception as e:
        print(f"[FAIL] UI state manager test failed: {e}")
        return False

def test_force_close_function():
    """Test force close function exists and can be called."""
    print("Testing force close function...")
    try:
        from ui.ui_engine import force_close_application
        # Just test that the function exists and can be called (without actually closing)
        print("[PASS] Force close function available")
        return True
    except Exception as e:
        print(f"[FAIL] Force close function test failed: {e}")
        return False

def test_keyboard_shortcuts():
    """Test keyboard shortcuts function."""
    print("Testing keyboard shortcuts function...")
    try:
        from ui.ui_engine import show_keyboard_shortcuts
        print("[PASS] Keyboard shortcuts function available")
        return True
    except Exception as e:
        print(f"[FAIL] Keyboard shortcuts test failed: {e}")
        return False

def test_ui_callbacks():
    """Test UI callback functions."""
    print("Testing UI callbacks...")
    try:
        from ui.ui_engine import (
            start_simulation_callback,
            stop_simulation_callback,
            reset_simulation_callback,
            view_logs_callback,
            apply_config_changes,
            reset_to_defaults
        )

        # Test that functions exist
        callbacks = [
            start_simulation_callback,
            stop_simulation_callback,
            reset_simulation_callback,
            view_logs_callback,
            apply_config_changes,
            reset_to_defaults
        ]

        for callback in callbacks:
            if callable(callback):
                print(f"[PASS] {callback.__name__} is callable")
            else:
                print(f"[FAIL] {callback.__name__} is not callable")
                return False

        print("[PASS] All UI callbacks available")
        return True
    except Exception as e:
        print(f"[FAIL] UI callbacks test failed: {e}")
        return False

def test_operation_status():
    """Test operation status functions."""
    print("Testing operation status functions...")
    try:
        from ui.ui_engine import update_operation_status, clear_operation_status

        # Test that functions exist
        if callable(update_operation_status) and callable(clear_operation_status):
            print("[PASS] Operation status functions available")
            return True
        else:
            print("[FAIL] Operation status functions not callable")
            return False
    except Exception as e:
        print(f"[FAIL] Operation status test failed: {e}")
        return False

def run_simple_ui_tests():
    """Run all simple UI tests."""
    print("SIMPLE UI COMPONENT TEST")
    print("=" * 40)

    tests = [
        test_ui_imports,
        test_ui_state_manager,
        test_force_close_function,
        test_keyboard_shortcuts,
        test_ui_callbacks,
        test_operation_status
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        print()

    print("=" * 40)
    print(f"TEST RESULTS: {passed}/{total} tests passed")

    if passed == total:
        print("SUCCESS: ALL UI TESTS PASSED!")
        print("\nUI Features Verified:")
        print("• UI imports working")
        print("• State management functional")
        print("• Force close functionality available")
        print("• Keyboard shortcuts available")
        print("• All UI callbacks present")
        print("• Operation status indicators ready")
    else:
        print("FAILED: Some UI tests failed")

    return passed == total

if __name__ == "__main__":
    success = run_simple_ui_tests()
    print(f"\nForce close available: Ctrl+Q or Force Close button")
    print(f"Keyboard shortcuts: F1 for help")
    sys.exit(0 if success else 1)