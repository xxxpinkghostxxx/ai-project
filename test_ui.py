"""
Simple UI test to verify DearPyGui is working correctly.
"""

import dearpygui.dearpygui as dpg

def test_ui():
    """Test basic DearPyGui functionality."""
    print("Creating DearPyGui context...")
    dpg.create_context()
    
    print("Creating test window...")
    with dpg.window(label="Test Window", tag="test_window"):
        dpg.add_text("Hello, this is a test window!")
        dpg.add_button(label="Test Button", callback=lambda: print("Button clicked!"))
    
    print("Creating viewport...")
    dpg.create_viewport(title="Test Window", width=400, height=300)
    dpg.setup_dearpygui()
    
    print("Showing viewport...")
    dpg.show_viewport()
    dpg.set_primary_window("test_window", True)
    
    print("Starting DearPyGui main loop...")
    try:
        dpg.start_dearpygui()
    except Exception as e:
        print(f"Error in DearPyGui: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        dpg.destroy_context()

if __name__ == "__main__":
    test_ui()
