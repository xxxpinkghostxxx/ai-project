import dearpygui.dearpygui as dpg
import faulthandler

faulthandler.enable()

def test_dearpygui():
    """
    Tests basic DearPyGui functionality.
    """
    try:
        print("Initializing DearPyGui...")
        dpg.create_context()
        dpg.create_viewport(title='DearPyGui Test', width=600, height=300)
        dpg.setup_dearpygui()

        with dpg.window(label="Example Window"):
            dpg.add_text("Hello, world")
            dpg.add_button(label="Save")
            dpg.add_input_text(label="string", default_value="Quick brown fox")
            dpg.add_slider_float(label="float", default_value=0.273, max_value=1)

        dpg.show_viewport()
        
        # Run one frame to ensure it initializes without crashing.
        dpg.render_dearpygui_frame()

        print("DearPyGui initialized successfully.")
        print("DearPyGui test PASSED.")
    except Exception as e:
        print(f"DearPyGui test FAILED: {e}")
    finally:
        if dpg.is_dearpygui_running():
            dpg.destroy_context()

if __name__ == "__main__":
    test_dearpygui()






