"""
Launch script for the AI Neural System UI.
This script ensures the UI launches correctly and is visible.
"""

import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize services before importing other modules
try:
    from service_initializer import initialize_services
    initialize_services()
    print("Services initialized successfully")
except (ImportError, AttributeError, RuntimeError) as e:
    print(f"Service initialization failed: {e}")
    print("Continuing with fallback initialization...")
except Exception as e:
    print(f"Unexpected service initialization error: {e}")
    print("Continuing with fallback initialization...")

def launch_ui():
    """Launch the UI with proper error handling and visibility."""
    try:
        print("Launching AI Neural System...")
        print("=" * 50)
        
        # Import and run the UI
        import ui_engine
        
        print("UI Engine imported successfully")
        print("Starting UI window...")
        
        # Run the UI
        ui_engine.run_ui()
        
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        print(f"Error launching UI: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Unexpected error launching UI: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative launch method
        print("\nTrying alternative launch method...")
        try:
            import dearpygui.dearpygui as dpg
            
            # Create a simple test window first
            dpg.create_context()
            
            with dpg.window(label="AI Neural System", tag="main_window"):
                dpg.add_text("AI Neural System is starting...")
                dpg.add_button(label="Launch Full System", callback=lambda: launch_full_system())
            
            dpg.create_viewport(title="AI Neural System", width=800, height=600)
            dpg.setup_dearpygui()
            dpg.show_viewport()
            dpg.set_primary_window("main_window", True)
            
            print("Alternative UI launched successfully")
            print("Click 'Launch Full System' to start the complete system")
            
            dpg.start_dearpygui()
            
        except (ImportError, AttributeError, RuntimeError, OSError) as e2:
            print(f"Alternative launch also failed: {e2}")
            input("Press Enter to exit...")
        except Exception as e2:
            print(f"Unexpected error in alternative launch: {e2}")
            input("Press Enter to exit...")

def launch_full_system():
    """Launch the full neural system."""
    try:
        import dearpygui.dearpygui as dpg
        dpg.destroy_context()
        
        # Now launch the full system
        import ui_engine
        ui_engine.run_ui()
        
    except (ImportError, AttributeError, RuntimeError, OSError) as e:
        print(f"Error launching full system: {e}")
    except Exception as e:
        print(f"Unexpected error launching full system: {e}")

if __name__ == "__main__":
    launch_ui()
