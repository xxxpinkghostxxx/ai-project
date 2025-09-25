import torch
import faulthandler

faulthandler.enable()

def test_pytorch():
    """
    Tests basic PyTorch functionality.
    """
    try:
        print("Initializing PyTorch tensor...")
        x = torch.rand(5, 3)
        print("PyTorch tensor initialized successfully:")
        print(x)
        print("PyTorch test PASSED.")
    except Exception as e:
        print(f"PyTorch test FAILED: {e}")

if __name__ == "__main__":
    test_pytorch()