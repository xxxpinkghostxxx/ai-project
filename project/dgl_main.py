# Wrapper to redirect DGL main to PyG implementation
import sys
sys.path.append('.')

from project.pyg_main import main

if __name__ == '__main__':
    main()