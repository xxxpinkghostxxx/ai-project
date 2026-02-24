# Documentation Index

Welcome to the documentation for the PyTorch Geometric Neural System project.

## Quick Navigation

### Core Documentation

- **[Main README](../README.md)** - Project overview, vision, and roadmap
- **[Getting Started](GETTING_STARTED.md)** - Installation, configuration, and first run
- **[Architecture](ARCHITECTURE.md)** - System design, node types, data flow, and hybrid engine
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project

### Technical Reference

- **[API Documentation](technical/API_DOCUMENTATION/)** - API reference by module
- **[Performance](PERFORMANCE.md)** - Benchmarks, bottlenecks, and optimization guide
- **[Troubleshooting](technical/TROUBLESHOOTING.md)** - Solutions for common issues

### Project Planning

- **[Roadmap](ROADMAP.md)** - Completed work, current priorities, and planned features
- **[Documentation Guide](development/DOCUMENTATION_GUIDE.md)** - Documentation standards

## Documentation Structure

```text
docs/
├── README.md                   # This index file
├── GETTING_STARTED.md          # Installation and first run
├── ARCHITECTURE.md             # System design and data flow
├── PERFORMANCE.md              # Benchmarks and optimization
├── ROADMAP.md                  # Development roadmap
├── CONTRIBUTING.md             # Contribution guidelines
├── technical/
│   ├── API_DOCUMENTATION/      # API docs by module
│   │   ├── common_imports.md
│   │   ├── neural_system.md
│   │   ├── vision.md
│   │   ├── config.md
│   │   ├── utils.md
│   │   └── ui.md
│   ├── API_DOCUMENTATION.md    # API overview
│   ├── TROUBLESHOOTING.md      # Problem-solving guide
│   └── myrig.md                # Hardware reference
├── development/
│   └── DOCUMENTATION_GUIDE.md  # Documentation standards
└── archive/                    # Historical docs and milestone reports
```

## Quick Reference

### Common Commands

```bash
# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r config/requirements.txt

# Run the application
python -m project.pyg_main

# Type checking
mypy --ignore-missing-imports src/project/

# Linting
pylint --disable=import-error src/project/
```

### Key Configuration Files

- `src/project/config.py` - Main configuration (Python module)
- `config/requirements.txt` - Python dependencies
- `src/project/pyg_config.json` - Runtime configuration (JSON, used by the application)

### Important Log Files

- `pyg_system.log` - Main application logs
- Console output - Real-time status and errors
