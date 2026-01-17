# Documentation Index

Welcome to the comprehensive documentation for the PyTorch Geometric Neural System project.

## Quick Navigation

### Core Documentation

- **[Main README](../README.md)** - Project overview, vision, and roadmap
- **[Setup Guide](README_SETUP.md)** - Installation and configuration instructions
- **[Getting Started Guide](GETTING_STARTED.md)** - Beginner's guide to the project
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to the project

### Technical Documentation

- **[API Documentation](technical/API_DOCUMENTATION/)** - Comprehensive API reference for all modules
- **[Troubleshooting Guide](TROUBLESHOOTING.md)** - Solutions for common issues and problems

## Documentation Structure

```text
docs/
├── README.md                 # This index file
├── GETTING_STARTED.md        # Beginner's guide
├── CONTRIBUTING.md           # Contribution guidelines
├── technical/
│   ├── API_DOCUMENTATION/    # Split API docs by module
│   │   ├── common_imports.md
│   │   ├── neural_system.md
│   │   ├── vision.md
│   │   ├── config.md
│   │   ├── utils.md
│   │   └── ui.md
│   ├── TROUBLESHOOTING.md    # Problem-solving guide
│   └── CONFIGURATION.md      # Configuration reference
├── tutorials/                # Learning materials
├── development/              # Development resources
│   ├── DOCUMENTATION_GUIDE.md # Documentation standards
│   └── MODERNIZATION_PLAN.md # Modernization plan
└── resources/                # Visual aids and examples
```

## Getting Started

### For New Users

1. Start with the [main README](../README.md) to understand the project
2. Follow the [setup guide](README_SETUP.md) for installation
3. Review the [troubleshooting guide](technical/TROUBLESHOOTING.md) if you encounter issues

### For Developers

1. Read the [contributing guide](CONTRIBUTING.md)
2. Study the [API documentation](technical/API_DOCUMENTATION/)
3. Review the [getting started guide](GETTING_STARTED.md) for setup instructions

### For Troubleshooting

1. Check the [troubleshooting guide](technical/TROUBLESHOOTING.md) first
2. Look at log files in the project directory
3. Review relevant sections in the [API documentation](technical/API_DOCUMENTATION/)

## Documentation Status

### ✅ Complete

- Main project documentation (README.md)
- Setup and installation guide
- API documentation for core modules (now split by module)
- Comprehensive troubleshooting guide
- Code analysis and cleanup report
- Getting started guide
- Documentation standards and guidelines

### 🔄 Ongoing

- Regular updates to match code changes
- Community feedback integration
- Performance optimization guides

## Feedback

If you find issues with the documentation or have suggestions for improvement:

1. Open an issue in the repository
2. Provide specific feedback about what needs clarification
3. Suggest additional topics that should be covered

## Quick Reference

### Common Commands

```bash
# Setup and activation
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
# Or use activation scripts:
config\environment\activate_env.bat  # Windows
source config/environment/activate_env.sh  # Linux/Mac

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

- `src/project/config.py` - Main configuration file (Python module)
- `config/requirements.txt` - Python dependencies
- `config/pyg_config.json` - Optional JSON configuration file
- `.vscode/settings.json` - VSCode configuration (if present)

### Important Log Files

- `pyg_system.log` - Main application logs
- Console output - Real-time status and errors

---

## Documentation Update Info

This documentation is maintained alongside the codebase. Last updated: 2025-12-09
