# File Header Template — Design Spec

## Goal

Standardize all Python source files in `src/project/` with a consistent three-section `#` comment header at the top of each file. Remove all inline `#` comments from the code body. Preserve `"""docstrings"""` unchanged.

## Decisions Made

| Decision | Choice |
|----------|--------|
| Section 1 detail level | Full outline: signature + params + returns + brief internal flow |
| Empty sections | Keep section with explicit `# None` marker |
| Inline comment removal scope | Remove all `#` comments; keep all `"""docstrings"""` |
| Rule line placement | Once, right after header ends, before imports |
| Section divider style | `# ===...===` (equals-sign separators) |
| Header vs docstring | Header is `#` comment block; module docstring preserved separately (trimmed to 1-2 lines) |

## Template

Every `.py` file in `src/project/` (excluding `__init__.py` files) will follow this structure:

```python
# =============================================================================
# CODE STRUCTURE
# =============================================================================
#
# Module-level Constants:
#   CONSTANT_NAME = value
#     Brief description if non-obvious
#
# Module-level Functions:
#   function_name(param: type, ...) -> return_type    [@ti.kernel if applicable]
#     Brief description of purpose and internal flow
#
# Classes:
#   ClassName:
#     class_attribute = value
#       Brief description if non-obvious
#
#     method_name(self, param: type, ...) -> return_type
#       Brief description of purpose and internal flow
#
#     property_name -> type
#       Brief description
#
# =============================================================================
# TODOS
# =============================================================================
#
# None
#   (or listed items, e.g.:)
# - [critical] Description of critical TODO
# - [hanging] Description of hanging/stale TODO
#
# =============================================================================
# KNOWN BUGS
# =============================================================================
#
# None
#   (or listed items, e.g.:)
# - Description of known bug and any workaround
#
# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.

"""Short 1-2 line module docstring."""

import ...
```

## Rules

### Section 1: CODE STRUCTURE

- Lists all module-level functions, classes, methods, and properties in **source order**
- Each entry includes: name, full signature (params + types + return type), and a brief description
- For functions with non-trivial internal flow, include a one-line summary of the flow (e.g., `grid map -> DNA transfer -> sync -> death/spawn -> clamp -> metrics`)
- Taichi kernel decorators (`@ti.kernel`, `@ti.func`) are annotated next to the function name
- Module-level constants are listed only if they are part of the public API or are non-obvious
- Nested/inner functions are not listed unless they are significant

### Section 2: TODOS

- Lists all critical and hanging TODOs that were previously scattered as inline comments
- New TODOs discovered during development go here, not inline
- Each TODO is prefixed with a severity tag: `[critical]`, `[hanging]`, `[minor]`
- If no TODOs exist, the section contains `# None`

### Section 3: KNOWN BUGS

- Documents any known bugs, edge cases, or discovered issues
- Each entry describes the bug and any known workaround
- If no bugs are known, the section contains `# None`

### Rule Line

- Placed once, immediately after the KNOWN BUGS section, before any blank line or module docstring
- Exact text: `# DO NOT ADD PROJECT NOTES BELOW — all notes go in the file header above.`

### Module Docstring

- The existing `"""..."""` module docstring is preserved but trimmed to 1-2 lines
- Detailed architecture notes, bit layouts, and encoding docs that currently live in module docstrings are either:
  - Moved to Section 1 entries where they describe specific functions/classes
  - Kept in function-level docstrings where they're most relevant
  - Dropped if they duplicate information already in function docstrings

### Inline Comment Removal

- All `#` comments in the code body are removed
- This includes: section separators (`# ===`), explanatory comments, parameter annotations, TODO/NOTE/FIXME markers
- Critical "why" information from removed comments is either:
  - Incorporated into the Section 1 description for the relevant function
  - Added to Section 3 if it describes a known limitation or bug
  - Added to the function's `"""docstring"""` if it's essential context for that function
- `"""Docstrings"""` at function, method, and class level are **not** touched

## Scope

### Files to refactor (31 files in `src/project/`):

**Phase 1 — Start file:**
1. `src/project/system/taichi_engine.py` (1,466 lines, heaviest file)

**Phase 2 — Core system files:**
2. `src/project/main.py` (1,186 lines)
3. `src/project/config.py` (138 lines)
4. `src/project/system/state_manager.py`
5. `src/project/system/global_storage.py`

**Phase 3 — UI files:**
6. `src/project/ui/modern_main_window.py` (1,973 lines)
7. `src/project/ui/modern_config_panel.py` (916 lines)

**Phase 4 — Workspace files:**
8. `src/project/workspace/workspace_system.py`
9. `src/project/workspace/workspace_node.py`
10. `src/project/workspace/renderer.py`
11. `src/project/workspace/visualization.py`
12. `src/project/workspace/visualization_integration.py`
13. `src/project/workspace/mapping.py`
14. `src/project/workspace/pixel_shading.py`
15. `src/project/workspace/config.py`

**Phase 5 — Utility files:**
16. `src/project/utils/config_manager.py`
17. `src/project/utils/error_handler.py`
18. `src/project/utils/performance_utils.py`
19. `src/project/utils/security_utils.py`
20. `src/project/utils/shutdown_utils.py`

**Phase 6 — I/O and visualization files:**
21. `src/project/audio_capture.py`
22. `src/project/audio_output.py`
23. `src/project/optimized_capture.py`
24. `src/project/vision.py`
25. `src/project/visualization/taichi_gui_manager.py`

**Excluded:**
- All `__init__.py` files (minimal content, no meaningful structure to map)
- All `tests/` files (test files follow their own conventions)
- All `examples/` files

## Validation

After each file is refactored:
1. No `#` comments remain in the code body (below the rule line), except inside string literals
2. All three header sections are present with correct `# ===` separators
3. The rule line appears exactly once, before imports
4. The module docstring is preserved (trimmed)
5. All function/class/method `"""docstrings"""` are unchanged
6. The file still imports and runs correctly (`python -c "import project.system.taichi_engine"` or equivalent)
