#!/usr/bin/env python3
"""
Script to analyze and detect unused imports across Python files in the project.
"""

import os
import ast
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import re


class ImportAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze import usage in Python files."""

    def __init__(self):
        self.imports = {}  # name -> (module, alias, line_number)
        self.used_names = set()
        self.current_scope = [set()]  # Stack of scopes for nested functions/classes

    def visit_Import(self, node):
        """Handle 'import module' statements."""
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = (alias.name, alias.asname, node.lineno)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Handle 'from module import name' statements."""
        module = node.module or ''
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports[name] = (f"{module}.{alias.name}", alias.asname, node.lineno)
        self.generic_visit(node)

    def visit_Name(self, node):
        """Track usage of imported names."""
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Track usage of module.attribute patterns."""
        if isinstance(node.ctx, ast.Load):
            # Handle cases like torch.tensor, np.array
            if isinstance(node.value, ast.Name):
                full_name = f"{node.value.id}.{node.attr}"
                self.used_names.add(full_name)
        self.generic_visit(node)

    def enter_scope(self):
        """Enter a new scope (function, class, etc.)."""
        self.current_scope.append(set())

    def exit_scope(self):
        """Exit current scope."""
        if self.current_scope:
            self.current_scope.pop()

    def visit_FunctionDef(self, node):
        self.enter_scope()
        self.generic_visit(node)
        self.exit_scope()

    def visit_AsyncFunctionDef(self, node):
        self.enter_scope()
        self.generic_visit(node)
        self.exit_scope()

    def visit_ClassDef(self, node):
        self.enter_scope()
        self.generic_visit(node)
        self.exit_scope()


def analyze_file(filepath: str) -> Dict[str, List[Tuple[str, int, bool]]]:
    """
    Analyze a single Python file for unused imports.

    Returns:
        Dict mapping import names to [(module, line_number, is_used), ...]
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        return {f"ERROR_READING_FILE_{filepath}": [(str(e), 0, False)]}

    try:
        tree = ast.parse(content, filepath)
    except SyntaxError as e:
        return {f"SYNTAX_ERROR_{filepath}": [(str(e), e.lineno or 0, False)]}

    analyzer = ImportAnalyzer()
    analyzer.visit(tree)

    # Check which imports are actually used
    unused_imports = {}
    for import_name, (module, alias, line_no) in analyzer.imports.items():
        is_used = False

        # Check if the import name is used directly
        if import_name in analyzer.used_names:
            is_used = True
        # Check if it's used as module.attribute (e.g., torch.tensor)
        elif f"{import_name}." in " ".join(analyzer.used_names):
            is_used = True
        # Special handling for common patterns
        else:
            # Check if any used name starts with the import name + "."
            for used_name in analyzer.used_names:
                if used_name.startswith(f"{import_name}."):
                    is_used = True
                    break

        unused_imports[import_name] = [(module, line_no, is_used)]

    return unused_imports


def find_python_files(directory: str) -> List[str]:
    """Find all Python files in the given directory recursively."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip common directories that shouldn't be analyzed
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'venv', '.venv', 'node_modules'}]

        for file in files:
            if file.endswith('.py') and not file.startswith('.'):
                python_files.append(os.path.join(root, file))

    return python_files


def main():
    """Main function to analyze all Python files in the project."""
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "."  # Current directory

    print(f"Analyzing Python files in: {directory}")
    python_files = find_python_files(directory)

    if not python_files:
        print("No Python files found!")
        return

    print(f"Found {len(python_files)} Python files")

    # Analyze all files
    all_unused = defaultdict(list)
    files_with_unused = []

    for filepath in python_files:
        unused_imports = analyze_file(filepath)

        file_unused = []
        for import_name, details in unused_imports.items():
            if not any(used for _, _, used in details):
                file_unused.append((import_name, details[0][1]))  # (name, line_number)
                all_unused[import_name].append((filepath, details[0][1]))

        if file_unused:
            files_with_unused.append((filepath, file_unused))

    # Print results
    print(f"\n{'='*80}")
    print("UNUSED IMPORTS ANALYSIS RESULTS")
    print(f"{'='*80}")

    if not files_with_unused:
        print("No unused imports found!")
        return

    print(f"\nSUMMARY:")
    print(f"Found unused imports in {len(files_with_unused)} files")
    print(f"Total unique unused import names: {len(all_unused)}")

    # Show most frequent unused imports
    print("\nMOST FREQUENT UNUSED IMPORTS:")
    sorted_unused = sorted(all_unused.items(), key=lambda x: len(x[1]), reverse=True)
    for import_name, occurrences in sorted_unused[:20]:  # Top 20
        print(f"  {import_name}: used in {len(occurrences)} files")

    # Show details per file
    print("\nDETAILED RESULTS PER FILE:")
    for filepath, unused in files_with_unused:
        print(f"\n{filepath}:")
        for import_name, line_no in sorted(unused, key=lambda x: x[1]):
            print(f"  Line {line_no}: {import_name}")

    # Save results to file
    results_file = "unused_imports_report.txt"
    with open(results_file, 'w') as f:
        f.write("UNUSED IMPORTS ANALYSIS REPORT\n")
        f.write("="*50 + "\n\n")

        f.write(f"Summary:\n")
        f.write(f"Files with unused imports: {len(files_with_unused)}\n")
        f.write(f"Total unique unused import names: {len(all_unused)}\n\n")

        f.write("Most frequent unused imports:\n")
        for import_name, occurrences in sorted_unused[:20]:
            f.write(f"  {import_name}: {len(occurrences)} files\n")

        f.write("\nDetailed results:\n")
        for filepath, unused in files_with_unused:
            f.write(f"\n{filepath}:\n")
            for import_name, line_no in sorted(unused, key=lambda x: x[1]):
                f.write(f"  Line {line_no}: {import_name}\n")

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()