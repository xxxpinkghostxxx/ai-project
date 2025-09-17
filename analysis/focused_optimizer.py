#!/usr/bin/env python3
"""
Focused Code Optimizer
Identifies and fixes specific duplicate patterns and redundant code.
"""

import os
import re
import ast

from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Result of an optimization operation."""
    file_path: str
    line_number: int
    pattern_type: str
    original_code: str
    optimized_code: str
    description: str


class FocusedOptimizer:
    """Focused optimizer for specific code patterns."""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = project_dir
        self.optimizations: List[OptimizationResult] = []
    
    def optimize_codebase(self) -> List[OptimizationResult]:
        """Run all optimization passes."""
        print("Starting focused code optimization...")
        
        # Find and fix redundant None checks
        self._fix_redundant_none_checks()
        
        # Find and fix duplicate logging patterns
        self._fix_duplicate_logging()
        
        # Find and fix unused imports
        self._fix_unused_imports()
        
        # Find and fix duplicate string constants
        self._fix_duplicate_strings()
        
        # Find and fix similar functions
        self._fix_similar_functions()
        
        return self.optimizations
    
    def _fix_redundant_none_checks(self):
        """Fix redundant None checks like 'if x and x is not None'."""
        print("Fixing redundant None checks...")
        
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._process_file_for_none_checks(file_path)
    
    def _process_file_for_none_checks(self, file_path: str):
        """Process a single file for redundant None checks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            modified = False
            
            for i, line in enumerate(lines):
                # Look for redundant None checks
                if 'is not None and' in line and 'is not None' in line:
                    # Count occurrences of 'is not None'
                    none_count = line.count('is not None')
                    if none_count > 1:
                        # Simplify redundant checks
                        simplified = re.sub(r'\s+is not None\s+and\s+', ' and ', line)
                        simplified = re.sub(r'and\s+is not None', '', simplified)
                        
                        if simplified != line:
                            lines[i] = simplified
                            modified = True
                            
                            self.optimizations.append(OptimizationResult(
                                file_path=file_path,
                                line_number=i + 1,
                                pattern_type="redundant_none_check",
                                original_code=line.strip(),
                                optimized_code=simplified.strip(),
                                description="Removed redundant None checks"
                            ))
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"Fixed redundant None checks in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _fix_duplicate_logging(self):
        """Fix duplicate logging patterns."""
        print("Fixing duplicate logging patterns...")
        
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._process_file_for_logging(file_path)
    
    def _process_file_for_logging(self, file_path: str):
        """Process a single file for duplicate logging patterns."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.split('\n')
            modified = False
            
            # Look for duplicate logging patterns
            for i, line in enumerate(lines):
                if 'logging.info(' in line and 'logging.info(' in content:
                    # Count similar logging.info calls
                    similar_logs = [l for l in lines if 'logging.info(' in l and l.strip() != line.strip()]
                    if len(similar_logs) > 1:
                        # This could be optimized by creating a logging utility
                        pass
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"Fixed logging patterns in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _fix_unused_imports(self):
        """Fix unused imports."""
        print("Fixing unused imports...")
        
        for root, dirs, files in os.walk(self.project_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    self._process_file_for_imports(file_path)
    
    def _process_file_for_imports(self, file_path: str):
        """Process a single file for unused imports."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            used_names = set()
            
            # Collect all imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append((node.lineno, alias.name, alias.asname or alias.name))
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        imports.append((node.lineno, alias.name, alias.asname or alias.name))
            
            # Collect all used names
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used_names.add(node.attr)
            
            # Find unused imports
            lines = content.split('\n')
            modified = False
            
            for line_num, import_name, alias_name in imports:
                if alias_name not in used_names and not import_name.startswith('_'):
                    # Remove unused import
                    if line_num <= len(lines):
                        original_line = lines[line_num - 1]
                        if original_line.strip():
                            lines[line_num - 1] = ""
                            modified = True
                            
                            self.optimizations.append(OptimizationResult(
                                file_path=file_path,
                                line_number=line_num,
                                pattern_type="unused_import",
                                original_code=original_line.strip(),
                                optimized_code="",
                                description=f"Removed unused import: {import_name}"
                            ))
            
            if modified:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"Fixed unused imports in {file_path}")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    def _fix_duplicate_strings(self):
        """Fix duplicate string constants."""
        print("Fixing duplicate string constants...")
        
        # This would require more complex analysis to identify duplicate strings
        # and extract them to a constants file
        pass
    
    def _fix_similar_functions(self):
        """Fix similar functions that could be consolidated."""
        print("Fixing similar functions...")
        
        # This would require AST analysis to find similar function patterns
        pass
    
    def generate_report(self) -> str:
        """Generate optimization report."""
        report = ["# Focused Code Optimization Report", "=" * 50, ""]
        
        if not self.optimizations:
            report.append("No optimizations applied.")
            return "\n".join(report)
        
        report.append(f"Total optimizations applied: {len(self.optimizations)}")
        report.append("")
        
        # Group by pattern type
        by_type = {}
        for opt in self.optimizations:
            if opt.pattern_type not in by_type:
                by_type[opt.pattern_type] = []
            by_type[opt.pattern_type].append(opt)
        
        for pattern_type, optimizations in by_type.items():
            report.append(f"## {pattern_type.replace('_', ' ').title()}")
            report.append(f"Count: {len(optimizations)}")
            report.append("")
            
            for opt in optimizations:
                report.append(f"**{opt.file_path}** (Line {opt.line_number})")
                report.append(f"- Description: {opt.description}")
                report.append(f"- Original: `{opt.original_code}`")
                if opt.optimized_code:
                    report.append(f"- Optimized: `{opt.optimized_code}`")
                report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run the optimizer."""
    optimizer = FocusedOptimizer()
    optimizations = optimizer.optimize_codebase()
    
    # Generate and save report
    report = optimizer.generate_report()
    print(report)
    
    with open('focused_optimization_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nOptimization report saved to: focused_optimization_report.txt")


if __name__ == "__main__":
    main()