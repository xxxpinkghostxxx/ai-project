#!/usr/bin/env python3
"""
Advanced Duplicate Code Detection and Optimization Tool
Identifies dense, duplicate, and similar code patterns for consolidation.
"""

import os
import ast
import time
from dataclasses import dataclass
from typing import List, Dict, Any
from collections import defaultdict

try:
    import pycode_similar
    PYCODE_SIMILAR_AVAILABLE = True
except ImportError:
    PYCODE_SIMILAR_AVAILABLE = False
    print("⚠️  pycode-similar not available, using basic analysis")


@dataclass
class CodePattern:
    """Represents a code pattern found in the codebase."""
    pattern_type: str
    file_path: str
    line_start: int
    line_end: int
    content: str
    similarity_score: float
    occurrences: int
    optimization_potential: str


class DuplicateCodeDetector:
    """Advanced duplicate code detection and analysis."""
    
    def __init__(self, project_dir: str = "."):
        self.project_dir = project_dir
        self.patterns: List[CodePattern] = []
        self.file_contents: Dict[str, str] = {}
        self.function_ast: Dict[str, List[ast.FunctionDef]] = {}
        self.class_ast: Dict[str, List[ast.ClassDef]] = {}
        
    def analyze_codebase(self) -> Dict[str, Any]:
        """Comprehensive analysis of the entire codebase."""
        print("🔍 Starting comprehensive duplicate code analysis...")
        
        # Load all Python files
        self._load_python_files()
        
        # Analyze different types of duplicates
        results = {
            'duplicate_functions': self._find_duplicate_functions(),
            'similar_code_blocks': self._find_similar_code_blocks(),
            'repeated_patterns': self._find_repeated_patterns(),
            'dense_functions': self._find_dense_functions(),
            'redundant_imports': self._find_redundant_imports(),
            'duplicate_strings': self._find_duplicate_strings(),
            'similar_classes': self._find_similar_classes(),
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
        
        return results
    
    def _load_python_files(self):
        """Load all Python files in the project."""
        print("📁 Loading Python files...")
        
        for root, dirs, files in os.walk(self.project_dir):
            # Skip virtual environment and cache directories
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'logs']]
            
            for file in files:
                if file.endswith('.py') and not file.startswith('.'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        self.file_contents[file_path] = content
                        
                        # Parse AST
                        tree = ast.parse(content)
                        self.function_ast[file_path] = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                        self.class_ast[file_path] = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                        
                    except Exception as e:
                        print(f"⚠️  Could not parse {file_path}: {e}")
    
    def _find_duplicate_functions(self) -> List[CodePattern]:
        """Find duplicate or very similar functions."""
        print("🔍 Analyzing duplicate functions...")
        
        duplicates = []
        function_signatures = defaultdict(list)
        
        for file_path, functions in self.function_ast.items():
            for func in functions:
                # Create function signature
                signature = self._create_function_signature(func)
                function_signatures[signature].append((file_path, func))
        
        # Find functions with same signature
        for signature, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                for file_path, func in occurrences:
                    duplicates.append(CodePattern(
                        pattern_type="duplicate_function",
                        file_path=file_path,
                        line_start=func.lineno,
                        line_end=func.end_lineno if hasattr(func, 'end_lineno') else func.lineno,
                        content=signature,
                        similarity_score=1.0,
                        occurrences=len(occurrences),
                        optimization_potential="Extract to common utility function"
                    ))
        
        return duplicates
    
    def _find_similar_code_blocks(self) -> List[CodePattern]:
        """Find similar code blocks using AST analysis."""
        print("🔍 Analyzing similar code blocks...")
        
        similar_blocks = []
        
        if not PYCODE_SIMILAR_AVAILABLE:
            # Fallback to basic AST-based comparison
            return self._find_similar_code_blocks_basic()
        
        # Compare all files pairwise
        file_paths = list(self.file_contents.keys())
        for i, file1 in enumerate(file_paths):
            for file2 in file_paths[i+1:]:
                try:
                    # Use pycode-similar for detailed comparison
                    result = pycode_similar.detect([self.file_contents[file1], self.file_contents[file2]])
                    
                    for match in result[0][1]:  # Similarities found
                        if match.ratio > 0.7:  # High similarity threshold
                            similar_blocks.append(CodePattern(
                                pattern_type="similar_code_block",
                                file_path=file1,
                                line_start=match.lineno,
                                line_end=match.end_lineno,
                                content=match.content[:100] + "..." if len(match.content) > 100 else match.content,
                                similarity_score=match.ratio,
                                occurrences=2,
                                optimization_potential="Extract common logic to shared module"
                            ))
                except Exception as e:
                    print(f"⚠️  Could not compare {file1} and {file2}: {e}")
        
        return similar_blocks
    
    def _find_similar_code_blocks_basic(self) -> List[CodePattern]:
        """Basic AST-based similarity detection."""
        similar_blocks = []
        
        # Compare function signatures across files
        function_signatures = defaultdict(list)
        
        for file_path, functions in self.function_ast.items():
            for func in functions:
                signature = self._create_function_signature(func)
                function_signatures[signature].append((file_path, func))
        
        # Find functions with similar signatures
        for signature, occurrences in function_signatures.items():
            if len(occurrences) > 1:
                for file_path, func in occurrences:
                    similar_blocks.append(CodePattern(
                        pattern_type="similar_code_block",
                        file_path=file_path,
                        line_start=func.lineno,
                        line_end=func.end_lineno if hasattr(func, 'end_lineno') else func.lineno,
                        content=signature,
                        similarity_score=0.8,
                        occurrences=len(occurrences),
                        optimization_potential="Extract common logic to shared module"
                    ))
        
        return similar_blocks
    
    def _find_repeated_patterns(self) -> List[CodePattern]:
        """Find repeated code patterns within files."""
        print("🔍 Analyzing repeated patterns...")
        
        patterns = []
        
        for file_path, content in self.file_contents.items():
            lines = content.split('\n')
            
            # Look for repeated line sequences
            for i in range(len(lines) - 3):  # Minimum 4 lines for a pattern
                for length in range(4, min(20, len(lines) - i)):  # Pattern length 4-20 lines
                    pattern = lines[i:i+length]
                    pattern_text = '\n'.join(pattern)
                    
                    # Count occurrences of this pattern
                    occurrences = 0
                    for j in range(len(lines) - length + 1):
                        if j != i and lines[j:j+length] == pattern:
                            occurrences += 1
                    
                    if occurrences >= 2:  # Found at least 2 occurrences
                        patterns.append(CodePattern(
                            pattern_type="repeated_pattern",
                            file_path=file_path,
                            line_start=i+1,
                            line_end=i+length,
                            content=pattern_text[:200] + "..." if len(pattern_text) > 200 else pattern_text,
                            similarity_score=1.0,
                            occurrences=occurrences + 1,
                            optimization_potential="Extract to helper function"
                        ))
        
        return patterns
    
    def _find_dense_functions(self) -> List[CodePattern]:
        """Find functions that are too dense or complex."""
        print("🔍 Analyzing dense functions...")
        
        dense_functions = []
        
        for file_path, functions in self.function_ast.items():
            for func in functions:
                # Calculate complexity metrics
                complexity = self._calculate_function_complexity(func)
                line_count = func.end_lineno - func.lineno + 1 if hasattr(func, 'end_lineno') else 1
                
                # Check if function is too dense
                if complexity > 10 or line_count > 50:
                    dense_functions.append(CodePattern(
                        pattern_type="dense_function",
                        file_path=file_path,
                        line_start=func.lineno,
                        line_end=func.end_lineno if hasattr(func, 'end_lineno') else func.lineno,
                        content=f"Function: {func.name} (Complexity: {complexity}, Lines: {line_count})",
                        similarity_score=complexity / 20.0,  # Normalize to 0-1
                        occurrences=1,
                        optimization_potential="Break down into smaller functions"
                    ))
        
        return dense_functions
    
    def _find_redundant_imports(self) -> List[CodePattern]:
        """Find redundant or unused imports."""
        print("🔍 Analyzing redundant imports...")
        
        redundant_imports = []
        
        for file_path, content in self.file_contents.items():
            try:
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
                for line_num, import_name, alias_name in imports:
                    if alias_name not in used_names and not import_name.startswith('_'):
                        redundant_imports.append(CodePattern(
                            pattern_type="redundant_import",
                            file_path=file_path,
                            line_start=line_num,
                            line_end=line_num,
                            content=f"import {import_name}",
                            similarity_score=1.0,
                            occurrences=1,
                            optimization_potential="Remove unused import"
                        ))
            except Exception as e:
                print(f"⚠️  Could not analyze imports in {file_path}: {e}")
        
        return redundant_imports
    
    def _find_duplicate_strings(self) -> List[CodePattern]:
        """Find duplicate string literals."""
        print("🔍 Analyzing duplicate strings...")
        
        string_usage = defaultdict(list)
        duplicate_strings = []
        
        for file_path, content in self.file_contents.items():
            try:
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Constant) and isinstance(node.value, str):
                        if len(node.value) > 10:  # Only consider strings longer than 10 chars
                            string_usage[node.value].append((file_path, node.lineno))
            except Exception as e:
                print(f"⚠️  Could not analyze strings in {file_path}: {e}")
        
        # Find strings used in multiple places
        for string_value, locations in string_usage.items():
            if len(locations) > 1:
                for file_path, line_num in locations:
                    duplicate_strings.append(CodePattern(
                        pattern_type="duplicate_string",
                        file_path=file_path,
                        line_start=line_num,
                        line_end=line_num,
                        content=string_value[:100] + "..." if len(string_value) > 100 else string_value,
                        similarity_score=1.0,
                        occurrences=len(locations),
                        optimization_potential="Extract to constants file"
                    ))
        
        return duplicate_strings
    
    def _find_similar_classes(self) -> List[CodePattern]:
        """Find similar or duplicate classes."""
        print("🔍 Analyzing similar classes...")
        
        similar_classes = []
        class_signatures = defaultdict(list)
        
        for file_path, classes in self.class_ast.items():
            for cls in classes:
                signature = self._create_class_signature(cls)
                class_signatures[signature].append((file_path, cls))
        
        # Find classes with similar signatures
        for signature, occurrences in class_signatures.items():
            if len(occurrences) > 1:
                for file_path, cls in occurrences:
                    similar_classes.append(CodePattern(
                        pattern_type="similar_class",
                        file_path=file_path,
                        line_start=cls.lineno,
                        line_end=cls.end_lineno if hasattr(cls, 'end_lineno') else cls.lineno,
                        content=signature,
                        similarity_score=1.0,
                        occurrences=len(occurrences),
                        optimization_potential="Create base class or mixin"
                    ))
        
        return similar_classes
    
    def _identify_optimization_opportunities(self) -> List[CodePattern]:
        """Identify general optimization opportunities."""
        print("🔍 Identifying optimization opportunities...")
        
        opportunities = []
        
        for file_path, content in self.file_contents.items():
            lines = content.split('\n')
            
            # Look for common optimization patterns
            for i, line in enumerate(lines, 1):
                # Long lines
                if len(line) > 120:
                    opportunities.append(CodePattern(
                        pattern_type="long_line",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        content=line[:100] + "...",
                        similarity_score=len(line) / 200.0,
                        occurrences=1,
                        optimization_potential="Break into multiple lines"
                    ))
                
                # Complex list comprehensions
                if '[' in line and 'for' in line and 'if' in line and line.count('[') > 1:
                    opportunities.append(CodePattern(
                        pattern_type="complex_comprehension",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        content=line.strip(),
                        similarity_score=0.8,
                        occurrences=1,
                        optimization_potential="Extract to separate function"
                    ))
                
                # Deeply nested conditions
                if line.count('if') > 2 or line.count('and') > 3 or line.count('or') > 3:
                    opportunities.append(CodePattern(
                        pattern_type="complex_condition",
                        file_path=file_path,
                        line_start=i,
                        line_end=i,
                        content=line.strip(),
                        similarity_score=0.7,
                        occurrences=1,
                        optimization_potential="Simplify with helper functions"
                    ))
        
        return opportunities
    
    def _create_function_signature(self, func: ast.FunctionDef) -> str:
        """Create a signature for function comparison."""
        args = [arg.arg for arg in func.args.args]
        return f"{func.name}({', '.join(args)})"
    
    def _create_class_signature(self, cls: ast.ClassDef) -> str:
        """Create a signature for class comparison."""
        methods = [node.name for node in cls.body if isinstance(node, ast.FunctionDef)]
        return f"{cls.name}: {', '.join(methods)}"
    
    def _calculate_function_complexity(self, func: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def generate_optimization_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive optimization report."""
        report = ["# Duplicate Code Detection and Optimization Report", "=" * 60, ""]
        
        total_issues = sum(len(patterns) for patterns in results.values())
        report.append(f"📊 **Total Issues Found**: {total_issues}")
        report.append("")
        
        # Summary by category
        for category, patterns in results.items():
            if patterns:
                report.append(f"## {category.replace('_', ' ').title()}")
                report.append(f"**Count**: {len(patterns)}")
                report.append("")
                
                # Group by file
                by_file = defaultdict(list)
                for pattern in patterns:
                    by_file[pattern.file_path].append(pattern)
                
                for file_path, file_patterns in by_file.items():
                    report.append(f"### {file_path}")
                    for pattern in file_patterns[:5]:  # Show first 5 per file
                        report.append(f"- **Line {pattern.line_start}**: {pattern.optimization_potential}")
                        report.append(f"  - Score: {pattern.similarity_score:.2f}")
                        report.append(f"  - Occurrences: {pattern.occurrences}")
                        report.append("")
                
                report.append("")
        
        # Top optimization opportunities
        report.append("## 🎯 Top Optimization Opportunities")
        report.append("")
        
        all_patterns = []
        for patterns in results.values():
            all_patterns.extend(patterns)
        
        # Sort by optimization potential
        all_patterns.sort(key=lambda x: x.similarity_score, reverse=True)
        
        for i, pattern in enumerate(all_patterns[:10], 1):
            report.append(f"{i}. **{pattern.file_path}** (Line {pattern.line_start})")
            report.append(f"   - Type: {pattern.pattern_type}")
            report.append(f"   - Optimization: {pattern.optimization_potential}")
            report.append(f"   - Score: {pattern.similarity_score:.2f}")
            report.append("")
        
        return "\n".join(report)
    
    def create_optimization_plan(self, results: Dict[str, Any]) -> str:
        """Create a step-by-step optimization plan."""
        plan = ["# Code Optimization Plan", "=" * 40, ""]
        
        # Prioritize by impact
        high_impact = []
        medium_impact = []
        low_impact = []
        
        for patterns in results.values():
            for pattern in patterns:
                if pattern.similarity_score > 0.8:
                    high_impact.append(pattern)
                elif pattern.similarity_score > 0.5:
                    medium_impact.append(pattern)
                else:
                    low_impact.append(pattern)
        
        plan.append("## 🚀 High Impact Optimizations (Do First)")
        plan.append("")
        for i, pattern in enumerate(high_impact[:5], 1):
            plan.append(f"{i}. **{pattern.file_path}** - {pattern.optimization_potential}")
            plan.append(f"   - Lines: {pattern.line_start}-{pattern.line_end}")
            plan.append(f"   - Impact: {pattern.similarity_score:.2f}")
            plan.append("")
        
        plan.append("## ⚡ Medium Impact Optimizations")
        plan.append("")
        for i, pattern in enumerate(medium_impact[:5], 1):
            plan.append(f"{i}. **{pattern.file_path}** - {pattern.optimization_potential}")
            plan.append(f"   - Lines: {pattern.line_start}-{pattern.line_end}")
            plan.append("")
        
        plan.append("## 🔧 Low Impact Optimizations (Do Last)")
        plan.append("")
        for i, pattern in enumerate(low_impact[:5], 1):
            plan.append(f"{i}. **{pattern.file_path}** - {pattern.optimization_potential}")
            plan.append("")
        
        return "\n".join(plan)


def main():
    """Main function to run duplicate code detection."""
    print("🚀 Advanced Duplicate Code Detection Tool")
    print("=" * 50)
    
    detector = DuplicateCodeDetector()
    
    # Run analysis
    start_time = time.time()
    results = detector.analyze_codebase()
    end_time = time.time()
    
    print(f"\n✅ Analysis completed in {end_time - start_time:.2f} seconds")
    
    # Generate reports
    report = detector.generate_optimization_report(results)
    plan = detector.create_optimization_plan(results)
    
    # Save reports
    with open('duplicate_code_report.md', 'w') as f:
        f.write(report)
    
    with open('optimization_plan.md', 'w') as f:
        f.write(plan)
    
    print("\n📄 Reports generated:")
    print("- duplicate_code_report.md")
    print("- optimization_plan.md")
    
    # Print summary
    total_issues = sum(len(patterns) for patterns in results.values())
    print(f"\n📊 Summary: {total_issues} optimization opportunities found")
    
    return results


if __name__ == "__main__":
    main()
