"""
NASA Power of Ten Rules Code Analyzer
Static analysis tool to ensure compliance with NASA's safety-critical coding standards.
"""

import ast
import os
import re

from dataclasses import dataclass


@dataclass
class AnalysisResult:
    """Result of code analysis."""
    rule_name: str
    severity: str  # 'error', 'warning', 'info'
    message: str
    line_number: int
    file_path: str
    suggestion: str = ""


class NASACodeAnalyzer:
    """Analyzer for NASA Power of Ten rules compliance."""
    
    def __init__(self):
        self.rules = {
            'rule_1': self._check_function_length,
            'rule_2': self._check_loop_bounds,
            'rule_3': self._check_dynamic_allocation,
            'rule_4': self._check_control_flow,
            'rule_5': self._check_data_scope,
            'rule_6': self._check_return_values,
            'rule_7': self._check_preprocessor_directives,
            'rule_8': self._check_pointer_usage,
            'rule_9': self._check_assertions,
            'rule_10': self._check_complexity
        }
    
    def analyze_file(self, file_path: str) -> List[AnalysisResult]:
        """Analyze a single Python file for NASA rule compliance."""
        results = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Run all rule checks
            for rule_name, check_func in self.rules.items():
                rule_results = check_func(tree, content, file_path)
                results.extend(rule_results)
                
        except Exception as e:
            results.append(AnalysisResult(
                rule_name='parser_error',
                severity='error',
                message=f"Failed to parse file: {e}",
                line_number=0,
                file_path=file_path,
                suggestion="Fix syntax errors before analysis"
            ))
        
        return results
    
    def _check_function_length(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 1: Functions should not exceed 60 lines."""
        results = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line
                function_length = end_line - start_line + 1
                
                if function_length > 60:
                    results.append(AnalysisResult(
                        rule_name='rule_1_function_length',
                        severity='error',
                        message=f"Function '{node.name}' is {function_length} lines long (max 60)",
                        line_number=start_line,
                        file_path=file_path,
                        suggestion="Break down into smaller functions"
                    ))
        
        return results
    
    def _check_loop_bounds(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 2: All loops must have fixed upper bounds."""
        results = []
        lines = content.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.While):
                # Check if while loop has explicit bounds
                has_bounds = False
                if isinstance(node.test, ast.Compare):
                    # Look for patterns like 'i < max_iterations'
                    if (isinstance(node.test.left, ast.Name) and 
                        isinstance(node.test.ops[0], (ast.Lt, ast.LtE))):
                        has_bounds = True
                
                if not has_bounds:
                    results.append(AnalysisResult(
                        rule_name='rule_2_loop_bounds',
                        severity='error',
                        message="While loop without explicit upper bound",
                        line_number=node.lineno,
                        file_path=file_path,
                        suggestion="Add explicit iteration limit (e.g., 'i < max_iterations')"
                    ))
        
        return results
    
    def _check_dynamic_allocation(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 3: Avoid dynamic memory allocation."""
        results = []
        lines = content.split('\n')
        
        # Look for dynamic allocation patterns
        dynamic_patterns = [
            r'\.append\(',
            r'\.extend\(',
            r'\[.*for.*in.*\]',  # List comprehensions
            r'list\(\)',
            r'dict\(\)',
            r'set\(\)'
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in dynamic_patterns:
                if re.search(pattern, line):
                    results.append(AnalysisResult(
                        rule_name='rule_3_dynamic_allocation',
                        severity='warning',
                        message=f"Potential dynamic allocation: {line.strip()[:50]}...",
                        line_number=i,
                        file_path=file_path,
                        suggestion="Consider using static allocation or pre-allocated buffers"
                    ))
        
        return results
    
    def _check_control_flow(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 4: Avoid complex control flow."""
        results = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for complex control structures
                has_nested_loops = False
                has_recursion = False
                
                for child in ast.walk(node):
                    if isinstance(child, ast.For) or isinstance(child, ast.While):
                        # Check for nested loops
                        for grandchild in ast.walk(child):
                            if isinstance(grandchild, (ast.For, ast.While)) and grandchild != child:
                                has_nested_loops = True
                                break
                    
                    # Check for recursion
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                        if child.func.id == node.name:
                            has_recursion = True
                
                if has_nested_loops:
                    results.append(AnalysisResult(
                        rule_name='rule_4_control_flow',
                        severity='warning',
                        message=f"Function '{node.name}' has nested loops",
                        line_number=node.lineno,
                        file_path=file_path,
                        suggestion="Simplify control flow by breaking into smaller functions"
                    ))
                
                if has_recursion:
                    results.append(AnalysisResult(
                        rule_name='rule_4_control_flow',
                        severity='error',
                        message=f"Function '{node.name}' uses recursion",
                        line_number=node.lineno,
                        file_path=file_path,
                        suggestion="Replace recursion with iterative approach"
                    ))
        
        return results
    
    def _check_data_scope(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 5: Restrict scope of data."""
        results = []
        
        # Check for global variables
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                results.append(AnalysisResult(
                    rule_name='rule_5_data_scope',
                    severity='warning',
                    message=f"Global variables declared: {', '.join(node.names)}",
                    line_number=node.lineno,
                    file_path=file_path,
                    suggestion="Minimize global state, use local variables or class attributes"
                ))
        
        return results
    
    def _check_return_values(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 6: Check return values of functions."""
        results = []
        lines = content.split('\n')
        
        # Look for function calls that might not check return values
        for i, line in enumerate(lines, 1):
            # Look for patterns like 'function_call()' without assignment
            if re.search(r'\w+\([^)]*\)$', line.strip()) and not re.search(r'^\s*#', line):
                if not re.search(r'=\s*\w+\(', line) and not re.search(r'if\s+\w+\(', line):
                    results.append(AnalysisResult(
                        rule_name='rule_6_return_values',
                        severity='warning',
                        message=f"Function call may not check return value: {line.strip()[:50]}...",
                        line_number=i,
                        file_path=file_path,
                        suggestion="Check return value or assign to variable"
                    ))
        
        return results
    
    def _check_preprocessor_directives(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 7: Minimize preprocessor directives."""
        results = []
        lines = content.split('\n')
        
        # Look for complex conditional compilation
        for i, line in enumerate(lines, 1):
            if re.search(r'if.*hasattr.*and.*hasattr', line):
                results.append(AnalysisResult(
                    rule_name='rule_7_preprocessor',
                    severity='info',
                    message="Complex conditional check detected",
                    line_number=i,
                    file_path=file_path,
                    suggestion="Simplify with helper methods or early returns"
                ))
        
        return results
    
    def _check_pointer_usage(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 8: Limit pointer usage."""
        results = []
        
        # Check for complex object references
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                # Check for chained attribute access (a.b.c.d)
                if isinstance(node.value, ast.Attribute):
                    results.append(AnalysisResult(
                        rule_name='rule_8_pointer_usage',
                        severity='warning',
                        message="Chained attribute access detected",
                        line_number=node.lineno,
                        file_path=file_path,
                        suggestion="Simplify object references"
                    ))
        
        return results
    
    def _check_assertions(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 9: Use assertions."""
        results = []
        
        function_count = 0
        assertion_count = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_count += 1
                for child in ast.walk(node):
                    if isinstance(child, ast.Assert):
                        assertion_count += 1
        
        if function_count > 0 and assertion_count < function_count * 2:
            results.append(AnalysisResult(
                rule_name='rule_9_assertions',
                severity='warning',
                message=f"Low assertion density: {assertion_count} assertions for {function_count} functions",
                line_number=1,
                file_path=file_path,
                suggestion="Add more assertions (target: 2+ per function)"
            ))
        
        return results
    
    def _check_complexity(self, tree: ast.AST, content: str, file_path: str) -> List[AnalysisResult]:
        """Rule 10: Limit complexity."""
        results = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > 10:
                    results.append(AnalysisResult(
                        rule_name='rule_10_complexity',
                        severity='error',
                        message=f"Function '{node.name}' has high complexity: {complexity}",
                        line_number=node.lineno,
                        file_path=file_path,
                        suggestion="Reduce complexity by breaking into smaller functions"
                    ))
        
        return results
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def analyze_directory(self, directory: str) -> Dict[str, List[AnalysisResult]]:
        """Analyze all Python files in a directory."""
        results = {}
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    results[file_path] = self.analyze_file(file_path)
        
        return results
    
    def generate_report(self, results: Dict[str, List[AnalysisResult]]) -> str:
        """Generate a comprehensive analysis report."""
        report = ["NASA Power of Ten Rules Analysis Report", "=" * 50, ""]
        
        total_files = len(results)
        total_issues = sum(len(file_results) for file_results in results.values())
        
        # Summary
        report.append(f"Files analyzed: {total_files}")
        report.append(f"Total issues found: {total_issues}")
        report.append("")
        
        # Issues by severity
        error_count = sum(1 for file_results in results.values() 
                         for result in file_results if result.severity == 'error')
        warning_count = sum(1 for file_results in results.values() 
                           for result in file_results if result.severity == 'warning')
        info_count = sum(1 for file_results in results.values() 
                        for result in file_results if result.severity == 'info')
        
        report.append(f"Errors: {error_count}")
        report.append(f"Warnings: {warning_count}")
        report.append(f"Info: {info_count}")
        report.append("")
        
        # Detailed results
        for file_path, file_results in results.items():
            if file_results:
                report.append(f"File: {file_path}")
                report.append("-" * len(file_path))
                
                for result in file_results:
                    report.append(f"  {result.severity.upper()}: {result.rule_name}")
                    report.append(f"    Line {result.line_number}: {result.message}")
                    if result.suggestion:
                        report.append(f"    Suggestion: {result.suggestion}")
                    report.append("")
        
        return "\n".join(report)


def main():
    """Main function to run the analyzer."""
    analyzer = NASACodeAnalyzer()
    
    # Analyze current directory
    results = analyzer.analyze_directory('.')
    
    # Generate and print report
    report = analyzer.generate_report(results)
    print(report)
    
    # Save report to file
    with open('nasa_analysis_report.txt', 'w') as f:
        f.write(report)
    
    print(f"\nReport saved to: nasa_analysis_report.txt")


if __name__ == "__main__":
    main()
