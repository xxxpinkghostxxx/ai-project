"""
input_validation.py

Comprehensive input validation utilities for secure data handling.
Provides validation, sanitization, and security checks for user input.
"""

import re
import os
import json
import hashlib
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging


class ValidationSeverity(Enum):
    """Validation severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    sanitized_value: Any
    errors: List[str]
    warnings: List[str]
    severity: ValidationSeverity = ValidationSeverity.LOW


class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    def __init__(self):
        """Initialize the input validator."""
        self.logger = logging.getLogger(__name__)
        
        # Common patterns for validation
        self.patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'alphanumeric': re.compile(r'^[a-zA-Z0-9]+$'),
            'safe_string': re.compile(r'^[a-zA-Z0-9\s\-_.,!?]+$'),
            'filename': re.compile(r'^[a-zA-Z0-9._-]+$'),
            'path_traversal': re.compile(r'\.\./|\.\.\\|\.\.%2f|\.\.%5c', re.IGNORECASE),
            'sql_injection': re.compile(r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION)\b)', re.IGNORECASE),
            'xss_patterns': re.compile(r'<script|javascript:|on\w+\s*=', re.IGNORECASE)
        }
        
        # Dangerous characters and patterns
        self.dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`', '$', '(', ')']
        self.dangerous_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'getattr\s*\(',
            r'setattr\s*\(',
            r'delattr\s*\(',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\('
        ]
    
    def validate_string(self, value: Any, 
                       min_length: int = 0, 
                       max_length: int = 1000,
                       allow_empty: bool = True,
                       pattern: Optional[str] = None,
                       sanitize: bool = True) -> ValidationResult:
        """
        Validate and sanitize string input.
        
        Args:
            value: Input value to validate
            min_length: Minimum allowed length
            max_length: Maximum allowed length
            allow_empty: Whether empty strings are allowed
            pattern: Regex pattern name to validate against
            sanitize: Whether to sanitize the input
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW
        
        # Type validation
        if not isinstance(value, str):
            if value is None and allow_empty:
                return ValidationResult(True, "", [], [])
            errors.append("Input must be a string")
            severity = ValidationSeverity.MEDIUM
        
        if errors:
            return ValidationResult(False, None, errors, warnings, severity)
        
        # Length validation
        if len(value) < min_length:
            errors.append(f"Input too short (minimum {min_length} characters)")
            severity = ValidationSeverity.MEDIUM
        
        if len(value) > max_length:
            errors.append(f"Input too long (maximum {max_length} characters)")
            severity = ValidationSeverity.MEDIUM
        
        # Empty string validation
        if not allow_empty and not value.strip():
            errors.append("Empty input not allowed")
            severity = ValidationSeverity.MEDIUM
        
        # Pattern validation
        if pattern and pattern in self.patterns:
            if not self.patterns[pattern].match(value):
                errors.append(f"Input does not match required pattern: {pattern}")
                severity = ValidationSeverity.MEDIUM
        
        # Security checks
        security_issues = self._check_security_threats(value)
        if security_issues:
            errors.extend(security_issues)
            severity = ValidationSeverity.HIGH
        
        # Sanitization
        sanitized_value = value
        if sanitize and not errors:
            sanitized_value = self._sanitize_string(value)
            if sanitized_value != value:
                warnings.append("Input was sanitized")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=sanitized_value,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_number(self, value: Any, 
                       min_value: Optional[float] = None,
                       max_value: Optional[float] = None,
                       integer_only: bool = False) -> ValidationResult:
        """
        Validate numeric input.
        
        Args:
            value: Input value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            integer_only: Whether only integers are allowed
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW
        
        # Type conversion and validation
        try:
            if integer_only:
                num_value = int(value)
            else:
                num_value = float(value)
        except (ValueError, TypeError):
            errors.append(f"Input must be a {'integer' if integer_only else 'number'}")
            severity = ValidationSeverity.MEDIUM
            return ValidationResult(False, None, errors, warnings, severity)
        
        # Range validation
        if min_value is not None and num_value < min_value:
            errors.append(f"Value too small (minimum {min_value})")
            severity = ValidationSeverity.MEDIUM
        
        if max_value is not None and num_value > max_value:
            errors.append(f"Value too large (maximum {max_value})")
            severity = ValidationSeverity.MEDIUM
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=num_value,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_file_path(self, path: str, 
                          allowed_extensions: Optional[List[str]] = None,
                          max_size: Optional[int] = None,
                          check_existence: bool = True) -> ValidationResult:
        """
        Validate file path input.
        
        Args:
            path: File path to validate
            allowed_extensions: List of allowed file extensions
            max_size: Maximum file size in bytes
            check_existence: Whether to check if file exists
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW
        
        # Basic string validation
        string_result = self.validate_string(path, max_length=500, sanitize=True)
        if not string_result.is_valid:
            return string_result
        
        path = string_result.sanitized_value
        
        # Path traversal check
        if self.patterns['path_traversal'].search(path):
            errors.append("Path traversal detected")
            severity = ValidationSeverity.HIGH
        
        # Filename validation
        filename = os.path.basename(path)
        if not self.patterns['filename'].match(filename):
            errors.append("Invalid filename characters")
            severity = ValidationSeverity.MEDIUM
        
        # Extension validation
        if allowed_extensions:
            _, ext = os.path.splitext(filename)
            if ext.lower() not in [e.lower() for e in allowed_extensions]:
                errors.append(f"File extension not allowed: {ext}")
                severity = ValidationSeverity.MEDIUM
        
        # File existence check
        if check_existence and not os.path.exists(path):
            errors.append("File does not exist")
            severity = ValidationSeverity.MEDIUM
        
        # File size check
        if max_size and os.path.exists(path):
            try:
                file_size = os.path.getsize(path)
                if file_size > max_size:
                    errors.append(f"File too large (maximum {max_size} bytes)")
                    severity = ValidationSeverity.MEDIUM
            except OSError:
                warnings.append("Could not check file size")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=path,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_json(self, value: str, 
                     schema: Optional[Dict] = None,
                     max_depth: int = 10) -> ValidationResult:
        """
        Validate JSON input.
        
        Args:
            value: JSON string to validate
            schema: Optional JSON schema for validation
            max_depth: Maximum nesting depth allowed
            
        Returns:
            ValidationResult with validation status and parsed value
        """
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW
        
        # Basic string validation
        string_result = self.validate_string(value, max_length=10000, sanitize=False)
        if not string_result.is_valid:
            return string_result
        
        # JSON parsing
        try:
            parsed_value = json.loads(value)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {str(e)}")
            severity = ValidationSeverity.MEDIUM
            return ValidationResult(False, None, errors, warnings, severity)
        
        # Depth check
        if self._get_json_depth(parsed_value) > max_depth:
            errors.append(f"JSON too deeply nested (maximum {max_depth} levels)")
            severity = ValidationSeverity.MEDIUM
        
        # Schema validation (basic)
        if schema:
            schema_errors = self._validate_json_schema(parsed_value, schema)
            if schema_errors:
                errors.extend(schema_errors)
                severity = ValidationSeverity.MEDIUM
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=parsed_value,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def validate_dict(self, value: Any, 
                     required_keys: Optional[List[str]] = None,
                     allowed_keys: Optional[List[str]] = None,
                     max_keys: int = 100) -> ValidationResult:
        """
        Validate dictionary input.
        
        Args:
            value: Dictionary to validate
            required_keys: List of required keys
            allowed_keys: List of allowed keys
            max_keys: Maximum number of keys allowed
            
        Returns:
            ValidationResult with validation status and sanitized value
        """
        errors = []
        warnings = []
        severity = ValidationSeverity.LOW
        
        # Type validation
        if not isinstance(value, dict):
            errors.append("Input must be a dictionary")
            severity = ValidationSeverity.MEDIUM
            return ValidationResult(False, None, errors, warnings, severity)
        
        # Key count validation
        if len(value) > max_keys:
            errors.append(f"Too many keys (maximum {max_keys})")
            severity = ValidationSeverity.MEDIUM
        
        # Required keys validation
        if required_keys:
            missing_keys = [key for key in required_keys if key not in value]
            if missing_keys:
                errors.append(f"Missing required keys: {missing_keys}")
                severity = ValidationSeverity.MEDIUM
        
        # Allowed keys validation
        if allowed_keys:
            invalid_keys = [key for key in value.keys() if key not in allowed_keys]
            if invalid_keys:
                errors.append(f"Invalid keys: {invalid_keys}")
                severity = ValidationSeverity.MEDIUM
        
        # Key validation
        for key in value.keys():
            if not isinstance(key, str):
                errors.append("Dictionary keys must be strings")
                severity = ValidationSeverity.MEDIUM
                break
            
            # Check for dangerous key patterns
            if any(char in key for char in self.dangerous_chars):
                errors.append(f"Dangerous characters in key: {key}")
                severity = ValidationSeverity.HIGH
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            sanitized_value=value,
            errors=errors,
            warnings=warnings,
            severity=severity
        )
    
    def _check_security_threats(self, value: str) -> List[str]:
        """Check for security threats in input."""
        threats = []
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, value, re.IGNORECASE):
                threats.append(f"Potentially dangerous pattern detected: {pattern}")
        
        # Check for SQL injection patterns
        if self.patterns['sql_injection'].search(value):
            threats.append("Potential SQL injection detected")
        
        # Check for XSS patterns
        if self.patterns['xss_patterns'].search(value):
            threats.append("Potential XSS attack detected")
        
        # Check for path traversal
        if self.patterns['path_traversal'].search(value):
            threats.append("Path traversal attempt detected")
        
        return threats
    
    def _sanitize_string(self, value: str) -> str:
        """Sanitize string input by removing dangerous characters."""
        # Remove dangerous characters
        sanitized = value
        for char in self.dangerous_chars:
            sanitized = sanitized.replace(char, '')
        
        # Remove dangerous patterns
        for pattern in self.dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _get_json_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of a JSON object."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_json_depth(value, current_depth + 1) for value in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._get_json_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth
    
    def _validate_json_schema(self, data: Any, schema: Dict) -> List[str]:
        """Basic JSON schema validation."""
        errors = []
        
        # Simple schema validation (can be extended)
        if 'type' in schema:
            expected_type = schema['type']
            if expected_type == 'object' and not isinstance(data, dict):
                errors.append(f"Expected object, got {type(data).__name__}")
            elif expected_type == 'array' and not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
            elif expected_type == 'string' and not isinstance(data, str):
                errors.append(f"Expected string, got {type(data).__name__}")
            elif expected_type == 'number' and not isinstance(data, (int, float)):
                errors.append(f"Expected number, got {type(data).__name__}")
        
        return errors


class SecureInputHandler:
    """High-level secure input handling with validation and sanitization."""
    
    def __init__(self):
        """Initialize the secure input handler."""
        self.validator = InputValidator()
        self.logger = logging.getLogger(__name__)
    
    def process_user_input(self, input_data: Any, 
                          input_type: str = "string",
                          **validation_kwargs) -> ValidationResult:
        """
        Process user input with appropriate validation.
        
        Args:
            input_data: Input data to process
            input_type: Type of input (string, number, file_path, json, dict)
            **validation_kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with processed input
        """
        try:
            if input_type == "string":
                result = self.validator.validate_string(input_data, **validation_kwargs)
            elif input_type == "number":
                result = self.validator.validate_number(input_data, **validation_kwargs)
            elif input_type == "file_path":
                result = self.validator.validate_file_path(input_data, **validation_kwargs)
            elif input_type == "json":
                result = self.validator.validate_json(input_data, **validation_kwargs)
            elif input_type == "dict":
                result = self.validator.validate_dict(input_data, **validation_kwargs)
            else:
                result = ValidationResult(
                    False, None, [f"Unknown input type: {input_type}"], [], ValidationSeverity.MEDIUM
                )
            
            # Log validation results
            if not result.is_valid:
                self.logger.warning(f"Input validation failed: {result.errors}")
            elif result.warnings:
                self.logger.info(f"Input validation warnings: {result.warnings}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return ValidationResult(
                False, None, [f"Validation error: {str(e)}"], [], ValidationSeverity.HIGH
            )
    
    def batch_validate(self, inputs: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """
        Validate multiple inputs in batch.
        
        Args:
            inputs: Dictionary of input names and their data
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for name, data in inputs.items():
            # Determine input type based on data
            if isinstance(data, str):
                if data.startswith('{') or data.startswith('['):
                    input_type = "json"
                else:
                    input_type = "string"
            elif isinstance(data, (int, float)):
                input_type = "number"
            elif isinstance(data, dict):
                input_type = "dict"
            else:
                input_type = "string"
            
            results[name] = self.process_user_input(data, input_type)
        
        return results


# Global instances for easy access
_global_validator = InputValidator()
_global_handler = SecureInputHandler()


def validate_input(input_data: Any, input_type: str = "string", **kwargs) -> ValidationResult:
    """Convenience function for input validation."""
    return _global_handler.process_user_input(input_data, input_type, **kwargs)


def validate_string(value: Any, **kwargs) -> ValidationResult:
    """Convenience function for string validation."""
    return _global_validator.validate_string(value, **kwargs)


def validate_number(value: Any, **kwargs) -> ValidationResult:
    """Convenience function for number validation."""
    return _global_validator.validate_number(value, **kwargs)


def validate_file_path(path: str, **kwargs) -> ValidationResult:
    """Convenience function for file path validation."""
    return _global_validator.validate_file_path(path, **kwargs)


def validate_json(value: str, **kwargs) -> ValidationResult:
    """Convenience function for JSON validation."""
    return _global_validator.validate_json(value, **kwargs)


def validate_dict(value: Any, **kwargs) -> ValidationResult:
    """Convenience function for dictionary validation."""
    return _global_validator.validate_dict(value, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Input Validation System created successfully!")
    print("Features include:")
    print("- String validation and sanitization")
    print("- Number validation with range checks")
    print("- File path validation with security checks")
    print("- JSON validation with depth limits")
    print("- Dictionary validation with key restrictions")
    print("- Security threat detection")
    print("- Batch validation support")
    
    # Test the validation system
    try:
        # Test string validation
        result = validate_string("Hello World!", min_length=5, max_length=20)
        print(f"\nString validation: {result.is_valid}")
        
        # Test number validation
        result = validate_number("42", min_value=0, max_value=100, integer_only=True)
        print(f"Number validation: {result.is_valid}")
        
        # Test file path validation
        result = validate_file_path("test.txt", allowed_extensions=['.txt', '.json'])
        print(f"File path validation: {result.is_valid}")
        
        # Test JSON validation
        result = validate_json('{"name": "test", "value": 123}')
        print(f"JSON validation: {result.is_valid}")
        
        # Test security threat detection
        result = validate_string("<script>alert('xss')</script>")
        print(f"Security threat detection: {not result.is_valid}")
        
    except Exception as e:
        print(f"Input validation test failed: {e}")
    
    print("\nInput validation system test completed!")
