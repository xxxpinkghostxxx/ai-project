#!/usr/bin/env python3
"""
Security enhancement utilities for the PyTorch Geometric Neural System.

This module provides input validation, sanitization, and security best practices
to enhance system security and prevent common vulnerabilities.
"""
import re
import html
from typing import Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class InputValidator:
    """
    Input validation utilities to prevent injection attacks and ensure data integrity.
    """

    # Patterns for common input validation
    PATTERNS = {
        'safe_string': re.compile(r'^[a-zA-Z0-9_\-\.]+$'),
        'numeric': re.compile(r'^-?\d+(\.\d+)?$'),
        'positive_integer': re.compile(r'^\d+$'),
        'file_safe': re.compile(r'^[a-zA-Z0-9_\-\.]+$'),
        'json_safe': re.compile(r'^[a-zA-Z0-9_\-\.\[\]\{\},": ]+$'),
        'path_safe': re.compile(r'^[a-zA-Z0-9_\-\./\\]+$')
    }

    @classmethod
    def validate_string(cls, value: str, pattern: str = 'safe_string',
                       min_length: int = 0, max_length: int = 255) -> bool:
        """
        Validate string input against patterns and length constraints.

        Args:
            value: String to validate
            pattern: Pattern name from PATTERNS dict
            min_length: Minimum allowed length
            max_length: Maximum allowed length

        Returns:
            True if valid, False otherwise
        """
        # Type check is redundant since value is already typed as str
        # if not isinstance(value, str):
        #     return False

        if len(value) < min_length or len(value) > max_length:
            return False

        if pattern not in cls.PATTERNS:
            logger.warning(f"Unknown validation pattern: {pattern}")
            return False

        return cls.PATTERNS[pattern].match(value) is not None

    @classmethod
    def validate_numeric(cls, value: int | float,
                        min_value: float | None = None,
                        max_value: float | None = None) -> bool:
        """
        Validate numeric input within range constraints.

        Args:
            value: Numeric value to validate
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            True if valid, False otherwise
        """
        # Type check is redundant since value is already typed as int | float
        # if not isinstance(value, (int, float)):
        #     return False

        if min_value is not None and value < min_value:
            return False

        if max_value is not None and value > max_value:
            return False

        return True

    @classmethod
    def validate_list(cls, value: list[Any], element_type: type = str,
                     min_elements: int = 0, max_elements: int = 100) -> bool:
        """
        Validate list input with type and size constraints.

        Args:
            value: List to validate
            element_type: Expected type of list elements
            min_elements: Minimum number of elements
            max_elements: Maximum number of elements

        Returns:
            True if valid, False otherwise
        """
        if len(value) < min_elements or len(value) > max_elements:
            return False

        return all(isinstance(item, element_type) for item in value)

class SecuritySanitizer:
    """
    Input sanitization utilities to remove dangerous content and prevent attacks.
    """

    @staticmethod
    def sanitize_html(value: str) -> str:
        """
        Sanitize HTML content by escaping special characters.

        Args:
            value: HTML string to sanitize

        Returns:
            Sanitized HTML string
        """
        return html.escape(value, quote=True)

    @staticmethod
    def sanitize_json(value: str) -> str:
        """
        Sanitize JSON string to prevent injection attacks.

        Args:
            value: JSON string to sanitize

        Returns:
            Sanitized JSON string
        """
        # Remove potentially dangerous characters
        sanitized = re.sub(r'[<>"\']', '', value)
        return sanitized

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Sanitize filename to prevent directory traversal and other attacks.

        Args:
            filename: Filename to sanitize

        Returns:
            Sanitized filename
        """
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)

        # Ensure it's not empty
        if not sanitized.strip():
            sanitized = "unnamed_file"

        # Limit length
        if len(sanitized) > 255:
            sanitized = sanitized[:255]

        return sanitized

    @staticmethod
    def sanitize_path(path: str) -> str:
        """
        Sanitize file path to prevent directory traversal attacks.

        Args:
            path: Path to sanitize

        Returns:
            Sanitized path
        """
        # Remove dangerous sequences
        sanitized = path.replace('..', '').replace('//', '/')

        # Remove path traversal attempts
        sanitized = re.sub(r'[/\\]\.\.[/\\]', '', sanitized)

        return sanitized.strip()

class ConfigurationSecurityValidator:
    """
    Security validation for configuration parameters.
    """

    # Security constraints for configuration values
    SECURITY_CONSTRAINTS: dict[str, dict[str, float]] = {
        'max_total_connections': {'min': 1, 'max': 10000},
        'max_connections_per_node': {'min': 1, 'max': 1000},
        'memory_limit_gb': {'min': 0.1, 'max': 1000.0},
        'cache_size': {'min': 1, 'max': 100000},
        'thread_pool_size': {'min': 1, 'max': 100},
        'timeout_seconds': {'min': 1, 'max': 3600},
        'port': {'min': 1024, 'max': 65535}
    }

    @classmethod
    def validate_config_value(cls, key: str, value: Any) -> tuple[bool, str]:
        """
        Validate configuration value against security constraints.

        Args:
            key: Configuration key
            value: Configuration value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if key not in cls.SECURITY_CONSTRAINTS:
            return True, ""  # Allow unknown keys for forward compatibility

        constraints = cls.SECURITY_CONSTRAINTS[key]

        if not isinstance(value, (int, float)):
            return False, f"Configuration '{key}' must be numeric"

        if 'min' in constraints and value < constraints['min']:
            return False, f"Configuration '{key}' must be >= {constraints['min']}"

        if 'max' in constraints and value > constraints['max']:
            return False, f"Configuration '{key}' must be <= {constraints['max']}"

        return True, ""

    @classmethod
    def validate_config_section(cls, config: dict[str, Any]) -> list[str]:
        """
        Validate entire configuration section.

        Args:
            config: Configuration dictionary to validate

        Returns:
            List of validation errors
        """
        errors: list[str] = []

        for key, value in config.items():
            is_valid, error_msg = cls.validate_config_value(key, value)
            if not is_valid:
                errors.append(f"Invalid configuration '{key}': {error_msg}")

        return errors

class SecureLogger:
    """
    Secure logging utilities to prevent information leakage.
    """

    SENSITIVE_PATTERNS = [
        r'password\s*[:=]\s*[\w]+',
        r'api_key\s*[:=]\s*[\w]+',
        r'secret\s*[:=]\s*[\w]+',
        r'token\s*[:=]\s*[\w]+'
    ]

    @classmethod
    def sanitize_log_message(cls, message: str) -> str:
        """
        Sanitize log message to remove sensitive information.

        Args:
            message: Log message to sanitize

        Returns:
            Sanitized log message
        """
        sanitized = message

        for pattern in cls.SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)

        return sanitized

    @classmethod
    def secure_info(cls, logger_instance: logging.Logger, message: str) -> None:
        """
        Log information securely without sensitive data.

        Args:
            logger_instance: Logger instance to use
            message: Message to log
        """
        sanitized = cls.sanitize_log_message(message)
        logger_instance.info(sanitized)

    @classmethod
    def secure_error(cls, logger_instance: logging.Logger, message: str) -> None:
        """
        Log errors securely without sensitive data.

        Args:
            logger_instance: Logger instance to use
            message: Error message to log
        """
        sanitized = cls.sanitize_log_message(message)
        logger_instance.error(sanitized)

class SecurityAudit:
    """
    Security audit utilities to scan for potential vulnerabilities.
    """

    SECURITY_RISKS = [
        ('hardcoded_credentials', re.compile(r'(password|api_key|secret|token)\s*=\s*["\'][^"\']+["\']', re.IGNORECASE)),
        ('sql_injection', re.compile(r'(SELECT|INSERT|UPDATE|DELETE).*[\'"]\+', re.IGNORECASE)),
        ('path_traversal', re.compile(r'\.\.[/\\]')),
        ('eval_usage', re.compile(r'\beval\s*\(')),
        ('exec_usage', re.compile(r'\bexec\s*\('))
    ]

    @classmethod
    def scan_file_for_security_issues(cls, file_path: Path) -> list[dict[str, Any]]:
        """
        Scan file for potential security issues.

        Args:
            file_path: Path to file to scan

        Returns:
            List of security issues found
        """
        issues: list[dict[str, Any]] = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')

            for line_num, line in enumerate(lines, 1):
                for issue_type, pattern in cls.SECURITY_RISKS:
                    if pattern.search(line):
                        issues.append({
                            'type': issue_type,
                            'file': str(file_path),
                            'line': line_num,
                            'content': line.strip()
                        })

        except Exception as e:
            logger.warning(f"Could not scan file {file_path}: {e}")

        return issues

    @classmethod
    def scan_directory_for_security_issues(cls, directory: Path) -> list[dict[str, Any]]:
        """
        Scan directory for security issues in all files.

        Args:
            directory: Directory to scan

        Returns:
            List of all security issues found
        """
        all_issues: list[dict[str, Any]] = []

        for file_path in directory.rglob('*.py'):
            issues = cls.scan_file_for_security_issues(file_path)
            all_issues.extend(issues)

        return all_issues

def validate_and_sanitize_input(input_value: Any, validation_type: str = 'safe_string',
                               sanitization_method: str | None = None) -> tuple[Any, list[str]]:
    """
    Convenience function to validate and sanitize input.

    Args:
        input_value: Value to validate and sanitize
        validation_type: Type of validation to perform
        sanitization_method: Method to use for sanitization

    Returns:
        Tuple of (sanitized_value, validation_errors)
    """
    errors: list[str] = []

    # Validate
    if validation_type == 'string':
        if not InputValidator.validate_string(str(input_value)):
            errors.append("Invalid string format")
            input_value = str(input_value)[:255]  # Truncate if too long
    elif validation_type == 'numeric':
        if not InputValidator.validate_numeric(float(input_value)):
            errors.append("Invalid numeric format")
            input_value = 0
    elif validation_type == 'path':
        if not InputValidator.validate_string(str(input_value), 'path_safe', 1, 1000):
            errors.append("Invalid path format")
            input_value = "unknown_path"

    # Sanitize
    if sanitization_method == 'html':
        input_value = SecuritySanitizer.sanitize_html(str(input_value))
    elif sanitization_method == 'json':
        input_value = SecuritySanitizer.sanitize_json(str(input_value))
    elif sanitization_method == 'filename':
        input_value = SecuritySanitizer.sanitize_filename(str(input_value))
    elif sanitization_method == 'path':
        input_value = SecuritySanitizer.sanitize_path(str(input_value))

    return input_value, errors