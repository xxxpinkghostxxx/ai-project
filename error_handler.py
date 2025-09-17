
import logging
import traceback
import time
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
from logging_utils import log_step


class ErrorHandler:

    def __init__(self):
        self.error_counts = {}
        self.recovery_attempts = {}
        self.system_health = 'healthy'
        self.last_error_time = 0
        self.error_callbacks = []
        self.recovery_callbacks = []
        self.max_errors_per_minute = 10
        self.max_recovery_attempts = 3
        self.error_cooldown = 60
        self.health_levels = {
            'healthy': 0,
            'warning': 1,
            'degraded': 2,
            'critical': 3,
            'failed': 4
        }
        log_step("ErrorHandler initialized")
    def handle_error(self, error: Exception, context: str = "",
                    recovery_func: Optional[Callable] = None,
                    critical: bool = False) -> bool:

        error_type = type(error).__name__
        error_key = f"{error_type}:{context}"
        current_time = time.time()
        if error_key not in self.error_counts:
            self.error_counts[error_key] = 0
        self.error_counts[error_key] += 1
        log_step(f"Error occurred: {error_type}",
                context=context,
                error_count=self.error_counts[error_key],
                critical=critical)
        logging.error(f"[ERROR_HANDLER] {error_type} in {context}: {str(error)}")
        logging.debug(f"[ERROR_HANDLER] Traceback: {traceback.format_exc()}")
        self._update_system_health(error_type, critical)
        if not critical and recovery_func:
            success = self._attempt_recovery(error_key, recovery_func, error)
            if success:
                log_step("Error recovery successful", error_type=error_type, context=context)
                return True
        for callback in self.error_callbacks:
            try:
                callback(error, context, critical)
            except (TypeError, ValueError, AttributeError) as callback_error:
                logging.error(f"Error callback failed: {callback_error}")
            except Exception as callback_error:
                logging.error(f"Unexpected error callback failure: {callback_error}")
        if critical:
            self.system_health = 'failed'
            log_step("Critical error - system marked as failed", error_type=error_type)
            return False
        return True
    def _update_system_health(self, error_type: str, critical: bool):
        current_time = time.time()
        recent_errors = sum(1 for count in self.error_counts.values()
                          if count > 0 and current_time - self.last_error_time < 60)
        if critical:
            self.system_health = 'failed'
        elif recent_errors > self.max_errors_per_minute:
            self.system_health = 'critical'
        elif recent_errors > self.max_errors_per_minute // 2:
            self.system_health = 'degraded'
        elif recent_errors > 0:
            self.system_health = 'warning'
        else:
            self.system_health = 'healthy'
        self.last_error_time = current_time
    def _attempt_recovery(self, error_key: str, recovery_func: Callable,
                         original_error: Exception) -> bool:
        if error_key not in self.recovery_attempts:
            self.recovery_attempts[error_key] = 0
        if self.recovery_attempts[error_key] >= self.max_recovery_attempts:
            log_step("Max recovery attempts reached", error_key=error_key)
            return False
        self.recovery_attempts[error_key] += 1
        try:
            log_step("Attempting error recovery",
                    error_key=error_key,
                    attempt=self.recovery_attempts[error_key])
            result = recovery_func(original_error)
            if result:
                self.error_counts[error_key] = 0
                self.recovery_attempts[error_key] = 0
                for callback in self.recovery_callbacks:
                    try:
                        callback(error_key, True)
                    except (TypeError, ValueError, AttributeError) as callback_error:
                        logging.error(f"Recovery callback failed: {callback_error}")
                    except Exception as callback_error:
                        logging.error(f"Unexpected recovery callback failure: {callback_error}")
                return True
            else:
                log_step("Recovery attempt failed", error_key=error_key)
                return False
        except (TypeError, ValueError, AttributeError, RuntimeError) as recovery_error:
            log_step("Recovery attempt caused error",
                    error_key=error_key,
                    recovery_error=str(recovery_error))
            return False
        except Exception as recovery_error:
            log_step("Recovery attempt caused unexpected error",
                    error_key=error_key,
                    recovery_error=str(recovery_error))
            return False
    def add_error_callback(self, callback: Callable):
        self.error_callbacks.append(callback)
    def add_recovery_callback(self, callback: Callable):
        self.recovery_callbacks.append(callback)
    def get_system_health(self) -> Dict[str, Any]:
        return {
            'status': self.system_health,
            'level': self.health_levels.get(self.system_health, 0),
            'error_counts': self.error_counts.copy(),
            'recovery_attempts': self.recovery_attempts.copy(),
            'last_error_time': self.last_error_time,
            'is_healthy': self.system_health == 'healthy'
        }
    def reset_error_counts(self):
        self.error_counts.clear()
        self.recovery_attempts.clear()
        self.system_health = 'healthy'
        self.last_error_time = 0
        log_step("Error counts reset")


def get_error_handler() -> ErrorHandler:
    return ErrorHandler()


def safe_execute(func: Callable, context: str = "",
                recovery_func: Optional[Callable] = None,
                critical: bool = False, default_return: Any = None):

    error_handler = get_error_handler()
    try:
        return func()
    except Exception as e:
        success = error_handler.handle_error(e, context, recovery_func, critical)
        if not success and critical:
            raise
        return default_return


def error_handler_decorator(context: str = "", recovery_func: Optional[Callable] = None,
                          critical: bool = False, default_return: Any = None):

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return safe_execute(
                lambda: func(*args, **kwargs),
                context or func.__name__,
                recovery_func,
                critical,
                default_return
            )
        return wrapper
    return decorator


def graceful_degradation(func: Callable, fallback_func: Callable,
                        context: str = "") -> Any:

    error_handler = get_error_handler()
    try:
        return func()
    except Exception as e:
        error_handler.handle_error(e, f"{context} (primary)", critical=False)
        try:
            log_step("Attempting graceful degradation", context=context)
            return fallback_func()
        except Exception as fallback_error:
            error_handler.handle_error(fallback_error, f"{context} (fallback)", critical=True)
            raise


def system_health_check() -> Dict[str, Any]:
    error_handler = get_error_handler()
    health = error_handler.get_system_health()
    health['timestamp'] = time.time()
    health['uptime'] = time.time() - error_handler.last_error_time if error_handler.last_error_time > 0 else 0
    return health
if __name__ == "__main__":
    print("ErrorHandler initialized successfully!")
    print("Features include:")
    print("- Centralized error handling")
    print("- Automatic recovery mechanisms")
    print("- System health monitoring")
    print("- Graceful degradation support")
    print("- Error callbacks and notifications")
    handler = ErrorHandler()
    print(f"Handler created with health status: {handler.system_health}")
    print("ErrorHandler is ready for integration!")
