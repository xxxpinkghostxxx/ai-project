import functools
import logging
import time
from threading import Lock

log_lines = []
MAX_LOG_LINES = 100
log_lock = Lock()


def append_log_line(line):
    global log_lines  # pylint: disable=global-statement
    with log_lock:
        log_lines.append(line)
        if len(log_lines) > MAX_LOG_LINES:
            log_lines = log_lines[-MAX_LOG_LINES:]


class UILogHandler(logging.Handler):

    def __init__(self, ui_callback=None):
        super().__init__()
        self.ui_callback = ui_callback
    def emit(self, record):
        try:
            msg = self.format(record)
            append_log_line(msg)
            if self.ui_callback:
                self.ui_callback(msg)
        except Exception as e:  # pylint: disable=broad-except
            print(f"ERROR in UILogHandler.emit: {e}")


def setup_logging(ui_callback=None, level=logging.INFO):

    handler = UILogHandler(ui_callback=ui_callback)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    logging.info("Logging system initialized.")
    return handler


def get_log_lines():

    with log_lock:
        return list(log_lines)


def log_runtime(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info("[RUNTIME] Entering %s", func.__name__)
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        elapsed = (end - start) * 1000
        logging.info("[RUNTIME] Exiting %s (runtime: %.2f ms)", func.__name__, elapsed)
        return result
    return wrapper


def log_step(step_desc, **kwargs):

    msg = f"[STEP] {step_desc}"
    if kwargs:
        details = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        msg += f" | {details}"
    logging.info(msg)


def log_node_state(node_label, prefix="[NODE_STATE]"):

    required_fields = ["type", "energy", "behavior", "state", "last_update"]
    for field in required_fields:
        assert field in node_label, f"Node label missing required field: {field}"
    node_id = node_label.get("id", node_label.get("x", "?"))
    msg = (
        f"{prefix} type={node_label['type']} id={node_id} energy={node_label['energy']:.2f} "
        f"behavior={node_label['behavior']} state={node_label['state']} last_update={node_label['last_update']}"
    )
    logging.info(msg)







