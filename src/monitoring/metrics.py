import time
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge, generate_latest


prediction_counter = Counter("ml_predictions_total", "Total number of predictions made")
prediction_latency = Histogram("ml_prediction_duration_seconds", "Time spent on predictions")
model_accuracy = Gauge("ml_model_accuracy", "Current model accuracy")
error_counter = Counter("ml_prediction_errors_total", "Total prediction errors")


def track_predictions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            prediction_counter.inc()
            prediction_latency.observe(time.time() - start_time)
            return result
        except Exception as e:
            error_counter.inc()
            raise e

    return wrapper


def export_metrics():
    return generate_latest()
