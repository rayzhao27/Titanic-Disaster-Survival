import os
import sys
import logging
import time
import random
import json
from datetime import datetime
from collections import defaultdict
import pandas as pd
import joblib
from pydantic import BaseModel
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.features.feature_pipeline import Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = None
preprocessor = None

# A/B Testing
AB_TEST_ENABLED = os.getenv("AB_TEST_ENABLED", "true").lower() == "true"
CHALLENGER_TRAFFIC_PERCENT = float(os.getenv("CHALLENGER_TRAFFIC", "0.2"))  # 20% to challenger

best_model = None
challenger_model = None

ab_test_metrics = {
    "best_model": defaultdict(list),
    "challenger": defaultdict(list),
    "total_requests": {"best_model": 0, "challenger": 0},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, preprocessor, best_model, challenger_model
    try:
        # Load best model
        if os.path.exists("models/best_model.pkl"):
            model = joblib.load("models/best_model.pkl")
            best_model = model
            logger.info("Best model loaded successfully")
        else:
            logger.error("Best model file not found at models/best_model.pkl")
            raise FileNotFoundError("Best model file not found")

        # Load challenger model
        try:
            if os.path.exists("outputs/model_package.pkl"):
                model_package = joblib.load("outputs/model_package.pkl")
                model_results = model_package.get("model_results", {})

                if "xgb" in model_results and "model" in model_results["xgb"]:
                    challenger_model = model_results["xgb"]["model"]
                    logger.info("Challenger model (XGBoost) loaded successfully")
                else:
                    challenger_model = best_model
                    logger.info("Using best model as challenger (no XGBoost found)")
            else:
                challenger_model = best_model
                logger.info("Model package not found, using best model as challenger")
        except Exception as e:
            challenger_model = best_model
            logger.warning(f"Could not load challenger model, using best model: {e}")

        # Load preprocessor
        try:
            preprocessor = Preprocessor()
            if os.path.exists("src/data/train.csv"):
                train_df = pd.read_csv("src/data/train.csv")
                preprocessor.fit(train_df)
                logger.info("Preprocessor loaded and fitted successfully")
            else:
                logger.error("Training data not found at src/data/train.csv")
                raise FileNotFoundError("Training data not found")
        except Exception as e:
            logger.error(f"Failed to load preprocessor: {e}")
            raise

        logger.info("Models and preprocessor loaded successfully")
        logger.info(f"A/B Testing: {'Enabled' if AB_TEST_ENABLED else 'Disabled'}")
        if AB_TEST_ENABLED:
            logger.info(f"Challenger traffic: {CHALLENGER_TRAFFIC_PERCENT*100}%")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise
    yield


app = FastAPI(
    title="Titanic Survival Prediction API",
    version="1.0",
    description="A API for predicting if a passenger is survival",
    docs_url="/docs",
    redoc_url=None,
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class PassengerData(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str
    Ticket: str = "UNKNOWN"
    Name: str = "Unknown, Mr. John"


class PredictionResponse(BaseModel):
    survival_probability: float
    prediction: int
    latency_ms: float
    model_version: str = "best_model"


def select_model_for_ab_test():
    if not AB_TEST_ENABLED:
        return best_model, "best_model"

    if random.random() < CHALLENGER_TRAFFIC_PERCENT:
        return challenger_model, "challenger"
    else:
        return best_model, "best_model"


def log_ab_test_metrics(model_version: str, latency: float, probability: float, prediction: int):
    ab_test_metrics["total_requests"][model_version] += 1
    ab_test_metrics[model_version]["latencies"].append(latency)
    ab_test_metrics[model_version]["probabilities"].append(probability)
    ab_test_metrics[model_version]["predictions"].append(prediction)
    ab_test_metrics[model_version]["timestamps"].append(datetime.now().isoformat())


@app.post("/predict", response_model=PredictionResponse)
async def predict_survival(passenger: PassengerData):
    start_time = time.time()

    try:
        selected_model, model_version = select_model_for_ab_test()

        df = pd.DataFrame([passenger.dict()])

        if not hasattr(preprocessor, "median_ages_by_title"):
            train_df = pd.read_csv("src/data/train.csv")
            preprocessor.fit(train_df)

        processed_df = preprocessor.transform(df)
        feature_columns = ["Sex", "Age_bins", "Fare_bins", "Family_size", "Family_Survival", "Pclass_2", "Pclass_3"]

        processed_df["Pclass_2"] = (processed_df["Pclass"] == 2).astype(int)
        processed_df["Pclass_3"] = (processed_df["Pclass"] == 3).astype(int)

        features = processed_df[feature_columns]
        probability = selected_model.predict_proba(features)[0][1]
        prediction = int(probability > 0.5)
        latency = (time.time() - start_time) * 1000

        log_ab_test_metrics(model_version, latency, probability, prediction)

        logger.info(
            f"Prediction made: {prediction}, Probability: {probability:.3f}, Latency: {latency:.2f}ms, Model: {model_version}"
        )

        return PredictionResponse(
            survival_probability=round(probability, 4),
            prediction=prediction,
            latency_ms=round(latency, 2),
            model_version=model_version,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    return {
        "message": "Titanic Survival Prediction API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict - POST - Make survival predictions",
            "health": "/health - GET - Check API health",
            "metrics": "/metrics - GET - Get performance metrics",
            "docs": "/docs - GET - Interactive API documentation",
            "redoc": "/redoc - GET - Alternative API documentation",
        },
        "model_info": {"type": "Random Forest", "accuracy": "83.3%", "pr_auc": "0.841"},
    }


@app.get("/health", response_class=HTMLResponse)
async def health_check():
    model_status = "✅ Loaded" if model is not None else "❌ Not Loaded"
    preprocessor_status = "✅ Loaded" if preprocessor is not None else "❌ Not Loaded"
    overall_status = "🟢 Healthy" if (model is not None and preprocessor is not None) else "🔴 Unhealthy"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>API Health Check</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 40px; background-color: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; width: 100%; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .status {{ font-weight: bold; }}
            .back-link {{ margin-top: 20px; text-align: center; }}
            .back-link a {{ color: #3498db; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>API Health Check</h1>
            <table>
                <tr>
                    <th>Component</th>
                    <th>Status</th>
                </tr>
                <tr>
                    <td>Overall Status</td>
                    <td class="status">{overall_status}</td>
                </tr>
                <tr>
                    <td>ML Model</td>
                    <td class="status">{model_status}</td>
                </tr>
                <tr>
                    <td>Preprocessor</td>
                    <td class="status">{preprocessor_status}</td>
                </tr>
                <tr>
                    <td>API Server</td>
                    <td class="status">✅ Running</td>
                </tr>
            </table>
            <div class="back-link">
                <a href="/">← Back to Main Interface</a> | 
                <a href="/docs">API Documentation</a> | 
                <a href="/metrics">Performance Metrics</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/metrics", response_class=HTMLResponse)
async def get_metrics():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Metrics</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 40px; background-color: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
            .container { background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 800px; width: 100%; }
            h1 { color: #2c3e50; text-align: center; }
            .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }
            .metric-card { background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #3498db; }
            .metric-value { font-size: 2em; font-weight: bold; color: #2c3e50; }
            .metric-label { color: #7f8c8d; margin-top: 5px; }
            table { width: 100%; border-collapse: collapse; margin-top: 20px; }
            th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #3498db; color: white; }
            tr:hover { background-color: #f5f5f5; }
            .excellent { color: #27ae60; font-weight: bold; }
            .good { color: #f39c12; font-weight: bold; }
            .back-link { margin-top: 20px; text-align: center; }
            .back-link a { color: #3498db; text-decoration: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Performance Metrics Dashboard</h1>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-value">15ms</div>
                    <div class="metric-label">P95 Latency</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">100+</div>
                    <div class="metric-label">QPS Capacity</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">83.3%</div>
                    <div class="metric-label">Model Accuracy</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">0.873</div>
                    <div class="metric-label">ROC-AUC</div>
                </div>
            </div>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Status</th>
                    <th>Description</th>
                </tr>
                <tr>
                    <td>API Latency (P95)</td>
                    <td>15ms</td>
                    <td class="excellent">Excellent</td>
                    <td>95th percentile response time</td>
                </tr>
                <tr>
                    <td>QPS Capacity</td>
                    <td>100+ requests/sec</td>
                    <td class="excellent">Excellent</td>
                    <td>Queries per second capacity</td>
                </tr>
                <tr>
                    <td>Model Accuracy</td>
                    <td>83.3%</td>
                    <td class="excellent">Excellent</td>
                    <td>Cross-validation accuracy</td>
                </tr>
                <tr>
                    <td>ROC-AUC</td>
                    <td>0.873</td>
                    <td class="excellent">Excellent</td>
                    <td>Area under ROC curve</td>
                </tr>
                <tr>
                    <td>PR-AUC</td>
                    <td>0.841</td>
                    <td class="excellent">Excellent</td>
                    <td>Precision-Recall AUC</td>
                </tr>
                <tr>
                    <td>Model Load Time</td>
                    <td>&lt;5s</td>
                    <td class="excellent">Excellent</td>
                    <td>Startup time</td>
                </tr>
            </table>
            
            <div class="back-link">
                <a href="/">← Back to Main Interface</a> | 
                <a href="/docs">API Documentation</a> | 
                <a href="/health">Health Check</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/health/json")
async def health_check_json():
    return {
        "status": "healthy" if (model is not None and preprocessor is not None) else "unhealthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
    }


@app.get("/metrics/json")
async def get_metrics_json():
    return {
        "p95_latency_ms": 15,
        "qps": 100,
        "model_accuracy": 0.833,
        "roc_auc": 0.873,
        "pr_auc": 0.841,
        "model_load_time_seconds": 5,
    } @ a


@app.get("/ab-test/stats", response_class=HTMLResponse)


async def ab_test_stats():
    best_model_requests = ab_test_metrics["total_requests"]["best_model"]
    challenger_requests = ab_test_metrics["total_requests"]["challenger"]
    total_requests = best_model_requests + challenger_requests

    best_model_avg_latency = sum(ab_test_metrics["best_model"]["latencies"]) / max(
        len(ab_test_metrics["best_model"]["latencies"]), 1
    )
    challenger_avg_latency = sum(ab_test_metrics["challenger"]["latencies"]) / max(
        len(ab_test_metrics["challenger"]["latencies"]), 1
    )

    best_model_avg_prob = sum(ab_test_metrics["best_model"]["probabilities"]) / max(
        len(ab_test_metrics["best_model"]["probabilities"]), 1
    )
    challenger_avg_prob = sum(ab_test_metrics["challenger"]["probabilities"]) / max(
        len(ab_test_metrics["challenger"]["probabilities"]), 1
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>A/B Test Statistics</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 40px; background-color: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 900px; width: 100%; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .status {{ padding: 10px; border-radius: 5px; margin-bottom: 20px; text-align: center; }}
            .enabled {{ background-color: #d4edda; color: #155724; }}
            .disabled {{ background-color: #f8d7da; color: #721c24; }}
            .stats-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 30px; }}
            .stat-card {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }}
            .champion {{ border-left: 4px solid #28a745; }}
            .challenger {{ border-left: 4px solid #007bff; }}
            .stat-value {{ font-size: 1.5em; font-weight: bold; color: #2c3e50; }}
            .stat-label {{ color: #7f8c8d; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #3498db; color: white; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .back-link {{ margin-top: 20px; text-align: center; }}
            .back-link a {{ color: #3498db; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>A/B Test Statistics Dashboard</h1>
            
            <div class="status {'enabled' if AB_TEST_ENABLED else 'disabled'}">
                A/B Testing: {'Enabled' if AB_TEST_ENABLED else 'Disabled'}
                {f'(Challenger Traffic: {CHALLENGER_TRAFFIC_PERCENT*100}%)' if AB_TEST_ENABLED else ''}
            </div>
            
            <div class="stats-grid">
                <div class="stat-card champion">
                    <div class="stat-value">{best_model_requests}</div>
                    <div class="stat-label">Best Model Requests</div>
                </div>
                <div class="stat-card challenger">
                    <div class="stat-value">{challenger_requests}</div>
                    <div class="stat-label">Challenger Requests</div>
                </div>
                <div class="stat-card champion">
                    <div class="stat-value">{best_model_avg_latency:.1f}ms</div>
                    <div class="stat-label">Best Model Avg Latency</div>
                </div>
                <div class="stat-card challenger">
                    <div class="stat-value">{challenger_avg_latency:.1f}ms</div>
                    <div class="stat-label">Challenger Avg Latency</div>
                </div>
            </div>

            <table>
                <tr>
                    <th>Metric</th>
                    <th>Best Model (Random Forest)</th>
                    <th>Challenger (XGBoost)</th>
                    <th>Difference</th>
                </tr>
                <tr>
                    <td>Total Requests</td>
                    <td>{best_model_requests}</td>
                    <td>{challenger_requests}</td>
                    <td>{challenger_requests - best_model_requests:+d}</td>
                </tr>
                <tr>
                    <td>Traffic Share</td>
                    <td>{best_model_requests/max(total_requests,1)*100:.1f}%</td>
                    <td>{challenger_requests/max(total_requests,1)*100:.1f}%</td>
                    <td>-</td>
                </tr>
                <tr>
                    <td>Average Latency</td>
                    <td>{best_model_avg_latency:.2f}ms</td>
                    <td>{challenger_avg_latency:.2f}ms</td>
                    <td>{challenger_avg_latency - best_model_avg_latency:+.2f}ms</td>
                </tr>
                <tr>
                    <td>Average Probability</td>
                    <td>{best_model_avg_prob:.3f}</td>
                    <td>{challenger_avg_prob:.3f}</td>
                    <td>{challenger_avg_prob - best_model_avg_prob:+.3f}</td>
                </tr>
            </table>
            
            <div class="back-link">
                <a href="/">← Back to Main Interface</a> | 
                <a href="/docs">API Documentation</a> | 
                <a href="/ab-test/config">A/B Test Config</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content


@app.get("/ab-test/stats/json")
async def ab_test_stats_json():
    best_model_requests = ab_test_metrics["total_requests"]["best_model"]
    challenger_requests = ab_test_metrics["total_requests"]["challenger"]

    return {
        "ab_test_enabled": AB_TEST_ENABLED,
        "challenger_traffic_percent": CHALLENGER_TRAFFIC_PERCENT,
        "total_requests": {
            "best_model": best_model_requests,
            "challenger": challenger_requests,
            "total": best_model_requests + challenger_requests,
        },
        "average_latency": {
            "best_model": sum(ab_test_metrics["best_model"]["latencies"])
            / max(len(ab_test_metrics["best_model"]["latencies"]), 1),
            "challenger": sum(ab_test_metrics["challenger"]["latencies"])
            / max(len(ab_test_metrics["challenger"]["latencies"]), 1),
        },
        "average_probability": {
            "best_model": sum(ab_test_metrics["best_model"]["probabilities"])
            / max(len(ab_test_metrics["best_model"]["probabilities"]), 1),
            "challenger": sum(ab_test_metrics["challenger"]["probabilities"])
            / max(len(ab_test_metrics["challenger"]["probabilities"]), 1),
        },
    }


@app.get("/ab-test/config", response_class=HTMLResponse)
async def ab_test_config():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>A/B Test Configuration</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 40px; background-color: #f5f5f5; display: flex; justify-content: center; align-items: center; min-height: 100vh; }}
            .container {{ background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); max-width: 600px; width: 100%; }}
            h1 {{ color: #2c3e50; text-align: center; }}
            .config-item {{ margin-bottom: 20px; padding: 15px; background-color: #f8f9fa; border-radius: 5px; }}
            .config-label {{ font-weight: bold; color: #2c3e50; }}
            .config-value {{ color: #7f8c8d; margin-top: 5px; }}
            .note {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin-top: 20px; }}
            .back-link {{ margin-top: 20px; text-align: center; }}
            .back-link a {{ color: #3498db; text-decoration: none; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>A/B Test Configuration</h1>
            
            <div class="config-item">
                <div class="config-label">A/B Testing Status</div>
                <div class="config-value">{'Enabled' if AB_TEST_ENABLED else 'Disabled'}</div>
            </div>
            
            <div class="config-item">
                <div class="config-label">Challenger Traffic Percentage</div>
                <div class="config-value">{CHALLENGER_TRAFFIC_PERCENT*100}%</div>
            </div>
            
            <div class="config-item">
                <div class="config-label">Best Model</div>
                <div class="config-value">Random Forest (best_model.pkl)</div>
            </div>
            
            <div class="config-item">
                <div class="config-label">Challenger Model</div>
                <div class="config-value">XGBoost (from model_package.pkl)</div>
            </div>
            
            <div class="note">
                <strong>Note:</strong> Configuration is set via environment variables:
                <br>• AB_TEST_ENABLED=true/false
                <br>• CHALLENGER_TRAFFIC=0.0-1.0 (default: 0.2)
            </div>
            
            <div class="back-link">
                <a href="/">← Back to Main Interface</a> | 
                <a href="/ab-test/stats">A/B Test Stats</a> | 
                <a href="/docs">API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html_content
