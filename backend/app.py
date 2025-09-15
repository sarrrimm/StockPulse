import io, csv
import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
from pathlib import Path
import asyncio
import logging
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, validator
from threading import Lock

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

executor = ThreadPoolExecutor(max_workers=4)

model_lock = Lock()
model_bundle = None
model = None
scaler = None
feature_columns = None
def init_db():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reports (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL CHECK(length(filename) > 0),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'processing' CHECK(status IN ('processing', 'completed', 'failed')),
                    total_records INTEGER DEFAULT 0 CHECK(total_records >= 0),
                    anomaly_count INTEGER DEFAULT 0 CHECK(anomaly_count >= 0),
                    error_message TEXT,
                    threshold_percentile REAL DEFAULT 8.0 CHECK(threshold_percentile BETWEEN 1.0 AND 25.0)
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close_price REAL NOT NULL CHECK(close_price > 0),
                    volume REAL NOT NULL CHECK(volume >= 0),
                    anomaly_score REAL NOT NULL,
                    is_anomaly BOOLEAN NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports (id) ON DELETE CASCADE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chart_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    report_id TEXT NOT NULL,
                    date TEXT NOT NULL,
                    close_price REAL NOT NULL CHECK(close_price > 0),
                    volume REAL NOT NULL CHECK(volume >= 0),
                    is_anomaly BOOLEAN NOT NULL,
                    anomaly_score REAL NOT NULL,
                    FOREIGN KEY (report_id) REFERENCES reports (id) ON DELETE CASCADE
                )
            """)
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS idx_anomalies_report_date ON anomalies(report_id, date)",
                "CREATE INDEX IF NOT EXISTS idx_anomalies_is_anomaly ON anomalies(is_anomaly)",
                "CREATE INDEX IF NOT EXISTS idx_chart_data_report_date ON chart_data(report_id, date)",
                "CREATE INDEX IF NOT EXISTS idx_chart_data_is_anomaly ON chart_data(is_anomaly)",
                "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports(created_at)"
            ]
            
            for index_sql in indexes:
                cursor.execute(index_sql)
            
            conn.commit()
            logger.info("Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise RuntimeError(f"Database initialization failed: {str(e)}")

def load_model_safely():
    global model_bundle, model, scaler, feature_columns
    
    with model_lock:
        if model_bundle is not None:
            return True
            
        try:
            model_path = Path("models/svm.joblib")
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            model_bundle = joblib.load(model_path)
            
            required_keys = ["model", "scaler", "features"]
            missing_keys = [key for key in required_keys if key not in model_bundle]
            if missing_keys:
                raise ValueError(f"Model bundle missing required keys: {missing_keys}")
            
            model = model_bundle["model"]
            scaler = model_bundle["scaler"]
            feature_columns = model_bundle["features"]
            
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Model loading failed: {str(e)}")

try:
    load_model_safely()
except Exception as e:
    logger.error(f"Critical error during startup: {e}")
    sys.exit(1)

app = FastAPI(
    title="Stock Anomaly Detection API",
    version="2.0.0",
    description="Advanced stock anomaly detection using machine learning",
    docs_url="/docs"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

DB_PATH = "anomaly_reports.db"
DB_TIMEOUT = 30.0

@contextmanager
def get_db_connection(timeout: float = DB_TIMEOUT):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=timeout)
        conn.execute("PRAGMA foreign_keys = ON") 
        conn.execute("PRAGMA journal_mode = WAL") 
        yield conn
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected database error: {e}")
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail="Internal database error")
    finally:
        if conn:
            conn.close()


class ReportStatus(BaseModel):
    id: str
    filename: str
    created_at: str
    status: str
    total_records: Optional[int] = 0
    anomaly_count: Optional[int] = 0
    error_message: Optional[str] = None
    threshold_percentile: Optional[float] = 8.0

class StatsResponse(BaseModel):
    total_records: int
    total_tickers: int
    last_update: Optional[str]

class AnomalySummary(BaseModel):
    date: str
    close: float
    volume: float
    anomaly_score: float
    is_anomaly: bool
    severity: str
    type: str
    change_percent: float
    
    @validator('close', 'volume')
    def validate_positive_numbers(cls, v):
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v

class ChartDataPoint(BaseModel):
    date: str
    close: float
    volume: float
    is_anomaly: bool
    anomaly_score: float

class ReportSummary(BaseModel):
    id: str
    filename: str
    created_at: str
    status: str
    total_records: int
    anomaly_count: int
    anomaly_percentage: float

class PaginatedAnomalies(BaseModel):
    anomalies: List[AnomalySummary]
    total: int
    page: int
    page_size: int
    total_pages: int

class UploadResponse(BaseModel):
    report_id: str
    status: str
    message: str
    threshold_percentile: Optional[float] = None

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize stock dataset columns with enhanced error handling"""
    try:
        # Clean column names
        original_columns = df.columns.tolist()
        cols = (
            df.columns.str.strip()
            .str.replace(r"[\$/]", "", regex=True)
            .str.replace(" ", "_")
            .str.lower()
        )
        df.columns = cols

        # Enhanced rename mapping
        rename_map = {
            "date": "Date", "index": "Index", "open": "Open", "open_price": "Open",
            "high": "High", "high_price": "High", "low": "Low", "low_price": "Low",
            "close": "Close", "close_last": "Close", "close/last": "Close", "closelast": "Close", 
            "closing_price": "Close", "volume": "Volume", "trading_volume": "Volume",
            "adj_close": "Adj_Close", "adjusted_close": "Adj_Close"
        }

        df = df.rename(columns=lambda c: rename_map.get(c, c))

        logger.debug(f"Original columns: {original_columns}")
        logger.debug(f"Normalized columns: {df.columns.tolist()}")

        # Check for required columns
        required = {"Date", "Open", "High", "Low", "Close", "Volume"}
        available = set(df.columns)
        missing = required - available
        
        if missing:
            # Try alternative column names
            alternatives = {
                "Date": ["date", "timestamp", "time"],
                "Open": ["opening_price", "open_price"],
                "High": ["high_price", "maximum"],
                "Low": ["low_price", "minimum"],
                "Close": ["closing_price", "close_price", "last_price"],
                "Volume": ["vol", "trading_volume", "shares_traded"]
            }
            
            for req_col in missing.copy():
                for alt in alternatives.get(req_col, []):
                    if alt in available:
                        df = df.rename(columns={alt: req_col})
                        missing.remove(req_col)
                        break
        
        if missing:
            raise ValueError(f"Missing required columns after normalization: {missing}")

        return df
        
    except Exception as e:
        logger.error(f"Column normalization failed: {e}")
        raise ValueError(f"Column normalization failed: {str(e)}")

def preprocess_data(csv_bytes: bytes) -> tuple[pd.DataFrame, np.ndarray]:
    try:
        try:
            df = pd.read_csv(io.BytesIO(csv_bytes))
        except pd.errors.EmptyDataError:
            raise ValueError("CSV file is empty")
        except pd.errors.ParserError as e:
            raise ValueError(f"CSV parsing error: {str(e)}")
        
        if df.empty:
            raise ValueError("CSV file contains no data")
        
        if len(df) < 30:  
            raise ValueError("Insufficient data points (minimum 30 required)")
        
        df = normalize_columns(df)
        
        price_columns = ["Open", "High", "Low", "Close"]
        for col in price_columns:
            try:
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r"[\$,]", "", regex=True)
                    .str.strip()
                )
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception as e:
                logger.error(f"Error processing {col}: {e}")
                raise ValueError(f"Error processing price column {col}: {str(e)}")
        
        try:
            df["Volume"] = (
                df["Volume"].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("", "0")
            )
            df["Volume"] = pd.to_numeric(df["Volume"], errors='coerce').fillna(0)
        except Exception as e:
            logger.error(f"Error processing Volume: {e}")
            raise ValueError(f"Error processing volume column: {str(e)}")
        
        try:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            if df["Date"].isna().all():
                raise ValueError("No valid dates found in Date column")
        except Exception as e:
            logger.error(f"Error processing dates: {e}")
            raise ValueError(f"Date parsing failed: {str(e)}")
        
        initial_count = len(df)
        df = df.dropna().reset_index(drop=True)
        
        for col in price_columns:
            if (df[col] <= 0).any():
                logger.warning(f"Found non-positive values in {col}, removing rows")
                df = df[df[col] > 0].reset_index(drop=True)
        
        if df["Volume"].lt(0).any():
            logger.warning("Found negative volume values, setting to 0")
            df.loc[df["Volume"] < 0, "Volume"] = 0
        
        final_count = len(df)
        if final_count < initial_count * 0.5: 
            logger.warning(f"Significant data loss during cleaning: {initial_count} -> {final_count}")
        
        if final_count < 30:
            raise ValueError(f"Insufficient valid data after cleaning: {final_count} rows")
        
        df = df.sort_values("Date").reset_index(drop=True)
        
        try:
            df["Daily_Return"] = df["Close"].pct_change()
            df["Price_Range"] = df["High"] - df["Low"]
            df["Price_Change_Pct"] = (df["Close"] - df["Open"]) / df["Open"]
            df["Volume_Change"] = df["Volume"].pct_change()
            
            for window in [5, 10, 20]:
                df[f"Volume_MA_{window}"] = df["Volume"].rolling(window=window, min_periods=1).mean()
                df[f"MA_{window}"] = df["Close"].rolling(window=window, min_periods=1).mean()
                df[f"MA_{window}_Ratio"] = df["Close"] / df[f"MA_{window}"]
            
            df["Volume_Ratio"] = df["Volume"] / df["Volume_MA_5"]
            
            df["Volatility_5"] = df["Daily_Return"].rolling(window=5, min_periods=1).std()
            df["Volatility_10"] = df["Daily_Return"].rolling(window=10, min_periods=1).std()
            
            def calculate_rsi(prices, window=14):
                delta = prices.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
                rs = gain / loss.replace(0, np.nan)
                return 100 - (100 / (1 + rs))

            df["RSI"] = calculate_rsi(df["Close"])
            
            exp1 = df["Close"].ewm(span=12).mean()
            exp2 = df["Close"].ewm(span=26).mean()
            df["MACD"] = exp1 - exp2
            df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
            df["MACD_Histogram"] = df["MACD"] - df["MACD_Signal"]

            df["BB_Middle"] = df["Close"].rolling(window=20, min_periods=1).mean()
            bb_std = df["Close"].rolling(window=20, min_periods=1).std()
            df["BB_Upper"] = df["BB_Middle"] + (bb_std * 2)
            df["BB_Lower"] = df["BB_Middle"] - (bb_std * 2)
            df["BB_Width"] = df["BB_Upper"] - df["BB_Lower"]
            
            bb_range = df["BB_Upper"] - df["BB_Lower"]
            df["BB_Position"] = np.where(
                bb_range > 0,
                (df["Close"] - df["BB_Lower"]) / bb_range,
                0.5 
            )
            
        except Exception as e:
            logger.error(f"Feature engineering error: {e}")
            raise ValueError(f"Feature engineering failed: {str(e)}")
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(method='ffill').fillna(method='bfill').dropna()
        
        if len(df) < 20:
            raise ValueError("Insufficient data after feature engineering")
        
        try:
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                logger.error(f"Missing features: {missing_features}")
                raise ValueError(f"Missing required features: {missing_features}")
            
            X = df[feature_columns]
            
            if X.isna().any().any():
                logger.warning("Found NaN values in features, filling with median")
                X = X.fillna(X.median())
            
            if np.isinf(X.values).any():
                logger.warning("Found infinite values in features, replacing with extreme values")
                X = X.replace([np.inf, -np.inf], [X.max().max(), X.min().min()])
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            raise ValueError(f"Feature extraction failed: {str(e)}")
        
        logger.info(f"Preprocessing completed: {len(df)} records, {len(feature_columns)} features")
        return df, X.values
        
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise

def detect_anomalies_with_threshold(model, X_scaled: np.ndarray, threshold_percentile: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    try:
        if model is None:
            raise ValueError("Model not loaded")
        
        if X_scaled.shape[0] == 0:
            raise ValueError("Empty input data")
        
        if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
            raise ValueError("Invalid values in scaled features")
        
        scores = model.decision_function(X_scaled)
        
        if np.isnan(scores).any() or np.isinf(scores).any():
            raise ValueError("Model produced invalid scores")
        
        threshold = np.percentile(scores, threshold_percentile)
        predictions = np.where(scores <= threshold, -1, 1)
        
        anomaly_count = np.sum(predictions == -1)
        anomaly_rate = anomaly_count / len(predictions) * 100
        
        logger.info(f"Anomaly detection completed:")
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Score range: [{scores.min():.4f}, {scores.max():.4f}]")
        logger.info(f"  Anomalies: {anomaly_count}/{len(predictions)} ({anomaly_rate:.2f}%)")
        
        if anomaly_rate > 50:
            logger.warning(f"High anomaly rate detected: {anomaly_rate:.2f}%")
        
        return predictions, scores
        
    except Exception as e:
        logger.error(f"Anomaly detection failed: {e}")
        raise ValueError(f"Anomaly detection failed: {str(e)}")

async def process_anomaly_detection(report_id: str, csv_bytes: bytes, filename: str, threshold_percentile: float = 8.0):
    logger.info(f"Starting processing for report {report_id} with threshold {threshold_percentile}%")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            try:
                loop = asyncio.get_event_loop()
                df, X = await loop.run_in_executor(executor, preprocess_data, csv_bytes)
                
                X_scaled = scaler.transform(X)
                
                preds, scores = await loop.run_in_executor(
                    executor, detect_anomalies_with_threshold, model, X_scaled, threshold_percentile
                )
                
                anomaly_count = int(np.sum(preds == -1))
                total_records = len(df)
                
                logger.info(f"Report {report_id}: {anomaly_count}/{total_records} anomalies ({100*anomaly_count/total_records:.2f}%)")
                
                anomalies_data = []
                chart_data = []
                
                for i, row in df.iterrows():
                    is_anomaly = bool(preds[i] == -1)
                    anomaly_score = float(scores[i])
                    date_str = row["Date"].strftime("%Y-%m-%d")
                    
                    close_price = float(row["Close"])
                    volume = float(row["Volume"])
                    
                    if close_price <= 0 or np.isnan(close_price):
                        logger.warning(f"Invalid close price at {date_str}: {close_price}")
                        continue
                    
                    if volume < 0 or np.isnan(volume):
                        volume = 0
                    
                    anomalies_data.append((
                        report_id, date_str, close_price, volume, anomaly_score, is_anomaly
                    ))
                    
                    chart_data.append((
                        report_id, date_str, close_price, volume, is_anomaly, anomaly_score
                    ))
                
                if not anomalies_data:
                    raise ValueError("No valid data points to insert")
                
                cursor.executemany(
                    "INSERT INTO anomalies (report_id, date, close_price, volume, anomaly_score, is_anomaly) VALUES (?, ?, ?, ?, ?, ?)",
                    anomalies_data
                )
                
                cursor.executemany(
                    "INSERT INTO chart_data (report_id, date, close_price, volume, is_anomaly, anomaly_score) VALUES (?, ?, ?, ?, ?, ?)",
                    chart_data
                )
                
                cursor.execute(
                    """UPDATE reports SET 
                       status = 'completed', 
                       total_records = ?, 
                       anomaly_count = ?,
                       threshold_percentile = ?
                       WHERE id = ?""",
                    (total_records, anomaly_count, threshold_percentile, report_id)
                )
                
                conn.commit()
                logger.info(f"Successfully completed processing for report {report_id}")
                
            except sqlite3.Error as e:
                logger.error(f"Database error in report {report_id}: {e}")
                cursor.execute(
                    "UPDATE reports SET status = 'failed', error_message = ? WHERE id = ?",
                    (f"Database error: {str(e)}", report_id)
                )
                conn.commit()
                
            except Exception as e:
                logger.error(f"Processing error in report {report_id}: {e}")
                cursor.execute(
                    "UPDATE reports SET status = 'failed', error_message = ? WHERE id = ?",
                    (str(e), report_id)
                )
                conn.commit()
                
    except Exception as e:
        logger.error(f"Critical error processing report {report_id}: {e}")
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE reports SET status = 'failed', error_message = ? WHERE id = ?",
                    (f"Critical error: {str(e)}", report_id)
                )
                conn.commit()
        except:
            logger.error(f"Failed to update error status for report {report_id}")

def validate_csv_file(file: UploadFile = File(...)) -> UploadFile:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    if not file.filename.lower().endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")
    
    max_size = 10 * 1024 * 1024  # 10MB
    if hasattr(file, 'size') and file.size > max_size:
        raise HTTPException(status_code=400, detail="File too large (max 10MB)")
    
    return file

@app.post("/upload-and-analyze", response_model=UploadResponse)
async def upload_and_analyze(
    background_tasks: BackgroundTasks, 
    file: UploadFile = Depends(validate_csv_file)
):
    """Upload CSV and start background processing with default threshold"""
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        report_id = str(uuid.uuid4())
        
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO reports (id, filename, status) VALUES (?, ?, ?)",
                    (report_id, file.filename, "processing")
                )
                conn.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=500, detail="Failed to create report record")
        
        background_tasks.add_task(process_anomaly_detection, report_id, content, file.filename,threshold_percentile=5)
        
        return UploadResponse(
            report_id=report_id,
            status="processing",
            message="Analysis started with default settings"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.post("/upload-and-analyze-custom", response_model=UploadResponse)
async def upload_and_analyze_custom(
    background_tasks: BackgroundTasks,
    threshold_percentile: float,
    file: UploadFile = Depends(validate_csv_file)
):
    if not 1.0 <= threshold_percentile <= 25.0:
        raise HTTPException(
            status_code=400, 
            detail="Threshold percentile must be between 1.0 and 25.0"
        )
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        report_id = str(uuid.uuid4())
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO reports (id, filename, status, threshold_percentile) VALUES (?, ?, ?, ?)",
                (report_id, file.filename, "processing", threshold_percentile)
            )
            conn.commit()
        
        background_tasks.add_task(
            process_anomaly_detection, 
            report_id, 
            content, 
            file.filename, 
            threshold_percentile
        )
        
        return UploadResponse(
            report_id=report_id,
            status="processing",
            message=f"Analysis started with {threshold_percentile}% threshold",
            threshold_percentile=threshold_percentile
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom upload error: {e}")
        raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

@app.get("/reports", response_model=List[ReportSummary])
async def get_reports():
    """Get all reports with enhanced error handling"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, filename, created_at, status, 
                       COALESCE(total_records, 0) as total_records,
                       COALESCE(anomaly_count, 0) as anomaly_count
                FROM reports 
                ORDER BY created_at DESC
                LIMIT 100
            """)
            
            reports = []
            for row in cursor.fetchall():
                anomaly_percentage = (row[5] / row[4] * 100) if row[4] > 0 else 0
                reports.append(ReportSummary(
                    id=row[0],
                    filename=row[1],
                    created_at=row[2],
                    status=row[3],
                    total_records=row[4],
                    anomaly_count=row[5],
                    anomaly_percentage=round(anomaly_percentage, 2)
                ))
            
            return reports
            
    except Exception as e:
        logger.error(f"Error fetching reports: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch reports")

@app.get("/reports/{report_id}/status", response_model=ReportStatus)
async def get_report_status(report_id: str):
    """Get status of a specific report with validation"""
    if not report_id or len(report_id) != 36:  # UUID length validation
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
            report = cursor.fetchone()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return ReportStatus(
                id=report[0],
                filename=report[1],
                created_at=report[2],
                status=report[3],
                total_records=report[4] or 0,
                anomaly_count=report[5] or 0,
                error_message=report[6],
                threshold_percentile=report[7] or 8.0
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching report status {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch report status")

@app.get("/reports/{report_id}/anomalies", response_model=PaginatedAnomalies)
async def get_anomalies(
    report_id: str,
    page: int = 1,
    page_size: int = 50,
    anomalies_only: bool = False
):
    if not report_id or len(report_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    if page < 1:
        raise HTTPException(status_code=400, detail="Page must be >= 1")
    
    if not 1 <= page_size <= 1000:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 1000")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT status FROM reports WHERE id = ?", (report_id,))
            report = cursor.fetchone()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            if report[0] == "processing":
                raise HTTPException(status_code=202, detail="Report is still processing")
            
            if report[0] == "failed":
                raise HTTPException(status_code=400, detail="Report processing failed")
            
            where_clause = "WHERE report_id = ?"
            params = [report_id]
            
            if anomalies_only:
                where_clause += " AND is_anomaly = 1"
            
            cursor.execute(f"SELECT COUNT(*) FROM anomalies {where_clause}", params)
            total = cursor.fetchone()[0]
            
            if total == 0:
                return PaginatedAnomalies(
                    anomalies=[],
                    total=0,
                    page=page,
                    page_size=page_size,
                    total_pages=0
                )
            
            total_pages = (total + page_size - 1) // page_size
            if page > total_pages:
                raise HTTPException(status_code=400, detail=f"Page {page} exceeds total pages {total_pages}")
            
            offset = (page - 1) * page_size
            cursor.execute(f"""
                SELECT date, close_price, volume, anomaly_score, is_anomaly
                FROM anomalies 
                {where_clause}
                ORDER BY date DESC
                LIMIT ? OFFSET ?
            """, params + [page_size, offset])
            
            anomalies = []
            for row in cursor.fetchall():
                try:
                    anomalies.append(AnomalySummary(
                        date=row[0],
                        close=row[1],
                        volume=row[2],
                        anomaly_score=row[3],
                        is_anomaly=bool(row[4])
                    ))
                except Exception as e:
                    logger.warning(f"Skipping invalid anomaly record: {e}")
                    continue
            
            return PaginatedAnomalies(
                anomalies=anomalies,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching anomalies for report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomalies")

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Retrieve global stats for dashboard header"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT SUM(total_records) FROM reports WHERE status = 'completed'")
            total_records = cursor.fetchone()[0] or 0

            cursor.execute("SELECT COUNT(DISTINCT filename) FROM reports WHERE status = 'completed'")
            total_tickers = cursor.fetchone()[0] or 0

            cursor.execute("SELECT MAX(created_at) FROM reports WHERE status = 'completed'")
            last_update = cursor.fetchone()[0]

            return StatsResponse(
                total_records=total_records,
                total_tickers=total_tickers,
                last_update=last_update if last_update else None
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")

@app.get("/anomalies", response_model=PaginatedAnomalies)
async def get_all_anomalies(
    page: int = Query(1, ge=1, description="Page number, must be >= 1"),
    page_size: int = Query(50, ge=1, le=1000, description="Items per page, 1–1000"),
    severity: str = Query(None, description="Filter by severity"),
    type_: str = Query(None, alias="type", description="Filter by anomaly type"),
    search: str = Query(None, description="Search anomalies by date or type"),
    sort_by: str = Query("date", description="Column to sort by"),
    sort_order: str = Query("desc", regex="^(asc|desc)$", description="Sort order"),
):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            base_query = """
                SELECT 
                    a.report_id,
                    a.date,
                    a.close_price,
                    a.volume,
                    a.anomaly_score,
                    a.is_anomaly,
                    r.threshold_percentile,
                    (
                        SELECT prev.close_price 
                        FROM anomalies AS prev
                        WHERE prev.report_id = a.report_id
                          AND prev.date < a.date
                        ORDER BY prev.date DESC
                        LIMIT 1
                    ) AS prev_close
                FROM anomalies a
                JOIN reports r ON a.report_id = r.id
                WHERE a.is_anomaly = 1
            """

            filters = []
            params = []

            if severity in ["low", "medium", "high"]:
                filters.append("1=1") 

            if type_ in ["price", "volume", "combined"]:
                filters.append("1=1") 

            if search:
                filters.append("a.date LIKE ?")
                params.append(f"%{search}%")

            if filters:
                base_query += " AND " + " AND ".join(filters)

            valid_sort_columns = {
                "date": "a.date",
                "close": "a.close_price",
                "anomaly_score": "a.anomaly_score"
            }
            sort_col = valid_sort_columns.get(sort_by, "a.date")
            sort_dir = "ASC" if sort_order == "asc" else "DESC"

            base_query += f" ORDER BY {sort_col} {sort_dir}"

            offset = (page - 1) * page_size
            base_query += " LIMIT ? OFFSET ?"
            params.extend([page_size, offset])

            cursor.execute(base_query, params)
            rows = cursor.fetchall()

            anomalies = []
            for row in rows:
                date, close, volume, score, is_anomaly, threshold, prev_close = (
                    row[1], row[2], row[3], row[4], row[5], row[6], row[7]
                )

                if abs(score) < 1.0:
                    sev = "low"
                elif abs(score) < 2.0:
                    sev = "medium"
                else:
                    sev = "high"

                change_percent = 0.0
                if prev_close and prev_close > 0:
                    change_percent = ((close - prev_close) / prev_close) * 100

                type_val = "combined"
                if abs(change_percent) > threshold and abs(volume) < 1.5 * 10**6:
                    type_val = "price"
                elif volume > 2 * 10**6 and abs(change_percent) < threshold:
                    type_val = "volume"

                if severity and sev != severity:
                    continue
                if type_ and type_val != type_:
                    continue

                anomalies.append({
                    "date": date,
                    "close": close,
                    "volume": volume,
                    "anomaly_score": score,
                    "is_anomaly": bool(is_anomaly),
                    "severity": sev,
                    "type": type_val,
                    "change_percent": round(change_percent, 2)
                })

            count_query = "SELECT COUNT(*) FROM anomalies a WHERE a.is_anomaly = 1"
            if search:
                count_query += " AND a.date LIKE ?"
                cursor.execute(count_query, [f"%{search}%"])
            else:
                cursor.execute(count_query)
            total = cursor.fetchone()[0]

            total_pages = (total + page_size - 1) // page_size

            return PaginatedAnomalies(
                anomalies=anomalies,
                total=total,
                page=page,
                page_size=page_size,
                total_pages=total_pages
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching enriched anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch anomalies")

@app.get("/reports/{report_id}/chart-data", response_model=List[ChartDataPoint])
async def get_chart_data(report_id: str, limit: int = 1000):
    """Get optimized chart data with intelligent sampling"""
    if not report_id or len(report_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    if not 100 <= limit <= 10000:
        raise HTTPException(status_code=400, detail="Limit must be between 100 and 10000")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT status, total_records FROM reports WHERE id = ?", (report_id,))
            report_info = cursor.fetchone()
            
            if not report_info:
                raise HTTPException(status_code=404, detail="Report not found")
            
            if report_info[0] == "processing":
                raise HTTPException(status_code=202, detail="Report is still processing")
            
            if report_info[0] == "failed":
                raise HTTPException(status_code=400, detail="Report processing failed")
            
            total_records = report_info[1] or 0
            
            if total_records == 0:
                return []
            
            if total_records <= limit:
                cursor.execute("""
                    SELECT date, close_price, volume, is_anomaly, anomaly_score
                    FROM chart_data 
                    WHERE report_id = ?
                    ORDER BY date
                """, (report_id,))
                all_points = cursor.fetchall()
                
            else:
                cursor.execute("""
                    SELECT date, close_price, volume, is_anomaly, anomaly_score
                    FROM chart_data 
                    WHERE report_id = ? AND is_anomaly = 1
                    ORDER BY date
                """, (report_id,))
                anomalies = cursor.fetchall()
                
                anomaly_count = len(anomalies)
                normal_limit = max(100, limit - anomaly_count)
                
                if normal_limit > 0:
                    cursor.execute("""
                        SELECT date, close_price, volume, is_anomaly, anomaly_score,
                               ROW_NUMBER() OVER (ORDER BY date) as rn,
                               COUNT(*) OVER () as total
                        FROM chart_data 
                        WHERE report_id = ? AND is_anomaly = 0
                        ORDER BY date
                    """, (report_id,))
                    
                    normal_data = cursor.fetchall()
                    if normal_data:
                        total_normal = normal_data[0][6]
                        step = max(1, total_normal // normal_limit)
                        
                        normal_points = [
                            row[:5] for i, row in enumerate(normal_data) 
                            if i % step == 0
                        ][:normal_limit]
                    else:
                        normal_points = []
                else:
                    normal_points = []
                
                all_points = anomalies + normal_points
                all_points.sort(key=lambda x: x[0])
            
            chart_data = []
            for row in all_points:
                try:
                    chart_data.append(ChartDataPoint(
                        date=row[0],
                        close=row[1],
                        volume=row[2],
                        is_anomaly=bool(row[3]),
                        anomaly_score=row[4]
                    ))
                except Exception as e:
                    logger.warning(f"Skipping invalid chart data point: {e}")
                    continue
            
            logger.info(f"Returning {len(chart_data)} chart points for report {report_id}")
            return chart_data
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chart data for report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chart data")

@app.delete("/reports/{report_id}")
async def delete_report(report_id: str):
    """Delete report with enhanced validation and cleanup"""
    if not report_id or len(report_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT id, status FROM reports WHERE id = ?", (report_id,))
            report = cursor.fetchone()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            if report[1] == "processing":
                raise HTTPException(status_code=400, detail="Cannot delete report while processing")
            
            tables_to_clean = ["chart_data", "anomalies", "reports"]
            
            for table in tables_to_clean:
                if table == "reports":
                    cursor.execute(f"DELETE FROM {table} WHERE id = ?", (report_id,))
                else:
                    cursor.execute(f"DELETE FROM {table} WHERE report_id = ?", (report_id,))
                deleted_count = cursor.rowcount
                logger.info(f"Deleted {deleted_count} rows from {table}")
            
            conn.commit()
            logger.info(f"Successfully deleted report {report_id}")
            
            return {"message": "Report deleted successfully", "report_id": report_id}
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete report")

@app.get("/reports/{report_id}", response_model=ReportStatus)
async def get_report(report_id: str):
    """Get a single report with enhanced validation and error handling"""
    if not report_id or len(report_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM reports WHERE id = ?", (report_id,))
            report = cursor.fetchone()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return ReportStatus(
                id=report[0],
                filename=report[1],
                created_at=report[2],
                status=report[3],
                total_records=report[4] or 0,
                anomaly_count=report[5] or 0,
                error_message=report[6],
                threshold_percentile=report[7] or 8.0
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch report")

@app.get("/reports/{report_id}/statistics")
async def get_report_statistics(report_id: str):
    if not report_id or len(report_id) != 36:
        raise HTTPException(status_code=400, detail="Invalid report ID format")
    
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT status, total_records, anomaly_count, threshold_percentile FROM reports WHERE id = ?", (report_id,))
            report = cursor.fetchone()
            
            if not report:
                raise HTTPException(status_code=404, detail="Report not found")
            
            if report[0] != "completed":
                raise HTTPException(status_code=400, detail="Report not completed")
            
            cursor.execute("""
                SELECT 
                    MIN(anomaly_score) as min_score,
                    MAX(anomaly_score) as max_score,
                    AVG(anomaly_score) as avg_score,
                    AVG(CASE WHEN is_anomaly = 1 THEN anomaly_score END) as avg_anomaly_score,
                    AVG(CASE WHEN is_anomaly = 0 THEN anomaly_score END) as avg_normal_score
                FROM anomalies 
                WHERE report_id = ?
            """, (report_id,))
            
            score_stats = cursor.fetchone()
            
            cursor.execute("""
                SELECT MIN(date), MAX(date) 
                FROM anomalies 
                WHERE report_id = ?
            """, (report_id,))
            
            date_range = cursor.fetchone()
            
            cursor.execute("""
                SELECT 
                    substr(date, 1, 7) as month,
                    COUNT(*) as total,
                    SUM(CASE WHEN is_anomaly = 1 THEN 1 ELSE 0 END) as anomalies
                FROM anomalies 
                WHERE report_id = ?
                GROUP BY substr(date, 1, 7)
                ORDER BY month
            """, (report_id,))
            
            monthly_stats = cursor.fetchall()
            
            return {
                "report_id": report_id,
                "total_records": report[1],
                "anomaly_count": report[2],
                "anomaly_percentage": round((report[2] / report[1]) * 100, 2) if report[1] > 0 else 0,
                "threshold_percentile": report[3],
                "date_range": {
                    "start": date_range[0],
                    "end": date_range[1]
                },
                "score_statistics": {
                    "min_score": round(score_stats[0], 4) if score_stats[0] else 0,
                    "max_score": round(score_stats[1], 4) if score_stats[1] else 0,
                    "avg_score": round(score_stats[2], 4) if score_stats[2] else 0,
                    "avg_anomaly_score": round(score_stats[3], 4) if score_stats[3] else 0,
                    "avg_normal_score": round(score_stats[4], 4) if score_stats[4] else 0
                },
                "monthly_distribution": [
                    {
                        "month": month,
                        "total": total,
                        "anomalies": anomalies,
                        "anomaly_rate": round((anomalies / total) * 100, 2) if total > 0 else 0
                    }
                    for month, total, anomalies in monthly_stats
                ]
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching statistics for report {report_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch report statistics")

@app.get("/health")
async def health_check():
    try:
        model_status = model is not None and scaler is not None
        
        db_status = False
        try:
            with get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                db_status = True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
        
        db_stats = {}
        if db_status:
            try:
                with get_db_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM reports")
                    total_reports = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM reports WHERE status = 'completed'")
                    completed_reports = cursor.fetchone()[0]
                    
                    cursor.execute("SELECT COUNT(*) FROM reports WHERE status = 'processing'")
                    processing_reports = cursor.fetchone()[0]
                    
                    db_stats = {
                        "total_reports": total_reports,
                        "completed_reports": completed_reports,
                        "processing_reports": processing_reports
                    }
            except Exception as e:
                logger.error(f"Database stats check failed: {e}")
                db_stats = {"error": "Failed to fetch database statistics"}
        
        overall_status = "healthy" if (model_status and db_status) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "components": {
                "model": "ok" if model_status else "error",
                "database": "ok" if db_status else "error"
            },
            "database_statistics": db_stats,
            "version": "2.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }

@app.get("/system/cleanup")
async def cleanup_old_reports(days_old: int = 30):
    """Clean up old reports (admin endpoint)"""
    if not 1 <= days_old <= 365:
        raise HTTPException(status_code=400, detail="Days must be between 1 and 365")
    
    try:
        cutoff_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id FROM reports 
                WHERE created_at < datetime('now', '-{} days')
            """.format(days_old))
            
            old_reports = [row[0] for row in cursor.fetchall()]
            
            if not old_reports:
                return {
                    "message": "No old reports found",
                    "deleted_count": 0,
                    "cutoff_days": days_old
                }
            
            deleted_count = 0
            for report_id in old_reports:
                try:
                    cursor.execute("DELETE FROM chart_data WHERE report_id = ?", (report_id,))
                    cursor.execute("DELETE FROM anomalies WHERE report_id = ?", (report_id,))
                    cursor.execute("DELETE FROM reports WHERE id = ?", (report_id,))
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete old report {report_id}: {e}")
                    continue
            
            conn.commit()
            logger.info(f"Cleaned up {deleted_count} old reports")
            
            return {
                "message": f"Cleaned up old reports successfully",
                "deleted_count": deleted_count,
                "cutoff_days": days_old,
                "cutoff_date": cutoff_date
            }
            
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail="Cleanup operation failed")

@app.get("/anomalies/export")
async def export_anomalies(report_id: str = Query("0", description="Report ID, or '0' for all reports")):
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()

            if report_id == "0":
                cursor.execute("""
                    SELECT 
                        a.report_id,
                        a.date,
                        a.close_price,
                        a.anomaly_score
                    FROM anomalies a
                    JOIN reports r ON a.report_id = r.id
                    WHERE a.is_anomaly = 1
                    ORDER BY a.date DESC
                """)
            else:
                cursor.execute("""
                    SELECT 
                        a.report_id,
                        a.date,
                        a.close_price,
                        a.anomaly_score
                    FROM anomalies a
                    JOIN reports r ON a.report_id = r.id
                    WHERE a.is_anomaly = 1 AND a.report_id = ?
                    ORDER BY a.date DESC
                """, (report_id,))

            rows = cursor.fetchall()
            if not rows:
                raise HTTPException(status_code=404, detail="No anomalies found for the given report")

            headers = [desc[0] for desc in cursor.description]

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(headers)
            writer.writerows(rows)

            response = StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv"
            )

            filename = "all_anomalies.csv" if report_id == "0" else f"report_{report_id}.csv"
            response.headers["Content-Disposition"] = f"attachment; filename={filename}"
            return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting anomalies: {e}")
        raise HTTPException(status_code=500, detail="Failed to export anomalies")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return HTTPException(status_code=500, detail="Internal server error occurred")

if __name__ == "__main__":
    import uvicorn
    
    try:
        load_model_safely()
        init_db()
        logger.info("Application startup validation completed successfully")
        init_db()
    except Exception as e:
        logger.error(f"Startup validation failed: {e}")
        sys.exit(1)
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True,
        log_level="info"
    )
    