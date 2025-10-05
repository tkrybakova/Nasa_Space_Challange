import io
import json
import logging
from contextlib import asynccontextmanager
from typing import List

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

# Локальные импорты
from exoplanet_api import models, schemas, crud
from exoplanet_api.database import engine, get_db
from exoplanet_api.ml_model import ml_model, cached_predict

# ----------------- Настройки логирования -----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------- Создаем таблицы -----------------
models.Base.metadata.create_all(bind=engine)

# ----------------- FastAPI и CORS -----------------
PARQUET_PATH = "exoplanet_api/data/combined_exoplanet_dataset_imputed.parquet"
COLUMNS = ["name", "ra", "dec", "pl_rade", "pl_masse", "planet_class"]

@asynccontextmanager
async def lifespan(app: FastAPI):  # <--- здесь нужно app
    try:
        ml_model.load_model()
        logger.info("ML model initialized successfully ✅")
    except Exception as e:
        logger.warning(f"ML model initialization failed: {str(e)}")
    yield
    logger.info("Shutting down app...")

app = FastAPI(title="Exoplanet Analysis API", version="1.0.0", lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # для фронта можно указать localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analysis_tasks = {}  # хранение состояния фоновых задач

# ----------------- Утилиты -----------------
def get_exoplanets(limit: int = 100):
    """Возвращает первые limit записей из Parquet"""
    df = pd.read_parquet(PARQUET_PATH, columns=COLUMNS)
    return df.head(limit).to_dict(orient="records")

# ----------------- Эндпоинты -----------------
@app.get("/")
async def root():
    return {"message": "Exoplanet Analysis API", "version": "1.0.0"}

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """Проверка состояния БД и ML модели"""
    db_status = True
    try:
        db.execute("SELECT 1")
    except Exception:
        db_status = False

    ml_status = False
    try:
        ml_model.load_model()
        ml_status = ml_model.model is not None
    except Exception:
        pass

    return {
        "status": "healthy",
        "database": db_status,
        "ml_model": ml_status,
        "timestamp": pd.Timestamp.now().isoformat()
    }
import pandas as pd
from fastapi import HTTPException

@app.get("/api/dataset")
def get_dataset():
    try:
        df = pd.read_csv("exoplanet_api/data/combined_exoplanet_dataset_extended.csv")

        # Заменяем бесконечности и NaN на None (чтобы JSON не падал)
        df = df.replace([float("inf"), float("-inf")], pd.NA)
        df = df.fillna(value="N/A")

        # Преобразуем всё в JSON-дружественный формат
        data = df.to_dict(orient="records")
        return data

    except FileNotFoundError:
        return {"detail": "Dataset file not found"}
    except Exception as e:
        return {"detail": f"Error loading dataset: {str(e)}"}


@app.get("/exoplanets/")
async def exoplanets(limit: int = Query(100, le=1000)):
    """Список экзопланет с ограничением limit"""
    data = get_exoplanets(limit)
    return {"count": len(data), "results": data}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user_id: int = 1, db: Session = Depends(get_db)):
    """Загрузка CSV/JSON файлов"""
    try:
        content = await file.read()

        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
            file_type = 'csv'
        elif file.filename.endswith('.json'):
            df = pd.DataFrame(json.loads(content))
            file_type = 'json'
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")

        required_columns = ['pl_orbper', 'pl_rade', 'pl_bmasse']
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

        file_record = schemas.UploadedFileCreate(
            user_id=user_id,
            filename=file.filename,
            file_type=file_type,
            content=content.decode('utf-8')
        )
        db_file = crud.create_uploaded_file(db, file_record)

        return {"file_id": db_file.id, "filename": db_file.filename, "rows": len(df), "columns": list(df.columns)}

    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Фоновая обработка анализа -----------------
def process_analysis_task(task_id: int, file_id: int = None, raw_data: List[dict] = None):
    from exoplanet_api.database import SessionLocal
    db = SessionLocal()

    try:
        analysis_tasks[task_id] = {"status": "processing"}

        if file_id:
            file_record = crud.get_uploaded_file(db, file_id)
            if file_record.file_type == 'csv':
                df = pd.read_csv(io.StringIO(file_record.content))
            else:
                df = pd.DataFrame(json.loads(file_record.content))
            planet_data = df.to_dict('records')
        else:
            planet_data = raw_data

        crud.update_analysis_session_status(db, task_id, "processing")

        predictions = cached_predict(planet_data)

        prediction_records = [
            schemas.ExoplanetPredictionCreate(
                session_id=task_id,
                planet_data=planet,
                prediction=pred['prediction'],
                confidence=pred['confidence'],
                features=pred
            )
            for planet, pred in zip(planet_data, predictions)
        ]

        crud.create_exoplanet_predictions(db, prediction_records)
        crud.update_analysis_session_status(db, task_id, "completed")
        analysis_tasks[task_id] = {"status": "completed", "results": predictions}

    except Exception as e:
        logger.error(f"Analysis task error: {str(e)}")
        crud.update_analysis_session_status(db, task_id, "failed")
        analysis_tasks[task_id] = {"status": "failed", "error": str(e)}
    finally:
        db.close()

@app.post("/api/analyze")
async def analyze_data(request: schemas.AnalysisRequest, background_tasks: BackgroundTasks, user_id: int = 1,
                       db: Session = Depends(get_db)):
    """Запуск анализа экзопланет в фоне"""
    try:
        session_data = schemas.AnalysisSessionCreate(
            user_id=user_id,
            filename=f"analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        )
        db_session = crud.create_analysis_session(db, session_data)

        background_tasks.add_task(process_analysis_task, db_session.id, request.file_id, request.raw_data)
        return schemas.AnalysisResponse(task_id=db_session.id, status="started")

    except Exception as e:
        logger.error(f"Analysis start error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/results/{task_id}")
async def get_results(task_id: int, db: Session = Depends(get_db)):
    db_session = crud.get_analysis_session(db, task_id)
    if not db_session:
        raise HTTPException(status_code=404, detail="Analysis session not found")

    predictions = [
        {
            "planet_data": pred.planet_data,
            "prediction": pred.prediction,
            "confidence": pred.confidence,
            "features": pred.features
        }
        for pred in db_session.predictions
    ]

    return {
        "task_id": task_id,
        "status": db_session.status,
        "created_at": db_session.created_at.isoformat(),
        "predictions": predictions,
        "background_status": analysis_tasks.get(task_id, {}).get("status", "unknown")
    }

# ----------------- Run -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("exoplanet_api.main:app", host="0.0.0.0", port=8000, reload=True)

