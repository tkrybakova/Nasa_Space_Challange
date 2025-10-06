from sqlalchemy.orm import Session
<<<<<<< HEAD
from typing import List, Optional
from exoplanet_api import models, schemas  # ← исправлено: относительный импорт

# 👤 Работа с пользователями
=======
from app import models, schemas
from typing import List

>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def create_user(db: Session, user: schemas.UserCreate):
    db_user = models.User(email=user.email)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

<<<<<<< HEAD

def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()


# 🧪 Работа с сессиями анализа
=======
def get_user(db: Session, user_id: int):
    return db.query(models.User).filter(models.User.id == user_id).first()

>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def create_analysis_session(db: Session, session: schemas.AnalysisSessionCreate):
    db_session = models.AnalysisSession(**session.dict())
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

<<<<<<< HEAD

def get_analysis_session(db: Session, session_id: int):
    return db.query(models.AnalysisSession).filter(models.AnalysisSession.id == session_id).first()


=======
def get_analysis_session(db: Session, session_id: int):
    return db.query(models.AnalysisSession).filter(models.AnalysisSession.id == session_id).first()

>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def update_analysis_session_status(db: Session, session_id: int, status: str):
    db_session = db.query(models.AnalysisSession).filter(models.AnalysisSession.id == session_id).first()
    if db_session:
        db_session.status = status
        db.commit()
        db.refresh(db_session)
    return db_session

<<<<<<< HEAD

# 🌍 Работа с предсказаниями
=======
>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def create_exoplanet_predictions(db: Session, predictions: List[schemas.ExoplanetPredictionCreate]):
    db_predictions = [models.ExoplanetPrediction(**pred.dict()) for pred in predictions]
    db.add_all(db_predictions)
    db.commit()
    for pred in db_predictions:
        db.refresh(pred)
    return db_predictions

<<<<<<< HEAD

# 📂 Работа с файлами
=======
>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def create_uploaded_file(db: Session, file: schemas.UploadedFileCreate):
    db_file = models.UploadedFile(**file.dict())
    db.add(db_file)
    db.commit()
    db.refresh(db_file)
    return db_file

<<<<<<< HEAD

def get_uploaded_file(db: Session, file_id: int):
    return db.query(models.UploadedFile).filter(models.UploadedFile.id == file_id).first()


# 📜 История анализа пользователя
def get_user_analysis_history(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return (
        db.query(models.AnalysisSession)
        .filter(models.AnalysisSession.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )


# ❌ Удаление сессии анализа
=======
def get_uploaded_file(db: Session, file_id: int):
    return db.query(models.UploadedFile).filter(models.UploadedFile.id == file_id).first()

def get_user_analysis_history(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return db.query(models.AnalysisSession).filter(
        models.AnalysisSession.user_id == user_id
    ).offset(skip).limit(limit).all()

>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def delete_analysis_session(db: Session, session_id: int):
    db_session = db.query(models.AnalysisSession).filter(models.AnalysisSession.id == session_id).first()
    if db_session:
        db.query(models.ExoplanetPrediction).filter(
            models.ExoplanetPrediction.session_id == session_id
        ).delete()
        db.delete(db_session)
        db.commit()
        return True
<<<<<<< HEAD
    return False
=======
    return False
>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
