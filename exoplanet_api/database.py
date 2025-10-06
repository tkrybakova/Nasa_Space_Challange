from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
<<<<<<< HEAD
from exoplanet_api.config import settings  # ← исправлено: относительный импорт

# Создание подключения к базе данных
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)

# Сессия
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Базовый класс для моделей
Base = declarative_base()


# Зависимость для получения сессии БД в эндпоинтах
=======
from config import settings

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


>>>>>>> 6d826bd89a7a829950e92fd0513ba4ef97d9d2f8
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
