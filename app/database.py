"""
Database module for Cardiac Detection System.
SQLite database with SQLAlchemy ORM for patient and analysis management.
Compatible with SQLAlchemy 2.x.
"""

from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import json

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text,
    DateTime, ForeignKey, event
)
from sqlalchemy.orm import (
    DeclarativeBase, sessionmaker, relationship, Session
)

# ── Database path ────────────────────────────────────────────────────────────
DB_DIR = Path(__file__).parent.parent / 'data'
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / 'cardiac_data.db'

# Enable WAL mode for better concurrent read performance
engine = create_engine(
    f'sqlite:///{DB_PATH}',
    echo=False,
    connect_args={"check_same_thread": False},
)


@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, _connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


# ── ORM Base ─────────────────────────────────────────────────────────────────
class Base(DeclarativeBase):
    pass


# ── Models ───────────────────────────────────────────────────────────────────
class Patient(Base):
    """Patient information table."""

    __tablename__ = 'patients'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    age = Column(Integer)
    gender = Column(String(10))
    contact_info = Column(String(200))          # NEW: email / phone
    medical_history = Column(Text)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)

    analyses = relationship(
        "ECGAnalysis",
        back_populates="patient",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'contact_info': self.contact_info,
            'medical_history': self.medical_history,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'total_analyses': len(self.analyses),
        }


class ECGAnalysis(Base):
    """ECG analysis results table."""

    __tablename__ = 'ecg_analyses'

    id = Column(Integer, primary_key=True, autoincrement=True)
    patient_id = Column(
        String(50), ForeignKey('patients.patient_id', ondelete='CASCADE'),
        nullable=False, index=True
    )
    prediction = Column(String(100), nullable=False)
    confidence = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    heart_rate = Column(Float)
    hrv_sdnn = Column(Float)
    signal_length = Column(Integer)             # NEW: number of ECG samples
    probabilities_json = Column(Text)
    features_json = Column(Text)
    recommendations_json = Column(Text)
    analysis_timestamp = Column(DateTime, default=datetime.now, index=True)
    ecg_data_path = Column(String(500))

    patient = relationship("Patient", back_populates="analyses")
    reports = relationship(
        "GeneratedReport",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'patient_id': self.patient_id,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'risk_level': self.risk_level,
            'heart_rate': self.heart_rate,
            'hrv_sdnn': self.hrv_sdnn,
            'signal_length': self.signal_length,
            'probabilities': json.loads(self.probabilities_json) if self.probabilities_json else {},
            'features': json.loads(self.features_json) if self.features_json else {},
            'recommendations': json.loads(self.recommendations_json) if self.recommendations_json else [],
            'analysis_timestamp': self.analysis_timestamp.isoformat() if self.analysis_timestamp else None,
            'ecg_data_path': self.ecg_data_path,
        }


class GeneratedReport(Base):
    """Generated PDF reports table."""

    __tablename__ = 'generated_reports'

    id = Column(Integer, primary_key=True, autoincrement=True)
    analysis_id = Column(
        Integer, ForeignKey('ecg_analyses.id', ondelete='CASCADE'),
        nullable=False, index=True
    )
    report_path = Column(String(500), nullable=False)
    generated_at = Column(DateTime, default=datetime.now)

    analysis = relationship("ECGAnalysis", back_populates="reports")

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'analysis_id': self.analysis_id,
            'report_path': self.report_path,
            'generated_at': self.generated_at.isoformat() if self.generated_at else None,
        }


# ── Session helpers ───────────────────────────────────────────────────────────
@contextmanager
def get_db() -> Session:
    """Context-manager database session (preferred)."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_session() -> Session:
    """Return a raw session (caller is responsible for close/rollback)."""
    return SessionLocal()


# ── Initialization ────────────────────────────────────────────────────────────
_SAMPLE_PATIENTS = [
    {
        'patient_id': 'P001',
        'name': 'John Doe',
        'age': 45,
        'gender': 'Male',
        'contact_info': 'john.doe@example.com',
        'medical_history': 'Hypertension, controlled with medication',
    },
    {
        'patient_id': 'P002',
        'name': 'Jane Smith',
        'age': 52,
        'gender': 'Female',
        'contact_info': 'jane.smith@example.com',
        'medical_history': 'Type 2 Diabetes, family history of heart disease',
    },
    {
        'patient_id': 'P003',
        'name': 'Robert Johnson',
        'age': 68,
        'gender': 'Male',
        'contact_info': 'r.johnson@example.com',
        'medical_history': 'Previous MI (2020), on anticoagulants',
    },
]


def init_database() -> None:
    """Create all tables and seed sample patients if the table is empty."""
    Base.metadata.create_all(engine)
    print(f"✅ Database initialized at {DB_PATH}")

    with get_db() as session:
        existing_ids = {
            row[0] for row in session.query(Patient.patient_id).all()
        }
        new_patients = [
            Patient(**p)
            for p in _SAMPLE_PATIENTS
            if p['patient_id'] not in existing_ids
        ]
        if new_patients:
            session.add_all(new_patients)
            print(f"✅ Seeded {len(new_patients)} sample patient(s)")


if __name__ == "__main__":
    init_database()
