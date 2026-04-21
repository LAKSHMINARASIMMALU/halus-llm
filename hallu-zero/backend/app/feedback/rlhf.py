"""
RLHF Feedback Loop:
  - Stores user feedback (thumbs up/down) with full query/response context
  - Aggregates reward signals
  - Generates fine-tuning prompts from high-quality positive/negative pairs
  - Exposes feedback stats for the dashboard
"""
import json
import os
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Boolean, DateTime, Text, Integer,
    create_engine, func, select
)
from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from config.settings import get_settings

settings = get_settings()


# ─── Database Models ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    pass


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id = Column(String, primary_key=True)
    query = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    context_sources = Column(Text, default="[]")  # JSON array
    rating = Column(Integer, nullable=False)       # 1 (positive) or -1 (negative)
    comment = Column(Text, default="")
    confidence_score = Column(Float, default=0.0)
    verification_passed = Column(Boolean, default=False)
    model_used = Column(String, default="")
    created_at = Column(DateTime, default=datetime.utcnow)
    used_for_training = Column(Boolean, default=False)


class TrainingPair(Base):
    __tablename__ = "training_pairs"

    id = Column(String, primary_key=True)
    query = Column(Text, nullable=False)
    chosen_response = Column(Text, nullable=False)    # positive example
    rejected_response = Column(Text, nullable=False)  # negative example
    reward_gap = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    exported = Column(Boolean, default=False)


# ─── Database Manager ─────────────────────────────────────────────────────────

class FeedbackDB:
    def __init__(self):
        db_path = settings.feedback_db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)

    def save_feedback(
        self,
        feedback_id: str,
        query: str,
        response: str,
        rating: int,
        confidence_score: float = 0.0,
        verification_passed: bool = False,
        context_sources: list[str] = None,
        comment: str = "",
        model_used: str = "",
    ) -> FeedbackRecord:
        record = FeedbackRecord(
            id=feedback_id,
            query=query,
            response=response,
            rating=rating,
            confidence_score=confidence_score,
            verification_passed=verification_passed,
            context_sources=json.dumps(context_sources or []),
            comment=comment,
            model_used=model_used,
        )
        with Session(self.engine) as session:
            session.add(record)
            session.commit()
            session.refresh(record)
        return record

    def get_stats(self) -> dict:
        with Session(self.engine) as session:
            total = session.scalar(select(func.count(FeedbackRecord.id)))
            positives = session.scalar(
                select(func.count(FeedbackRecord.id)).where(FeedbackRecord.rating == 1)
            )
            negatives = session.scalar(
                select(func.count(FeedbackRecord.id)).where(FeedbackRecord.rating == -1)
            )
            avg_conf = session.scalar(
                select(func.avg(FeedbackRecord.confidence_score))
            ) or 0.0
            verified = session.scalar(
                select(func.count(FeedbackRecord.id)).where(
                    FeedbackRecord.verification_passed == True
                )
            )
            pairs = session.scalar(select(func.count(TrainingPair.id)))

        return {
            "total_feedback": total or 0,
            "positives": positives or 0,
            "negatives": negatives or 0,
            "avg_confidence": round(avg_conf, 3),
            "verification_pass_rate": round((verified or 0) / max(total or 1, 1), 3),
            "training_pairs_available": pairs or 0,
            "satisfaction_rate": round((positives or 0) / max(total or 1, 1), 3),
        }

    def get_recent_feedback(self, limit: int = 20) -> list[dict]:
        with Session(self.engine) as session:
            records = session.execute(
                select(FeedbackRecord)
                .order_by(FeedbackRecord.created_at.desc())
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "id": r.id,
                    "query": r.query[:100],
                    "rating": r.rating,
                    "confidence_score": r.confidence_score,
                    "verification_passed": r.verification_passed,
                    "created_at": r.created_at.isoformat(),
                }
                for r in records
            ]

    def generate_training_pairs(self) -> int:
        """
        Matches positive/negative feedback on similar queries to create
        DPO-style (chosen, rejected) training pairs.
        """
        with Session(self.engine) as session:
            positives = session.execute(
                select(FeedbackRecord).where(
                    FeedbackRecord.rating == 1,
                    FeedbackRecord.used_for_training == False,
                )
            ).scalars().all()

            negatives = session.execute(
                select(FeedbackRecord).where(
                    FeedbackRecord.rating == -1,
                    FeedbackRecord.used_for_training == False,
                )
            ).scalars().all()

            pairs_created = 0
            import uuid
            neg_by_query = {n.query[:50]: n for n in negatives}

            for pos in positives:
                key = pos.query[:50]
                if key in neg_by_query:
                    neg = neg_by_query[key]
                    pair = TrainingPair(
                        id=str(uuid.uuid4()),
                        query=pos.query,
                        chosen_response=pos.response,
                        rejected_response=neg.response,
                        reward_gap=pos.confidence_score - neg.confidence_score,
                    )
                    session.add(pair)
                    pos.used_for_training = True
                    neg.used_for_training = True
                    pairs_created += 1

            session.commit()
        return pairs_created

    def export_training_data(self) -> list[dict]:
        """Export training pairs in Alpaca/DPO format for fine-tuning."""
        with Session(self.engine) as session:
            pairs = session.execute(
                select(TrainingPair).where(TrainingPair.exported == False)
            ).scalars().all()

            data = []
            for p in pairs:
                data.append({
                    "prompt": p.query,
                    "chosen": p.chosen_response,
                    "rejected": p.rejected_response,
                    "reward_gap": p.reward_gap,
                })
                p.exported = True
            session.commit()
        return data


# Singleton
_db: FeedbackDB | None = None


def get_feedback_db() -> FeedbackDB:
    global _db
    if _db is None:
        _db = FeedbackDB()
    return _db
