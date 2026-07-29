import pytest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from fake_supabase import FakeSupabaseClient
import database
from database import init_db, save_feedback, get_training_feedback, mark_used_feedback


@pytest.fixture
def fake_supabase(monkeypatch):
    client = FakeSupabaseClient()
    # database.py's functions all call _get_client(), which returns whatever
    # database._client already is rather than creating a real one — so patching
    # this one module-level attribute is enough to isolate every test below
    # from the real, live Supabase database, with no real network calls made.
    monkeypatch.setattr(database, "_client", client)
    return client


def test_init_db_is_a_noop():
    # Table creation now happens once via Supabase's own SQL Editor, not this
    # function (supabase-py's client library doesn't support DDL like CREATE
    # TABLE) — so this just confirms it's still safe to call.
    init_db()


def test_save_feedback_returns_id(fake_supabase):
    feedback_id = save_feedback(
        text="I always mess everything up",
        predicted_distortion="Overgeneralization",
        user_correction=None,
        is_accepted=True,
        confidence=0.85,
    )
    assert isinstance(feedback_id, int)
    assert feedback_id > 0


def test_get_training_feedback_returns_untrained_rows(fake_supabase):
    save_feedback("I never do anything right", "Emotional Reasoning", "Overgeneralization", False, 0.76)
    save_feedback("Everything is fine", "No Distortion", None, True, 0.91)

    results = get_training_feedback()

    assert len(results) == 1
    assert results[0][0] == "I never do anything right"


def test_mark_used_feedback(fake_supabase):
    save_feedback(
        text="I feel bad so everything is bad",
        predicted_distortion="Overgeneralization",
        user_correction="Emotional Reasoning",
        is_accepted=False,
        confidence=0.76,
    )
    save_feedback(
        text="Nothing good ever happens to me",
        predicted_distortion="Overgeneralization",
        user_correction=None,
        is_accepted=True,
        confidence=0.76,
    )
    mark_used_feedback()

    rows = fake_supabase.tables["feedback"]
    row1 = next(r for r in rows if r["text"] == "I feel bad so everything is bad")
    row2 = next(r for r in rows if r["text"] == "Nothing good ever happens to me")

    assert row1["used_in_training"] is True
    assert row2["used_in_training"] is True
