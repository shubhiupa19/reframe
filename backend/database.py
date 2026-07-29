import os

from supabase import create_client

_client = None


def _get_client():
    # Created lazily (not at import time) so importing this module doesn't
    # require SUPABASE_URL/SUPABASE_KEY to already be set — e.g. load_dotenv()
    # in app.py runs before any of these functions are actually called.
    global _client
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    return _client


def init_db():
    # No-op: table creation is a one-time setup step done through Supabase's
    # own SQL Editor, not something supabase-py's client library supports at
    # runtime (it only does data operations — insert/select/update — not DDL
    # like CREATE TABLE, unlike the sqlite3 version this replaced).
    pass


def save_feedback(text, predicted_distortion, user_correction, is_accepted, confidence) -> int:
    supabase = _get_client()
    result = (
        supabase.table("feedback")
        .insert(
            {
                "text": text,
                "predicted_distortion": predicted_distortion,
                "user_correction": user_correction,
                "is_accepted": is_accepted,
                "confidence": confidence,
            }
        )
        .execute()
    )
    return result.data[0]["id"]


def save_prediction(submission_id, input_text, prediction, confidence):
    supabase = _get_client()
    supabase.table("predictions").insert(
        {
            "submission_id": submission_id,
            "input": input_text,
            "prediction": prediction,
            "confidence": confidence,
        }
    ).execute()


def get_training_feedback():
    supabase = _get_client()
    result = (
        supabase.table("feedback")
        .select("text, user_correction")
        .eq("is_accepted", False)
        .eq("used_in_training", False)
        .execute()
    )
    return [(row["text"], row["user_correction"]) for row in result.data]


def mark_used_feedback():
    supabase = _get_client()
    supabase.table("feedback").update({"used_in_training": True}).eq("used_in_training", False).execute()


def save_model_version(version_number, training_samples, accuracy, notes):
    supabase = _get_client()
    result = (
        supabase.table("model_versions")
        .insert(
            {
                "version_number": version_number,
                "training_samples": training_samples,
                "accuracy": accuracy,
                "notes": notes,
            }
        )
        .execute()
    )
    return result.data[0]["id"]


def get_latest_version():
    supabase = _get_client()
    result = (
        supabase.table("model_versions")
        .select("version_number")
        .order("version_number", desc=True)
        .limit(1)
        .execute()
    )
    if not result.data:
        return 1
    return result.data[0]["version_number"]


if __name__ == "__main__":
    print("Tables are created via the Supabase SQL Editor, not this script.")
