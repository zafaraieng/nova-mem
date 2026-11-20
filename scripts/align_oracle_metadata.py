#!/usr/bin/env python3
"""Create a versioned Oracle variant with cleaned-S question/evidence dates.

The source benchmark files are never modified. The derived file is intended for
re-evaluating the Oracle condition against cleaned-S under aligned metadata.
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ORACLE_SHA256 = "821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c"
CLEANED_S_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_json_array(path: Path):
    """Yield records from a top-level JSON array without loading it all."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    first = True
    chunk_size = 1024 * 1024

    with path.open("r", encoding="utf-8") as f:
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                more = f.read(chunk_size)
                if not more:
                    return
                buffer = buffer[position:] + more
                position = 0

            if first:
                if buffer[position] != "[":
                    raise ValueError(f"{path} must contain a top-level JSON array")
                position += 1
                first = False
                continue

            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position >= len(buffer):
                continue
            if buffer[position] == "]":
                return

            try:
                record, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                more = f.read(chunk_size)
                if not more:
                    raise
                buffer = buffer[position:] + more
                position = 0
                continue

            yield record
            position = end
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                more = f.read(chunk_size)
                if not more:
                    return
                buffer = buffer[position:] + more
                position = 0

            if buffer[position] == ",":
                position += 1
            elif buffer[position] == "]":
                return
            else:
                raise ValueError(f"Expected comma or closing bracket in {path}")

            if position > chunk_size:
                buffer = buffer[position:]
                position = 0


def normalized_evidence_ids(record: dict) -> set[str]:
    return {item.replace("noans_", "answer_") for item in record.get("answer_session_ids", [])}


def session_content_hash(messages: list[dict]) -> str:
    normalized = [
        {key: value for key, value in message.items() if key != "has_answer"}
        for message in messages
    ]
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_cleaned_metadata(path: Path, expected_question_ids: set[str]) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    for record in iter_json_array(path):
        question_id = record["question_id"]
        if question_id in metadata:
            raise ValueError(f"Duplicate cleaned-S question id: {question_id}")
        dates = record.get("haystack_dates", [])
        session_ids = record.get("haystack_session_ids", [])
        sessions = record.get("haystack_sessions", [])
        if not (len(dates) == len(session_ids) == len(sessions)):
            raise ValueError(f"Unaligned cleaned-S session arrays for {question_id}")
        evidence_ids = normalized_evidence_ids(record)
        evidence_sessions: dict[str, dict[str, str]] = {}
        for session_id, date, messages in zip(session_ids, dates, sessions):
            if session_id not in evidence_ids:
                continue
            if session_id in evidence_sessions:
                raise ValueError(
                    f"Duplicate cleaned-S evidence session {session_id} for {question_id}"
                )
            evidence_sessions[session_id] = {
                "date": date,
                "content_sha256": session_content_hash(messages),
            }
        metadata[question_id] = {
            "question": record["question"],
            "answer": record["answer"],
            "question_type": record["question_type"],
            "question_date": record["question_date"],
            "evidence_ids": evidence_ids,
            "evidence_sessions": evidence_sessions,
        }

    if set(metadata) != expected_question_ids:
        raise ValueError("Oracle and cleaned-S question ID sets differ")
    return metadata


def align_oracle_records(
    oracle_records: list[dict], cleaned_metadata: dict[str, dict]
) -> tuple[list[dict], dict]:
    aligned = copy.deepcopy(oracle_records)
    seen: set[str] = set()
    changes_by_question = []

    for record in aligned:
        question_id = record["question_id"]
        if question_id in seen:
            raise ValueError(f"Duplicate Oracle question id: {question_id}")
        seen.add(question_id)
        if question_id not in cleaned_metadata:
            raise ValueError(f"Oracle question is missing from cleaned-S: {question_id}")

        cleaned = cleaned_metadata[question_id]
        for field in ("question", "answer", "question_type"):
            if record[field] != cleaned[field]:
                raise ValueError(f"{field} differs between source files for {question_id}")

        oracle_ids = record.get("haystack_session_ids", [])
        oracle_dates = record.get("haystack_dates", [])
        oracle_sessions = record.get("haystack_sessions", [])
        if not (len(oracle_ids) == len(oracle_dates) == len(oracle_sessions)):
            raise ValueError(f"Unaligned Oracle session arrays for {question_id}")
        oracle_evidence_ids = normalized_evidence_ids(record)
        if len(oracle_ids) != len(set(oracle_ids)):
            raise ValueError(f"Duplicate Oracle evidence session for {question_id}")
        if set(oracle_ids) != oracle_evidence_ids or oracle_evidence_ids != cleaned["evidence_ids"]:
            raise ValueError(f"Oracle evidence IDs do not align with cleaned-S for {question_id}")

        old_question_date = record["question_date"]
        record["question_date"] = cleaned["question_date"]
        updated_session_ids = []
        for index, (session_id, messages) in enumerate(zip(oracle_ids, oracle_sessions)):
            cleaned_session = cleaned["evidence_sessions"].get(session_id)
            if cleaned_session is None:
                raise ValueError(
                    f"Evidence session {session_id} is missing from cleaned-S for {question_id}"
                )
            if session_content_hash(messages) != cleaned_session["content_sha256"]:
                raise ValueError(f"Evidence content differs for {question_id}/{session_id}")
            if oracle_dates[index] != cleaned_session["date"]:
                updated_session_ids.append(session_id)
                record["haystack_dates"][index] = cleaned_session["date"]

        if old_question_date != record["question_date"] or updated_session_ids:
            changes_by_question.append(
                {
                    "question_id": question_id,
                    "question_date_changed": old_question_date != record["question_date"],
                    "evidence_session_dates_changed": updated_session_ids,
                }
            )

    if seen != set(cleaned_metadata):
        raise ValueError("Oracle and cleaned-S question ID sets differ")

    manifest = {
        "variant": "oracle_s_aligned_v1",
        "transformation": (
            "For each question_id, copy question_date from cleaned-S; for each "
            "Oracle evidence session, copy haystack_dates from the same "
            "question_id/session_id in cleaned-S after verifying identical "
            "question, answer, type, evidence IDs, and message content "
            "(ignoring only cleaned-S has_answer annotation keys)."
        ),
        "source_files": {
            "longmemeval_oracle.json": ORACLE_SHA256,
            "longmemeval_s_cleaned.json": CLEANED_S_SHA256,
        },
        "question_count": len(aligned),
        "question_date_changes": sum(item["question_date_changed"] for item in changes_by_question),
        "evidence_session_date_changes": sum(
            len(item["evidence_session_dates_changed"]) for item in changes_by_question
        ),
        "affected_question_ids": [item["question_id"] for item in changes_by_question],
        "changes_by_question": changes_by_question,
    }
    return aligned, manifest


def main() -> None:
    data_dir = REPO / "data"
    oracle_path = data_dir / "longmemeval_oracle.json"
    cleaned_path = data_dir / "longmemeval_s_cleaned.json"
    if sha256_file(oracle_path) != ORACLE_SHA256:
        raise ValueError("Oracle source checksum differs from the audited source")
    if sha256_file(cleaned_path) != CLEANED_S_SHA256:
        raise ValueError("Cleaned-S source checksum differs from the audited source")

    oracle_records = json.loads(oracle_path.read_text(encoding="utf-8"))
    metadata = build_cleaned_metadata(cleaned_path, {r["question_id"] for r in oracle_records})
    aligned, manifest = align_oracle_records(oracle_records, metadata)

    output_dir = data_dir / "derived"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "longmemeval_oracle_s_aligned_v1.json"
    manifest_path = output_dir / "oracle_s_aligned_v1_manifest.json"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as f:
        json.dump(aligned, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\r\n")  # CRLF reproduces the audited derived-file byte layout (Windows origin)
    manifest["derived_file"] = output_path.name
    manifest["derived_sha256"] = sha256_file(temporary_path)

    if output_path.exists():
        if sha256_file(output_path) != manifest["derived_sha256"]:
            temporary_path.unlink()
            raise FileExistsError(f"Refusing to overwrite a different derived file: {output_path}")
        temporary_path.unlink()
    else:
        temporary_path.replace(output_path)

    serialized_manifest = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    if manifest_path.exists():
        if manifest_path.read_text(encoding="utf-8") != serialized_manifest:
            raise FileExistsError(f"Refusing to overwrite a different manifest: {manifest_path}")
    else:
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
    print(
        json.dumps(
            {
                "derived_file": str(output_path),
                "derived_sha256": manifest["derived_sha256"],
                "question_count": manifest["question_count"],
                "question_date_changes": manifest["question_date_changes"],
                "evidence_session_date_changes": manifest["evidence_session_date_changes"],
                "affected_question_count": len(manifest["affected_question_ids"]),
                "manifest": str(manifest_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
