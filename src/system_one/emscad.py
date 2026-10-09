"""EMSCAD contract v1: checksum-pinned CSV, shared evidence, grouped frozen splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal
from urllib.request import urlopen

import yaml
from pydantic import Field
from sklearn.model_selection import StratifiedGroupKFold

from system_one.config import ExperimentSettings, NonemptyString, PositiveInt, StrictConfig
from system_one.runs import create_run, finish_run

FIELDS = (
    "title",
    "description",
    "requirements",
    "company_profile",
    "benefits",
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
    "telecommuting",
    "has_company_logo",
    "has_questions",
)
SPLIT_FOLDS = {
    "train": range(6),
    "validation": range(6, 7),
    "calibration": range(7, 8),
    "test": range(8, 10),
}


class SourceConfig(StrictConfig):
    version: PositiveInt
    url: NonemptyString
    csv_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    expected_rows: PositiveInt
    expected_fraud_rows: PositiveInt


class EMSCADConfig(ExperimentSettings):
    source: SourceConfig
    evidence_template: Literal["job_posting_v1"]
    group_description_min_chars: PositiveInt


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def acquire_source(
    source: SourceConfig,
    cache_dir: Path,
    *,
    local_csv: Path | None = None,
) -> tuple[bytes, dict]:
    """Download the named ZIP member or accept a local CSV; always check its content."""
    archive_sha256 = None
    if local_csv is not None:
        raw = local_csv.read_bytes()
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
        archive = cache_dir / f"kaggle-v{source.version}.zip"
        if not archive.exists():
            with urlopen(source.url, timeout=60) as response:
                downloaded = response.read()
            # Validate before caching; a web error page must not become a cached dataset.
            with zipfile.ZipFile(io.BytesIO(downloaded)) as zipped:
                zipped.read("fake_job_postings.csv")
            archive.write_bytes(downloaded)
        archive_bytes = archive.read_bytes()
        archive_sha256 = sha256(archive_bytes)
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as zipped:
            raw = zipped.read("fake_job_postings.csv")
    actual = sha256(raw)
    if actual != source.csv_sha256:
        raise ValueError(f"CSV checksum mismatch: expected {source.csv_sha256}, got {actual}")
    return raw, {
        "url": source.url,
        "version": source.version,
        "csv_sha256": actual,
        "archive_sha256": archive_sha256,
        "acquisition": "local_csv" if local_csv is not None else "kaggle_zip",
    }


def clean_text(value: str) -> str:
    """Strip HTML tags, decode entities, and collapse whitespace without filling values."""
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", value))).strip()


def format_evidence(fields: dict[str, str]) -> str:
    """A stable field order; IDs and fraud labels never enter the model input."""
    return "\n".join(f"{field}: {fields[field] or '[missing]'}" for field in FIELDS)


def prepare_cases(raw: bytes, config: EMSCADConfig) -> tuple[list[dict], dict]:
    """Check source rows once, retain lowest-ID exact duplicates, and group descriptions."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if set(reader.fieldnames or []) != {"job_id", "fraudulent", *FIELDS}:
        raise ValueError("Unexpected EMSCAD CSV columns")
    rows = list(reader)
    ids = [row["job_id"] for row in rows]
    if len(set(ids)) != len(ids) or any(not value.isdigit() for value in ids):
        raise ValueError("Expected unique integer job IDs")
    if any(row["fraudulent"] not in {"0", "1"} for row in rows):
        raise ValueError("Fraud labels must be 0 or 1")
    if any(
        row[field] not in {"0", "1"}
        for row in rows
        for field in ("telecommuting", "has_company_logo", "has_questions")
    ):
        raise ValueError("Posting flags must be 0 or 1")
    fraud_rows = sum(row["fraudulent"] == "1" for row in rows)
    if len(rows) != config.source.expected_rows or fraud_rows != config.source.expected_fraud_rows:
        raise ValueError(f"Unexpected source counts: {len(rows)} rows, {fraud_rows} fraud")

    retained: dict[str, dict] = {}
    duplicates = []
    for row in sorted(rows, key=lambda row: int(row["job_id"])):
        fields = {field: clean_text(row[field]) for field in FIELDS}
        key = sha256(json.dumps([fields[f].casefold() for f in FIELDS]).encode())
        label = int(row["fraudulent"])
        sample_id = f"emscad:{row['job_id']}"
        if key in retained:
            original = retained[key]
            if label != original["label"]:
                raise ValueError(f"Conflicting labels for identical evidence: {sample_id}")
            duplicates.append({"excluded_id": sample_id, "retained_id": original["id"]})
            continue
        description = fields["description"].casefold()
        group_id = (
            "description:" + sha256(description.encode())
            if len(description) >= config.group_description_min_chars
            else "content:" + key
        )
        retained[key] = {
            "id": sample_id,
            "label": label,
            "evidence": format_evidence(fields),
            "group_id": group_id,
        }
    cases = list(retained.values())
    groups = Counter(case["group_id"] for case in cases)
    lengths = sorted(len(case["evidence"]) for case in cases)
    audit = {
        "source_rows": len(rows),
        "source_fraud_rows": fraud_rows,
        "retained_rows": len(cases),
        "retained_fraud_rows": sum(c["label"] for c in cases),
        "exact_duplicate_rows_removed": len(duplicates),
        "duplicates": duplicates,
        "missing_fields": {f: sum(not clean_text(row[f]) for row in rows) for f in FIELDS},
        "groups": len(groups),
        "repeated_groups": sum(n > 1 for n in groups.values()),
        "largest_group_rows": max(groups.values()),
        "evidence_characters": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "max": lengths[-1],
        },
        "limitations": [
            "Groups match normalized descriptions, not fuzzy paraphrases or employer identity.",
            "Partitions are grouped random splits, not temporal or client holdouts.",
            "Source URL/email placeholders are retained; this is an anonymized public text task.",
            "Evidence is untruncated; model-specific token accounting belongs to inference.",
        ],
    }
    return cases, audit


def freeze_splits(cases: list[dict], config: EMSCADConfig) -> dict:
    """Ten stratified group folds, assigned once under the declared 60/10/10/20 recipe."""
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=config.experiment.seed)
    folds = list(
        splitter.split(
            X=[0] * len(cases),
            y=[c["label"] for c in cases],
            groups=[c["group_id"] for c in cases],
        )
    )
    splits = {
        name: [cases[i]["id"] for fold in fold_ids for i in folds[fold][1]]
        for name, fold_ids in SPLIT_FOLDS.items()
    }
    return {
        "contract_version": "emscad_v1",
        "evidence_template": config.evidence_template,
        "split_seed": config.experiment.seed,
        "csv_sha256": config.source.csv_sha256,
        "dataset_version": f"kaggle-v{config.source.version}",
        "group_description_min_chars": config.group_description_min_chars,
        "split_recipe": "StratifiedGroupKFold(10): train=0..5,val=6,calibration=7,test=8..9",
        "cases_file": "cases.jsonl",
        "splits": splits,
    }


def prepare_dataset(
    config: EMSCADConfig, run_dir: Path, cache_dir: Path, *, local_csv: Path | None = None
) -> dict:
    raw, source = acquire_source(config.source, cache_dir, local_csv=local_csv)
    cases, audit = prepare_cases(raw, config)
    case_data = "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases).encode()
    manifest = freeze_splits(cases, config)
    manifest["cases_sha256"] = sha256(case_data)
    manifest["split_id"] = sha256(json.dumps(manifest, sort_keys=True).encode())
    lookup = {case["id"]: case for case in cases}
    audit["splits"] = {}
    seen_groups: set[str] = set()
    for name, ids in manifest["splits"].items():
        partition = [lookup[sample_id] for sample_id in ids]
        groups = {case["group_id"] for case in partition}
        if seen_groups & groups:
            raise ValueError("Duplicate group crossed partitions")
        seen_groups.update(groups)
        fraud = sum(case["label"] for case in partition)
        if not 0 < fraud < len(partition):
            raise ValueError(f"Partition {name} must contain both labels")
        audit["splits"][name] = {
            "rows": len(partition),
            "fraud_rows": fraud,
            "fraud_rate": fraud / len(partition),
            "groups": len(groups),
        }
    (run_dir / "cases.jsonl").write_bytes(case_data)
    for name, data in (("splits.json", manifest), ("audit.json", audit)):
        (run_dir / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return {
        "source": source,
        "split_id": manifest["split_id"],
        "dataset_audit": {key: value for key, value in audit.items() if key != "duplicates"},
        "artifacts": {
            "cases": "cases.jsonl",
            "split_manifest": "splits.json",
            "audit": "audit.json",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare frozen EMSCAD data (no model execution)")
    parser.add_argument("--config", type=Path, default=Path("configs/emscad_data.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/emscad/raw"))
    parser.add_argument("--csv", type=Path, help="Use a local CSV with the same required checksum")
    args = parser.parse_args(argv)
    run_dir = None
    try:
        config = EMSCADConfig.model_validate(
            yaml.safe_load(args.config.read_text(encoding="utf-8"))
        )
        run_dir = create_run(config, stage="data", run_root=args.results_dir)
        evidence = prepare_dataset(config, run_dir, args.cache_dir, local_csv=args.csv)
        finish_run(run_dir, **evidence, input_csv=str(args.csv) if args.csv else None)
        print(f"Prepared EMSCAD: {run_dir}")
        print(f"Frozen split ID: {evidence['split_id']}")
    except (ValueError, OSError, yaml.YAMLError, zipfile.BadZipFile, KeyError) as exc:
        if run_dir is not None:
            finish_run(run_dir, status="failed", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
