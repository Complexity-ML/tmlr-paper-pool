#!/usr/bin/env python3
"""Build a reproducible local o200k token shard from FineWeb-Edu sample-10BT."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import requests
import tiktoken
from huggingface_hub import HfApi, hf_hub_url


REPO_ID = "HuggingFaceFW/fineweb-edu"
DATASET_PREFIX = "sample/10BT/"
TOKEN_BIN = "tokens.bin"
TOKEN_INDEX = "tokens.idx.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-tokens", type=int, default=1_050_000_000)
    parser.add_argument("--encoding", default="o200k_base")
    parser.add_argument("--threads", type=int, default=max(1, min(16, os.cpu_count() or 1)))
    parser.add_argument("--document-batch-size", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    return parser


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    partial.replace(destination)


def main() -> None:
    args = build_parser().parse_args()
    if args.target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    if args.document_batch_size <= 0:
        raise ValueError("document_batch_size must be positive")

    output_dir = args.output_dir.resolve()
    index_path = output_dir / TOKEN_INDEX
    bin_path = output_dir / TOKEN_BIN
    if index_path.exists() and not args.force:
        raise FileExistsError(f"Verified shard already exists: {index_path}")
    if args.force and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = output_dir / ".download"
    download_dir.mkdir()

    api = HfApi()
    dataset_info = api.dataset_info(REPO_ID)
    revision = dataset_info.sha
    files = sorted(
        filename
        for filename in api.list_repo_files(REPO_ID, repo_type="dataset", revision=revision)
        if filename.startswith(DATASET_PREFIX) and filename.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"No Parquet files found under {DATASET_PREFIX} at {revision}")

    encoding = tiktoken.get_encoding(args.encoding)
    eos_id = encoding._special_tokens.get("<|endoftext|>")
    digest = hashlib.sha256()
    token_count = 0
    document_count = 0
    max_token_id = -1
    source_files: list[str] = []

    try:
        with bin_path.open("wb") as output:
            for filename in files:
                if token_count >= args.target_tokens:
                    break
                local_path = download_dir / Path(filename).name
                url = hf_hub_url(
                    REPO_ID,
                    filename=filename,
                    repo_type="dataset",
                    revision=revision,
                )
                print(f"download {filename}", flush=True)
                download_file(url, local_path)
                source_files.append(filename)

                parquet = pq.ParquetFile(local_path)
                for record_batch in parquet.iter_batches(
                    batch_size=args.document_batch_size,
                    columns=["text"],
                ):
                    texts = [text or "" for text in record_batch.column(0).to_pylist()]
                    encoded = encoding.encode_batch(
                        texts,
                        num_threads=args.threads,
                        disallowed_special=(),
                    )
                    for token_ids in encoded:
                        document_count += 1
                        if eos_id is not None:
                            token_ids.append(int(eos_id))
                        remaining = args.target_tokens - token_count
                        if len(token_ids) > remaining:
                            token_ids = token_ids[:remaining]
                        if token_ids:
                            array = np.asarray(token_ids, dtype="<u4")
                            max_token_id = max(max_token_id, int(array.max()))
                            payload = array.tobytes()
                            output.write(payload)
                            digest.update(payload)
                            token_count += int(array.size)
                        if token_count >= args.target_tokens:
                            break
                    if token_count >= args.target_tokens:
                        break
                local_path.unlink(missing_ok=True)
                print(
                    f"progress tokens={token_count:,}/{args.target_tokens:,} documents={document_count:,}",
                    flush=True,
                )
            output.flush()
            os.fsync(output.fileno())
    finally:
        shutil.rmtree(download_dir, ignore_errors=True)

    if token_count < args.target_tokens:
        raise RuntimeError(
            f"Dataset ended at {token_count:,} tokens before target {args.target_tokens:,}"
        )

    metadata = {
        "format": "complexity-token-shard-v1",
        "bin": TOKEN_BIN,
        "dtype": np.dtype("<u4").str,
        "num_tokens": token_count,
        "max_token_id": max_token_id,
        "vocab_size": encoding.n_vocab,
        "tokenizer": f"tiktoken:{args.encoding}",
        "sha256": digest.hexdigest(),
        "source_repo": REPO_ID,
        "source_revision": revision,
        "source_subset": "sample-10BT",
        "source_files": source_files,
        "documents": document_count,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary_index = index_path.with_suffix(".json.partial")
    temporary_index.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    temporary_index.replace(index_path)
    print(index_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
