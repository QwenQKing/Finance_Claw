import argparse, json, os
from pathlib import Path

from src.adapters.simple import _extract_primary_text, _split_chunks
from src.retrieval.dense_hybrid import build_dense_index_for_asset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--asset-dir", required=True)
    args = parser.parse_args()

    rows = []
    with open(args.train_path, encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    asset_dir = Path(args.asset_dir)
    corpus_dir = asset_dir / "corpus"
    seed_dir = asset_dir / "hypergraph_seed"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    seed_dir.mkdir(parents=True, exist_ok=True)

    chunks, findings = [], []
    for sample in rows:
        text = _extract_primary_text(sample)
        for idx, chunk in enumerate(_split_chunks(text)):
            chunks.append({
                "chunk_id": f"doc_{args.dataset}__{sample['id']}__chunk_{idx}",
                "doc_id": f"doc_{args.dataset}__{sample['id']}",
                "sample_id": sample["id"],
                "text": chunk,
            })
        for idx, chunk in enumerate(_split_chunks(text)[:6]):
            findings.append({
                "sentence": chunk,
                "entities": [],
                "confidence": min(0.95, 0.55 + idx * 0.05),
                "source_tool": "kb_seed",
                "data_timestamp": sample.get("created_at", ""),
            })

    with open(corpus_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(seed_dir / "findings.jsonl", "w", encoding="utf-8") as f:
        for fnd in findings:
            f.write(json.dumps(fnd, ensure_ascii=False) + "\n")

    print(f"{len(chunks)} chunks, {len(findings)} memory seeds")

    result = build_dense_index_for_asset(asset_dir, force=True)
    print(f"dense index: {result['status']} ({result['success_chunk_count']} vectors)")

if __name__ == "__main__":
    main()
