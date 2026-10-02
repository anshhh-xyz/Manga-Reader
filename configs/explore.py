import sys
from collections import Counter
from configs import data


def show(seq_id):
    s = data.get_sequence(seq_id)
    print("=" * 70, "\n", seq_id)
    for page_no, (img, lines) in enumerate(zip(s["image_paths"], s["labels"]), 1):
        print(f"\n--- page {page_no}: {img}")
        if not lines:
            print("   (empty)")
        for ln in lines:
            print(f"   [{ln['speaker']}] {ln['text']}")


def stats():
    empty_pages = total_pages = narr = lines_total = 0
    speakers_per_seq = []
    for i in data.dev_ids():
        pages = data.get_sequence(i)["labels"]
        total_pages += 3
        empty_pages += sum(not p for p in pages)
        spk = Counter(l["speaker"] for p in pages for l in p)
        speakers_per_seq.append(len(spk))
        narr += spk.get("NARRATION", 0)
        lines_total += sum(spk.values())
    print(f"dev sequences: {len(data.dev_ids())} | test: {len(data.test_ids())}")
    print(f"empty pages: {empty_pages}/{total_pages}")
    print(f"lines: {lines_total} | NARRATION lines: {narr}")
    print(f"speakers per sequence: min {min(speakers_per_seq)}, "
          f"max {max(speakers_per_seq)}, mean {sum(speakers_per_seq)/len(speakers_per_seq):.1f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        for sid in sys.argv[1:]:
            show(sid)
    else:
        stats()