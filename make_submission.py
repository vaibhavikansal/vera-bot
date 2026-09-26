"""
Builds submission.jsonl — one line per canonical test pair (30 lines).

  python make_submission.py            # uses Groq if GROQ_API_KEY is set, else templates only
"""
import asyncio
import glob
import json
import os
import subprocess
import sys

import llm
from composer import compose_async

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "expanded")
if not os.path.isdir(EXP):
    subprocess.run([sys.executable, os.path.join(HERE, "dataset", "generate_dataset.py"),
                    "--seed-dir", os.path.join(HERE, "dataset"), "--out", EXP], check=True, stdout=subprocess.DEVNULL)


def load(sub, key):
    out = {}
    for f in glob.glob(f"{EXP}/{sub}/*.json"):
        d = json.load(open(f))
        out[d[key]] = d
    return out


async def main():
    cats, ms, cs, ts = load("categories", "slug"), load("merchants", "merchant_id"), load("customers", "customer_id"), load("triggers", "id")
    pairs = json.load(open(f"{EXP}/test_pairs.json"))["pairs"]
    lines, used_llm = [], 0
    for p in pairs:
        m = ms[p["merchant_id"]]
        r = await compose_async(cats[m["category_slug"]], m, ts[p["trigger_id"]],
                                cs.get(p["customer_id"]) if p.get("customer_id") else None, llm_timeout=20)
        used_llm += r["composer"] == "llm"
        lines.append({"test_id": p["test_id"], "body": r["body"], "cta": r["cta"], "send_as": r["send_as"],
                      "suppression_key": r["suppression_key"], "rationale": r["rationale"]})
        print(f"{p['test_id']} [{r['composer']}] {r['body'][:90]}...")
        if llm.enabled():
            await asyncio.sleep(2.5)  # stay under Groq free-tier rate limits
    with open(os.path.join(HERE, "submission.jsonl"), "w", encoding="utf-8") as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    print(f"\nWrote submission.jsonl ({len(lines)} lines, {used_llm} LLM-polished, {len(lines) - used_llm} template)")


if __name__ == "__main__":
    asyncio.run(main())
