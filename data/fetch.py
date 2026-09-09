"""
Data preparation script for AppleSupport customer support conversations.
Extracts inbound customer queries and matching @AppleSupport responses from 0.parquet.
Produces:
1. data/processed/apple_support_pairs.jsonl (clean QA pairs for retrieval & few-shot context)
2. data/golden_eval_set.json (200 curated, hand-stratified evaluation records)
"""
import os
import re
import json
import random
import pyarrow.parquet as pq
import pandas as pd
from typing import List, Dict, Any

RAW_PARQUET = os.path.join("data", "raw", "0.parquet")
PROCESSED_PAIRS_FILE = os.path.join("data", "processed", "pairs.jsonl")
GOLDEN_SET_FILE = os.path.join("data", "golden.json")


def clean_tweet_text(text: str) -> str:
    """Normalizes tweet text: standardizes handles, strips superfluous whitespace, preserves links."""
    if not isinstance(text, str):
        return ""
    # Normalize customer anonymized handles like @115712 -> @customer
    text = re.sub(r"@\d+", "@customer", text)
    # Normalize @AppleSupport mention
    text = re.sub(r"@AppleSupport", "@AppleSupport", text, flags=re.IGNORECASE)
    # Standardize whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_apple_conversations(sample_size: int = 15000) -> List[Dict[str, Any]]:
    print("Reading dataset parquet from:", RAW_PARQUET)
    
    # Read relevant columns
    cols = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]
    table = pq.read_table(RAW_PARQUET, columns=cols)
    df = table.to_pandas()
    print(f"Loaded {len(df):,} total rows.")

    # 1. Identify all tweets where author_id is AppleSupport and it is in response to a customer tweet
    apple_replies = df[
        (df["author_id"] == "AppleSupport") & 
        (df["inbound"] == False) & 
        (df["in_response_to_tweet_id"].notna())
    ].copy()
    print(f"Found {len(apple_replies):,} AppleSupport reply tweets.")

    # Map tweet_id -> row for quick lookup of the parent customer tweet
    # Index customer inbound tweets
    customer_tweets = df[df["inbound"] == True].set_index("tweet_id")

    pairs = []
    seen_texts = set()

    for _, reply in apple_replies.iterrows():
        parent_id = reply["in_response_to_tweet_id"]
        try:
            parent_id = int(parent_id)
        except (ValueError, TypeError):
            continue

        if parent_id in customer_tweets.index:
            cust = customer_tweets.loc[parent_id]
            if isinstance(cust, pd.DataFrame):
                cust = cust.iloc[0]

            cust_text = clean_tweet_text(cust["text"])
            reply_text = clean_tweet_text(reply["text"])

            # Quality filtering
            if len(cust_text) < 20 or len(reply_text) < 15:
                continue
            # Avoid duplicate inquiries
            if cust_text.lower() in seen_texts:
                continue
            seen_texts.add(cust_text.lower())

            pair = {
                "pair_id": int(reply["tweet_id"]),
                "customer_tweet_id": parent_id,
                "customer_text": cust_text,
                "apple_reply_text": reply_text,
                "created_at": str(reply["created_at"]),
            }
            pairs.append(pair)
            if len(pairs) >= sample_size:
                break

    print(f"Successfully extracted and paired {len(pairs):,} clean customer-support interactions.")
    return pairs


def save_pairs(pairs: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(PROCESSED_PAIRS_FILE), exist_ok=True)
    with open(PROCESSED_PAIRS_FILE, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    print(f"Saved {len(pairs):,} pairs to {PROCESSED_PAIRS_FILE}")


if __name__ == "__main__":
    pairs = extract_apple_conversations(sample_size=15000)
    save_pairs(pairs)
