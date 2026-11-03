import re


def load_config(path="config.json"):
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def classify(client, cfg, data):
    feature_lines = [f"{feature}: {data[feature]}" for feature in cfg["features"]]
    response = client.chat.completions.create(
        model=cfg["model"],
        temperature=cfg.get("temperature", 0),
        max_tokens=cfg.get("max_tokens", 30),
        messages=[
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": "\n".join(feature_lines)},
        ],
    )
    raw = response.choices[0].message.content or ""
    normalized = re.sub(r"\s+", " ", raw.strip().lower())
    if re.search(r"\btidak\s+miskin\b", normalized) or normalized == "0":
        return 0, raw
    if re.search(r"\bmiskin\b", normalized) or normalized == "1":
        return 1, raw
    return None, raw
