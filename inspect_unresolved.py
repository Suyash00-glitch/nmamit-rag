import json

TARGETS = {
    "State Level Junior Power Lifting Championship 2024",
    "Sri Dilip Shetty",
    "Mr. Ramesh Kumar",
}

with open("graph_data.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

for target in TARGETS:

    print("\n" + "=" * 80)
    print("TARGET:", target)
    print("=" * 80)

    found = False

    for chunk in chunks:

        # Check entities
        for entity in chunk.get("entities", []):
            if not isinstance(entity, dict):
                continue

            if entity.get("name", "").strip() == target:
                found = True

                print("\nCHUNK:", chunk.get("chunk_id"))

                print("\nENTITY:")
                print(json.dumps(entity, indent=2, ensure_ascii=False))

                print("\nRELATIONSHIPS:")
                for rel in chunk.get("relationships", []):
                    if not isinstance(rel, dict):
                        continue

                    if (
                        rel.get("source", "").strip() == target
                        or rel.get("target", "").strip() == target
                    ):
                        print(json.dumps(
                            rel,
                            indent=2,
                            ensure_ascii=False
                        ))

        # Check relationships even if entity extraction missed it
        for rel in chunk.get("relationships", []):
            if not isinstance(rel, dict):
                continue

            if (
                rel.get("source", "").strip() == target
                or rel.get("target", "").strip() == target
            ):
                found = True

                print("\nCHUNK:", chunk.get("chunk_id"))

                print("\nRELATIONSHIP:")
                print(json.dumps(
                    rel,
                    indent=2,
                    ensure_ascii=False
                ))

    if not found:
        print("NOT FOUND")
