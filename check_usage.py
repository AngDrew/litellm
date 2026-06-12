#!/usr/bin/env python3
import requests
import sys

PROXY_URL = "http://localhost:4000"
API_KEY = sys.argv[1] if len(sys.argv) > 1 else "sk-YOUR_KEY"

resp = requests.get(
    f"{PROXY_URL}/key/info",
    headers={"Authorization": f"Bearer {API_KEY}"},
)
resp.raise_for_status()
data = resp.json()["info"]

spend = data.get("spend", 0.0)
limits = data.get("budget_limits") or []

if not limits:
    print("No budget_limits found")
    sys.exit(1)

print(f"Total spend: ${spend:.4f}\n")
print(f"{'Window':<10} {'Max Budget':>10} {'Reset At':>25} {'Usage %':>10}")
print("-" * 60)

for bl in limits:
    dur = bl["budget_duration"]
    max_b = bl["max_budget"]
    reset = bl.get("reset_at", "N/A")
    pct = (spend / max_b * 100) if max_b > 0 else 0.0
    print(f"{dur:<10} ${max_b:>9.2f} {reset:>25} {pct:>9.2f}%")
