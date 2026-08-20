"""cloud/scripts/seed_device.py — provision a station + token.

Mirrors the field-provisioning flow in the main README: someone hands
the field engineer a CW_SYNC_TOKEN. This is that "someone."
"""
import argparse
import secrets

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--node-id", required=True)
    p.add_argument("--org", required=True)
    p.add_argument("--site", required=True)
    p.add_argument("--zone", default=None)
    p.add_argument("--api", default="http://localhost:8000")
    args = p.parse_args()

    token = secrets.token_urlsafe(32)
    resp = requests.post(f"{args.api}/stations", json={
        "node_id": args.node_id, "org": args.org, "site": args.site,
        "zone": args.zone, "token": token,
    })
    resp.raise_for_status()
    print(f"station id: {resp.json()['id']}")
    print(f"export CW_NODE_ID={args.node_id}")
    print(f"export CW_SYNC_TOKEN={token}")
    print("(save this token now — the backend only ever stores its hash)")


if __name__ == "__main__":
    main()