#!/usr/bin/env python3
"""Emergency incident-response tool: delete share objects directly from S3.

Not part of the deployed application, and not reachable via any API endpoint.
For use when share links have leaked externally (e.g. indexed by a search
engine) and only share_id values are known — no deletion_id is available in
that scenario, so the normal POST /v1/share/delete endpoint can't be used.

Deletes only shares/{share_id} objects. It does not touch the corresponding
deletions/{deletion_id} sidecar objects (there is no share_id -> deletion_id
index), which are left to expire via the bucket's lifecycle policy.

CREDENTIALS: run this with AWS credentials that have s3:DeleteObject on
shares/* in the target bucket, e.g. via aws-vault:
    aws-vault exec <profile> -- python scripts/emergency_delete_shares.py \\
        --bucket <bucket> <share_id> [<share_id> ...]

Usage:
    python scripts/emergency_delete_shares.py --bucket <bucket> <share_id> [<share_id> ...]
    python scripts/emergency_delete_shares.py --bucket <bucket> --file ids.txt
    python scripts/emergency_delete_shares.py --bucket <bucket> --file ids.txt --confirm
"""

import argparse
import sys
from dataclasses import dataclass

import boto3
from botocore.exceptions import ClientError


@dataclass
class ShareResult:
    share_id: str
    existed: bool | None = None
    deleted: bool = False
    error: str | None = None


def load_share_ids(share_ids: list[str], file_path: str | None) -> list[str]:
    ids = list(share_ids)
    if file_path:
        with open(file_path, encoding="utf-8") as f:
            ids.extend(line.strip() for line in f if line.strip())

    seen = set()
    deduped = []
    for share_id in ids:
        if share_id not in seen:
            seen.add(share_id)
            deduped.append(share_id)
    return deduped


def check_exists(s3, bucket: str, share_id: str) -> ShareResult:
    try:
        s3.head_object(Bucket=bucket, Key=f"shares/{share_id}")
        return ShareResult(share_id=share_id, existed=True)
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return ShareResult(share_id=share_id, existed=False)
        return ShareResult(share_id=share_id, error=str(e))


def delete_share(s3, bucket: str, share_id: str) -> ShareResult:
    try:
        s3.delete_object(Bucket=bucket, Key=f"shares/{share_id}")
        return ShareResult(share_id=share_id, deleted=True)
    except ClientError as e:
        return ShareResult(share_id=share_id, error=str(e))


def run_dry_run(s3, bucket: str, share_ids: list[str]) -> int:
    print(
        f"DRY RUN — {len(share_ids)} share_id(s), bucket={bucket}. "
        "Pass --confirm to actually delete.\n"
    )
    results = [check_exists(s3, bucket, share_id) for share_id in share_ids]
    for r in results:
        if r.error:
            print(f"  ERROR     {r.share_id}: {r.error}")
        elif r.existed:
            print(f"  FOUND     {r.share_id}")
        else:
            print(f"  NOT FOUND {r.share_id}")

    found = sum(1 for r in results if r.existed)
    print(
        f"\n{found}/{len(share_ids)} share_id(s) currently exist and would be deleted."
    )
    return 0


def run_delete(s3, bucket: str, share_ids: list[str]) -> int:
    print(f"Deleting {len(share_ids)} share_id(s) from bucket={bucket}...\n")
    results = [delete_share(s3, bucket, share_id) for share_id in share_ids]
    for r in results:
        if r.error:
            print(f"  FAILED  {r.share_id}: {r.error}")
        else:
            print(f"  DELETED {r.share_id}")

    deleted = sum(1 for r in results if r.deleted)
    failed = len(results) - deleted
    print(f"\n{deleted}/{len(share_ids)} deleted, {failed} failed.")
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("share_ids", nargs="*", help="share_id values to delete")
    parser.add_argument("--file", help="Path to a file with one share_id per line")
    parser.add_argument(
        "--bucket", required=True, help="S3 bucket name (target environment)"
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete. Without this flag, only reports what would be deleted.",
    )
    args = parser.parse_args()

    share_ids = load_share_ids(args.share_ids, args.file)
    if not share_ids:
        print(
            "No share_ids provided (pass as arguments and/or --file).", file=sys.stderr
        )
        return 1

    s3 = boto3.client("s3")

    if not args.confirm:
        return run_dry_run(s3, args.bucket, share_ids)
    return run_delete(s3, args.bucket, share_ids)


if __name__ == "__main__":
    sys.exit(main())
