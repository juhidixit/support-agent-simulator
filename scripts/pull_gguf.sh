#!/usr/bin/env bash
# Usage: pull_gguf.sh <repo e.g. ai/qwen2.5> <tag> <out.gguf>
set -euo pipefail
REPO=$1; TAG=$2; OUT=$3
T=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$REPO:pull" | python3 -c 'import sys,json;print(json.load(sys.stdin)["token"])')
ACC='application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.index.v1+json'
MAN=$(curl -s -H "Authorization: Bearer $T" -H "Accept: $ACC" "https://registry-1.docker.io/v2/$REPO/manifests/$TAG")
DIG=$(echo "$MAN" | python3 -c '
import sys,json; m=json.load(sys.stdin)
ls=[l for l in m.get("layers",[]) if "gguf" in l.get("mediaType","") or "weight" in l.get("mediaType","")]
print(max(ls,key=lambda l:l["size"])["digest"])')
curl -sL -H "Authorization: Bearer $T" -o "$OUT" "https://registry-1.docker.io/v2/$REPO/blobs/$DIG"
ls -la "$OUT"
