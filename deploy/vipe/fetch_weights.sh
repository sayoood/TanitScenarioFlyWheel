#!/usr/bin/env bash
# Resumable, segmented prefetch of ViPE weights into the torch-hub cache the image reads (/cache/torch/hub).
# The link on thor3 is shared and slow, so big files use N parallel range requests.
#   bash fetch_weights.sh            # pose_only essentials: MoGe-2-L depth prior + GeoCalib
#   bash fetch_weights.sh instance   # + GroundingDINO and SAM-B (dynamic-object masks)
#   bash fetch_weights.sh wheels     # big wheels (rerun-sdk, imageio-ffmpeg, opencv, transformers) for the build
# DROID (Google Drive), the AOT tracker and BERT are fetched by ViPE itself on first use (small / gdown / HF).
set -u
VIPE_ROOT=${VIPE_ROOT:-$HOME/tsf/dev/vipe}
H=$VIPE_ROOT/cache/torch/hub

dl() {  # url dest [sha256]
    local url=$1 dest=$2 sha=${3:-} N=${VIPE_CONN:-4} size chunk k s e have
    [ -s "$dest" ] && return 0
    mkdir -p "$(dirname "$dest")"
    size=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
    chunk=$(( (size + N - 1) / N ))
    for k in $(seq 0 $((N - 1))); do
        (
            s=$((k * chunk)); e=$((s + chunk - 1)); [ $e -ge $size ] && e=$((size - 1))
            until [ "$(stat -c %s "$dest.p$k" 2>/dev/null || echo 0)" -ge $((e - s + 1)) ]; do
                have=$(stat -c %s "$dest.p$k" 2>/dev/null || echo 0)
                curl -sSL --retry 5 --retry-delay 3 -r $((s + have))-$e "$url" >> "$dest.p$k" || sleep 3
            done
        ) &
    done
    wait
    cat $(for k in $(seq 0 $((N - 1))); do echo "$dest.p$k"; done) > "$dest.tmp" && mv "$dest.tmp" "$dest" \
        && rm -f "$dest".p*
    [ "$(stat -c %s "$dest")" -eq "$size" ] || { echo "size mismatch: $dest"; mv "$dest" "$dest.bad"; return 1; }
    [ -z "$sha" ] || [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]         || { echo "sha256 mismatch: $dest"; mv "$dest" "$dest.bad"; return 1; }
}

pypi_wheel() {  # package [version] -> URL of its aarch64 manylinux cp312/abi3 wheel (or pure-python wheel)
    python3 - "$@" <<'PY'
import json, sys, urllib.request
pkg, ver = sys.argv[1], (sys.argv[2:] or [""])[0]
d = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{pkg}/{ver + '/' if ver else ''}json", timeout=30))
w = [f for f in d["urls"] if f["filename"].endswith(".whl") and (
    ("aarch64" in f["filename"] and "manylinux" in f["filename"]
     and any(k in f["filename"] for k in ("cp312", "abi3", "py3-none"))) or "none-any" in f["filename"])]
print(w[0]["url"])
PY
}

if [ "${1:-}" = wheels ]; then
    mkdir -p "$VIPE_ROOT/wheels"
    for p in "rerun-sdk" "imageio-ffmpeg" "opencv-python-headless 4.14.0.94" "transformers 4.57.6"; do
        url=$(pypi_wheel $p) && dl "$url" "$VIPE_ROOT/wheels/$(basename "$url")"
    done
    echo "wheels ready in $VIPE_ROOT/wheels"
    exit 0
fi

VIPE_CONN=${VIPE_CONN:-8} dl https://huggingface.co/Ruicheng/moge-2-vitl/resolve/main/model.pt     "$H/moge2/moge-2-vitl.pt" 3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5
dl https://github.com/cvg/GeoCalib/releases/download/v1.0/geocalib-pinhole.tar "$H/geocalib/pinhole.tar"
if [ "${1:-}" = instance ]; then
    dl https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth \
        "$H/checkpoints/groundingdino_swint_ogc.pth"
    dl https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "$H/sam/sam_vit_b_01ec64.pth"
fi
echo "weights ready in $H"
