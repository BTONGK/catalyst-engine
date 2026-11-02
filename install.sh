#!/bin/sh
# Install the catalyst engine app and Meta's UMA model on a Mac (Apple silicon, macOS 14 or later) or on Linux:
#
#     curl -LsSf https://raw.githubusercontent.com/BTONGK/catalyst-engine/main/install.sh | sh
#
# Run the same command again to update. Your screens, reactions and lab data stay in place.
# On Windows, run it inside WSL (Ubuntu).
#
# What it does, in order:
#   1. puts the app in ~/catalyst-engine (git clone, or GitHub's archive on computers without git)
#   2. installs uv (astral.sh/uv), which brings the right Python versions
#   3. builds the app's Python environment (.venv) from uv.lock
#   4. builds the UMA model's own Python environment (.uma) from uma-requirements.txt
#   5. downloads the UMA model file (1.2 GB) from Hugging Face with your token, after Meta approves your access
#   6. saves your Materials Project API key in ~/.config/.pmgrc.yaml, if you give one
#   7. writes the catalyst-engine command to ~/.local/bin and opens the app
#
# Settings, all optional (pass them to sh, e.g. `curl ... | CE_HOME=~/apps/ce sh`):
#   CE_HOME     the app's folder (default ~/catalyst-engine)
#   CE_BIN      where the catalyst-engine command goes (default ~/.local/bin)
#   CE_UMA_DIR  where the UMA model file goes (default ~/uma/checkpoints)
#   CE_REPO     the git repository (default the public GitHub repository)
#   CE_BRANCH   its branch (default main)
#   CE_TARBALL  a .tar.gz of the repository, used when git is missing (default GitHub's archive of CE_BRANCH)
#   HF_TOKEN    a Hugging Face token with access to facebook/UMA (asked for when needed)
#   MP_API_KEY  a Materials Project API key (asked for when needed)
#   CE_START=0  finish with the app closed
set -eu

REPO_URL="${CE_REPO:-https://github.com/BTONGK/catalyst-engine.git}"
BRANCH="${CE_BRANCH:-main}"
TARBALL="${CE_TARBALL:-https://codeload.github.com/BTONGK/catalyst-engine/tar.gz/refs/heads/$BRANCH}"
absolute() { case "$1" in /*) printf '%s' "$1" ;; *) printf '%s/%s' "$PWD" "$1" ;; esac; }
APP_DIR=$(absolute "${CE_HOME:-$HOME/catalyst-engine}")
BIN_DIR=$(absolute "${CE_BIN:-$HOME/.local/bin}")
UMA_DIR=$(absolute "${CE_UMA_DIR:-$HOME/uma/checkpoints}")
cd "$HOME"   # an update from GitHub's archive swaps the app folder, so work from outside it
START="${CE_START:-1}"

MODEL_NAME="uma-s-1p1.pt"
MODEL_URL="https://huggingface.co/facebook/UMA/resolve/main/checkpoints/$MODEL_NAME"
MODEL_PAGE="https://huggingface.co/facebook/UMA"
MODEL_SHA256="07068e9c76702ca173d13155095f2117c1b327ec228557e64cd2709c777b824a"
UMA_PYTHON_VERSION="3.14"
MARKER=".catalyst-engine-install"

if [ -t 1 ]; then B=$(printf '\033[1m'); G=$(printf '\033[32m'); Y=$(printf '\033[33m'); R=$(printf '\033[31m'); N=$(printf '\033[0m')
else B=""; G=""; Y=""; R=""; N=""; fi
say()  { printf '%s==>%s %s\n' "$B" "$N" "$*"; }
note() { printf '    %s\n' "$*"; }
good() { printf '    %s%s%s\n' "$G" "$*" "$N"; }
warn() { printf '    %s%s%s\n' "$Y" "$*" "$N"; }
stop() { printf '%sStopped:%s %s\n' "$R" "$N" "$*" >&2; exit 1; }

# Questions go to the terminal itself, so they work when this script arrives through a pipe.
if (exec </dev/tty) 2>/dev/null && [ -z "${CE_QUIET:-}" ]; then TTY=1; else TTY=0; fi
ask() { printf '%s' "$1" >/dev/tty; IFS= read -r REPLY </dev/tty || REPLY=""; }
ask_secret() {
    printf '%s' "$1" >/dev/tty
    stty -echo </dev/tty 2>/dev/null || true
    IFS= read -r REPLY </dev/tty || REPLY=""
    stty echo </dev/tty 2>/dev/null || true
    printf '\n' >/dev/tty
}
[ "$TTY" = 1 ] && trap 'stty echo </dev/tty 2>/dev/null || true' EXIT INT TERM

sha256() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

# ---------------------------------------------------------------- 0. this computer
OS=$(uname -s)
ARCH=$(uname -m)
case "$OS/$ARCH" in
    Darwin/arm64)
        major=$(sw_vers -productVersion | cut -d. -f1)
        [ "$major" -ge 14 ] || stop "The UMA model's PyTorch release needs macOS 14 (Sonoma) or later. This Mac runs $(sw_vers -productVersion)."
        ;;
    Darwin/x86_64)
        if [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
            stop "This Terminal runs under Rosetta. Open Terminal natively (Get Info on Terminal, clear 'Open using Rosetta') and run the command again."
        fi
        stop "This installer runs on Apple-silicon Macs (M1 and later) and on Linux, where the UMA model's PyTorch release ships."
        ;;
    Linux/x86_64 | Linux/aarch64 | Linux/arm64) ;;
    *) stop "This installer runs on macOS (Apple silicon) and Linux. On Windows, install WSL (https://learn.microsoft.com/windows/wsl/install), open Ubuntu and run the command there." ;;
esac
command -v curl >/dev/null 2>&1 || stop "This installer uses curl. Install it (e.g. sudo apt install curl) and run the command again."
command -v tar >/dev/null 2>&1 || stop "This installer uses tar. Install it and run the command again."

has_git() {
    command -v git >/dev/null 2>&1 || return 1
    # On a Mac, /usr/bin/git is a stub until the command-line tools are installed.
    [ "$OS" != Darwin ] || xcode-select -p >/dev/null 2>&1
}

printf '\n%sCatalyst engine installer%s\n' "$B" "$N"
note "App folder:   $APP_DIR"
note "UMA model:    $UMA_DIR/$MODEL_NAME"
note "Command:      $BIN_DIR/catalyst-engine"
printf '\n'

# ---------------------------------------------------------------- 1. the app's code
if [ -e "$APP_DIR" ] && [ ! -f "$APP_DIR/$MARKER" ]; then
    stop "$APP_DIR is taken by another folder. Pick a new place for the app, e.g. run: curl -LsSf https://raw.githubusercontent.com/BTONGK/catalyst-engine/main/install.sh | CE_HOME=~/catalyst-engine-app sh"
fi

from_archive() {
    parent=$(dirname "$APP_DIR")
    mkdir -p "$parent"
    tmp=$(mktemp -d "$parent/.catalyst-engine-download.XXXXXX")
    curl -fsSL "$TARBALL" | tar -xz -C "$tmp" || { rm -rf "$tmp"; stop "Downloading the app from $TARBALL failed. Check the connection and run the command again."; }
    src=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)
    if [ -d "$APP_DIR" ]; then
        # Keep the user's data, both Python environments and the marker; replace the code.
        for keep in data .venv .uma "$MARKER"; do
            if [ -e "$APP_DIR/$keep" ]; then mv "$APP_DIR/$keep" "$src/$keep"; fi
        done
        mv "$APP_DIR" "$tmp/previous"
    fi
    mv "$src" "$APP_DIR"
    rm -rf "$tmp"
}

if [ -d "$APP_DIR/.git" ]; then
    say "Updating the app in $APP_DIR"
    git -C "$APP_DIR" pull --ff-only --quiet origin "$BRANCH" ||
        stop "Updating $APP_DIR needs its files as downloaded; one of them was edited. Move that edit aside and run the command again."
elif [ -d "$APP_DIR" ]; then
    say "Updating the app in $APP_DIR"
    from_archive
elif has_git; then
    say "Downloading the app into $APP_DIR"
    git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
else
    say "Downloading the app into $APP_DIR"
    from_archive
fi
touch "$APP_DIR/$MARKER"
good "App code ready."

# ---------------------------------------------------------------- 2. uv
find_uv() {
    for candidate in "$(command -v uv 2>/dev/null || true)" "${XDG_BIN_HOME:-}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then echo "$candidate"; return 0; fi
    done
    return 1
}
if UV=$(find_uv); then
    say "Using uv at $UV"
else
    say "Installing uv, the Python manager the app runs on (https://astral.sh/uv)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV=$(find_uv) || stop "uv installed, yet its command is outside the usual folders. Open a new terminal and run the command again."
fi

# ---------------------------------------------------------------- 3. the app's Python
say "Building the app's Python environment (first run: a few minutes)"
(cd "$APP_DIR" && "$UV" sync --frozen --no-dev --quiet)
good "App environment ready."

# ---------------------------------------------------------------- 4. UMA's Python
say "Building the UMA model's Python environment (first run: about 1 GB of PyTorch and friends)"
"$UV" venv --quiet --allow-existing --python "$UMA_PYTHON_VERSION" "$APP_DIR/.uma"
if [ "$OS" = Linux ]; then
    TORCH="--torch-backend cpu"   # the app runs UMA on the CPU, so Linux skips PyTorch's GPU libraries
else
    TORCH=""
fi
# shellcheck disable=SC2086
"$UV" pip install --quiet --python "$APP_DIR/.uma/bin/python" $TORCH -r "$APP_DIR/uma-requirements.txt"
good "UMA environment ready."

# ---------------------------------------------------------------- 5. the UMA model file
MODEL="$UMA_DIR/$MODEL_NAME"
MODEL_READY=0

probe() {  # ok | waiting | token: can this token download the model?
    code=$(printf 'Authorization: Bearer %s\n' "$1" | curl -s -o /dev/null -w '%{http_code}' -I -H @- "$MODEL_URL" || echo 000)
    case "$code" in
        200 | 302 | 307) echo ok ;;
        403) echo waiting ;;
        *) echo token ;;
    esac
}

if [ -f "$MODEL" ] && [ "$(sha256 "$MODEL")" = "$MODEL_SHA256" ]; then
    say "UMA model already in place"
    MODEL_READY=1
else
    say "Getting Meta's UMA model from Hugging Face"
    token="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    hf_token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
    if [ -z "$token" ] && [ -f "$hf_token_file" ]; then token=$(cat "$hf_token_file"); fi
    state=token
    if [ -n "$token" ]; then state=$(probe "$token"); fi
    if [ "$state" != ok ] && [ "$TTY" = 1 ]; then
        note "Meta shares the UMA model through Hugging Face, one approved person at a time:"
        note "  1. Sign in (or sign up free) at https://huggingface.co"
        note "  2. Open $MODEL_PAGE, accept the license and request access"
        note "  3. Make a token with the Read role at https://huggingface.co/settings/tokens"
        tries=0
        while [ "$state" != ok ] && [ "$tries" -lt 3 ]; do
            tries=$((tries + 1))
            ask_secret "    Paste your Hugging Face token (Enter to finish this step later): "
            [ -n "$REPLY" ] || break
            token=$REPLY
            state=$(probe "$token")
            case "$state" in
                waiting) warn "Hugging Face knows this token; Meta's approval of your access request is still on its way." ; break ;;
                token) warn "Hugging Face turned this token down. Copy it again from https://huggingface.co/settings/tokens." ;;
            esac
        done
    fi
    if [ -f "$MODEL.part" ] && [ "$(sha256 "$MODEL.part")" = "$MODEL_SHA256" ]; then
        mv "$MODEL.part" "$MODEL"   # an earlier run finished the download
        MODEL_READY=1
        good "UMA model ready (checksum matches the file the app's checks used)."
    elif [ "$state" = ok ]; then
        mkdir -p "$UMA_DIR"
        note "Downloading 1.2 GB to $MODEL"
        if [ -t 2 ]; then PROGRESS="--progress-bar"; else PROGRESS="-sS"; fi
        printf 'Authorization: Bearer %s\n' "$token" |
            curl -fL --retry 3 -C - "$PROGRESS" -H @- -o "$MODEL.part" "$MODEL_URL" ||
            stop "The model download stopped partway. Run the command again to resume it."
        if [ "$(sha256 "$MODEL.part")" = "$MODEL_SHA256" ]; then
            mv "$MODEL.part" "$MODEL"
            MODEL_READY=1
            good "UMA model ready (checksum matches the file the app's checks used)."
        else
            rm -f "$MODEL.part"
            stop "The downloaded model's checksum differs from the expected file. Run the command again for a fresh download."
        fi
    elif [ "$state" = waiting ]; then
        warn "Once Hugging Face emails you that Meta approved access, run the install command again to fetch the model."
    else
        warn "The model step is waiting for your Hugging Face token. Run the install command again when you have it."
    fi
fi

# ---------------------------------------------------------------- 6. Materials Project key
PMGRC="$HOME/.config/.pmgrc.yaml"
if grep -qs '^PMG_MAPI_KEY:' "$PMGRC"; then
    say "Materials Project key found in $PMGRC"
else
    mp_key="${MP_API_KEY:-${PMG_MAPI_KEY:-}}"
    if [ -z "$mp_key" ] && [ "$TTY" = 1 ]; then
        say "Materials Project key (optional)"
        note "Pure metals work right away. A free key adds every alloy and oxide in the Materials Project:"
        note "sign in at https://next-gen.materialsproject.org/api and copy your API key."
        ask_secret "    Paste it here (Enter to add it later): "
        mp_key=$REPLY
    fi
    if [ -n "$mp_key" ]; then
        code=$(printf 'X-API-KEY: %s\n' "$mp_key" | curl -s -o /dev/null -w '%{http_code}' -H @- \
            "https://api.materialsproject.org/materials/summary/?material_ids=mp-149&_fields=material_id" || echo 000)
        if [ "$code" = 200 ]; then
            mkdir -p "$(dirname "$PMGRC")"
            if [ -s "$PMGRC" ] && [ -n "$(tail -c 1 "$PMGRC")" ]; then printf '\n' >>"$PMGRC"; fi
            printf 'PMG_MAPI_KEY: %s\n' "$mp_key" >>"$PMGRC"
            chmod 600 "$PMGRC"
            good "Key saved in $PMGRC."
        else
            warn "The Materials Project answered $code to that key. Add it later as a line 'PMG_MAPI_KEY: your-key' in $PMGRC."
        fi
    else
        note "Add a key any time as a line 'PMG_MAPI_KEY: your-key' in $PMGRC."
    fi
fi

# ---------------------------------------------------------------- 7. the catalyst-engine command
mkdir -p "$BIN_DIR"
cat >"$BIN_DIR/catalyst-engine" <<EOF
#!/bin/sh
# Opens the catalyst engine app in your browser (written by install.sh). Option: --port 8600
export CE_UMA_PYTHON="$APP_DIR/.uma/bin/python"
export CE_UMA_CHECKPOINT="$MODEL"
cd "$APP_DIR" && exec "$UV" run --frozen --no-dev --quiet ce ui "\$@"
EOF
chmod +x "$BIN_DIR/catalyst-engine"

case ":$PATH:" in
    *":$BIN_DIR:"*) OPEN="catalyst-engine" ;;
    *) OPEN="$BIN_DIR/catalyst-engine" ;;
esac

printf '\n%sInstalled.%s\n' "$G" "$N"
note "Open the app any time with:   $OPEN"
note "Update it with the same install command."
if [ "$MODEL_READY" = 1 ]; then
    note "Everything is ready, including designing catalysts with UMA."
else
    note "Results, playgrounds and calculators work now; designing new catalysts starts once the UMA model arrives."
fi
printf '\n'

if [ "$START" = 1 ] && [ "$TTY" = 1 ]; then
    say "Opening the app in your browser. Press Control-C here to close it."
    exec "$BIN_DIR/catalyst-engine" </dev/tty
fi
