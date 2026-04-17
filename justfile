default: build

build:
    #!/usr/bin/env bash
    set -euo pipefail
    version=$(./derive-semver.py)
    sed "s|__VERSION_PLACEHOLDER__|${version}|" derive-semver.py > derive-semver
    chmod +x derive-semver
    echo "built derive-semver ${version}"

clean:
    rm -f derive-semver

lint *args:
    #!/usr/bin/env bash
    set -euo pipefail
    python_bin=".venv/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        python_bin="python3"
    fi
    "${python_bin}" -m ruff check . "$@"

format *args:
    #!/usr/bin/env bash
    set -euo pipefail
    python_bin=".venv/bin/python"
    if [[ ! -x "${python_bin}" ]]; then
        python_bin="python3"
    fi
    "${python_bin}" -m ruff format . "$@"
