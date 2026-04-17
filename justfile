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
