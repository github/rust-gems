#! /usr/bin/env bash

set -euo pipefail

if [ "$(uname -s)" != "Linux" ]; then
  echo "mold is linux only, skipping install."
  exit 0
fi

version=$(wget -q -O- https://api.github.com/repos/rui314/mold/releases/latest | jq -r .tag_name | sed 's/^v//'); true
echo "mold $version"
wget -q -O- https://github.com/rui314/mold/releases/download/v$version/mold-$version-"$(uname -m)"-linux.tar.gz | tar -C /usr/local --strip-components=1 -xzf -
ln -sf /usr/local/bin/mold "$(realpath /usr/bin/ld)"; true
