#!/usr/bin/env bash
# Clone gnuradio4 at gr4/REVISION into DIR (default: build/gnuradio4) and apply gr4/patches/ as commits.
#   gr4/apply.sh [DIR]
# SPDX-FileCopyrightText: 2026 GSI Helmholtz Centre for Heavy Ion Research GmbH
# SPDX-License-Identifier: LGPL-3.0-or-later WITH LGPL-3.0-linking-exception
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=${1:-$HERE/../build/gnuradio4}
REV=$(cat "$HERE/REVISION")
[ -d "$DIR/.git" ] || git clone https://github.com/fair-acc/gnuradio4.git "$DIR"
git -C "$DIR" checkout -q -B stockham "$REV"
git -C "$DIR" am -q "$HERE"/patches/*.patch
git -C "$DIR" log --oneline "$REV"..HEAD
