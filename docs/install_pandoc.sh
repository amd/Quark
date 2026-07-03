#!/bin/bash

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Install pandoc binary required by nbconvert (if not already available)
# This script tries multiple installation methods with fallbacks:
# 1. Check if pandoc is already installed
# 2. Try pypandoc download (may hit GitHub API rate limits)
# 3. Fall back to apt-get
# 4. Fall back to conda

set -e

install_pandoc() {
    if command -v pandoc &> /dev/null; then
        echo "[QUARK-INFO] Pandoc is already installed: $(pandoc --version | head -n1)"
        return 0
    fi

    echo "[QUARK-INFO] Pandoc not found, installing via pypandoc..."
    if python -c "from pypandoc.pandoc_download import download_pandoc; download_pandoc()" 2>/dev/null; then
        echo "[QUARK-INFO] Pandoc installed successfully via pypandoc"
        # Pandoc is installed by default on $HOME/bin when using pypandoc
        export PATH=~/bin:${PATH}
        return 0
    fi

    echo "[QUARK-INFO] pypandoc download failed (likely rate limited), trying apt-get..."
    if apt-get update && apt-get install -y pandoc 2>/dev/null; then
        echo "[QUARK-INFO] Pandoc installed successfully via apt-get"
        return 0
    fi

    echo "[QUARK-INFO] apt-get failed, trying conda..."
    if conda install -y -c conda-forge pandoc 2>/dev/null; then
        echo "[QUARK-INFO] Pandoc installed successfully via conda"
        return 0
    fi

    echo "[QUARK-ERROR] All pandoc installation methods failed."
    return 1
}

install_pandoc
