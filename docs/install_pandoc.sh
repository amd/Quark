#!/bin/bash

#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Install the pandoc binary required by nbconvert (if not already available).
#
# pandoc ships as a self-contained static binary, so the preferred path is a
# direct binary download -- no package manager, no dependency solving. Every
# method below is wrapped in `timeout` so a rate-limited download (GitHub
# HTTP 429) or a stalled conda/dnf solve can never hang the build for hours.
# (Previously an unbounded `conda install -c conda-forge pandoc` fallback could
# sit "Solving environment..." until the multi-hour CI job timeout.)
#
# Best long-term fix: pre-install pandoc in the build docker image so
# `command -v pandoc` short-circuits this script entirely and no network
# download is needed.
#
# Fallback order (each time-bounded):
#   0. already installed
#   1. pypandoc direct binary download (retried; rides out transient 429)
#   2. system package manager: dnf/yum (+EPEL) on RHEL/manylinux, apt-get on Debian
#   3. conda (last resort; repodata fetch + env solve can be slow)

set -e

# Per-method wall-clock cap in seconds (override via env if needed).
PANDOC_INSTALL_TIMEOUT="${PANDOC_INSTALL_TIMEOUT:-300}"
# Retries for the pypandoc download to ride out transient GitHub rate limiting.
PANDOC_DOWNLOAD_ATTEMPTS="${PANDOC_DOWNLOAD_ATTEMPTS:-3}"

_have() { command -v "$1" &> /dev/null; }

install_pandoc() {
    if _have pandoc; then
        echo "[QUARK-INFO] Pandoc is already installed: $(pandoc --version | head -n1)"
        return 0
    fi

    # 1. Direct static-binary download via pypandoc (pulls the pandoc release
    #    from GitHub). 429-prone on shared CI egress IPs, so retry with backoff.
    echo "[QUARK-INFO] Pandoc not found, installing via pypandoc (direct binary download)..."
    local attempt
    for attempt in $(seq 1 "${PANDOC_DOWNLOAD_ATTEMPTS}"); do
        if timeout "${PANDOC_INSTALL_TIMEOUT}" python -c "from pypandoc.pandoc_download import download_pandoc; download_pandoc()"; then
            # pypandoc drops the binary under ~/bin by default.
            export PATH=~/bin:${PATH}
            echo "[QUARK-INFO] Pandoc installed successfully via pypandoc"
            return 0
        fi
        echo "[QUARK-INFO] pypandoc download attempt ${attempt}/${PANDOC_DOWNLOAD_ATTEMPTS} failed (likely rate limited)."
        if [ "${attempt}" -lt "${PANDOC_DOWNLOAD_ATTEMPTS}" ]; then
            sleep "$((attempt * 15))"
        fi
    done

    # 2. System package manager. The manylinux build image is RHEL-based
    #    (dnf/yum, no apt-get); pandoc lives in EPEL there. Other images
    #    (Debian/Ubuntu) use apt-get.
    if _have dnf; then
        echo "[QUARK-INFO] pypandoc failed, trying dnf (+EPEL)..."
        if timeout "${PANDOC_INSTALL_TIMEOUT}" bash -c 'dnf install -y epel-release || true; dnf install -y pandoc'; then
            echo "[QUARK-INFO] Pandoc installed successfully via dnf"
            return 0
        fi
    elif _have yum; then
        echo "[QUARK-INFO] pypandoc failed, trying yum (+EPEL)..."
        if timeout "${PANDOC_INSTALL_TIMEOUT}" bash -c 'yum install -y epel-release || true; yum install -y pandoc'; then
            echo "[QUARK-INFO] Pandoc installed successfully via yum"
            return 0
        fi
    elif _have apt-get; then
        echo "[QUARK-INFO] pypandoc failed, trying apt-get..."
        if timeout "${PANDOC_INSTALL_TIMEOUT}" bash -c 'apt-get update && apt-get install -y pandoc'; then
            echo "[QUARK-INFO] Pandoc installed successfully via apt-get"
            return 0
        fi
    fi

    # 3. conda -- last resort. Its repodata fetch and environment solve can be
    #    very slow, so keep it strictly time-bounded.
    if _have conda; then
        echo "[QUARK-INFO] system package manager failed, trying conda (time-bounded)..."
        if timeout "${PANDOC_INSTALL_TIMEOUT}" conda install -y -c conda-forge pandoc; then
            echo "[QUARK-INFO] Pandoc installed successfully via conda"
            return 0
        fi
    fi

    echo "[QUARK-ERROR] All pandoc installation methods failed (see logs above)." >&2
    return 1
}

install_pandoc
