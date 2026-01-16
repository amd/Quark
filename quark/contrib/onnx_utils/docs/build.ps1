# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

function Invoke-SafeCommand {
    param (
        [ScriptBlock]$Command,
        [string]$action,
        [bool]$native
    )
    Write-Host "Start ${action} ..."
    try {
        & $Command
        if ($native) {
            if ($? -ne 1) {
                Write-Host "Native $Action failed."
                exit 1
            } else {
                Write-Host "${action} success."
            }
        } else {
            if ($LASTEXITCODE -ne 0) {
                Write-Host "$Action failed."
                exit 1
            } else {
                Write-Host "${action} success."
            }
        }
    } catch {
        Write-Host "${action} failed: $($_.Exception.Message)"
        exit 1
    }
}

Invoke-SafeCommand { mkdir -Force build | Out-Null } "mkdir" $true
Invoke-SafeCommand { Remove-Item build/docs -Recurse -Force | Out-Null } "delete old docs" $true
Invoke-SafeCommand { git clone --depth 1 --single-branch --branch gh-pages https://gitenterprise.xilinx.com/varunsh/onnx_utils.git build/docs/html } "git clone" $false
Invoke-SafeCommand { cd build/docs/html } "changing directories" $true

# delete all existing files
Invoke-SafeCommand { rm .git/index } "Delete git files" $true
Invoke-SafeCommand { git clean -fdx } "Clean git files" $false

Invoke-SafeCommand { cd .. } "cd up" $true
Invoke-SafeCommand { sphinx-build -M html ../../docs . -W } "build docs" $false
Invoke-SafeCommand { cd ../.. } "restore path" $true
