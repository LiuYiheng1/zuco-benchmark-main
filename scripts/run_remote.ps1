param(
    [string]$RemoteScript = "scripts/run_remote.sh"
)

$ErrorActionPreference = "Stop"

$ProjectDir = "D:\pycharmproject\zuco-benchmark-main"
$RemoteHost = "yiheng-server"
$RemoteProjectDir = "/home/yiheng/projects/zuco-benchmark-main"

function Resolve-GitExe {
    $gitCommand = Get-Command git -ErrorAction SilentlyContinue
    if ($gitCommand) {
        return $gitCommand.Source
    }

    $fallback = "C:\Program Files\Git\cmd\git.exe"
    if (Test-Path -LiteralPath $fallback) {
        return $fallback
    }

    throw "Git was not found in PATH or at $fallback. Please reopen PowerShell or add Git to PATH."
}

Set-Location -LiteralPath $ProjectDir

$GitExe = Resolve-GitExe
$status = & $GitExe status --porcelain
if ($LASTEXITCODE -ne 0) {
    throw "git status --porcelain failed."
}

if ($status) {
    Write-Host "Local working tree has uncommitted changes. Please commit before starting the remote experiment."
    Write-Host ""
    $status | ForEach-Object { Write-Host $_ }
    exit 1
}

ssh $RemoteHost "test -d '$RemoteProjectDir'"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Remote project directory does not exist: $RemoteProjectDir"
    Write-Host "Please clone the project on the server first. This script will not create an empty fake clone."
    exit 1
}

ssh $RemoteHost "cd '$RemoteProjectDir' && bash '$RemoteScript'"
exit $LASTEXITCODE
