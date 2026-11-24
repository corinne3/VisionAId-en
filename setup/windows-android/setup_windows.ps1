# VisionAId - setup for a Windows PC (CPU only) + Android phone
# Author: Corinne Bergès - MIT licence (this script only, see LICENSE-ADDITIONS)
# Run it from the root of this repository:
#   powershell -ExecutionPolicy Bypass -File setup\windows-android\setup_windows.ps1

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $Root

foreach ($tool in @("git", "uv")) {
    if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
        Write-Host "Missing tool: $tool. Install it first (see setup\windows-android\README.md)." -ForegroundColor Red
        exit 1
    }
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "Warning: ffmpeg not found. Voice requests with Whisper will not work (winget install --id Gyan.FFmpeg -e)." -ForegroundColor Yellow
}

# 1. Dependencies (CPU PyTorch on Windows)
uv sync

# 2. Environment variables (new PowerShell windows, and this one)
setx PYTHONUTF8 1 | Out-Null
setx HF_HUB_DISABLE_XET 1 | Out-Null
$env:PYTHONUTF8 = "1"
$env:HF_HUB_DISABLE_XET = "1"

# 3. Local settings for a slow machine (UTF-8 without BOM). Not meant to be committed.
$utf8 = New-Object System.Text.UTF8Encoding $false
function Edit-File($path, $pairs) {
    $full = Join-Path $Root $path
    $text = [IO.File]::ReadAllText($full)
    foreach ($p in $pairs) {
        if ($text.Contains($p[0])) {
            $text = $text.Replace($p[0], $p[1])
            Write-Host "  $path : '$($p[0])' -> '$($p[1])'"
        } elseif ($text.Contains($p[1])) {
            Write-Host "  $path : already set ($($p[1]))"
        } else {
            Write-Host "  $path : '$($p[0])' not found (code changed?)" -ForegroundColor Yellow
        }
    }
    [IO.File]::WriteAllText($full, $text, $utf8)
}

Write-Host "Applying local settings:"
Edit-File "config\default.yaml" @(
    @("device: iphone12", "device: samsung"),
    @("max_fps: 15.0", "max_fps: 3.0"),
    @("openai/whisper-small", "openai/whisper-base")
)
# the leading comma keeps a one-pair list as a list (PowerShell would otherwise unroll it)
Edit-File "src\visionaid\apps\live_blocks.py" @(
    ,@("stale_s: float = 1.5", "stale_s: float = 6.0")
)
git --no-pager diff --stat

Write-Host ""
Write-Host "Done. Next steps:" -ForegroundColor Green
Write-Host "  1. Firewall (PowerShell as administrator, once):"
Write-Host '     New-NetFirewallRule -DisplayName "VisionAId" -Direction Inbound -Protocol TCP -LocalPort 8080,8443 -Action Allow -Profile Any'
Write-Host "  2. Connect the PC to the phone hotspot."
Write-Host '  3. uv run python -m visionaid.apps.run_live --target "où est ma gourde" --height 130 --depth-width 364'
Write-Host "  To undo the local settings: git checkout -- config/default.yaml src/visionaid/apps/live_blocks.py"
