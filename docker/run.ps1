<#
.SYNOPSIS
One-click launcher for the YOLO dev container (Windows PowerShell / host machine).

.DESCRIPTION
Mounts the current project directory into /workspace and attaches the GPU.
Removes the container on exit (--rm) unless -Keep is used.

.EXAMPLE
run.ps1                                    # default image yoloworkbench:cuda12.8
run.ps1 -Image yolo-dev:cuda12.8           # pick another image
run.ps1 -Keep                              # keep container (reuse via docker start -i yolo)
run.ps1 -NoGpu                             # disable GPU (debugging)
#>

param(
    [string]$Image = "yoloworkbench:cuda12.8",
    [string]$Name  = "yoloworkbench",
    [switch]$NoGpu,
    [switch]$Keep
)

$ErrorActionPreference = "Stop"

# ============================================================
# 0. Check Docker availability
# ============================================================
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "[ERROR] docker command not found. Install Docker Desktop (WSL2 backend)." -ForegroundColor Red
    exit 1
}

# ============================================================
# 1. Check image exists
# ============================================================
$imageId = docker images -q $Image
if (-not $imageId) {
    Write-Host "[ERROR] Image not found: $Image" -ForegroundColor Red
    Write-Host "        Build it first: docker build -t $Image -f docker/Dockerfile ." -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] Using image: $Image" -ForegroundColor Green

# ============================================================
# 2. Resolve project root (parent of the docker/ folder)
# ============================================================
$Root = Split-Path -Parent $PSScriptRoot
Write-Host "[OK] Mount: $Root -> /workspace" -ForegroundColor Green

# ============================================================
# 3. Assemble docker run arguments
# ============================================================
$dockerArgs = @("run", "-it")

if ($Keep) {
    $dockerArgs += "--name"; $dockerArgs += $Name
    Write-Host "[OK] Keep container: $Name (re-enter with: docker start -i $Name)" -ForegroundColor Green
} else {
    $dockerArgs += "--rm"
}

if ($NoGpu) {
    Write-Host "[WARN] GPU not attached (-NoGpu)" -ForegroundColor Yellow
} else {
    $dockerArgs += "--gpus"; $dockerArgs += "all"
}

$dockerArgs += "-v";    $dockerArgs += "${Root}:/workspace"
$dockerArgs += "-w";    $dockerArgs += "/workspace"
$dockerArgs += $Image

# ============================================================
# 4. Run
# ============================================================
Write-Host ""
Write-Host "[RUN] docker $($dockerArgs -join ' ')" -ForegroundColor Cyan
Write-Host "----------------------------------------" -ForegroundColor DarkGray

docker @dockerArgs

exit $LASTEXITCODE
