$ErrorActionPreference = "Stop"

$Python = "C:\AI\npu-env\Scripts\python.exe"
$ServerDir = "C:\AI\server"
$LogDir = "C:\AI\logs"

. "C:\AI\load-qdrant-api-key.ps1"

# NAS services are on the local LAN; do not send them through an HTTP proxy.
$env:NO_PROXY = "192.168.1.3,127.0.0.1,localhost"
# The embedding model is already cached locally on this host.
$env:HF_HUB_OFFLINE = "1"

$ModelXml = "C:\AI\models\lfm2-1.2b-npu-v2\openvino_model.xml"
$ModelBin = "C:\AI\models\lfm2-1.2b-npu-v2\openvino_model.bin"

$StdOut = Join-Path $LogDir "fastapi-stdout-v5.log"
$StdErr = Join-Path $LogDir "fastapi-stderr-v5.log"

if (-not (Test-Path $LogDir)) {
    New-Item -Path $LogDir -ItemType Directory -Force | Out-Null
}

if (-not (Test-Path $Python)) {
    Write-Error "Python executable not found: $Python"
    exit 1
}

if (-not (Test-Path $ServerDir)) {
    Write-Error "Server directory not found: $ServerDir"
    exit 1
}

if (-not (Test-Path $ModelXml) -or -not (Test-Path $ModelBin)) {
    Write-Error "NPU model files not found."
    exit 2
}

Set-Location $ServerDir

# load-qdrant-api-key.ps1 is dot-sourced above and re-sets this to Stop, which
# turns uvicorn's stderr logging into a terminating NativeCommandError.
$ErrorActionPreference = "Continue"

& $Python `
    -m uvicorn `
    app:app `
    --app-dir C:\AI\server `
    --host 0.0.0.0 `
    --port 8000 `
    1>> $StdOut `
    2>> $StdErr

exit $LASTEXITCODE
