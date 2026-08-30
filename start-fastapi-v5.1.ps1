$ErrorActionPreference = "Continue"

$Root = "C:\AI"
$Python = Join-Path $Root "npu-env\Scripts\python.exe"
$ServerDir = Join-Path $Root "server"
$ModelXml = Join-Path $Root "models\lfm2-1.2b-npu-v2\openvino_model.xml"
$ModelBin = Join-Path $Root "models\lfm2-1.2b-npu-v2\openvino_model.bin"
$LogDir = Join-Path $Root "logs"

. (Join-Path $Root "load-qdrant-api-key.ps1")

# NAS services are on the local LAN; do not send them through an HTTP proxy.
$env:NO_PROXY = "192.168.1.3,127.0.0.1,localhost"
# The embedding model is already cached locally on this host.
$env:HF_HUB_OFFLINE = "1"

$StdOut = Join-Path $LogDir "fastapi-stdout-v5.1.log"
$StdErr = Join-Path $LogDir "fastapi-stderr-v5.1.log"
$RunnerLog = Join-Path $LogDir "fastapi-runner-v5.1.log"

if (-not (Test-Path $LogDir)) {
    New-Item -Path $LogDir -ItemType Directory -Force | Out-Null
}

function Write-RunnerLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format s), $Message
    try {
        Add-Content -Path $RunnerLog -Value $line -Encoding UTF8
    }
    catch {
    }
}

function Test-Port8000 {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $async = $client.BeginConnect("127.0.0.1", 8000, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne(300)) {
            $client.Close()
            return $false
        }
        $client.EndConnect($async)
        $client.Close()
        return $true
    }
    catch {
        return $false
    }
}

if (Test-Port8000) {
    Write-RunnerLog "FastAPI already listening on port 8000."
    exit 0
}

for ($i = 1; $i -le 24; $i++) {
    $ready = (
        (Test-Path $Python) -and
        (Test-Path $ServerDir) -and
        (Test-Path $ModelXml) -and
        (Test-Path $ModelBin)
    )
    if ($ready) {
        break
    }
    Start-Sleep -Seconds 5
}

if (
    -not (Test-Path $Python) -or
    -not (Test-Path $ServerDir) -or
    -not (Test-Path $ModelXml) -or
    -not (Test-Path $ModelBin)
) {
    Write-RunnerLog "Required Python/server/model files are not available."
    exit 2
}

for ($attempt = 1; $attempt -le 3; $attempt++) {

    if (Test-Port8000) {
        Write-RunnerLog "FastAPI became ready before attempt $attempt."
        exit 0
    }

    Write-RunnerLog "FastAPI startup attempt $attempt/3."

    try {
        $proc = Start-Process `
            -FilePath $Python `
            -ArgumentList "-m uvicorn app:app --app-dir C:\AI\server --host 0.0.0.0 --port 8000" `
            -WorkingDirectory $ServerDir `
            -RedirectStandardOutput $StdOut `
            -RedirectStandardError $StdErr `
            -WindowStyle Hidden `
            -PassThru
    }
    catch {
        Write-RunnerLog ("Start-Process failed: " + $_.Exception.Message)
        Start-Sleep -Seconds 10
        continue
    }

    # NPU cold-start can exceed two minutes after a full process restart.
    for ($wait = 1; $wait -le 300; $wait++) {
        Start-Sleep -Seconds 1

        if (Test-Port8000) {
            Write-RunnerLog "FastAPI READY on attempt $attempt. PID=$($proc.Id)"
            exit 0
        }

        try {
            $proc.Refresh()
        }
        catch {
        }

        if ($proc.HasExited) {
            Write-RunnerLog "FastAPI process exited during attempt $attempt. ExitCode=$($proc.ExitCode)"
            break
        }
    }

    if (-not $proc.HasExited) {
        try {
            Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
            Write-RunnerLog "Stopped timed-out FastAPI PID=$($proc.Id)"
        }
        catch {
        }
    }

    if ($attempt -lt 3) {
        Write-RunnerLog "Retrying in 15 seconds."
        Start-Sleep -Seconds 15
    }
}

Write-RunnerLog "FastAPI failed after 3 attempts."
exit 3
