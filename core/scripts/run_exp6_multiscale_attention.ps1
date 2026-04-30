$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $repoRoot

$logRoot = Join-Path $repoRoot "core\storage\outputs\multiscale_attention\run_logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null

$startedAt = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryLog = Join-Path $logRoot "exp6_multiscale_attention_$startedAt.summary.log"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name,

        [Parameter(Mandatory = $true)]
        [string] $LogName,

        [Parameter(Mandatory = $true)]
        [string[]] $Arguments
    )

    $stepLog = Join-Path $logRoot "$startedAt.$LogName.log"
    "[$(Get-Date -Format o)] START $Name" | Tee-Object -FilePath $summaryLog -Append
    "Command: uv $($Arguments -join ' ')" | Tee-Object -FilePath $summaryLog -Append

    & uv @Arguments 2>&1 | Tee-Object -FilePath $stepLog -Append
    $exitCode = $LASTEXITCODE

    if ($exitCode -ne 0) {
        "[$(Get-Date -Format o)] FAIL $Name exit=$exitCode log=$stepLog" | Tee-Object -FilePath $summaryLog -Append
        throw "$Name failed with exit code $exitCode"
    }

    "[$(Get-Date -Format o)] DONE $Name log=$stepLog" | Tee-Object -FilePath $summaryLog -Append
}

"[$(Get-Date -Format o)] Exp6 multiscale_attention run started in $repoRoot" | Tee-Object -FilePath $summaryLog -Append

Invoke-Step `
    -Name "train market1501" `
    -LogName "01_train_market1501" `
    -Arguments @(
        "run", "python", "-m", "core.tools.train",
        "--config_file", "./core/configs/exp6_multiscale_attention.yml",
        "DATASETS.NAMES", "('market1501',)",
        "OUTPUT_DIR", "./core/storage/outputs/multiscale_attention/train/market1501"
    )

Invoke-Step `
    -Name "train dukemtmc" `
    -LogName "02_train_dukemtmc" `
    -Arguments @(
        "run", "python", "-m", "core.tools.train",
        "--config_file", "./core/configs/exp6_multiscale_attention.yml",
        "DATASETS.NAMES", "('dukemtmc',)",
        "OUTPUT_DIR", "./core/storage/outputs/multiscale_attention/train/dukemtmc"
    )

Invoke-Step `
    -Name "test market1501" `
    -LogName "03_test_market1501" `
    -Arguments @(
        "run", "python", "-m", "core.tools.test",
        "--config_file", "./core/configs/exp6_multiscale_attention.yml",
        "DATASETS.NAMES", "('market1501',)",
        "TEST.WEIGHT", "./core/storage/outputs/multiscale_attention/train/market1501/resnet50_model_60.pth",
        "OUTPUT_DIR", "./core/storage/outputs/multiscale_attention/test/market1501"
    )

Invoke-Step `
    -Name "test dukemtmc" `
    -LogName "04_test_dukemtmc" `
    -Arguments @(
        "run", "python", "-m", "core.tools.test",
        "--config_file", "./core/configs/exp6_multiscale_attention.yml",
        "DATASETS.NAMES", "('dukemtmc',)",
        "TEST.WEIGHT", "./core/storage/outputs/multiscale_attention/train/dukemtmc/resnet50_model_60.pth",
        "OUTPUT_DIR", "./core/storage/outputs/multiscale_attention/test/dukemtmc"
    )

"[$(Get-Date -Format o)] Exp6 multiscale_attention run completed" | Tee-Object -FilePath $summaryLog -Append
