# Train every core model for seeds 0-2 in sequence, then evaluate.
# Usage (from repo root, venv active):  .\scripts\train_all.ps1
# Add -ColorConstancy to train the colour-constancy variant of one model instead.
param(
    [string[]]$Models = @("small_cnn", "efficientnet_b0", "mobilenet_v2"),
    [int[]]$Seeds = @(0, 1, 2),
    [switch]$ColorConstancy
)
$ErrorActionPreference = "Stop"
foreach ($m in $Models) {
    foreach ($s in $Seeds) {
        $extra = @()
        if ($ColorConstancy) { $extra += "--color-constancy" }
        Write-Host "=== $m seed $s $extra ==="
        python -m ml.train --model $m --seed $s @extra
        if ($LASTEXITCODE -ne 0) { throw "training failed: $m seed $s" }
    }
}
python -m ml.evaluate
