# STCH-Set-BO Overnight Benchmark Suite
# Run from: projects/stch-botorch/
# Estimated runtime: 8-12 hours depending on hardware

$ErrorActionPreference = "Continue"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = "benchmarks/results/overnight_$timestamp"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "STCH-Set-BO Overnight Benchmark Suite"
Write-Host "Started: $(Get-Date)"
Write-Host "Log dir: $logDir"
Write-Host "============================================" -ForegroundColor Cyan

# Activate conda env if needed
# conda activate stch-botorch

# ---------------------------------------------------------------------------
# 1. DTLZ2 m=5: Main comparison (5 seeds, ~2-3 hours)
# ---------------------------------------------------------------------------
Write-Host "`n[1/6] DTLZ2 m=5 - Main comparison (5 seeds)" -ForegroundColor Yellow
python benchmarks/dtlz_benchmark_v2.py `
    --problem DTLZ2 --m 5 --seeds 5 --iters 30 --K 5 --mu 0.1 `
    --methods stch_set stch_nparego qnparego qehvi random `
    --output "$logDir/dtlz2_m5_main.json" `
    2>&1 | Tee-Object "$logDir/dtlz2_m5_main.log"

# ---------------------------------------------------------------------------
# 2. DTLZ2 m=5: K ablation (5 seeds, ~2-3 hours)
# ---------------------------------------------------------------------------
Write-Host "`n[2/6] DTLZ2 m=5 - K ablation {3,5,10}" -ForegroundColor Yellow
python benchmarks/dtlz_benchmark_v2.py `
    --problem DTLZ2 --m 5 --seeds 5 --iters 30 --mu 0.1 `
    --ablation k `
    --output "$logDir/dtlz2_m5_ablation_k.json" `
    2>&1 | Tee-Object "$logDir/dtlz2_m5_ablation_k.log"

# ---------------------------------------------------------------------------
# 3. DTLZ2 m=5: mu ablation (5 seeds, ~2-3 hours)
# ---------------------------------------------------------------------------
Write-Host "`n[3/6] DTLZ2 m=5 - mu ablation {0.01, 0.1, 0.5, 1.0}" -ForegroundColor Yellow
python benchmarks/dtlz_benchmark_v2.py `
    --problem DTLZ2 --m 5 --seeds 5 --iters 30 --K 5 `
    --ablation mu `
    --output "$logDir/dtlz2_m5_ablation_mu.json" `
    2>&1 | Tee-Object "$logDir/dtlz2_m5_ablation_mu.log"

# ---------------------------------------------------------------------------
# 4. DTLZ2 m=8: High-dimensional objectives (3 seeds, skip qEHVI)
# ---------------------------------------------------------------------------
Write-Host "`n[4/6] DTLZ2 m=8 - Many objectives (3 seeds, no qEHVI)" -ForegroundColor Yellow
python benchmarks/dtlz_benchmark_v2.py `
    --problem DTLZ2 --m 8 --seeds 3 --iters 30 --K 8 --mu 0.1 `
    --skip-qehvi `
    --methods stch_set stch_nparego qnparego random `
    --output "$logDir/dtlz2_m8_main.json" `
    2>&1 | Tee-Object "$logDir/dtlz2_m8_main.log"

# ---------------------------------------------------------------------------
# 5. ZDT2 m=2: Bi-objective baseline (3 seeds)
# ---------------------------------------------------------------------------
Write-Host "`n[5/6] ZDT2 m=2 - Bi-objective (3 seeds)" -ForegroundColor Yellow
python benchmarks/dtlz_benchmark_v2.py `
    --problem ZDT2 --m 2 --seeds 3 --iters 30 --K 3 --mu 0.1 `
    --methods stch_set stch_nparego qnparego qehvi random `
    --output "$logDir/zdt2_m2_main.json" `
    2>&1 | Tee-Object "$logDir/zdt2_m2_main.log"

# ---------------------------------------------------------------------------
# 6. ZDT3 m=2: Disconnected Pareto front (3 seeds)
# ---------------------------------------------------------------------------
Write-Host "`n[6/6] ZDT3 m=2 - Disconnected front (3 seeds)" -ForegroundColor Yellow
python benchmarks/dtlz_benchmark_v2.py `
    --problem ZDT3 --m 2 --seeds 3 --iters 30 --K 3 --mu 0.1 `
    --methods stch_set stch_nparego qnparego qehvi random `
    --output "$logDir/zdt3_m2_main.json" `
    2>&1 | Tee-Object "$logDir/zdt3_m2_main.log"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host "`n============================================" -ForegroundColor Cyan
Write-Host "Overnight benchmarks complete!"
Write-Host "Finished: $(Get-Date)"
Write-Host "Results in: $logDir"
Write-Host "============================================" -ForegroundColor Cyan

# List result files
Get-ChildItem "$logDir/*.json" | ForEach-Object {
    $size = [math]::Round($_.Length / 1024, 1)
    Write-Host "  $($_.Name) (${size}KB)"
}
