param(
    [int]$Steps = 100000,
    [int]$Seeds = 10,
    [int]$Repeats = 5,
    [int]$Workers = 8,
    [int]$SampleInterval = 500,
    [int]$CompetenceCatches = 10,
    [string]$OutputRoot = "benchmark_results\\studies_full"
)

$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

python run_frog_study.py `
    --steps $Steps `
    --seeds $Seeds `
    --repeats $Repeats `
    --workers $Workers `
    --sample-interval $SampleInterval `
    --competence-catches $CompetenceCatches `
    --output-root $OutputRoot
