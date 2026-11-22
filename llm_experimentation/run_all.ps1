# Runs every strategy x model x inputs cell of the LLM baseline (seed 0, color, red + blue: run_benchmarks.py's benchmark1).
# Usage, from the repo root:
#   $env:AWS_BEARER_TOKEN_BEDROCK = "<bedrock api key>"
#   .\llm_experimentation\run_all.ps1 -MaxUsd 25
#   .\llm_experimentation\run_all.ps1 -Models sonnet,opus -Strategies open -Inputs image
param(
    [string]$Region = "us-east-2",
    [double]$MaxUsd = 25,                                  # per run
    [int]$Rounds = 30,
    [string[]]$Models = @("sonnet", "terra", "opus"),
    [string[]]$Strategies = @("schema", "open", "direct"),
    [string[]]$Inputs = @("image", "image+text", "text")
)

if (-not $env:AWS_BEARER_TOKEN_BEDROCK) {
    Write-Error "Set `$env:AWS_BEARER_TOKEN_BEDROCK first (the Bedrock API key)."
    exit 1
}

foreach ($strategy in $Strategies) {
    foreach ($inputs in $Inputs) {
        foreach ($model in $Models) {
            Write-Host "=== $strategy / $model / $inputs ==="
            python llm_experimentation/run_experiment.py --strategy $strategy --model $model --inputs $inputs `
                --region $Region --rounds $Rounds --max-usd $MaxUsd
            if ($LASTEXITCODE -ne 0) { Write-Warning "$strategy / $model / $inputs failed (exit $LASTEXITCODE); continuing" }
        }
    }
}

python llm_experimentation/compare_results.py --config color_red-blue
