param(
    [string]$ArtifactSource = 'C:\Stock Price Predictor\solar-filament-working\deadline\fusion_384_crop_fresh',
    [string]$RepoName = 'solar-filament-segmentation-2026'
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
gh auth status
if ($LASTEXITCODE -ne 0) { throw 'Run gh auth login, then rerun this script.' }

$names = @('maskrcnn_weights.pt', 'tiled_cnn_weights.pt', 'lightgbm_model.txt',
    'lightgbm_without_crop.txt', 'inference_config.json', 'runtime.txt')
$config = Get-Content -Raw (Join-Path $ArtifactSource 'inference_config.json') | ConvertFrom-Json
if (-not $config.crop_weight_files) { throw 'No crop CNN fold files are listed in inference_config.json.' }
$names += @($config.crop_weight_files)
$files = $names | ForEach-Object {
    $path = Join-Path $ArtifactSource $_
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing artifact: $path" }
    $path
}
$requirements = Join-Path $ArtifactSource 'requirements.txt'
if (-not (Test-Path -LiteralPath $requirements -PathType Leaf)) { throw "Missing exact requirements: $requirements" }
Copy-Item -LiteralPath $requirements -Destination 'requirements.txt' -Force
Compress-Archive -LiteralPath $files -DestinationPath 'model-artifacts.zip' -Force

if (-not (Test-Path .git)) {
    git init -b main
    if ($LASTEXITCODE -ne 0) { throw 'git init failed.' }
}
git add -- .
if ($LASTEXITCODE -ne 0) { throw 'git add failed.' }
git diff --cached --quiet
if ($LASTEXITCODE -eq 1) {
    git commit -m 'Package scored solar filament segmentation solution'
    if ($LASTEXITCODE -ne 0) { throw 'git commit failed. Check your Git name and email.' }
}
$remotes = @(git remote)
if ($LASTEXITCODE -ne 0) { throw 'Could not list Git remotes.' }
if ($remotes -notcontains 'origin') {
    gh repo create $RepoName --public --source . --remote origin --push
    if ($LASTEXITCODE -ne 0) { throw 'GitHub repository creation failed.' }
} else {
    git push -u origin main
    if ($LASTEXITCODE -ne 0) { throw 'Git push failed.' }
}
$releaseTags = @(gh release list --limit 100 --json tagName --jq '.[].tagName')
if ($LASTEXITCODE -ne 0) { throw 'Could not list GitHub releases.' }
if ($releaseTags -contains 'v1.0.0') {
    gh release upload v1.0.0 'model-artifacts.zip' --clobber
} else {
    gh release create v1.0.0 'model-artifacts.zip' --target main --title 'Scored submission model artifacts' --notes 'Inference weights, verifier models and configuration for the scored CSV.'
}
if ($LASTEXITCODE -ne 0) { throw 'Release upload failed.' }
gh repo view --json url,visibility --jq '.url + " (" + .visibility + ")"'
