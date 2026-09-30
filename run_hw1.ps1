# 从本目录执行 .\run_hw1.ps1；已完成的训练按结果文件跳过。
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$resolved = (Resolve-Path -LiteralPath $PSScriptRoot).Path
$drive = $resolved.Substring(0, 1).ToLowerInvariant()
$relative = $resolved.Substring(2).Replace('\', '/')
$linuxRoot = "/mnt/$drive$relative"

if (-not (Test-Path -LiteralPath 'results/completed/train.txt')) {
    python fetch_corpus.py
    if ($LASTEXITCODE -ne 0) { throw '语料获取或校验失败' }
    python run_assignment.py --prepare-only
    if ($LASTEXITCODE -ne 0) { throw '语料预处理失败' }
}
wsl.exe -d Ubuntu-22.04 -- bash "$linuxRoot/env/ensure_full.sh"
if ($LASTEXITCODE -ne 0) { throw 'KenLM 环境检查失败' }
foreach ($script in @('run_full.sh', 'run_full_stupid.sh')) {
    wsl.exe -d Ubuntu-22.04 -- bash "$linuxRoot/env/$script"
    if ($LASTEXITCODE -ne 0) { throw "执行失败：$script" }
}
if (-not (Test-Path -LiteralPath 'results/full/generation.json')) {
    wsl.exe -d Ubuntu-22.04 -- bash "$linuxRoot/env/run_full_compare.sh"
    if ($LASTEXITCODE -ne 0) { throw '排序与续写失败' }
}
python build_full_report.py
if ($LASTEXITCODE -ne 0) { throw '报告生成失败' }
