# COELANOX Windows install: copies coelanox.exe to user bin dir and adds to PATH
$dest = Join-Path $env:LOCALAPPDATA "coelanox\bin"
New-Item -ItemType Directory -Force -Path $dest | Out-Null
$exe = Join-Path $PSScriptRoot "coelanox.exe"
if (Test-Path $exe) {
    Copy-Item -Force $exe (Join-Path $dest "coelanox.exe")
    $path = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($path -notlike "*$dest*") {
        [Environment]::SetEnvironmentVariable("Path", "$path;$dest", "User")
        Write-Host "Added $dest to PATH. Restart your terminal for changes to take effect."
    }
    Write-Host "Installed coelanox to $dest"
} else {
    Write-Host "coelanox.exe not found in $PSScriptRoot"
    exit 1
}
