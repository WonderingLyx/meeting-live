param(
    [string[]]$Path
)

$ErrorActionPreference = "Stop"

foreach ($raw in $Path) {
    foreach ($item in ([string]$raw -split ",")) {
        $item = $item.Trim()
        if (-not $item) {
            continue
        }
    $errors = $null
    [System.Management.Automation.PSParser]::Tokenize(
        (Get-Content -Raw -LiteralPath $item),
        [ref]$errors
    ) | Out-Null
    if ($errors) {
        throw "PowerShell parse failed for ${item}: $errors"
    }
    Write-Host "OK $item"
    }
}
