param([string]$Message = "")
$PI_USER = "evoroot"
$PI_HOST = "192.168.10.2"
$PI_DIR  = "~/deck-checker"
function Green($t)  { Write-Host $t -ForegroundColor Green }
function Yellow($t) { Write-Host $t -ForegroundColor Yellow }
function Red($t)    { Write-Host $t -ForegroundColor Red }
function Cyan($t)   { Write-Host $t -ForegroundColor Cyan }
Cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
Cyan "  Deck Checker - Auto Sync"
Cyan "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
$status = git status --porcelain
if ($status) {
    if ($Message -eq "") { $Message = "Auto sync $(Get-Date -Format 'yyyy-MM-dd HH:mm')" }
    Yellow "[ 1/3 ] Committing: $Message"
    git add .
    git commit -m $Message
    if ($LASTEXITCODE -ne 0) { Red "Commit failed!"; exit 1 }
    Yellow "[ 2/3 ] Pushing to GitHub..."
    git push
    if ($LASTEXITCODE -ne 0) { Red "Push failed!"; exit 1 }
    Green "        Pushed OK"
} else {
    Yellow "Nothing to commit - checking Pi..."
}
Write-Host ""
Yellow "[ 3/3 ] Pulling on Raspberry Pi ($PI_HOST)..."
ssh "${PI_USER}@${PI_HOST}" "cd $PI_DIR && source .venv/bin/activate && git pull && echo '--- Tests ---' && pytest tests/ -q 2>&1 | tail -3"
if ($LASTEXITCODE -ne 0) { Red "Pi sync failed!"; exit 1 }
Write-Host ""
Green "  All done!"
