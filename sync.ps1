param(
    [string]$msg = "update: " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
)

git add .
git commit -m $msg
git push
Write-Host ">>> Code pushed to GitHub successfully! <<<" -ForegroundColor Green
