$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

Write-Host 'Installing npm dependencies...'
npm install

if (-not (Test-Path '.env') -and (Test-Path '.env.example')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'Created .env from .env.example'
}

Write-Host ''
Write-Host 'Setup complete.'
Write-Host 'Next step: add your OPENAI_API_KEY to .env, then run npm run dev'
