$ErrorActionPreference = "Stop"

if (-not $env:OPENAI_API_KEY) {
    $env:OPENAI_API_KEY = "test-key"
}

python -m ruff check app tests
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

python -m pytest -q
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
