"""Local opt-in presentation of the graph APIs; no backend or model initialization."""

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router = APIRouter()
_STATIC = Path(__file__).parent / 'static'
_HEADERS = {
    'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff',
    'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; "
                               "connect-src 'self'; img-src 'self' data:; object-src 'none'; "
                               "base-uri 'none'; frame-ancestors 'none'",
    'Referrer-Policy': 'no-referrer',
}
_ASSETS = {'demo.js': 'text/javascript', 'panorama.js': 'text/javascript', 'demo.css': 'text/css'}


@router.get('/demo', include_in_schema=False)
def graph_demo():
    return FileResponse(_STATIC / 'index.html', media_type='text/html', headers=_HEADERS)


@router.get('/demo/assets/{name}', include_in_schema=False)
def graph_demo_asset(name: str):
    if name not in _ASSETS:
        raise HTTPException(status_code=404, detail='Asset not found')
    return FileResponse(_STATIC / name, media_type=_ASSETS[name], headers=_HEADERS)
