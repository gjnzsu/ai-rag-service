"""Explicit model access through AI Gateway; no provider credential fallback."""

from urllib.parse import urlsplit
from contextvars import ContextVar

import httpx

from app.config import Settings, settings

request_context: ContextVar[dict[str, str]] = ContextVar('gateway_request_context', default={})
CONTEXT_HEADERS = ('x-request-id', 'traceparent', 'tracestate', 'x-ai-application-id',
                   'x-ai-project-id', 'x-ai-team-id', 'x-ai-use-case', 'x-ai-feature')


def _forward_context(request: httpx.Request) -> None:
    # Resolve at send time: pooled clients must not retain another request's identity.
    for key, value in request_context.get().items():
        request.headers[key] = value


def gateway_http_client(*, client=None, **kwargs) -> httpx.Client:
    if client is None:
        return httpx.Client(event_hooks={'request': [_forward_context]}, **kwargs)
    client.event_hooks = {'request': [_forward_context]}
    return client


def gateway_options(config: Settings = settings) -> dict:
    endpoint = config.ai_gateway_base_url.rstrip('/')
    parsed = urlsplit(endpoint)
    credential = config.ai_gateway_api_key.get_secret_value()
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or not parsed.path.endswith('/v1') or not credential.strip()):
        raise ValueError('AI_GATEWAY_BASE_URL must be an HTTP(S) /v1 endpoint and '
                         'AI_GATEWAY_API_KEY must be configured')
    return {
        'base_url': endpoint,
        'api_key': credential,
        'default_headers': {
            'x-consumer-service': 'ai-rag-service',
            'x-ai-application-id': config.ai_gateway_application_id,
        },
    }
