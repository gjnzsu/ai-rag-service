import httpx
import pytest

from app.config import Settings


def test_gateway_options_ignore_provider_environment(monkeypatch):
    from app.model_access import gateway_options

    monkeypatch.setenv('OPENAI_BASE_URL', 'https://provider.invalid/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'provider-secret')
    config = Settings(_env_file=None, ai_gateway_base_url='http://gateway:4000/v1',
                      ai_gateway_api_key='gateway-secret')
    options = gateway_options(config)
    assert options['base_url'] == 'http://gateway:4000/v1'
    assert options['api_key'] == 'gateway-secret'
    assert options['default_headers']['x-consumer-service'] == 'ai-rag-service'


def test_missing_gateway_configuration_fails_closed():
    from app.model_access import gateway_options

    with pytest.raises(ValueError, match='AI_GATEWAY'):
        gateway_options(Settings(_env_file=None, ai_gateway_base_url='', ai_gateway_api_key=''))


def test_embedding_request_uses_gateway_transport(monkeypatch):
    from openai import OpenAI
    from app.model_access import gateway_options

    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(200, json={'object': 'list', 'model': 'text-embedding-3-small',
            'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.1, 0.2]}],
            'usage': {'prompt_tokens': 2, 'total_tokens': 2}})

    config = Settings(_env_file=None, ai_gateway_base_url='http://gateway:4000/v1',
                      ai_gateway_api_key='gateway-secret')
    with OpenAI(**gateway_options(config), http_client=httpx.Client(
            transport=httpx.MockTransport(respond))) as client:
        result = client.embeddings.create(model='text-embedding-3-small', input=['hello'],
                                          encoding_format='float')
    assert result.data[0].embedding == [0.1, 0.2]
    assert str(seen[0].url) == 'http://gateway:4000/v1/embeddings'
    assert seen[0].headers['authorization'] == 'Bearer gateway-secret'


def test_reused_transport_gets_current_request_context():
    from app.model_access import gateway_http_client, request_context

    seen = []
    with gateway_http_client(transport=httpx.MockTransport(
            lambda request: seen.append(dict(request.headers)) or httpx.Response(200))) as client:
        for identity in ('request-a', 'request-b'):
            token = request_context.set({'x-request-id': identity, 'x-ai-application-id': identity})
            try:
                client.get('http://gateway.test/v1/models')
            finally:
                request_context.reset(token)
        client.get('http://gateway.test/v1/models')
    assert seen[0]['x-request-id'] == 'request-a'
    assert seen[1]['x-ai-application-id'] == 'request-b'
    assert 'x-request-id' not in seen[2]


def test_rag_boundary_forwards_context_to_sync_model_call():
    from fastapi.testclient import TestClient
    from app.main import create_app
    from app.model_access import gateway_http_client

    seen = []
    app = create_app()
    with gateway_http_client(transport=httpx.MockTransport(
            lambda request: seen.append(dict(request.headers)) or httpx.Response(200))) as transport:
        @app.get('/test-model-call')
        def call_model():
            transport.get('http://gateway.test/v1/models')
            return {'ok': True}

        with TestClient(app) as client:
            response = client.get('/test-model-call', headers={
                'x-request-id': 'rag-request', 'x-ai-application-id': 'requirement-tool',
                'traceparent': '00-0123456789abcdef0123456789abcdef-0123456789abcdef-01'})
            generated = client.get('/test-model-call')
    assert response.headers['x-request-id'] == 'rag-request'
    assert seen[0]['x-ai-application-id'] == 'requirement-tool'
    assert seen[0]['traceparent'].endswith('-01')
    assert generated.headers['x-request-id'] == seen[1]['x-request-id']
    assert seen[1]['x-request-id'] != 'rag-request'
    assert 'x-ai-application-id' not in seen[1]
