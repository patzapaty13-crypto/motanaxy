"""Run against the local Next.js server: python check_local_app.py."""
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request(path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = Request('http://127.0.0.1:3000' + path, data=data,
                  headers={'Content-Type': 'application/json'})
    with urlopen(req, timeout=60) as response:
        return response.read().decode('utf-8')


def main():
    assert 'MOTANAXY' in request('/')
    info = json.loads(request('/api/model'))
    assert info['parameters'] == 858880
    assert info['run'] == 'knowledge_v1'
    assert info['training_documents'] == 96
    result = json.loads(request('/api/chat', {
        'message': 'def add(a, b):', 'max_tokens': 12, 'top_k': 1,
    }))
    assert result['code'].startswith('def add(a, b):\n    ')
    assert result['code'].endswith(result['raw_code'])
    assert result['elapsed_seconds'] >= 0
    for invalid in [{'message': '   '}, {'message': 'x', 'temperature': 0},
                    {'message': 'x', 'max_tokens': 501}, {'message': 'x', 'top_k': 257}]:
        try:
            request('/api/chat', invalid)
        except HTTPError as exc:
            assert exc.code == 422, exc.code
        else:
            raise AssertionError(f'Invalid input accepted: {invalid}')
    print('PASS: Next.js homepage, API proxy, model metadata, real inference, input validation')


if __name__ == '__main__':
    main()
