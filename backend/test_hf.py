import urllib.request
import urllib.error
import json

url = 'https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english'
headers = {
    'Authorization': 'Bearer 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',
    'Content-Type': 'application/json'
}
data = json.dumps({'inputs': 'Test text.'}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers=headers)

try:
    res = urllib.request.urlopen(req)
    print(res.read().decode())
except urllib.error.HTTPError as e:
    print(f"HTTPError {e.code}: {e.read().decode()}")
