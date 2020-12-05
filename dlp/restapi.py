import requests
import json

def get(url, query, token):
	headers = {}
	if token is not None:
		headers['Authorization'] = token

	res = requests.get(url=url, params=query, headers=headers, timeout=20)
	try:
		msg = json.loads(res.text)
		return msg['msgCode'], msg['msgResp']
	except Exception as e:
		return 2001, res.status_code

def post(url, query, body, token):
	headers = {'Content-Type': 'application/json'}
	if token is not None:
		headers['Authorization'] = token
		
	res = requests.post(url=url, params=query, json=body, headers=headers, timeout=20)
	try:
		msg = json.loads(res.text)
		return msg['msgCode'], msg['msgResp']
	except Exception as e:
		return 2001, res.status_code

def post_file(url, query, files, data, token):
	headers = {'Content-Type': 'application/octet-stream'}
	if token is not None:
		headers['Authorization'] = token

	res = requests.post(url=url, params=query, files=files, data=data, timeout=20)
	try:
		msg = json.loads(res.text)
		return msg['msgCode'], msg['msgResp']
	except Exception as e:
		return 2001, res.status_code

def put(url, query, body, token):
	headers = {'Content-Type': 'application/json'}
	if token is not None:
		headers['Authorization'] = token
		
	res = requests.put(url=url, params=query, json=body, headers=headers, timeout=20)
	try:
		msg = json.loads(res.text)
		return msg['msgCode'], msg['msgResp']
	except Exception as e:
		return 2001, res.status_code

def patch(url, query, body, token):
	headers = {'Content-Type': 'application/json'}
	if token is not None:
		headers['Authorization'] = token
		
	res = requests.patch(url=url, params=query, json=body, headers=headers, timeout=20)
	try:
		msg = json.loads(res.text)
		return msg['msgCode'], msg['msgResp']
	except Exception as e:
		return 2001, res.status_code

def delete(url, query, token):
	headers = {}
	if token is not None:
		headers['Authorization'] = token

	res = requests.delete(url=url, params=query, headers=headers, timeout=20)
	try:
		msg = json.loads(res.text)
		return msg['msgCode'], msg['msgResp']
	except Exception as e:
		return 2001, res.status_code

def update_train_result(encoded_token, weight_file_path, weights_file_name, train_result):
	token = json.loads(encoded_token)
	id = token['id']
	jwt_token = token['jwtToken']

	files = {'file': (weights_file_name, open(weight_file_path, 'rb'))}
	msg_code, msg_resp = post_file(url='https://ai-designer.io/upload/weights', query={}, files=files, data={}, token=None)
	if msg_code != 1000:
		print(msg_resp)

	body = {
		'weights': msg_resp['url'],
		'trainResult': train_result,
	}
	msg_code, msg_resp = patch(url='https://ai-designer.io/api/aimodel/update?id='+id, query={}, body=body, token=jwt_token)
	if msg_code != 1000:
		print(msg_resp)
