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

