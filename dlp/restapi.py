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

dlp_services_base_url = 'https://dlp-services.co-bee.com:8003'

def set_dlp_services_base_url(url):
	global dlp_services_base_url
	dlp_services_base_url = url

def download_weights(encoded_token, weights_file_path):
	global dlp_services_base_url

	token = json.loads(encoded_token)
	id = token['id']
	jwt_token = token['jwtToken']

	msg_code, msg_resp = get(url=dlp_services_base_url+'/get-aimodel', query={'id': id}, token=jwt_token)
	if msg_code % 100 == 0:
		weights_url = msg_resp['weights']
		if not weights_url or 'https://' not in weights_url:
			return

		with requests.get(weights_url, stream=True) as r:
			# r.raise_for_status()
			with open(weights_file_path, 'wb') as f:
				for chunk in r.iter_content(chunk_size=8192):
					# If you have chunk encoded response uncomment if
					# and set chunk_size parameter to None.
					#if chunk: 
					f.write(chunk)

def update_train_result(encoded_token, weights_file_path, weights_file_name, epoch_train_loss, epoch_test_loss, epoch_tp, epoch_fp, epoch_fn):
	global dlp_services_base_url

	token = json.loads(encoded_token)
	id = token['id']
	jwt_token = token['jwtToken']

	files = {'file': (weights_file_name, open(weights_file_path, 'rb'))}
	msg_code, msg_resp = post_file(url=dlp_services_base_url+'/upload-weights', query={}, files=files, data={}, token=None)
	if msg_code % 100 != 0:
		return False

	body = {
		'weights': msg_resp['url'],
		'eTrainLoss': epoch_train_loss,
		'eTestLoss': epoch_test_loss,
		'eTP': epoch_tp,
		'eFP': epoch_fp,
		'eFN': epoch_fn,
	}
	msg_code, msg_resp = patch(url=dlp_services_base_url+'/update-aimodel-with-train-result?id='+id, query={}, body=body, token=jwt_token)
	if msg_code % 100 != 0:
		return False

	return True
