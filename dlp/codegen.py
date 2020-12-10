import numpy as np
import json
import os
import pkg_resources


def read_json_model(file):
	graph_json = open(file, 'r').read()
	graph = json.loads(graph_json)
	nodes = graph['vertices']
	connection = graph['connection']
	return nodes, connection

def write_codegen(code_lines, output_file_path):
	file = open(output_file_path, 'w')
	for code_line in code_lines:
		file.write(code_line+'\n')

def get_datagen_node(nodes):
	datagen_node = None
	datagen_vertex = -1
	for i in range(len(nodes)):
		node = nodes[i]
		blockType = node['blockType']
		if blockType == 'IMAGE_CLASSIFICATION_DATAGEN':
			datagen_node = node
			datagen_vertex = i
			break
		elif blockType == 'HEATMAP_REGRESSION_DATAGEN':
			datagen_node = node
			datagen_vertex = i
			break
		elif blockType == 'OBJECT_DETECTION_DATAGEN':
			datagen_node = node
			datagen_vertex = i
			break

	return datagen_node, datagen_vertex

def get_input_node(nodes):
	input_node = None
	input_vertex = -1
	for i in range(len(nodes)):
		node = nodes[i]
		blockType = node['blockType']
		if blockType == 'INPUT_LAYER':
			input_node = node
			input_vertex = i
			break

	return input_node, input_vertex

def traverse(nodes, serialisation, conn2d, prev_vertex, vertex):
	serialisation.append([prev_vertex, vertex, nodes[vertex]])

	col = conn2d[:, vertex]
	total_conns = np.sum(col)
	if total_conns > 1:
		return

	row = conn2d[vertex, :]
	for next_vertex in range(len(row)):
		if conn2d[vertex, next_vertex] == 1:
			traverse(
				nodes=nodes,
				serialisation=serialisation, 
				conn2d=conn2d, 
				prev_vertex=vertex, 
				vertex=next_vertex)
			conn2d[vertex, next_vertex] = 0

def gen_model_part(serialisation, current_code_lines, inference=None):
	input_tensor_name = None
	output_tensor_name = None
	loss_func_name = None
	code_lines = []

	code_lines.append('def build_model():')

	# Add tensor none var
	code_lines.append('\ttensorNone = None')

	# Initialise zero tensor for ADD layers
	for i in range(len(serialisation)):
		_, vertex, node = serialisation[i]
		conn_type = node['type']
		if 'MANY_' in conn_type: 
			code_line = '\t'+'tensor'+str(vertex)+' = None'
			if code_line not in code_lines:
				code_lines.append(code_line);

	# Generate layers code
	for i in range(len(serialisation)):
		prev_vertex, vertex, node = serialisation[i]
		func_name = node['blockType']
		params = node['params']
		conn_type = node['type']

		if 'MANY_' in conn_type: 
			tensor_name = 'tensor'+str(vertex)
			_tensor_name = 'tensor'+str(prev_vertex)
			if func_name == 'ADD_LAYER':
				code_line = tensor_name+' = blocks.'+func_name+'(tensor1='+tensor_name+', tensor2='+_tensor_name+')'
			elif func_name == 'CONCAT_LAYER':
				code_line = tensor_name+' = blocks.'+func_name+'(tensor1='+tensor_name+', tensor2='+_tensor_name+', axis='+str(params['axis'])+')'
		else:
			func_input = 'input_tensor=tensor'+str(prev_vertex)
			for param in params:
				value = params[param]

				if param == 'shape':
					continue

				if type(value) is str:
					value = "'"+value+"'"

				if type(value) is list:
					value = '['+', '.join([str(elem) for elem in value])+']'

				if type(value) is int or type(value) is float:
					value = str(value)

				if inference is not None and param in ['trainable', 'bn_trainable']:
					value = '0'

				func_input += ', '+param+'='+value

			prev_var_name = 'tensor'+str(prev_vertex)
			var_name = 'tensor'+str(vertex)

			if node['blockType'] == 'INPUT_LAYER':
				input_tensor_name = var_name

			if 'LOSS_FUNC' in node['blockType']:
				output_tensor_name = prev_var_name
				var_name = 'loss_func'+str(vertex)
				loss_func_name = var_name

			code_line = var_name+' = blocks.'+func_name+'('+func_input+')'
		
		code_lines.append('\t'+code_line)

	if inference is not None:
		procedure = inference['procedure']
		if procedure == 'heatmap_regression':
			code_lines.append('')
			code_lines.append('\theatmap_4dtensor = '+output_tensor_name+' - '+input_tensor_name)
			code_lines.append('\ttensor = heatmap_4dtensor[0] # (h, w, 5), batch_size = 1')
			code_lines.append('\ttensor = tf.where(condition=tf.math.greater(x=tensor, y=192), x=tensor, y=tensor*0)')
			code_lines.append('\ttensor = tf.transpose(a=tensor, perm=[2, 0, 1]) # (5, h, w)')
			code_lines.append('\thm1 = tf.reshape(tensor=tensor[0], shape=[-1])')
			code_lines.append('\thm2 = tf.reshape(tensor=tensor[1], shape=[-1])')
			code_lines.append('\thm3 = tf.reshape(tensor=tensor[2], shape=[-1])')
			code_lines.append('\thm4 = tf.reshape(tensor=tensor[3], shape=[-1])')
			code_lines.append('\thm5 = tf.reshape(tensor=tensor[4], shape=[-1])')
			code_lines.append('\ta = tf.math.argmax(input=hm1)')
			code_lines.append('\tb = tf.math.argmax(input=hm2)')
			code_lines.append('\tc = tf.math.argmax(input=hm3)')
			code_lines.append('\td = tf.math.argmax(input=hm4)')
			code_lines.append('\te = tf.math.argmax(input=hm5)')
			code_lines.append('\ttensor = [a, b, c, d, e]')
			code_lines.append('\t'+output_tensor_name+' = tensor')
			code_lines.append('\t'+loss_func_name+' = None')
			code_lines.append('')

	# Generate mode code
	code_lines.append('\tmodel = tf.keras.models.Model(inputs='+input_tensor_name+', outputs='+output_tensor_name+')')
	code_lines.append('\tmodel.compile(optimizer=tf.keras.optimizers.Adam(), loss='+loss_func_name+')')
	code_lines.append('\treturn model')
	code_lines.append('')

	current_code_lines += code_lines

def gen_train_part(datagen_node, code_lines):
	train_procedure = datagen_node['params']['train_procedure'].lower()
	file = open(pkg_resources.resource_filename(__name__, 'code_templates/train_'+train_procedure+'.py'), 'r')
	lines = file.readlines()
	for i in range(len(lines)):
		code_line = lines[i][:-1]
		code_lines.append(code_line);

	code_lines.append('');

	if train_procedure == 'image_classification':
		code_lines.append('train(dataset_name='+json.dumps(datagen_node['params']['dataset_name'])+
			', image_shape='+json.dumps(datagen_node['params']['image_shape'])+
			', epochs='+str(datagen_node['params']['epochs'])+
			', total_train_examples='+str(datagen_node['params']['total_train_examples'])+
			', total_test_examples='+str(datagen_node['params']['total_test_examples'])+
			', batch_size='+str(datagen_node['params']['batch_size'])+
			')');
		code_lines.append('')
	elif train_procedure == 'object_detection':
		code_lines.append('train(dataset_name='+json.dumps(datagen_node['params']['dataset_name'])+
			', image_shape='+json.dumps(datagen_node['params']['image_shape'])+
			', scale_sizes='+json.dumps(datagen_node['params']['scale_sizes'])+
			', anchor_sizes='+json.dumps(datagen_node['params']['anchor_sizes'])+
			', iou_thresholds='+json.dumps(datagen_node['params']['iou_thresholds'])+
			', anchor_sampling='+json.dumps(datagen_node['params']['anchor_sampling'])+
			', epochs='+str(datagen_node['params']['epochs'])+')');
		code_lines.append('')
	elif train_procedure == 'object_detection_4tiers':
		code_lines.append('train(dataset_name='+json.dumps(datagen_node['params']['dataset_name'])+
			', image_shape='+json.dumps(datagen_node['params']['image_shape'])+
			', scale_sizes='+json.dumps(datagen_node['params']['scale_sizes'])+
			', anchor_sizes='+json.dumps(datagen_node['params']['anchor_sizes'])+
			', iou_thresholds='+json.dumps(datagen_node['params']['iou_thresholds'])+
			', anchor_sampling='+json.dumps(datagen_node['params']['anchor_sampling'])+
			', epochs='+str(datagen_node['params']['epochs'])+')');
		code_lines.append('')
	elif train_procedure == 'heatmap_regression':
		code_lines.append('train(dataset_name='+json.dumps(datagen_node['params']['dataset_name'])+
			', image_shape='+json.dumps(datagen_node['params']['image_shape'])+
			', epochs='+str(datagen_node['params']['epochs'])+
			', total_train_examples='+str(datagen_node['params']['total_train_examples'])+
			', total_test_examples='+str(datagen_node['params']['total_test_examples'])+
			', batch_size='+str(datagen_node['params']['batch_size'])+
			')');
		code_lines.append('')

def gen_convert_part(datagen_node, code_lines, weights_file_path, output_path, settings):
	train_procedure = datagen_node['params']['train_procedure'].lower()
	file = open(pkg_resources.resource_filename(__name__, 'code_templates/convert_'+train_procedure+'.py'), 'r')
	lines = file.readlines()
	for i in range(len(lines)):
		code_line = lines[i][:-1]
		code_lines.append(code_line);

	code_lines.append('');

	if train_procedure == 'image_classification':
		code_lines.append('convert(weights_file_path=\''+weights_file_path+
			'\', output_path=\''+output_path+'\''+
			')');
		code_lines.append('')
	elif train_procedure == 'object_detection':
		code_lines.append('convert(weights_file_path=\''+weights_file_path+
			'\', output_path=\''+output_path+'\''+
			')');
		code_lines.append('')
	elif train_procedure == 'object_detection_4tiers':
		code_lines.append('convert(weights_file_path=\''+weights_file_path+
			'\', output_path=\''+output_path+'\''+
			')');
		code_lines.append('')
	elif train_procedure == 'heatmap_regression':
		code_lines.append('convert(weights_file_path=\''+weights_file_path+
			'\', output_path=\''+output_path+'\''+
			')');
		code_lines.append('')

def generate_code_for_train(json_model_file, output_path, encoded_token):
	nodes, connection = read_json_model(file=json_model_file)
	datagen_node, datagen_vertex = get_datagen_node(nodes=nodes)
	input_node, input_vertex = get_input_node(nodes=nodes)

	serialisation = []
	conn2d = np.array(connection)
	traverse(
		nodes=nodes,
		serialisation=serialisation, 
		conn2d=conn2d, 
		prev_vertex=None, 
		vertex=input_vertex)

	code_lines = []

	code_lines.append('import os')
	code_lines.append('os.system(\'pip install dlp\')')
	code_lines.append('import tensorflow as tf')
	code_lines.append('import numpy as np')
	code_lines.append('import json')
	code_lines.append('import dlp.blocks as blocks')
	code_lines.append('import dlp.utils as utils')
	code_lines.append('import dlp.restapi as restapi')
	code_lines.append('')
	code_lines.append('encoded_token = \''+encoded_token+'\'')
	code_lines.append('')

	gen_model_part(serialisation=serialisation, current_code_lines=code_lines, inference=None)
	gen_train_part(datagen_node=datagen_node, code_lines=code_lines)

	# Write to file
	write_codegen(code_lines=code_lines, output_file_path=output_path+'/train.py')

def generate_code_for_convert(json_model_file, output_path, weights_file_path, jSettings):
	nodes, connection = read_json_model(file=json_model_file)
	datagen_node, datagen_vertex = get_datagen_node(nodes=nodes)
	input_node, input_vertex = get_input_node(nodes=nodes)

	serialisation = []
	conn2d = np.array(connection)
	traverse(
		nodes=nodes,
		serialisation=serialisation, 
		conn2d=conn2d, 
		prev_vertex=None, 
		vertex=input_vertex)

	code_lines = []

	code_lines.append('import os')
	code_lines.append('os.system(\'pip install dlp\')')
	code_lines.append('import tensorflow as tf')
	code_lines.append('import numpy as np')
	code_lines.append('import json')
	code_lines.append('import dlp.blocks as blocks')
	code_lines.append('import dlp.utils as utils')
	code_lines.append('import dlp.restapi as restapi')
	code_lines.append('')

	settings = json.loads(jSettings)

	gen_model_part(serialisation=serialisation, current_code_lines=code_lines, inference=settings['inference'])
	gen_convert_part(datagen_node=datagen_node, code_lines=code_lines, weights_file_path=weights_file_path, output_path=output_path, settings=settings)

	# Write to file
	write_codegen(code_lines=code_lines, output_file_path=output_path+'/convert.py')
