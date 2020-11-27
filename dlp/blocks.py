import tensorflow as tf

def INPUT_LAYER(input_tensor, batch_size, dtype, input_shape):
	return tf.keras.layers.Input(batch_size=batch_size, shape=input_shape, dtype=dtype)

def CONV2D_LAYER(input_tensor, filters, kernel_size, strides, padding, use_bias, trainable, name):
	use_bias = True if use_bias == 1 else False
	trainable = True if trainable == 1 else False
	return tf.keras.layers.Conv2D(
		filters=filters, 
		kernel_size=kernel_size, 
		strides=strides,
		padding=padding, 
		use_bias=use_bias, 
		trainable=trainable, 
		kernel_regularizer=tf.keras.regularizers.l2(0.0))(input_tensor)

def MAXPOOL2D_LAYER(input_tensor, pool_size, strides, padding):
	return tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=strides, padding=padding)(input_tensor)

def UPSAMPLING2D_LAYER(input_tensor, size):
	return tf.keras.layers.UpSampling2D(size=size, interpolation='bilinear')(input_tensor)

def BATCH_NORM_LAYER(input_tensor, trainable):
	return tf.keras.layers.BatchNormalization(trainable=trainable, name='lateral_P2_bn')(input_tensor)

def ACTIVATION_LAYER(input_tensor, activation):
	return tf.keras.layers.Activation(activation)(input_tensor)

def SPLIT_LAYER(input_tensor, axis, size_splits):
	return tf.split(value=input_tensor, num_or_size_splits=size_splits, axis=axis)

def SPLITTED_LAYER(input_tensor, order):
	return input_tensor[order]

def ADD_LAYER(tensor1, tensor2):
	if tensor1 == None:
		return tensor2

	if tensor2 == None:
		return tensor1

	return tf.keras.layers.Add()([tensor1, tensor2])

def FLATTEN_LAYER(input_tensor):
	return tf.keras.layers.Flatten()(input_tensor)

def DENSE_LAYER(input_tensor, units, trainable, use_bias, name):
	use_bias = True if use_bias == 1 else False
	return tf.keras.layers.Dense(
		units=units, 
		use_bias=use_bias, 
		trainable=trainable, 
		name=name)(input_tensor)

def DROPOUT_LAYER(input_tensor, rate, trainable, name):
	trainable = True if trainable == 1 else False
	return tf.keras.layers.Dropout(rate=rate)(input_tensor, training=trainable)

def CONCAT_LAYER(tensor1, tensor2, axis):
	if tensor1 == None:
		return tensor2

	if tensor2 == None:
		return tensor1

	return tf.concat(values=[tensor1, tensor2], axis=axis)

def RESHAPE_LAYER(input_tensor, new_shape):
	return tf.reshape(tensor=input_tensor, shape=new_shape)

def CAST_LAYER(tensor, dtype):
	return tf.cast(x=tensor, dtype=dtype)

def LOSS_FUNC_HMR(input_tensor, name):
	def heatmap_loss(y_true, y_pred):
		'''
		Arguments
			y_true: (batch_size, h, w, total_heatpoints)
			y_pred: (batch_size, h, w, total_heatpoints)
		Return
			loss
		'''

		_, h, w, total_heatpoints = y_pred.shape

		y_true = tf.reshape(tensor=y_true, shape=[-1, h*w*total_heatpoints])
		y_pred = tf.reshape(tensor=y_pred, shape=[-1, h*w*total_heatpoints])

		diff = y_true - y_pred # (batch_size, h*w*total_heatpoints)
		diff = tf.math.abs(x=diff) # (batch_size, h*w*total_heatpoints)
		loss = tf.math.reduce_sum(input_tensor=diff, axis=-1) # (batch_size,)
		loss = tf.math.reduce_mean(input_tensor=loss, axis=-1) 

		return loss

	return heatmap_loss

def LOSS_FUNC_IC(input_tensor, name):
	return tf.keras.losses.categorical_crossentropy

def LOSS_FUNC_OD4(input_tensor, name, total_classes, lamda=1.0):
	'''
	'''

	def smooth_l1(y_true, y_pred):
		'''
		'''

		HUBER_DELTA = 1.0

		x = tf.math.abs(y_true - y_pred)
		x = tf.keras.backend.switch(x < HUBER_DELTA, 0.5*x**2, HUBER_DELTA*(x - 0.5*HUBER_DELTA))
		return  x

	def balanced_l1(y_true, y_pred):
		'''
		https://arxiv.org/pdf/1904.02701.pdf
		'''

		alpha = 0.5
		gamma = 1.5
		b = 19.085
		C = 0

		x = tf.math.abs(y_true - y_pred)
		x = tf.keras.backend.switch(x < 1.0, (alpha*x + alpha/b)*tf.math.log(b*x + 1) - alpha*x, gamma*x + C)
		return  x

	def ssd_loss(y_true, y_pred):
		'''
		https://arxiv.org/pdf/1512.02325.pdf
		Arguments
			y_true: (h*w*k, total_classes+1+4)
			y_pred: (h*w*k, total_classes+1+4)
		Return
			loss
		'''

		true_clz_2dtensor = y_true[:, :total_classes+1] # (h*w*k, total_classes+1)
		pred_clz_2dtensor = y_pred[:, :total_classes+1] # (h*w*k, total_classes+1)
		true_loc_2dtensor = y_true[:, total_classes+1:] # (h*w*k, 4)
		pred_loc_2dtensor = y_pred[:, total_classes+1:] # (h*w*k, 4)

		sum_true_clz_2dtensor = tf.math.reduce_sum(input_tensor=true_clz_2dtensor, axis=-1) # (h*w*k,)
		selected_clz_indices = tf.where(
			condition=tf.math.equal(x=sum_true_clz_2dtensor, y=1)) # foreground, background
		selected_loc_indices = tf.where(
			condition=tf.math.logical_and(
				x=tf.math.equal(x=sum_true_clz_2dtensor, y=1),
				y=tf.math.not_equal(x=true_clz_2dtensor[:, -1], y=1))) # foreground

		true_clz_2dtensor = tf.gather_nd(params=true_clz_2dtensor, indices=selected_clz_indices) # (fb, total_classes+1)
		pred_clz_2dtensor = tf.gather_nd(params=pred_clz_2dtensor, indices=selected_clz_indices) # (fb, total_classes+1)
		true_loc_2dtensor = tf.gather_nd(params=true_loc_2dtensor, indices=selected_loc_indices) # (f, 4)
		pred_loc_2dtensor = tf.gather_nd(params=pred_loc_2dtensor, indices=selected_loc_indices) # (f, 4)

		clz_loss = tf.keras.backend.categorical_crossentropy(true_clz_2dtensor, pred_clz_2dtensor) # (fb,)
		loc_loss = tf.math.reduce_sum(input_tensor=smooth_l1(true_loc_2dtensor, pred_loc_2dtensor), axis=-1) # (f,)
		loss = tf.math.reduce_mean(clz_loss) + lamda*tf.math.reduce_mean(loc_loss)

		return loss

	return ssd_loss

def CONV2D_BLOCK(input_tensor, filters, kernel_size, strides, padding, use_bias, trainable, bn_trainable, activation, block_name):
	use_bias = True if use_bias == 1 else False
	trainable = True if trainable == 1 else False
	bn_trainable = True if bn_trainable == 1 else False

	tensor = tf.keras.layers.Conv2D(
		filters=filters, 
		kernel_size=kernel_size, 
		strides=strides,
		padding=padding, 
		use_bias=use_bias,
		trainable=trainable, 
		name=block_name+'_conv',
		kernel_regularizer=tf.keras.regularizers.l2(0.0))(input_tensor)
	tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_bn')(tensor)
	tensor = tf.keras.layers.Activation(activation)(tensor)
	return tensor

def HOURGLASS_BLOCK(input_tensor, block_name, depth, use_bias, trainable, bn_trainable, repeat):
	use_bias = True if use_bias == 1 else False
	trainable = True if trainable == 1 else False
	bn_trainable = True if bn_trainable == 1 else False
	
	tensor = input_tensor
	filters = tensor.shape[-1];

	for blk in range(repeat):
		tensor = tf.keras.layers.Conv2D(
			filters=filters, 
			kernel_size=[3, 3], 
			strides=[1, 1], 
			padding='same',
			use_bias=use_bias, 
			kernel_regularizer=tf.keras.regularizers.l2(0.0), 
			trainable=trainable, 
			name=block_name+str(blk)+'_first_hourglass_conv')(tensor)
		tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+str(blk)+'_first_hourglass_bn')(tensor)
		tensor = tf.keras.layers.Activation('relu')(tensor)
		first_conv_tensor = tensor

		tensors = []
		for i in range(depth):
			tensor = tf.keras.layers.Conv2D(
				filters=(2**i)*filters, 
				kernel_size=[3, 3], 
				strides=[2, 2], 
				padding='same',
				use_bias=use_bias, 
				kernel_regularizer=tf.keras.regularizers.l2(0.0), 
				trainable=trainable, 
				name=block_name+str(blk)+'_encoder_stage_'+str(i)+'_conv')(tensor)
			tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+str(blk)+'_encoder_stage_'+str(i)+'_bn')(tensor)
			tensor = tf.keras.layers.Activation('relu')(tensor)
			tensors.append(tensor)

		for i in range(depth-1, 0, -1):
			tensor = tf.keras.layers.Conv2D(
				filters=(2**i)*filters, 
				kernel_size=[3, 3], 
				strides=[1, 1], 
				padding='same',
				use_bias=use_bias, 
				kernel_regularizer=tf.keras.regularizers.l2(0.0), 
				trainable=trainable, 
				name=block_name+str(blk)+'_decoder_stage_'+str(i)+'_conv1')(tensor)
			tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+str(blk)+'_decoder_stage_'+str(i)+'_bn1')(tensor)
			tensor = tf.keras.layers.Activation('relu')(tensor)

			tensor = tf.keras.layers.Conv2D(
				filters=(2**(i-1))*filters, 
				kernel_size=[3, 3], 
				strides=[1, 1], 
				padding='same',
				use_bias=use_bias, 
				kernel_regularizer=tf.keras.regularizers.l2(0.0), 
				trainable=trainable, 
				name=block_name+str(blk)+'_decoder_stage_'+str(i)+'_conv2')(tensor)
			tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+str(blk)+'_decoder_stage_'+str(i)+'_bn2')(tensor)
			tensor = tf.keras.layers.Activation('relu')(tensor)

			tensor = tf.keras.layers.UpSampling2D(size=(2, 2))(tensor)
			tensor = tf.keras.layers.Add()([tensor, tensors[i-1]])

		tensor = tf.keras.layers.Conv2D(
			filters=filters, 
			kernel_size=[3, 3], 
			strides=[1, 1], 
			padding='same',
			use_bias=use_bias, 
			kernel_regularizer=tf.keras.regularizers.l2(0.0), 
			trainable=trainable, 
			name=block_name+str(blk)+'_decoder_stage_0_conv1')(tensor)
		tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+str(blk)+'_decoder_stage_0_bn1')(tensor)
		tensor = tf.keras.layers.Activation('relu')(tensor)

		tensor = tf.keras.layers.Conv2D(
			filters=filters, 
			kernel_size=[3, 3], 
			strides=[1, 1], 
			padding='same',
			use_bias=use_bias, 
			kernel_regularizer=tf.keras.regularizers.l2(0.0), 
			trainable=trainable, 
			name=block_name+str(blk)+'_decoder_stage_0_conv2')(tensor)
		tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+str(blk)+'_decoder_stage_0_bn2')(tensor)
		tensor = tf.keras.layers.Activation('relu')(tensor)

		tensor = tf.keras.layers.UpSampling2D(size=(2, 2))(tensor)
		tensor = tf.keras.layers.Add()([tensor, first_conv_tensor])

	return tensor

def NMS(abox_2dtensor, prediction, nsm_iou_threshold, nsm_score_threshold, nsm_max_output_size, total_classes):
	'''
	'''

	loc_2dtensor = prediction[:, total_classes+1:] # (h*w*k, 4)
	pbox_2dtensor = loc2box2d(box_2dtensor=abox_2dtensor, bbe_2dtensor=loc_2dtensor) # (h*w*k, 4)

	clz_2dtensor = prediction[:, :total_classes+1] # (h*w*k, total_classes+1)
	clz_1dtensor = tf.math.argmax(input=clz_2dtensor, axis=-1) # (h*w*k,)

	cancel = tf.where(
		condition=tf.math.less(x=clz_1dtensor, y=total_classes),
		x=1.0,
		y=0.0) # (h*w*k,)

	score_1dtensor = tf.math.reduce_max(input_tensor=clz_2dtensor, axis=-1) # (h*w*k,)
	score_1dtensor *= cancel # (h*w*k,)

	selected_indices, valid_outputs = tf.image.non_max_suppression_padded(
		boxes=pbox_2dtensor,
		scores=score_1dtensor,
		max_output_size=nsm_max_output_size,
		iou_threshold=nsm_iou_threshold,
		score_threshold=nsm_score_threshold,
		pad_to_max_output_size=True)

	box_2dtensor = tf.gather(params=pbox_2dtensor, indices=selected_indices) # (nsm_max_output_size, 4)
	clz_1dtensor = tf.gather(params=clz_1dtensor, indices=selected_indices) # (nsm_max_output_size,)
	clz_2dtensor = tf.expand_dims(input=clz_1dtensor, axis=1) # (nsm_max_output_size, 1)
	box_2dtensor = tf.cast(x=box_2dtensor, dtype='int32')
	clz_2dtensor = tf.cast(x=clz_2dtensor, dtype='int32')

	boxclz_2dtensor = tf.concat(values=[box_2dtensor, clz_2dtensor], axis=-1)

	return boxclz_2dtensor, valid_outputs

def RESNET_IDENTITY_BLOCK(input_tensor, kernel_size, filters, block_name, use_bias, trainable, bn_trainable, repeat):
	use_bias = True if use_bias == 1 else False
	trainable = True if trainable == 1 else False
	bn_trainable = True if bn_trainable == 1 else False

	def identity_block(input_tensor, kernel_size, filters, block_name, use_bias, trainable, bn_trainable):
		'''
		https://arxiv.org/pdf/1512.03385.pdf
		Bottleneck architecture
		Arguments
			input_tensor:
			kernel_size:
			filters:
			trainable:
		Return
			tensor:
		'''

		weight_decay = 0.0
		filters1, filters2, filters3 = filters

		tensor = tf.keras.layers.Conv2D(
			filters=filters1, 
			kernel_size=[1, 1], 
			use_bias=use_bias, 
			kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=block_name+'_conv1')(input_tensor)
		tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv1_bn')(tensor)
		tensor = tf.keras.layers.Activation('relu')(tensor)

		tensor = tf.keras.layers.Conv2D(
			filters=filters2, 
			kernel_size=kernel_size, 
			padding='same', 
			use_bias=use_bias, 
			kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=block_name+'_conv2')(tensor)
		tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv2_bn')(tensor)
		tensor = tf.keras.layers.Activation('relu')(tensor)

		tensor = tf.keras.layers.Conv2D(
			filters=filters3, 
			kernel_size=[1, 1], 
			use_bias=use_bias, 
			kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
			trainable=trainable, 
			name=block_name+'_conv3')(tensor)
		tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv3_bn')(tensor)
		tensor = tf.keras.layers.Add()([tensor, input_tensor])
		tensor = tf.keras.layers.Activation('relu')(tensor)

		return tensor

	tensor = input_tensor
	for n in range(repeat):
		tensor = identity_block(
			input_tensor=tensor, 
			kernel_size=kernel_size, 
			filters=filters, 
			block_name=block_name+'_'+str(n)+'_', 
			use_bias=use_bias, 
			trainable=trainable,
			bn_trainable=bn_trainable)

	return tensor

def RESNET_SIDENTITY_BLOCK(input_tensor, kernel_size, filters, strides, block_name, use_bias, trainable, bn_trainable):
	'''
	https://arxiv.org/pdf/1512.03385.pdf
	Bottleneck architecture
	Arguments
		input_tensor:
		kernel_size:
		filters:
		strides:
		trainable:
	Return
		tensor:
	'''

	use_bias = True if use_bias == 1 else False
	trainable = True if trainable == 1 else False
	bn_trainable = True if bn_trainable == 1 else False
	weight_decay = 0.0

	filters1, filters2, filters3 = filters

	tensor = tf.keras.layers.Conv2D(
		filters=filters1, 
		kernel_size=[1, 1], 
		strides=strides, 
		use_bias=use_bias, 
		kernel_regularizer=tf.keras.regularizers.l2(0.0), 
		trainable=trainable, 
		name=block_name+'_conv1')(input_tensor)
	tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv1_bn')(tensor)
	tensor = tf.keras.layers.Activation('relu')(tensor)

	tensor = tf.keras.layers.Conv2D(
		filters=filters2, 
		kernel_size=kernel_size, 
		padding='same', 
		use_bias=use_bias, 
		kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv2')(tensor)
	tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv2_bn')(tensor)
	tensor = tf.keras.layers.Activation('relu')(tensor)

	tensor = tf.keras.layers.Conv2D(
		filters=filters3, 
		kernel_size=[1, 1], 
		use_bias=use_bias, 
		kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv3')(tensor)
	tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv3_bn')(tensor)

	input_tensor = tf.keras.layers.Conv2D(
		filters=filters3, 
		kernel_size=[1, 1], 
		strides=strides, 
		use_bias=use_bias, 
		kernel_regularizer=tf.keras.regularizers.l2(weight_decay), 
		trainable=trainable, 
		name=block_name+'_conv4')(input_tensor)
	input_tensor = tf.keras.layers.BatchNormalization(trainable=bn_trainable, name=block_name+'_conv4_bn')(input_tensor, training=trainable)

	tensor = tf.keras.layers.Add()([tensor, input_tensor])
	tensor = tf.keras.layers.Activation('relu')(tensor)

	return tensor

def RFE_BLOCK(input_tensor, block_name, use_bias, trainable, bn_trainable):
	return input_tensor