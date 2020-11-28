def train(dataset_name, image_shape, scale_sizes, anchor_sizes, iou_thresholds, anchor_sampling, epochs):
	dataset_info = utils.get_dataset_info(dataset_name)
	output_path = './outputs'
	train_anno_file_path = dataset_info['train_anno_file_path']
	train_image_dir_path = dataset_info['train_image_dir_path']
	ishape = image_shape
	ssize = scale_sizes
	asizes = anchor_sizes
	total_classes = dataset_info['total_classes']
	total_epoches = epochs
	total_train_examples = dataset_info['total_train_examples']

	abox_2dtensor = tf.constant(value=utils.genanchors(isize=ishape[:2], ssize=ssize, asizes=asizes), dtype='float32') # (h*w*k, 4)

	model = build_model()
	model.summary()

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	weight_file_path = output_path+'/weights_'+dataset_name+'.h5'
	if os.path.isdir(weight_file_path):
		model.load_weights(weight_file_path, by_name=True)

	train_dataset = utils.load_object_detection_dataset(anno_file_path=train_anno_file_path, total_classes=total_classes)

	for epoch in range(total_epoches):
		gen = utils.genxy_od(
			dataset=train_dataset, 
			image_dir=train_image_dir_path, 
			ishape=ishape, 
			abox_2dtensor=abox_2dtensor, 
			iou_thresholds=iou_thresholds, 
			total_examples=total_train_examples,
			total_classes=total_classes, 
			anchor_sampling=anchor_sampling)

		print('\nTrain epoch {}'.format(epoch))
		loss = np.zeros(total_train_examples)

		for batch in range(total_train_examples):
			batchx_4dtensor, batchy_2dtensor, _ = next(gen)
			batch_loss = model.train_on_batch(batchx_4dtensor, batchy_2dtensor)
			loss[batch] = batch_loss

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_train_examples), end='\n')

		mean_loss = float(np.mean(loss, axis=-1))
		print('\nLoss: {:.3f}'.format(mean_loss))

		model.save_weights(weight_file_path)
