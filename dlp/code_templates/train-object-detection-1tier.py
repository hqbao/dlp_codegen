def train(dataset_name, image_shape, scale_sizes, anchor_sizes, iou_thresholds, anchor_sampling, scale_range, epochs):
	dataset_info = utils.get_dataset_info(dataset_name)
	output_path = './outputs'
	train_anno_file_path = dataset_info['train_anno_file_path']
	train_image_dir_path = dataset_info['train_image_dir_path']
	test_anno_file_path = dataset_info['test_anno_file_path']
	test_image_dir_path = dataset_info['test_image_dir_path']
	total_train_examples = dataset_info['total_train_examples']
	total_test_examples = dataset_info['total_test_examples']
	total_classes = dataset_info['total_classes']
	ishape = image_shape
	ssizes = scale_sizes
	asizes = anchor_sizes
	total_epoches = epochs

	abox_2dtensor = tf.constant(value=utils.genanchors(isize=ishape[:2], ssize=ssizes, asizes=asizes), dtype='float32') # (h*w*k, 4)

	model = build_model()
	model.summary()

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	weights_file_name = 'weights_'+dataset_name+'.h5'
	weights_file_path = output_path+'/'+weights_file_name
	restapi.download_weights(encoded_token=encoded_token, weights_file_path=weights_file_path)
	if os.path.isfile(weights_file_path):
		model.load_weights(weights_file_path, by_name=True)

	train_dataset = utils.load_object_detection_dataset(anno_file_path=train_anno_file_path, total_classes=total_classes)
	test_dataset = utils.load_object_detection_dataset(anno_file_path=test_anno_file_path, total_classes=total_classes)

	train_loss = np.zeros((total_epoches, total_train_examples))
	test_loss = np.zeros((total_epoches, total_test_examples))
	true_positive = np.zeros((total_epoches, total_test_examples))
	false_positive = np.zeros((total_epoches, total_test_examples))
	false_negative = np.zeros((total_epoches, total_test_examples))

	for epoch in range(total_epoches):
		np.random.shuffle(train_dataset)
		gen = utils.genxy_od(
			dataset=train_dataset, 
			image_dir=train_image_dir_path, 
			ishape=ishape, 
			abox_2dtensor=abox_2dtensor, 
			iou_thresholds=iou_thresholds, 
			total_examples=total_train_examples,
			total_classes=total_classes, 
			anchor_sampling=anchor_sampling,
			scale_range=scale_range)

		print('\nEpoch '+str(epoch)+'\nTrain')
		for batch in range(total_train_examples):
			batchx_4dtensor, batchy_3dtensor, _ = next(gen)
			batch_loss = model.train_on_batch(batchx_4dtensor, batchy_3dtensor)
			train_loss[epoch, batch] = batch_loss

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_train_examples), end='\n')

		print('\nLoss: {:.3f}'.format(float(np.mean(train_loss[epoch], axis=-1))))

		model.save_weights(weights_file_path)

		gen = utils.genxy_od(
			dataset=test_dataset, 
			image_dir=test_image_dir_path, 
			ishape=ishape, 
			abox_2dtensor=abox_2dtensor, 
			iou_thresholds=iou_thresholds, 
			total_examples=total_test_examples,
			total_classes=total_classes, 
			anchor_sampling=anchor_sampling,
			scale_range=scale_range)

		print('\nTest')
		for batch in range(total_test_examples):
			batchx_4dtensor, batchy_3dtensor, bboxes = next(gen)
			batch_loss = model.train_on_batch(batchx_4dtensor, batchy_3dtensor)
			test_loss[epoch, batch] = batch_loss
			prediction = model.predict_on_batch(batchx_4dtensor)
			boxclz_2dtensor, valid_outputs = utils.nms(
				abox_2dtensor=abox_2dtensor, 
				prediction=prediction, 
				nsm_iou_threshold=0.2,
				nsm_score_threshold=0.8,
				nsm_max_output_size=100,
				total_classes=total_classes)
			boxclz_2dtensor = boxclz_2dtensor[:valid_outputs]
			pboxes = list(boxclz_2dtensor.numpy())
			tp, fp, fn = utils.match_od(boxes=bboxes, pboxes=pboxes, iou_threshold=0.5)
			true_positive[epoch, batch] = tp
			false_positive[epoch, batch] = fp
			false_negative[epoch, batch] = fn

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_test_examples), end='\n')

		print('\nLoss: {:.3f}'.format(float(np.mean(test_loss[epoch], axis=-1))))

		updated = restapi.update_train_result(
			encoded_token=encoded_token,
			weights_file_path=weights_file_path, 
			weights_file_name=weights_file_name, 
			epoch_train_loss=train_loss[epoch].tolist(),
			epoch_test_loss=test_loss[epoch].tolist(),
			epoch_tp=true_positive[epoch].tolist(),
			epoch_fp=false_positive[epoch].tolist(),
			epoch_fn=false_negative[epoch].tolist())

		if updated is not True:
			print('Can not update train result')
