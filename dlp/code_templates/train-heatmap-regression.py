def train(dataset_name, image_shape, total_train_examples, total_test_examples, batch_size, epochs):
	dataset_info = utils.get_dataset_info(dataset_name)
	output_path = './outputs'
	train_anno_file_path = dataset_info['train_anno_file_path']
	train_image_dir_path = dataset_info['train_image_dir_path']
	# test_anno_file_path = dataset_info['test_anno_file_path']
	# test_image_dir_path = dataset_info['test_image_dir_path']
	ishape = image_shape
	total_heatpoints = dataset_info['total_heatpoints']
	total_epoches = epochs
	total_train_batches = total_train_examples//batch_size
	total_test_batches = total_test_examples//batch_size

	model = build_model()
	model.summary()

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	weights_file_name = 'weights_'+dataset_name+'.h5'
	weights_file_path = output_path+'/'+weights_file_name
	restapi.download_weights(encoded_token=encoded_token, weights_file_path=weights_file_path)
	if os.path.isfile(weights_file_path):
		model.load_weights(weights_file_path, by_name=True)

	train_dataset = utils.load_heatmap_regression_dataset(anno_file_path=train_anno_file_path, total_heatpoints=total_heatpoints)
	# test_dataset = utils.load_heatmap_regression_dataset(anno_file_path=test_anno_file_path, total_heatpoints=total_heatpoints)

	train_loss = np.zeros((total_epoches, total_train_batches))
	# test_loss = np.zeros((total_epoches, total_test_batches))

	for epoch in range(total_epoches):
		np.random.shuffle(train_dataset)
		gen = utils.genxy_hmr(
			dataset=train_dataset, 
			image_dir_path=train_image_dir_path, 
			ishape=ishape, 
			total_examples=total_train_examples, 
			batch_size=batch_size)

		print('\nEpoch '+str(epoch)+'\nTrain')
		loss = np.zeros(total_train_batches)

		for batch in range(total_train_batches):
			batchx4d, batchy4d = next(gen)
			batch_loss = model.train_on_batch(batchx4d, batchy4d)
			train_loss[epoch, batch] = batch_loss

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_train_batches), end='\n')

		print('\nLoss: {:.3f}'.format(float(np.mean(train_loss[epoch], axis=-1))))

		model.save_weights(weights_file_path)

		# gen = utils.genxy_hmr(
		# 	dataset=test_dataset, 
		# 	image_dir_path=test_image_dir_path, 
		# 	ishape=ishape, 
		# 	total_examples=total_test_examples, 
		# 	batch_size=batch_size)

		# print('\nEpoch \nTrain {}'.format(epoch))
		# loss = np.zeros(total_test_batches)

		# for batch in range(total_test_batches):
		# 	batchx4d, batchy4d = next(gen)
		# 	batch_loss = model.test_on_batch(batchx4d, batchy4d)
		# 	test_loss[epoch, batch] = batch_loss

		# 	print('-', end='')
		# 	if batch%100==99:
		# 		print('{:.2f}%'.format((batch+1)*100/total_test_batches), end='\n')

		# print('\nLoss: {:.3f}'.format(float(np.mean(test_loss[epoch], axis=-1))))

		updated = restapi.update_train_result(
			encoded_token=encoded_token,
			weights_file_path=weights_file_path, 
			weights_file_name=weights_file_name, 
			epoch_train_loss=train_loss[epoch].tolist(),
			epoch_test_loss=[0 for _ in range(total_train_batches)],
			epoch_tp=[0 for _ in range(total_train_batches)],
			epoch_fp=[0 for _ in range(total_train_batches)],
			epoch_fn=[0 for _ in range(total_train_batches)])
		
		if updated is not True:
			print('Can not update train result')
