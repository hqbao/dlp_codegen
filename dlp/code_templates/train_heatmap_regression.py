def train(dataset_name, image_shape, epochs, total_train_examples, batch_size):
	dataset_info = utils.get_dataset_info(dataset_name)
	output_path = '/outputs'
	train_anno_file_path = dataset_info['train_anno_file_path']
	train_image_dir_path = dataset_info['train_image_dir_path']
	ishape = image_shape
	total_heatpoints = dataset_info['total_heatpoints']
	total_epoches = epochs
	total_train_batches = total_train_examples//batch_size

	model = build_model()
	model.summary()

	weight_file_path = output_path+'/weights_'+dataset_name+'.h5'
	if path.isdir(weight_file_path):
		model.load_weights(weight_file_path, by_name=True)

	train_dataset = utils.load_heatmap_regression_dataset(anno_file_path=train_anno_file_path, total_heatpoints=total_heatpoints)

	for epoch in range(total_epoches):
		np.random.shuffle(train_dataset)
		gen = utils.genxy_hmr(
			dataset=train_dataset, 
			image_dir_path=train_image_dir_path, 
			ishape=ishape, 
			total_examples=total_train_examples, 
			batch_size=batch_size)

		print('\nTrain epoch {}'.format(epoch))
		loss = np.zeros(total_train_batches)

		for batch in range(total_train_batches):
			batchx4d, batchy4d = next(gen)
			batch_loss = model.train_on_batch(batchx4d, batchy4d)
			loss[batch] = batch_loss

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_train_batches), end='\n')

		mean_loss = float(np.mean(loss, axis=-1))
		print('\nLoss: {:.3f}'.format(mean_loss))

		model.save_weights(output_path+'/weights_'+dataset_name+'.h5')
