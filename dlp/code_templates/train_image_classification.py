def train(dataset_name, image_shape, epochs, total_train_examples, total_test_examples, batch_size):
	dataset_info = utils.get_dataset_info(dataset_name)
	output_path = './outputs'
	train_anno_file_path = dataset_info['train_anno_file_path']
	train_image_dir_path = dataset_info['train_image_dir_path']
	test_anno_file_path = dataset_info['test_anno_file_path']
	test_image_dir_path = dataset_info['test_image_dir_path']
	ishape = image_shape
	total_classes = dataset_info['total_classes']
	total_epoches = epochs
	total_train_batches = total_train_examples//batch_size
	total_test_batches = total_test_examples//batch_size

	model = build_model()
	model.summary()

	if not os.path.exists(output_path):
		os.makedirs(output_path)
		
	weights_file_name = 'weights_'+dataset_name+'.h5'
	weight_file_path = output_path+'/'+weights_file_name
	if os.path.isdir(weight_file_path):
		model.load_weights(weight_file_path, by_name=True)

	train_dataset = utils.load_image_classification_dataset(anno_file_path=train_anno_file_path)
	test_dataset = utils.load_image_classification_dataset(anno_file_path=test_anno_file_path)

	train_loss = np.zeros((total_epoches, total_train_batches))
	test_loss = np.zeros((total_epoches, total_test_batches))
	true_positive = np.zeros((total_epoches, total_test_batches))
	false_positive = np.zeros((total_epoches, total_test_batches))
	false_negative = np.zeros((total_epoches, total_test_batches))

	for epoch in range(total_epoches):
		np.random.shuffle(train_dataset)
		gen = utils.genxy_ic(
			dataset=train_dataset, 
			image_dir=train_image_dir_path, 
			ishape=ishape, 
			total_classes=total_classes,
			total_examples=total_train_examples,
			batch_size=batch_size)

		print('\nEpoch '+str(epoch)+'\nTrain')
		for batch in range(total_train_batches):
			batchx_3dtensor, batchy_2dtensor = next(gen)
			batch_loss = model.train_on_batch(batchx_3dtensor, batchy_2dtensor)
			train_loss[epoch, batch] = batch_loss

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_train_batches), end='\n')

		print('\nLoss: {:.3f}'.format(float(np.mean(train_loss[epoch], axis=-1))))

		model.save_weights(weight_file_path)

		gen = utils.genxy_ic(
			dataset=test_dataset, 
			image_dir=test_image_dir_path, 
			ishape=ishape, 
			total_classes=total_classes,
			total_examples=total_test_examples,
			batch_size=batch_size)

		print('\nTest')
		for batch in range(total_test_batches):
			batchx_3dtensor, batchy_2dtensor = next(gen)
			batch_loss = model.test_on_batch(batchx_3dtensor, batchy_2dtensor)
			test_loss[epoch, batch] = batch_loss
			prediction = model.predict_on_batch(batchx_3dtensor)
			tp, fp, fn = utils.match_ic(prediction=prediction, batchy_2dtensor=batchy_2dtensor)
			true_positive[epoch, batch] = tp
			false_positive[epoch, batch] = fp
			false_negative[epoch, batch] = fn

			print('-', end='')
			if batch%100==99:
				print('{:.2f}%'.format((batch+1)*100/total_test_batches), end='\n')

		print('\nLoss: {:.3f}'.format(float(np.mean(test_loss[epoch], axis=-1))))

		restapi.update_train_result(
			encoded_token=encoded_token,
			weight_file_path=weight_file_path, 
			weights_file_name=weights_file_name, 
			train_result=json.dumps({
				"trainLoss": train_loss[:epoch+1].tolist(),
				"testLoss": test_loss[:epoch+1].tolist(),
				"truePositive": true_positive[:epoch+1].tolist(),
				"falsePositive": false_positive[:epoch+1].tolist(),
				"falseNegative": false_negative[:epoch+1].tolist(),
			}))
