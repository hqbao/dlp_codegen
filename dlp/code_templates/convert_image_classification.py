def convert(dataset_name, weights_file_path, output_path):
	model = build_model()
	model.summary()
	model.load_weights(weights_file_path)
	model.save(output_path+'/model')
