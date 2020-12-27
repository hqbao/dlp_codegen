def convert(dataset_name, weights_file_path, output_path, image_shape, scale_sizes, anchor_sizes):
	dataset_info = utils.get_dataset_info(dataset_name)
	total_classes = dataset_info['total_classes']
	model = build_model(total_classes=total_classes, image_shape=image_shape, scale_sizes=scale_sizes, anchor_sizes=anchor_sizes)
	model.summary()
	model.load_weights(weights_file_path, by_name=True)
	model.save(output_path+'/model')
