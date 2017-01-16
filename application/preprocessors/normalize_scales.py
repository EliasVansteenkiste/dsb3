

def preprocess_normscale(patient_data, result, index, augment=True,
                         metadata=None,
                         normscale_resize_and_augment_function=normscale_resize_and_augment,
                         testaug=False):
    zoom_factor = None

    # Iterate over different sorts of data
    for tag, data in patient_data.iteritems():
        if tag in metadata:
            metadata_tag = metadata[tag]
        desired_shape = result[tag][index].shape

        cleaning_processes = getattr(config(), 'cleaning_processes', [])
        cleaning_processes_post = getattr(config(), 'cleaning_processes_post', [])

        if tag.startswith("sliced:data:singleslice"):
            # Cleaning data before extracting a patch
            data = clean_images(
                [patient_data[tag]], metadata=metadata_tag,
                cleaning_processes=cleaning_processes)

            # Augment and extract patch
            # Decide which roi to use.
            shift_center = (None, None)
            if getattr(config(), 'use_hough_roi', False):
                shift_center = metadata_tag["hough_roi"]

            patient_3d_tensor = normscale_resize_and_augment_function(
                data, output_shape=desired_shape[-2:],
                augment=augmentation_params,
                pixel_spacing=metadata_tag["PixelSpacing"],
                shift_center=shift_center[::-1])[0]

            if augmentation_params is not None:
                zoom_factor = augmentation_params["zoom_x"] * augmentation_params["zoom_y"]
            else:
                zoom_factor = 1.0

            # Clean data further
            patient_3d_tensor = clean_images(
                patient_3d_tensor, metadata=metadata_tag,
                cleaning_processes=cleaning_processes_post)

            if "area_per_pixel:sax" in result:
                raise NotImplementedError()

            if augmentation_params and not augmentation_params.get("change_brightness", 0) == 0:
                patient_3d_tensor = augment_brightness(patient_3d_tensor, augmentation_params["change_brightness"])

            put_in_the_middle(result[tag][index], patient_3d_tensor, True)
