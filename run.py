from keras_segmentation.models.all_models import model_from_name
from imgaug import augmenters as iaa

################# Configure Here ################
path_base = "/home/mirap/database_folder/Menziesdata/ROI_examples_5_fold_whitehole/foldset2/"
train_images_path = path_base+ "train_images"
train_annotations_path = path_base + "train_annotation"
val_images_path = path_base + "val_images"
val_annotations_path = path_base + "val_annotation"
checkpoints_saving_path = "checkpoints/"
Data_Validation = False
dataset_abbr = "MBf2"

model_list = [
#            "fcn_16_vgg",
#            "fcn_8_vgg",
            "fcn_32_vgg",
#            "fcn_8_resnet50",  # big size over 11GB
#            "fcn_16_resnet50",
#            "fcn_32_resnet50", # big size over 11GB
#            "fcn_8_mobilenet",
#            "fcn_16_mobilenet",
#            "fcn_32_mobilenet",

#            "pspnet", # core dump error
#            "vgg_pspnet", # core dump error
#            "resnet50_pspnet", # core dump error
#            "pspnet_50", # big size over 11GB
#            "pspnet_101",

#            "unet_mini",
#            "unet",
#            "vgg_unet",
#            "resnet50_unet",
#            "mobilenet_unet",

#            "segnet",
#            "vgg_segnet",
#            "resnet50_segnet",
#            "mobilenet_segnet"
            ]

DO_Augment = True
def custom_augmentation():
    return  iaa.Sequential(
        [
            # apply the following augmenters to most images
            # https://imgaug.readthedocs.io/en/latest/source/overview/arithmetic.html
            iaa.AddToBrightness((-40, 40)),  
            #iaa.CropAndPad(percent=(-0.25, 0.25)),
            #iaa.ContrastNormalization(0.5),
            #iaa.AllChannelsHistogramEqualization(),
            #iaa.Affine(rotate=(-40, 40))
        ])

CLASSES = 7 #MAX33 -> want to increase? go to data_loader.py
EPOCH = 100
batch_size=1
steps_per_epoch=512

save_result = True

test_image_path = "//test_images/0408.vsi.jpg"
################################################

for model_name in model_list:
    # model define
    print("------------ Define Model:"+model_name+" ------------")
    model = model_from_name[model_name](n_classes = CLASSES
                                      #,input_height=648 
                                      #,input_width=432
                                      )

    # start training
    print("------------ start training ------------")
    checkpoints_model_saving_path= checkpoints_saving_path+dataset_abbr+model_name
    result = model.train( 
        verify_dataset=Data_Validation,
        train_images =  train_images_path,
        train_annotations = train_annotations_path,
        # input_height=224,
        # input_width=224,
        checkpoints_path = checkpoints_model_saving_path, 
        epochs=EPOCH,
        batch_size = batch_size,
        validate=True,
        val_images=val_images_path,
        val_annotations=val_annotations_path,
        val_batch_size=batch_size,
        load_weights=None,
        steps_per_epoch=steps_per_epoch,
        val_steps_per_epoch=steps_per_epoch,
        ignore_zero_class=False,
        optimizer_name='adam',
        do_augment=DO_Augment,
        augmentation_name="aug_all",
        custom_augmentation=custom_augmentation,
        preprocessing=None
    )

    if save_result:
        # save the result
        print("------------ save the result ------------")
        import pandas as pd
        hist_df = pd.DataFrame(result.history)
        history_csv_file = "checkpoints/"+model_name+"_"+dataset_abbr+".csv"
        with open(history_csv_file, mode='w') as f:
            hist_df.to_csv(f)
        #save_weights_path = "checkpoints/"+model_name+"_"+dataset_abbr+".h5"
        #model.save_weights(save_weights_path, overwrite=True)

        # Predict a image
        print("------------ Predict a image ------------")
        out = model.predict_segmentation(
            inp = test_image_path,
            out_fname = "out_frame/"+dataset_abbr+"/"+dataset_abbr+"_"+model_name+".png",
            overlay_img=True
        )

print("end of run.py")

# evaluating the model 
# print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
