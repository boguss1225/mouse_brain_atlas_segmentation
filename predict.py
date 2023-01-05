from keras_segmentation.predict import predict,predict_multiple,predict_video
from keras_segmentation.models.all_models import model_from_name

test_image_path = "/home/mirap/database_folder/Menziesdata/ROI_examples_5_fold_whitehole/test_images/"
checkpoints_saving_path = "checkpoints/"
dataset_abbr = "MBf"
out_folder = "out_frame/"+dataset_abbr

model_list = [
#            "fcn_16_vgg",
#            "fcn_32_vgg",
#            "fcn_8_vgg",
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
            "vgg_unet",
#            "resnet50_unet",
#            "mobilenet_unet",

#            "segnet",
#            "vgg_segnet",
#            "resnet50_segnet",
#            "mobilenet_segnet"
            ]

# load model
# model_weight_path = 'model.h5'
# model = vgg_unet(n_classes=6,  input_height=640, input_width=640)
# model.load_weights(model_weight_path, by_name=True)

for i in range(1,6):
    for model_name in model_list:
        print("--------- predict",dataset_abbr,i,"_",model_name,"------------")
        # Single Predict
        # predict( 
        #     checkpoints_path="checkpoints/mobilenet_segnet", 
        #     inp="database/IMAS_Salmon/train_images/untitled-10.jpg", 
        #     out_fname="out_frame/output_SaMobilenet_segnet_Predic.png",
        #     overlay_img=True
        # )
        print("--- using",checkpoints_saving_path+dataset_abbr+str(i)+model_name,"---")
        # Multi Predict
        predict_multiple( 
           checkpoints_path=checkpoints_saving_path+dataset_abbr+str(i)+model_name, 
           inp_dir=test_image_path, 
           out_dir=out_folder+str(i)+"/",
           overlay_img=True,
           class_names=None, show_legends=False,
           prediction_width=None, prediction_height=None,
        )

        # Video Predict
        # predict_video(
        #     checkpoints_path="checkpoints/vgg_unet_1", 
        #     inp=test_image_path, # should be avi file! 
        #     out_fname="output.avi"
        # )

