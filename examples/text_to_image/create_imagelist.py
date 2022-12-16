from pathlib import Path
import pickle

def create_image_list():
    img_dir='/home/arif/Documents/design/sandpit/vgg2faces/VGG-Face2/data/sdEyesTraining'
    save_dir='/home/arif/Documents/design/sandpit/vgg2faces/VGG-Face2/data'
    images_file_list = 'sd_images_list.pickle'
    image_list=[]
    images = list(Path(img_dir).iterdir())
    for image in images:
        name_sections=image.name.split('_')
        label_name = 'eyesm_' + name_sections[1] + '_label_' + name_sections[3]
        imgobj={'train':image.name,'label':label_name}
        image_list.append(imgobj)

    with open(save_dir + '/'+ images_file_list, 'wb') as handle:
        pickle.dump(image_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
create_image_list()