import os
import os.path as osp
import shutil
from tqdm import tqdm


dataset_path = '/media/alex/80CA308ECA308288/alex_dataset/ecological-assets'
split_dataset_path = osp.join('/media/alex/80CA308ECA308288/alex_dataset/ecological-assets', 'split_dataset')


save_imgs = osp.join(dataset_path, 'images')
save_masks = osp.join(dataset_path, 'masks')

ann_dir_train= osp.join(split_dataset_path, 'ann_dir', 'train')
ann_dir_val=osp.join(split_dataset_path, 'ann_dir', 'val')
ann_dir_train_val=osp.join(split_dataset_path, 'ann_dir', 'train_val')
img_dir_train=osp.join(split_dataset_path, 'img_dir', 'train')
img_dir_val=osp.join(split_dataset_path, 'img_dir', 'val')
img_dir_train_val=osp.join(split_dataset_path, 'img_dir', 'train_val')


def main():

    if not os.path.exists(ann_dir_train): os.makedirs(ann_dir_train)
    if not os.path.exists(ann_dir_val): os.makedirs(ann_dir_val)
    if not os.path.exists(ann_dir_train_val): os.makedirs(ann_dir_train_val)
    if not os.path.exists(img_dir_train): os.makedirs(img_dir_train)
    if not os.path.exists(img_dir_val): os.makedirs(img_dir_val)
    if not os.path.exists(img_dir_train_val): os.makedirs(img_dir_train_val)

    val_ratio = 0.15
    val_interval = int((1 / val_ratio))
    train_size = 0
    val_size = 0

    names = os.listdir(save_imgs)
    for i in tqdm(range(len(names))):
        name = names[i]
        mask_name = name[:-4] + '.png'
        if i % val_interval == 0:
            shutil.copy(os.path.join(save_imgs, name), os.path.join(img_dir_val, name))
            shutil.copy(os.path.join(save_masks, mask_name), os.path.join(ann_dir_val, mask_name))
            val_size += 1
        else:
            shutil.copy(os.path.join(save_imgs, name), os.path.join(img_dir_train, name))
            shutil.copy(os.path.join(save_masks, mask_name), os.path.join(ann_dir_train, mask_name))
            train_size += 1
        shutil.copy(os.path.join(save_imgs, name), os.path.join(img_dir_train_val, name))
        shutil.copy(os.path.join(save_masks, mask_name), os.path.join(ann_dir_train_val, mask_name))
    print("train size:{},val size:{}".format(train_size, val_size))


if __name__ == "__main__":
    main()