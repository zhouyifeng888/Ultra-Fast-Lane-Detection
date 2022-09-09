
import os

from PIL import Image

from mindspore import dataset as ds


class LaneDataset:
    def __init__(self, root_path, list_path):

        self.root_path = root_path

        with open(list_path, 'r') as f:
            self.list = f.readlines()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.root_path, label_name)
        label = Image.open(label_path)

        img_path = os.path.join(self.root_path, img_name)
        img = Image.open(img_path)

        return img, label

    def __len__(self):
        return len(self.list)


def create_lane_dataset(data_root_path, data_list_path, batch_size,
                              is_train=True,num_workers=8, rank_size=1, rank_id=0):

    lane_dataset = LaneDataset(
        data_root_path, os.path.join(data_root_path, data_list_path))
    
    if is_train:
        shuffle=True
    else:
        shuffle=False
    
    dataset = ds.GeneratorDataset(source=lane_dataset, column_names=["image", 'label'],
                                  num_parallel_workers=num_workers, shuffle=shuffle,
                                  num_shards=rank_size, shard_id=rank_id)
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    
    return dataset

if __name__ == "__main__":
    dataset = create_lane_dataset('../../../dataset/Tusimple/train_set/','train_gt.txt', 16, num_workers=1)
    data = next(dataset.create_dict_iterator())
