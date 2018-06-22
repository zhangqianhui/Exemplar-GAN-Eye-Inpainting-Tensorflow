import os
import errno
import numpy as np
import scipy
import scipy.misc

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def get_image(image_path , image_size, is_crop= True, resize_w= 64, is_grayscale= False, is_test=False):
    return transform(imread(image_path , is_grayscale), image_size, is_crop , resize_w, is_test=is_test)

def transform(image, npx = 64 , is_crop=False, resize_w=64, is_test=False):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image , npx , resize_w=resize_w, is_test=is_test)
    else:
        cropped_image = image
        cropped_image = scipy.misc.imresize(cropped_image ,
                            [resize_w , resize_w])
    return np.array(cropped_image)/127.5 - 1

def center_crop(x, crop_h , crop_w=None, resize_w=64, is_test=False):

    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))

    if not is_test:
        rate = np.random.uniform(0, 1, size=1)
        if rate < 0.5:
            x = np.fliplr(x)
    # return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
    #                            [resize_w, resize_w])
    return scipy.misc.imresize(x[20:218 - 20, 0: 178], [resize_w, resize_w])

def save_images(images, size, image_path, is_ouput=False):
    return imsave(inverse_transform(images, is_ouput), size, image_path)

def imread(path, is_grayscale=False):

    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def merge(images, size):

    if size[0] + size[1] == 2:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    else:
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1], 3))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def inverse_transform(image, is_ouput=False):

    if is_ouput == True:
        print image[0]
    result = ((image + 1) * 127.5).astype(np.uint8)
    if is_ouput == True:
        print result
    return result

def read_image_list_for_Eyes(category, is_test=False):

    filenames = []
    exemplar_names = []
    eye_pos_str = []
    print("list file")
    list = os.listdir(category)
    for file in list:

        if '-1' in file:

            flag = 1


            if len(file.split('-', 11)) == 11:
                identity = file.split('-', 11)[0] + "-" + file.split('-', 11)[1]

                eye_pos = []
                for i in range(3, 11):
                    eye_pos.append(int((file.split('-', 11)[i]).split('.', 2)[0]))

                eye_pos_str.append(eye_pos)

                filenames.append(category + "/" + file)

            elif len(file.split('-', 10)) == 10:

                identity = file.split('-', 10)[0]
                eye_pos = []
                for i in range(2, 10):
                    eye_pos.append(int((file.split('-', 10)[i]).split('.', 2)[0]))
                eye_pos_str.append(eye_pos)

                filenames.append(category + "/" + file)

            else:

                flag = 0

            if flag == 1:

                exemplar_1 = category + "/" + identity + "-2-0-0-0-0-0-0-0-0.jpg"
                exemplar_2 = category + "/" + identity + "-3-0-0-0-0-0-0-0-0.jpg"
                exemplar_3 = category + "/" + identity + "-4-0-0-0-0-0-0-0-0.jpg"
                exemplar_4 = category + "/" + identity + "-5-0-0-0-0-0-0-0-0.jpg"

                if os.path.exists(exemplar_1):
                    exemplar_names.append(exemplar_1)

                elif os.path.exists(exemplar_2):
                    exemplar_names.append(exemplar_2)
                elif os.path.exists(exemplar_3):
                    exemplar_names.append(exemplar_3)
                elif os.path.exists(exemplar_4):
                    exemplar_names.append(exemplar_4)
                else:

                    filenames.remove(category + "/" + file)
                    eye_pos_str.remove(eye_pos)

    print "len", len(filenames), len(exemplar_names), len(eye_pos_str)
    assert len(exemplar_names) == len(filenames)

    if is_test:
        return filenames[0:1000], exemplar_names[0:1000], eye_pos_str[0:1000]
    else:
        return filenames[1000:-1], exemplar_names[1000:-1], eye_pos_str[1000:-1]

class Eyes(object):

    def __init__(self, image_path):

        self.dataname = "Eyes"
        self.image_size = 128
        self.channel = 3
        self.dims = self.image_size*self.image_size
        self.shape = [self.image_size, self.image_size, self.channel]
        self.train_data_list, self.train_data_list2, self.eye_pos_str = self.load_Eyes(image_path)
        self.test_data_list, self.test_data_list2, _ = self.load_test_Eyes(image_path)

    def load_Eyes(self, image_path):

        # get the list of image path
        images_list, image_list2, eye_pos_str = read_image_list_for_Eyes(image_path, is_test=False)

        return images_list, image_list2, eye_pos_str

    def load_test_Eyes(self, image_path):

        # get the list of image path
        images_list, image_list2, eye_pos_str = read_image_list_for_Eyes(image_path, is_test=True)
        return images_list, image_list2, eye_pos_str

    def getShapeForData(self, filenames, is_test=False):

        array = [get_image(batch_file, 108, is_crop=False, resize_w=128,
                           is_grayscale=False, is_test=is_test) for batch_file in filenames]
        sample_images = np.array(array)
        # return sub_image_mean(array , IMG_CHANNEL)
        return sample_images

    def getNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.train_data_list) / batch_size
        if batch_num % ro_num == 0 and is_shuffle:

            length = len(self.train_data_list)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.train_data_list = np.array(self.train_data_list)
            self.train_data_list = self.train_data_list[perm]

            self.train_data_list2 = np.array(self.train_data_list2)
            self.train_data_list2 = self.train_data_list2[perm]

            self.eye_pos_str = np.array(self.eye_pos_str)
            self.eye_pos_str = self.eye_pos_str[perm]

        return self.train_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.train_data_list2[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.eye_pos_str[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]

    def getTestNextBatch(self, batch_num=0, batch_size=64, is_shuffle=True):

        ro_num = len(self.test_data_list) / batch_size
        if batch_num == 0 and is_shuffle:

            length = len(self.test_data_list)
            perm = np.arange(length)
            np.random.shuffle(perm)

            self.test_data_list = np.array(self.test_data_list)
            self.test_data_list = self.test_data_list[perm]

            self.test_data_list2 = np.array(self.test_data_list2)
            self.test_data_list2 = self.test_data_list2[perm]

        return self.test_data_list[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size], \
               self.test_data_list2[(batch_num % ro_num) * batch_size: (batch_num % ro_num + 1) * batch_size]
