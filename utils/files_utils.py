import os
import json
import shutil

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import pickle
from custom_types import *



PATH = Tuple[str, str, str]
PATHS = List[PATH]

def add_suffix(path: str, suffix: str) -> str:
    if len(path) < len(suffix) or path[-len(suffix):] != suffix:
        path = f'{path}{suffix}'
    return path


def add_prefix(path: str, prefix: str) -> str:
    if len(path) < len(prefix) or path[:len(prefix)] != prefix:
        path = f'{prefix}{path}'
    return path


def is_file(path: str):
    return os.path.isfile(path)


def init_folders(*folders):
    for f in folders:
        dir_name = os.path.dirname(f)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)


def path_init(suffix: str, path_arg_ind: int, is_save: bool):

    def wrapper(func):

        def do(*args, **kwargs):
            path = add_suffix(args[path_arg_ind], suffix)
            if is_save:
                init_folders(path)
            args = [args[i] if i != path_arg_ind else path for i in range(len(args))]
            return func(*args, **kwargs)

        return do

    return wrapper


def image_to_display(img) -> ARRAY:
    if type(img) is str:
        img = Image.open(str(img))
    if type(img) is not V:
        img = V(img)
    if img.max() <= 1:
        img = (255 * img).astype(np.uint8)
    return


def load_image(path: str, to_np: bool = True,
               background_color=(255, 255, 255)) -> Union[Image.Image, ARRAY]:
    for suffix in ('.png', '.jpg'):
        path_ = add_suffix(path, suffix)
        if os.path.isfile(path_):
            path = path_
            break
    image = Image.open(path)
        #.convert(color_type)
    if image.mode == 'RGBA':
        background = Image.new(image.mode[:-1], image.size, background_color)
        background.paste(image, image.split()[-1])  # omit transparency
        image = background
    if to_np:
        image = V(image)
    if image.ndim == 2:
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    return image


@path_init('.png', 1, True)
def save_image(image: Union[ARRAY, Image.Image], path: str):
    if type(image) is ARRAY:
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        image = Image.fromarray(image)
    image.save(path)


def show_images(image):
    if type(image) is str:
        image = load_image(image)
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close('all')


@path_init('.npy', 1, True)
def save_np(arr: ARRAY, path):
    np.save(path[:-4], arr)


def delete_rec(path: str) -> None:
    if DEBUG:
        return
    if os.path.isfile(path):
        os.remove(path)
    elif os.path.isdir(path):
        shutil.rmtree(path)

@path_init('.pkl', 0, False)
def load_pickle(path: str):
    data = None
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
    return data


@path_init('.pkl', 1, True)
def save_pickle(obj, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def collect(root: str, *suffix, prefix='') -> PATHS:

    paths = []
    root = add_suffix(root, '/')
    if not os.path.isdir(root):
        print(f'Warning: trying to collect from {root} but dir isn\'t exist')
    else:
        p_len = len(prefix)
        for path, _, files in os.walk(root):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                p_len_ = min(p_len, len(file_name))
                if file_extension in suffix and file_name[:p_len_] == prefix:
                    paths.append((f'{add_suffix(path, "/")}', file_name, file_extension))
        paths.sort(key=lambda x: os.path.join(x[1], x[2]))
    return paths


def load_512(image_path):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512), resample=Image.Resampling.BICUBIC))
    return image


def coco_main():
    anno_root ="/home/jupyter/data/coco/annotations/captions_val2014.json"
    with open(anno_root, "r") as f:
        data = json.load(f)["annotations"]
        select_items = np.random.choice(len(data), 200, False)
        visit = set()
        counter = 0
        eval_data = []
        for i in select_items:
            item = data[i]
            caption = item['caption']
            image_id = item["image_id"]
            if image_id in visit:
                continue
            counter += 1
            image_path = f'/home/jupyter/data/coco/val2014/COCO_val2014_{item["image_id"]:012d}.jpg'
            same_image = [item_ for item_ in data if item_["image_id"] == image_id]
            captions = [item_['caption'] for item_ in same_image]

            if (counter + 1) % 10 ==  0:
                image = Image.open(image_path)
                print(caption)
                show_images(np.array(image))
            eval_data.append({'image_id': image_id, 'image_path': image_path, 'captions': captions})
            if counter == 100:
                break
    save_pickle(eval_data, "/home/jupyter/data/amirhertz/evaluation/null_text_coco_metadata.pkl")


@path_init('', 1, True)
def save_model(model, model_path: str):
    init_folders(model_path)
    torch.save(model.state_dict(), model_path)


def load_model(model: nn.Module, model_path: str, device: D, verbose: bool = True, strict: bool = True):
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=strict)
        if verbose:
            print(f'loading {type(model).__name__} from {model_path}')
    elif verbose:
        print(f'init {type(model).__name__}')
    return


@path_init('.json', 1, True)
def save_json(obj, path: str):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=4)


@path_init('.json', 0, False)
def load_json(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def images_to_grid(*data: ARRAY, num_rows=1, offset_ratio=.02):
    images = []
    for im in data:
        if type(im) is list:
            im += images
        elif im.ndim == 4:
            if im.dtype == 'float32':
                im = np.array(im)
                if im.min() < 0:
                    im = (im + 1. / 2)
                im = (im * 255).astype(np.uint8)
            if im.shape[1] == 3:
                im = np.transpose(im, (0, 2, 3, 1))
            im = [im[i] for i in range(im.shape[0])]
        else:
            if im.shape[0] == 3:
                im = np.transpose(im, (1, 2, 0))
            im = [im]
        images += im

    num_items = len(images)
    if num_items > 1:
        h, w, c = images[0].shape
        if type(offset_ratio) is int:
            offset = offset_ratio
        else:
            offset = int(h * offset_ratio)
        num_cols = num_items // num_rows
        image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                          w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
        for i in range(num_rows):
            for j in range(num_cols):
                image_[i * (h + offset): i * (h + offset) + h :, j * (w + offset): j * (w + offset) + w] = images[i * num_cols + j]
    else:
        image_ = images[0]
    return image_


if __name__ == '__main__':
    coco_main()
