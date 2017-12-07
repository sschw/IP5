# splits the images in the train folder into train (70%), test (15%), valid(15%)
# and creates corresponding load-files for tensorflow
import os
from numpy import random
from scale_and_resize_on_object import scale_and_resize


def create_import_list(top):
    # creates a directory file containing image filenames and class labels, 1 file per line
    
    entries = []
    for root, dirs, files in os.walk(top):
        for fn in files:
            if fn.endswith(".PNG"):
                fn = os.path.join(root, fn)
                folder_path = os.path.dirname(fn)
                _, folder_name = os.path.split(folder_path)
                entries.append(fn + ' ' + folder_name)
    # shuffle list to improve training
    f = open(os.path.join(top, 'files.txt'), 'w')
    random.shuffle(entries)
    f.write('\n'.join(entries))
    f.close()


def make_data(source, dest):
    # resizes and crops images, then splits images into train, validation and test datasets
    print('loading existing file index to prevent recreating files')

    existing = []
    if os.path.isfile(os.path.join(dest, 'train', 'files.txt')):
        f = open(os.path.join(dest, 'train', 'files.txt'), 'r')
        for line in f:
            existing.append('/'.join(line.split(' ')[0].split('/')[-2:]))

        f.close()

    if os.path.isfile(os.path.join(dest, 'test', 'files.txt')):
        f = open(os.path.join(dest, 'test', 'files.txt'), 'r')
        for line in f:
            existing.append('/'.join(line.split(' ')[0].split('/')[-2:]))

        f.close()

    if os.path.isfile(os.path.join(dest, 'validation', 'files.txt')):
        f = open(os.path.join(dest, 'validation', 'files.txt'), 'r')
        for line in f:
            existing.append('/'.join(line.split(' ')[0].split('/')[-2:]))

        f.close()
    
    print('Analyse object and scale on it')
    scale_and_resize(source, os.path.join(dest, 'train/'), existing)

    # copy images to test and validation
    print('Randomly copying images to test and validation directories')
    i = 0
    for root, dirs, files in os.walk(os.path.join(dest, 'train')):
        for fn in files:
            fn = os.path.join(root, fn)
            if not '/'.join(fn.split('/')[-2:]) in existing and fn.endswith(".PNG"):
                i += 1
                rand = random.random()
                new_fn = ''

                if rand > 0.9:
                    # move to test
                    new_fn = fn.replace('\\train', '\\test')
                elif rand > 0.8:
                    # move to valid
                    new_fn = fn.replace('\\train', '\\validation')
                else:
                    continue

                if not os.path.exists(os.path.dirname(new_fn)):
                    os.makedirs(os.path.dirname(new_fn))
                os.rename(fn, new_fn)
                end = ""
                
                if i % 100 == 0:
                    end = "\n"
                print('.', end=end, flush=True)
    print('.', end="\n", flush=True)

    print('Creating import file lists')
    create_import_list(os.path.join(dest, 'train'))
    create_import_list(os.path.join(dest, 'test'))
    create_import_list(os.path.join(dest, 'validation'))
