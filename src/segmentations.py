# Reference: https://github.com/jayelm/compexp/blob/master/vision/loader/data_loader/broden.py

import os
import csv
import re
import zipfile
from collections import OrderedDict
from urllib.request import urlopen

import numpy
import torch
from torchvision.datasets.folder import default_loader
from scipy import ndimage


class BrodenDataset(torch.utils.data.Dataset):
    '''
    A multicategory segmentation data set.

    Returns three streams:
    (1) The image (3, h, w).
    (2) The multicategory segmentation (labelcount, h, w).
    (3) A bincount of pixels in the segmentation (labelcount).

    Net dissect also assumes that the dataset object has three properties
    with human-readable labels:

    ds.labels = ['red', 'black', 'car', 'tree', 'grid', ...]
    ds.categories = ['color', 'part', 'object', 'texture']
    ds.label_category = [0, 0, 2, 2, 3, ...] # The category for each label
    '''
    def __init__(
            self, directory='dataset/broden', resolution=384,
            subset='ade20k', categories=None,
            transform_image=None, transform_segment=None,
            size=None, include_bincount=True,
            broden_version=1, max_segment_depth=6):
        assert resolution in [224, 227, 384]
        self.directory = directory
        self.resolution = resolution
        self.resdir = os.path.join(directory, 'broden%d_%d' %
                                   (broden_version, resolution))
        self.loader = default_loader
        self.transform_image = transform_image
        self.transform_segment = transform_segment
        self.include_bincount = include_bincount
        # The maximum number of multilabel layers that coexist at an image.
        self.max_segment_depth = max_segment_depth
        with open(os.path.join(self.resdir, 'category.csv'),
                  encoding='utf-8') as f:
            self.category_info = OrderedDict()
            for row in csv.DictReader(f):
                self.category_info[row['name']] = row
        if categories is not None:
            # Filter out unused categories
            categories = set(
                [c for c in categories if c in self.category_info])
            for cat in list(self.category_info.keys()):
                if cat not in categories:
                    del self.category_info[cat]
        categories = list(self.category_info.keys())
        self.categories = categories

        # Filter out unneeded images.
        index_file = self.resdir + f'/index_{subset}.csv'
        if not os.path.exists(index_file):
            create_subset_index(subset, os.path.join(self.resdir, 'index.csv'))

        with open(index_file,
                  encoding='utf-8') as f:
            all_images = [decode_index_dict(r) for r in csv.DictReader(f)]
        self.image = [row for row in all_images
                      if index_has_any_data(
                            row, categories)]
        if size is not None:
            self.image = self.image[:size]
        with open(os.path.join(self.resdir, 'label.csv'),
                  encoding='utf-8') as f:
            self.label_info = build_dense_label_array([
                decode_label_dict(r) for r in csv.DictReader(f)])
            self.labels = [l['name'] for l in self.label_info]
        # Build dense remapping arrays for labels, so that you can
        # get dense ranges of labels for each category.
        self.category_map = {}
        self.category_unmap = {}
        self.category_label = {}
        for cat in self.categories:
            with open(os.path.join(self.resdir, 'c_%s.csv' % cat),
                      encoding='utf-8') as f:
                c_data = [decode_label_dict(r) for r in csv.DictReader(f)]
            self.category_unmap[cat], self.category_map[cat] = (
                    build_numpy_category_map(c_data))
            self.category_label[cat] = build_dense_label_array(
                    c_data, key='code')
        self.num_labels = len(self.labels)
        # Primary categories for each label is the category in which it
        # appears with the maximum coverage.
        self.label_category = numpy.zeros(self.num_labels, dtype=int)
        for i in range(self.num_labels):
            _, self.label_category[i] = max(
               (self.category_label[cat][self.category_map[cat][i]]['coverage']
                if i < len(self.category_map[cat])
                   and self.category_map[cat][i] else 0, ic)
               for ic, cat in enumerate(categories))

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        record = self.image[idx]
        # example record: {
        #    'image': 'opensurfaces/25605.jpg', 'split': 'train',
        #    'ih': 384, 'iw': 384, 'sh': 192, 'sw': 192,
        #    'color': ['opensurfaces/25605_color.png'],
        #    'object': [], 'part': [],
        #    'material': ['opensurfaces/25605_material.png'],
        #    'scene': [], 'texture': []}
        image = self.loader(os.path.join(
                    self.resdir, 'images',
                    record['image']))
        segment = numpy.zeros(shape=(
                    self.max_segment_depth,
                    record['sh'], record['sw']), dtype=int)
        if self.include_bincount:
            bincount = numpy.zeros(shape=(self.num_labels,), dtype=int)
        depth = 0
        for cat in self.categories:
            for layer in record[cat]:
                if isinstance(layer, int):
                    segment[depth, :, :] = layer
                    if self.include_bincount:
                        bincount[layer] += segment.shape[1] * segment.shape[2]
                else:
                    png = numpy.asarray(self.loader(os.path.join(
                        self.resdir, 'images', layer)))
                    segment[depth, :, :] = png[:, :, 0] + png[:, :, 1] * 256
                    if self.include_bincount:
                        bincount += numpy.bincount(
                            segment[depth, :, :].flatten(),
                            minlength=self.num_labels)
                depth += 1
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_segment:
            segment = self.transform_segment(segment)
        if self.include_bincount:
            bincount[0] = 0
            return (image, segment, bincount)
        else:
            return (image, segment)


def build_dense_label_array(label_data, key='number', allow_none=False):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    result = [None] * (max([d[key] for d in label_data]) + 1)
    for d in label_data:
        result[d[key]] = d
    # Fill in none
    if not allow_none:
        example = label_data[0]
        def make_empty(k):
            return dict((c, k if c is key else type(v)())
                        for c, v in example.items())
        for i, d in enumerate(result):
            if d is None:
                result[i] = dict(make_empty(i))
    return result


def build_numpy_category_map(map_data, key1='code', key2='number'):
    '''
    Input: set of rows with 'number' fields (or another field name key).
    Output: array such that a[number] = the row with the given number.
    '''
    results = list(numpy.zeros((max([d[key] for d in map_data]) + 1),
                   dtype=numpy.int16) for key in (key1, key2))
    for d in map_data:
        results[0][d[key1]] = d[key2]
        results[1][d[key2]] = d[key1]
    return results


def index_has_any_data(row, categories):
    for c in categories:
        for data in row[c]:
            if data:
                return True
    return False


def decode_label_dict(row):
    result = {}
    for key, val in row.items():
        if key == 'category':
            result[key] = dict(
                (c, int(n))
                for c, n in [re.match('^([^(]*)\(([^)]*)\)$', f).groups()
                             for f in val.split(';')])
        elif key == 'name':
            result[key] = val
        elif key == 'syns':
            result[key] = val.split(';')
        elif re.match('^\d+$', val):
            result[key] = int(val)
        elif re.match('^\d+\.\d*$', val):
            result[key] = float(val)
        else:
            result[key] = val
    return result


def decode_index_dict(row):
    result = {}
    for key, val in row.items():
        if key in ['image', 'split']:
            result[key] = val
        elif key in ['sw', 'sh', 'iw', 'ih']:
            result[key] = int(val)
        else:
            item = [s for s in val.split(';') if s]
            for i, v in enumerate(item):
                if re.match('^\d+$', v):
                    item[i] = int(v)
            result[key] = item
    return result


def create_subset_index(subset_name, index_file):
    dir_path = os.path.dirname(os.path.realpath(index_file))
    with open(index_file) as infile, open(f'{dir_path}/index_{subset_name}.csv', 'w') as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
        writer.writeheader()
        for r in reader:
            if subset_name in r['image']:
                writer.writerow(r)


class ScaleSegmentation:
    '''
    Utility for scaling segmentations, using nearest-neighbor zooming.
    '''
    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width
    
    def __call__(self, seg):
        ratio = (1, self.target_height / float(seg.shape[1]),
                 self.target_width / float(seg.shape[2]))
        return ndimage.zoom(seg, ratio, order=0)


def scatter_batch(seg, num_labels, omit_zero=True, dtype=torch.uint8):
    '''
    Utility for scattering semgentations into a one-hot representation.
    '''
    result = torch.zeros(*((seg.shape[0], num_labels,) + seg.shape[2:]),
                         dtype=dtype, device=seg.device)
    result.scatter_(1, seg, 1)
    if omit_zero:
        result[:, 0] = 0
    return result






