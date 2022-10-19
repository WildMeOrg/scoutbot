# -*- coding: utf-8 -*-
"""
docker run \
    -d \
    -p 6035:5000 \
    --gpus '"device=0,1,2,3"' \
    --no-healthcheck \
    --shm-size=8g \
    --name scout \
    -v /lifeboat/ibeis/Scout_MVP:/data/db \
    -v /nas/raw/unprocessed/scout:/data/nas \
    -v /data/public/models:/data/public/models \
    -v /data/cache:/cache \
    --env-file /opt/wbia/.env \
    --restart unless-stopped \
    wildme/wbia:latest \
        --container-name scout \
        --engine-fast-lane-workers 0 \
        --engine-slow-lane-workers 0
"""
import csv
import json
import random
import time
from os.path import split, splitext

import numpy as np
import tqdm
import utool as ut
import wbia
from wbia.detecttools.directory import Directory

ibs = wbia.opendb('/data/db')

direct = Directory('/data/nas', recursive=True, images=True)
filepaths = list(direct.files())  # 322805

csv_filepaths = ut.glob('/data/nas/csv/*.csv')
lines = []
for csv_filepath in csv_filepaths:
    with open(csv_filepath, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        csv_header = None
        for csv_line in csv_reader:
            if csv_header is None:
                csv_header = csv_line
            else:
                lines.append(dict(zip(csv_header, csv_line)))

globals().update(locals())
filenames = [split(filepath)[1] for filepath in filepaths]
filenames = [filename.replace('.JPG', '.jpg') for filename in filenames]
ut.dict_hist(ut.dict_hist(filenames).values())  # {1: 315491, 2: 3657}

mapping = {}
for filepath, filename in zip(filepaths, filenames):
    if filename not in mapping:
        mapping[filename] = []
    mapping[filename].append(filepath)

filename_mapping = {}
count = 0
for filename in mapping:
    data = sorted(mapping[filename])
    assert len(data) in [1, 2]
    if len(data) > 1:
        if (
            'tsavo-2017-core-annotated/Tsavo 2017, Core, all even-code positives, copy for compress'
            in data[0]
            and 'tsavo-mar16-212017/Tsavo 16-21 March, Core, AKP only' in data[1]
        ):
            data = [data[0]]
        elif (
            'tsavo-2017-core-annotated/Tsavo 2017, Core, all even-code positives, copy for compress'
            in data[0]
            and 'tsavo-mar12-152017/Tsavo 12-15 March, Core, AKP only' in data[1]
        ):
            data = [data[0]]
        elif (
            'murchison-sep2015/TEP SURVEY SEP 2015 AIRPHOTOS 30 Sep Left and Right, 1 Oct Left Only'
            in data[0]
            and 'murchison-sep2015/TEP SURVEY SEP 2015, AIRPHOTOS 1 Oct Right, 2 Oct Left and Right A'
            in data[1]
        ):
            data = [data[0]]
        elif (
            'murchison-apr2016-all/TEP SURVEY APR 2016, ALL AIRPHOTOS, CODED' in data[0]
            and 'murchison-apr2016-annotated/TEP April 2016 ALL animals positive images even for annotation compressed Review'
            in data[1]
        ):
            data = [data[0]]
        else:
            print(data)
            count += 1
    assert len(data) == 1
    filepath = data[0]
    assert filename not in filename_mapping
    filename_mapping[filename] = filepath
print(count)

filepaths_ = list(filename_mapping.values())
assert len(filepaths_) == len(set(filepaths_))

missing = []
invalid = []
annot_filepaths = []
annot_bboxes = []
annot_species = []
for line in lines:
    filename = line.get('filename', None)
    region_shape_attributes = line.get('region_shape_attributes', None)
    region_attributes = line.get('region_attributes', None)

    filename = filename.replace('.JPG', '.jpg')

    filepath = filename_mapping.get(filename, None)
    if filepath is None:
        missing.append(filename)
        continue

    assert region_shape_attributes is not None
    assert region_attributes is not None

    bbox_data = json.loads(region_shape_attributes)
    species_data = json.loads(region_attributes)

    bbox = (
        bbox_data.get('x', None),
        bbox_data.get('y', None),
        bbox_data.get('width', None),
        bbox_data.get('height', None),
    )
    species = species_data.get('Name', None)

    if None in bbox or None in [filepath, species]:
        invalid.append(filepath)
        continue

    annot_filepaths.append(filepath)
    annot_bboxes.append(bbox)
    annot_species.append(species)

assert len(annot_filepaths) == len(annot_species)
assert len(annot_filepaths) == len(annot_bboxes)

len(missing)  # 0
len(invalid)  # 257

#########

all_filepaths = set(filename_mapping.values())  # 319148
valid_filepaths = all_filepaths - set(invalid)  # 318949
positive_filepaths = set(annot_filepaths)  # 8211
negative_filepaths = valid_filepaths - positive_filepaths  # 310741

buckets = {}
for positive_filepath in positive_filepaths:
    bucket = '/'.join(positive_filepath.split('/')[:4])
    if bucket not in buckets:
        buckets[bucket] = []
    buckets[bucket].append(positive_filepath)

print('positives')
for bucket in sorted(buckets.keys()):
    bucket_filepaths = buckets[bucket]
    print(bucket, len(bucket_filepaths))

buckets = {}
for negative_filepath in negative_filepaths:
    bucket = '/'.join(negative_filepath.split('/')[:4])
    if bucket not in buckets:
        buckets[bucket] = []
    buckets[bucket].append(negative_filepath)

print('negatives')
random.seed(1)
sampling_filepaths = []
for bucket in sorted(buckets.keys()):
    bucket_filepaths = sorted(buckets[bucket])
    random.shuffle(bucket_filepaths)
    index = max(100, int(len(bucket_filepaths) * 0.01))
    print(bucket, len(bucket_filepaths), index)
    bucket_filepaths = bucket_filepaths[:index]
    sampling_filepaths += bucket_filepaths

assert len(sampling_filepaths) == len(set(sampling_filepaths))
sampling_filepaths = set(sampling_filepaths)

# 10%
#     positives
#         /data/nas/murchison-2019, 2127
#         /data/nas/murchison-apr2016-all, 1029
#         /data/nas/murchison-apr2016-annotated, 446
#         /data/nas/murchison-dec2015, 1067
#         /data/nas/murchison-sep2015, 1204
#         /data/nas/queenelizabeth-oct01-022018, 605
#         /data/nas/queenelizabeth-sep29-302018, 603
#         /data/nas/tsavo-2017-core-annotated, 1130

#     negatives
#         /data/nas/murchison-2019, 60535, 6053
#         /data/nas/murchison-apr2016-all, 20749, 2074
#         /data/nas/murchison-dec2015, 31696, 3169
#         /data/nas/murchison-sep2015, 32949, 3294
#         /data/nas/queenelizabeth-oct01-022018, 19111, 1911
#         /data/nas/queenelizabeth-sep29-302018, 29707, 2970
#         /data/nas/tsavo-2017-non-core-annotated, 946, 94
#         /data/nas/tsavo-mar12-152017, 47947, 4794
#         /data/nas/tsavo-mar16-212017, 67101, 6710

# 1%, max 100
#     positives
#         /data/nas/murchison-2019, 2127
#         /data/nas/murchison-apr2016-all, 1029
#         /data/nas/murchison-apr2016-annotated, 446
#         /data/nas/murchison-dec2015, 1067
#         /data/nas/murchison-sep2015, 1204
#         /data/nas/queenelizabeth-oct01-022018, 605
#         /data/nas/queenelizabeth-sep29-302018, 603
#         /data/nas/tsavo-2017-core-annotated, 1130
#     negatives
#         /data/nas/murchison-2019, 60535, 605
#         /data/nas/murchison-apr2016-all, 20749, 207
#         /data/nas/murchison-dec2015, 31696, 316
#         /data/nas/murchison-sep2015, 32949, 329
#         /data/nas/queenelizabeth-oct01-022018, 19111, 191
#         /data/nas/queenelizabeth-sep29-302018, 29707, 297
#         /data/nas/tsavo-2017-non-core-annotated, 946, 100
#         /data/nas/tsavo-mar12-152017, 47947, 479
#         /data/nas/tsavo-mar16-212017, 67101, 671

assert len(positive_filepaths & sampling_filepaths) == 0

final_filepaths = sorted(positive_filepaths | sampling_filepaths)
ut.dict_hist([splitext(final_filepath)[1] for final_filepath in final_filepaths])

ut.save_cPkl('/data/db/filepaths.positive.pkl', positive_filepaths)
ut.save_cPkl('/data/db/filepaths.negative.pkl', sampling_filepaths)
ut.save_cPkl('/data/db/filepaths.final.pkl', final_filepaths)

# {'.JPG': 38829, '.jpg': 451}
# {'.JPG': 10958, '.jpg': 448}

######

gids = ibs.get_valid_gids()
processed = set(ibs.get_image_uris_original(gids))
filepaths_ = sorted(list(set(final_filepaths) - processed))

chunks = ut.ichunks(filepaths_, 100)
for filepath_chunk in tqdm.tqdm(chunks):
    try:
        gids = ibs.add_images(
            filepath_chunk,
            auto_localize=True,
            ensure_loadable=False,
            ensure_exif=False,
        )
    except Exception:
        pass

#######

# positive_filepaths = ut.load_cPkl('/data/db/filepaths.positive.pkl')
# sampling_filepaths = ut.load_cPkl('/data/db/filepaths.negative.pkl')
# final_filepaths = ut.load_cPkl('/data/db/filepaths.final.pkl')

gids = ibs.get_valid_gids()
uris_original = ibs.get_image_uris_original(gids)
globals().update(locals())
flags = [uri_original not in final_filepaths for uri_original in uris_original]
delete_gids = ut.compress(gids, flags)
delete_tids = ut.flatten(ibs.get_tile_children_gids(delete_gids))
assert set(ibs.get_tile_flags(delete_gids)) == {False}
assert set(ibs.get_tile_flags(delete_tids)) == {True}
ibs.delete_images(delete_tids, trash_images=False)
ibs.delete_images(delete_gids, trash_images=False)

#######

gids = ibs.get_valid_gids()
uris = ibs.get_image_uris_original(gids)
filepath_map = dict(zip(uris, gids))

globals().update(locals())
annot_gids = [filepath_map.get(annot_filepath) for annot_filepath in annot_filepaths]
aids = ibs.add_annots(annot_gids, bbox_list=annot_bboxes, species_list=annot_species)

len(aids)  # 67566
len(set(aids))  # 59754

#######

gids = sorted(ibs.get_valid_gids())

tile_size = 256
tile_overlap = 64
tile_offset = (tile_size - tile_overlap) // 2

config = {
    'tile_width': tile_size,
    'tile_height': tile_size,
    'tile_overlap': tile_overlap,
}
tids_grid1 = ibs.compute_tiles(gid_list=gids, **config)

# config = {
#     'ext': '.jpg',
#     'tile_width': tile_size,
#     'tile_height': tile_size,
#     'tile_overlap': tile_overlap,
#     'tile_offset': tile_offset,
#     'allow_borders': False,
# }
# tids_grid2 = ibs.compute_tiles(gid_list=gids, **config)

# The above compute takes a long time, monitor it with the below code
while True:
    ibs.depc_image.tables
    tiles = ibs.depc_image.tables[-2]
    rowids = tiles._get_all_rowids()
    tids = ibs.get_valid_gids(is_tile=True)

    print(len(rowids))
    print(len(tids))
    time.sleep(60)

########

gids = ibs.get_valid_gids()
tids = ibs.scout_get_valid_tile_rowids()
len(gids)  # 11,406
len(tids)  # 8,063,156

flags = ibs.get_tile_flags(tids)
assert False not in set(flags)

aids_list = ibs.get_tile_aids(tids)
flags = [len(aid_list) > 0 for aid_list in aids_list]
positive_tids = ut.compress(tids, flags)
negative_tids = ut.filterfalse_items(tids, flags)

random.shuffle(negative_tids)
index = int(len(negative_tids) * 0.1)
sample_tids = negative_tids[:index]
delete_tids = negative_tids[index:]
ibs.delete_images(delete_tids, trash_images=False)

len(positive_tids)  # 115414
len(negative_tids)  # 7947742
len(sample_tids)  # 794774

len(positive_tids) / len(negative_tids)  # 0.014521608778946272
len(positive_tids) / len(sample_tids)  # 0.014521608778946272

ibs.scout_imageset_train_test_split(recompute_split=True)

########

# train WIC with boosting rounds

models = ibs.scout_wic_train()

restart_config_dict = {
    'scout-mvp-boost0': 'https://cthulhu.dyn.wildme.io/public/models/classifier2.scout.mvp.0.zip',
    'scout-mvp-boost1': 'https://cthulhu.dyn.wildme.io/public/models/classifier2.scout.mvp.1.zip',
    'scout-mvp-boost2': 'https://cthulhu.dyn.wildme.io/public/models/classifier2.scout.mvp.2.zip',
}

config_list = [
    {
        'label': 'WIC mvp Round 0',
        'classifier_algo': 'densenet',
        'classifier_weight_filepath': 'scout-mvp-boost0',
    },
    {
        'label': 'WIC mvp Round 1',
        'classifier_algo': 'densenet',
        'classifier_weight_filepath': 'scout-mvp-boost1',
    },
    {
        'label': 'WIC mvp Round 2',
        'classifier_algo': 'densenet',
        'classifier_weight_filepath': 'scout-mvp-boost2',
    },
]
ibs.scout_wic_validate(config_list)

########

(nid,) = ibs.get_imageset_imgsetids_from_text(['NEGATIVE'])

nid_tids = ibs.get_imageset_gids(nid)
assert set(ibs.get_tile_flags(nid_tids)) == {True}
model_tag = 'scout-mvp-boost2'
confidences = ibs.scout_wic_test(nid_tids, model_tag=model_tag)
confidences = np.array(confidences)

globals().update(locals())
flags = [conf >= 0.07 for conf in confidences]
hard_nid_tids = ut.compress(nid_tids, flags)

########

ibs.scout_localizer_train()

ibs.scout_localizer_validate()

# Start training manually

# CUDA_VISIBLE_DEVICES=1,3 /virtualenv/env3/bin/python /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/bin/train.py -c -n /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/cfg/yolo.py -b /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/backup /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/darknet19_448.conv.23.pt

# Resume training manually

# CUDA_VISIBLE_DEVICES=2 /virtualenv/env3/bin/python /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/bin/train.py -c -n /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/cfg/yolo.py -b /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/backup /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/backup/weights_30000.state.pt
# CUDA_VISIBLE_DEVICES=1 /virtualenv/env3/bin/python /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/bin/test.py --fast-pr -c -n /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/cfg/yolo.py --results /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/results.500.o.txt /data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/backup/weights_*.state.pt

# Validation Results

# 0.2722,
# 2.1196953,
# '/data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/backup/weights_58000.state.pt'

# 0.27790000000000004,
# 2.14777691,
# '/data/db/_ibsdb/_ibeis_cache/training/lightnet/lightnet-training-mvp-892b8c24f52400ff/backup/weights_30000.state.pt'
