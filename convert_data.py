from collections import OrderedDict, defaultdict
import numpy as np
import os
import os.path as osp
import json
from pathlib import Path
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='Conversion of ActionGenome to style suitable for us')
# Dataset related arguments
parser.add_argument('--in_json', required=True,
                    type=str,help='Input JSON, PKL of ActionGenome converted to JSON')
parser.add_argument('--out_json', required=True,
                    type=str,
                    help='Path to output JSON, which is cleaned up, suitable for COCO model')
parser.add_argument('--data_dir', required=True,
                    type=str, help='Path to where ActionGenome frames are')

args = parser.parse_args()


def get_detection_json(train_dict, class_to_id):
	detection_json = {}
	im_id = 0
	for k1, v1 in tqdm(train_dict.items()):
		for k2, v2 in v1.items():
			im_path = osp.join(k1, k2)

			## Check if the frames have been extracted.
			## Many frames don't exist due to bug in
			## Author's(Action Genome) extraction code.

			if not osp.isfile(osp.join(args.data_dir, im_path)):
				continue
			detections = defaultdict(list)
			for dets in v2:
				if 'class' in dets:
					if dets['class'] in class_to_id:
						class_id = class_to_id[dets['class']]
					else:
						class_id = list(class_to_id.values())[-1] + 1
						class_to_id[dets['class']] = class_id
					class_name = dets['class']
					is_crowd = not dets['visible']
					cur_bbox = dets['bbox']
					cur_rel  = dets['contacting_relationship']
				else:
					assert 'person' in dets
					class_id = class_to_id['person']
					cur_bbox = dets['person']['bbox']
					is_crowd = False
					class_name = 'person'
				if cur_bbox is None:
					continue
				if len(cur_bbox) == 0:
					continue

				## Objects have synonyms
				## We only use the first word.
				if '/' in class_name:
					class_name = class_name.split('/')[0]

				im_id += 1
				cur_bbox = np.array(cur_bbox)
				detections['area'].append(np.product(cur_bbox))
				cur_bbox[2:4] += cur_bbox[0:2]
				detections['is_crowd'].append(is_crowd)
				detections['bbox'].append(cur_bbox.tolist())
				detections['labels'].append(class_id)
				detections['class_name'].append(class_name)

				if cur_rel is not None:
					detections['relation'].append(cur_rel[0])
				detections['image_id'] = im_id
			detection_json[im_path] = detections
	return detection_json



if __name__ == '__main__':
	
	in_dict = json.load(open(args.in_json, 'r'))

	class_id_json = osp.join(Path(args.in_json).parent, 'class_id.json')
	if osp.isfile(class_id_json):
		class_to_id = json.load(open(class_id_json, 'r'))
	else:
		class_to_id = OrderedDict({'person': 1})
	
	converted_json = get_detection_json(in_dict, class_to_id)
	print("Converted! New dataset contains : " + str(len(converted_json)) + " frames")
	with open(args.out_json, 'w') as fp:
		json.dump(converted_json, fp)

