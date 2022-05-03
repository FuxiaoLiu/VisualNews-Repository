
from pycocotools.coco import COCO
from pycocoevalcap1.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys, pdb

def myeval(input_json, annFile):
	coco = COCO(annFile)
	valids = coco.getImgIds()
	if isinstance(input_json, list): # pass preds list from test method in train.py
		preds = input_json
	elif isinstance(input_json, str): # pass filename of predictions
		preds = json.load(open(input_json, 'r'))
	else:
		checkpoint = json.load(open(input_json, 'r')) # pass name of checkpoint(although no need i think)
		preds = checkpoint['val_predictions']

	# filter results to only those in MSCOCO validation set (will be about a third)
	preds_filt = [p for p in preds if p['image_id'] in valids]
	print('using %d/%d predictions' % (len(preds_filt), len(preds)))
	json.dump(preds_filt, open('tmp.json', 'w')) # serialize to temporary json file. Sigh, COCO API...

	resFile = 'tmp.json'
	cocoRes = coco.loadRes(resFile)
	cocoEval = COCOEvalCap(coco, cocoRes)
	cocoEval.params['image_id'] = cocoRes.getImgIds()
	cocoEval.evaluate()

	# create output dictionary
	out = {}
	for metric, score in cocoEval.eval.items():
		out[metric] = score
	# serialize to file, to be read from Lua
	print(out)
	# json.dump(out, open(input_json + '_out.json', 'w'))
	return out

# scores: ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]
