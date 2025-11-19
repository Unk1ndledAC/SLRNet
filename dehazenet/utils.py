import os, glob, random

def get_pairs(root_dir):
    gt_dir   = os.path.join(root_dir, 'gt')
    hazy_dir = os.path.join(root_dir, 'hazy')
    gt_list  = sorted(glob.glob(os.path.join(gt_dir, '*')))
    hazy_dict = {}
    for p in glob.glob(os.path.join(hazy_dir, '*')):
        key = os.path.splitext(os.path.basename(p))[0]
        hazy_dict.setdefault(key, []).append(p)

    pairs = []
    for gp in gt_list:
        key = os.path.splitext(os.path.basename(gp))[0]
        if key in hazy_dict:
            pairs.append((gp, random.choice(hazy_dict[key])))
    return pairs
