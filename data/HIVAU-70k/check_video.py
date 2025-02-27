import os
import json
from tqdm import tqdm

def check(subset):
    video_root = './videos'
    instruction_data = "./instruction/merge_instruction_{}_final.jsonl".format(subset)
    
    n = 0
    with open(instruction_data, 'r') as f:
        raw_data = f.readlines()
        for item in tqdm(raw_data):
            item = json.loads(item)
            vid_path = os.path.join(video_root, item['video'])
            if not os.path.exists(vid_path):
                n += 1
                print(vid_path, "does not exist.")
    if n > 0:
        print(n, " videos do not exsit, please check the download/split process")
    else:
        print("Success, all videos are ready!")
    

if __name__ == "__main__":
    check('train')
    check('test')