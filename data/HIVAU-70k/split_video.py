import os
import json
from tqdm import tqdm
import decord
import cv2
import random
from multiprocessing import Process

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_video_decord(video_reader, segment, save_dir, is_frame=False):
    fps = video_reader.get_avg_fps()
    if is_frame:
        start_frame_index, end_frame_index = segment
        end_frame_index = min(end_frame_index, len(video_reader)-1)
    else:
        start_time, end_time = segment
        start_frame_index = int(start_time * fps)
        end_frame_index = min(int(end_time * fps), len(video_reader)-1)
    frames = video_reader.get_batch(range(start_frame_index, end_frame_index)).asnumpy()
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(save_dir, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) 
        video_writer.write(frame)
    video_writer.release()

def get_vid_list(json_data, instruction_data):
    ins_vid_list = []
    with open(instruction_data, 'r') as f:
        raw_data = f.readlines()
        for item in raw_data:
            item = json.loads(item)
            if item['type'] == 'video':
                vid_name = item['video'].split('/')[-1][:-4]
                if vid_name not in ins_vid_list: ins_vid_list.append(vid_name)
    vid_list = []
    for vid_name in json_data.keys():
        if vid_name in ins_vid_list:
            vid_list.append(vid_name)
    return vid_list

def process(vid_list, video_root, json_data, event_save_path, clip_save_path):
    p_bar = tqdm(vid_list)
    for vid_name in p_bar:
        video_path = os.path.join(video_root, vid_name+'.mp4')
        video_reader = decord.VideoReader(video_path)
        vid_events = json_data[vid_name]['events']
        vid_clips = json_data[vid_name]['clips']
        for e_idx, event in enumerate(vid_events):
            e_dir = os.path.join(event_save_path, "{}_E{}.mp4".format(vid_name, e_idx))
            if not os.path.exists(e_dir): 
                split_video_decord(video_reader, event, e_dir, is_frame=False)
            p_bar.set_description("[Processing-{}-E{}]".format(e_idx, vid_name))
            for c_idx, clip in enumerate(vid_clips[e_idx]):
                c_dir = os.path.join(clip_save_path, "{}_E{}C{}.mp4".format(vid_name, e_idx, c_idx))
                if not os.path.exists(c_dir):
                    split_video_decord(video_reader, clip, c_dir, is_frame=False)
                p_bar.set_description("[Processing-{}-E{}C{}]".format(vid_name, e_idx, c_idx))


def segment(dataset, subset, n_thread=1):
    instruction_data = "./instruction/merge_instruction_{}_final.jsonl".format(subset)
    if 'ucf' in dataset:
        json_data = './raw_annotations/ucf_database_{}.json'.format(subset)
    elif 'xd' in dataset:
        json_data = './raw_annotations/xd_database_{}.json'.format(subset)
    video_root = './videos/{}/videos/{}'.format(dataset, subset)
    event_save_path = './videos/{}/events/{}'.format(dataset, subset)
    clip_save_path = './videos/{}/clips/{}'.format(dataset, subset)
    mkdir(clip_save_path)
    mkdir(event_save_path)
    
    with open(json_data, 'r') as f:
        json_data = json.load(f)
    vid_list = get_vid_list(json_data, instruction_data)
    random.shuffle(vid_list)

    batch = len(vid_list) // n_thread
    process_list = []
    for n in range(n_thread):
        videos = vid_list[batch*n:batch*(n+1)]
        p = Process(target=process, args=(videos, video_root, json_data, event_save_path, clip_save_path))
        p.start()
        process_list.append(p)
    if len(vid_list)%n_thread != 0:
        videos = vid_list[batch*n_thread:]
        p = Process(target=process, args=(videos, video_root, json_data, event_save_path, clip_save_path))
        p.start()
        process_list.append(p)

    for p in process_list:
        p.join()

    print('Done')

if __name__ == '__main__':
    import time
    t_start = time.perf_counter() 
    
    segment('ucf-crime', 'test', n_thread=16)
    segment('xd-violence', 'test', n_thread=16)
    segment('ucf-crime', 'train', n_thread=16)
    segment('xd-violence', 'train', n_thread=16)

    t_end = time.perf_counter()
    print('Running time: %s Seconds'%(t_end-t_start))
