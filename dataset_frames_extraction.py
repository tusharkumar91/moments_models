import os
import json
import glob
import sys
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict


def extract_video_frames(video_dir, frames_dir):
    with open('../NymbleData/COIN.json', 'r') as f:
        coin_data = json.load(f)
    add_verbs = ['add', 'combine', 'add-to', 'pour']
    #mixing_verbs = ['mix', 'beat', 'mix-around', 'stir-with', 'whisk', 'stir', 'blend', 'mix-in', 'stir-in']
    mixing_verbs = ['beat', 'stir-with', 'whisk', 'stir', 'mix-in', 'stir-in', 'mix']

    mix_segments = {}
    add_segments = {}
    mix_segment_info = {}
    keys = list(coin_data['database'].keys())
    for key in keys:
        add_segments_list = []
        mix_segments_list = []
        if coin_data['database'][key]['class'].startswith('Make'):
            for ann in coin_data['database'][key]['annotation']:
                mix_verb_found = False
                for mix_verb in mixing_verbs:
                    if mix_verb in ann['label']:
                        mix_verb_found = True
                        
                add_verb_found = False
                for add_verb in add_verbs:
                    if add_verb in ann['label']:
                        add_verb_found = True
            
                if mix_verb_found and not add_verb_found:
                    mix_segments_list.append(ann['segment'])
            
                if add_verb_found and not mix_verb_found:
                    add_segments_list.append(ann['segment'])
        
        if len(mix_segments_list) > 0:
            mix_segments[key] = mix_segments_list
        if len(add_segments_list) > 0:
            add_segments[key] = add_segments_list
    add_videos = set()
    for key in add_segments:
        add_videos.add(key)
    # mix_videos = set()
    # for key in mix_segments:
    #     mix_videos.add(key)
    # print(len(mix_videos))
    found = 0
    for video in tqdm(add_videos):
        video_file = glob.glob(os.path.join(video_dir, video + '*'))
        if len(video_file) > 0:
            #print('found')
            if not os.path.isdir(os.path.join(frames_dir, video)):
                os.makedirs(os.path.join(frames_dir, video))
            #print(mix_segments[video])
            for ann in add_segments[video]:
                start = np.ceil(ann[0])
                end = np.floor(ann[1])
                length = (end - start)/2
                #print(length, video)
                # start_time = time.strftime('%H:%M:%S', time.gmtime(int(np.ceil(start))))
                # end_time = time.strftime('%H:%M:%S', time.gmtime(int(np.floor(end))))
                #print(start, end)
                os.system('ffmpeg -ss {} -t {} -i {} -q:v 2 -r 0.5 -f image2 {}/{}_{}_%06d.jpeg >/dev/null 2>&1'.
                      format((start+end)/2, length, video_file[0], os.path.join(frames_dir, video), start, end))
                
            #os.system('ffmpeg -i {} -f image2 %06d.jpeg ')
            #exit(0)
    print(found)
    
    
    
    
if __name__ == '__main__':
    extract_video_frames('../NymbleData/yt_videos', '../NymbleData/yt_frames_add')