def get_xdv_info(test=False,train=False,train_alter=False):
    " 'test' or 'train' "
    
    if test:
        mp4_paths,mp4_labels,*_ = load_xdv_test()
        txt_fn = '/raid/DATASETS/.zuble/vigia/zurgb00/dataset-xdv-info/test.txt'
    elif train:
        mp4_paths,mp4_labels = load_xdv_train()
        txt_fn = '/raid/DATASETS/.zuble/vigia/zurgb00/dataset-xdv-info/train.txt'
    elif train_alter: 
        mp4_paths,mp4_labels = load_xdv_train_alter()
        txt_fn = '/raid/DATASETS/.zuble/vigia/zurgb00/dataset-xdv-info/train_alter.txt'
    else: raise Exception("not a valid string")
    print(np.shape(mp4_paths))

    frames_arrA , frame_arrBG=[],[]
    data = '';aux=0;total=0;line=''
    for path in mp4_paths:
        fname = os.path.splitext(os.path.basename(path))[0]
        video = cv2.VideoCapture(path)
        frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = video.get(cv2.CAP_PROP_FPS)
        video_time = frames/fps
        video.release()

        if "A" in fname.split("label")[1]:frames_arrA.append(frames)
        else:frame_arrBG.append(frames)
        aux+=1;total+=frames

        line=str(round(frames))+' frames | '+str(round(video_time))+' secs | '+fname+'\n'
        data+=line
        print(line)

    sorted_frame_arrA = sorted(frames_arrA, reverse=True)
    topA_n = sorted_frame_arrA[:20]
    sorted_frame_arrBG = sorted(frame_arrBG, reverse=True)
    topBG_n = sorted_frame_arrBG[:20]
    
    line = "\nmean of frames per video: "+str(total/aux)
    data+=line

    line0 = "\n\ntop_max_frames A\n"
    line1 = ' '.join(map(str, topA_n))
    data+=line0;data+=line1
    
    line0 = "\n\ntop_max_frames BG\n"
    line1 = ' '.join(map(str, topBG_n))
    data+=line0;data+=line1

    f = open(txt_fn, 'w')        
    f.write(data)
    f.close()  


def get_testxdvanom_info():    
    print('\nOPENING annotations',)
    txt = open('/raid/DATASETS/anomaly/XD_Violence/annotations.txt','r')
    txt_data = txt.read()
    txt.close()

    video_list = [line.split() for line in txt_data.split("\n") if line]
    total_anom_frame_count = 0
    for video_j in range(len(video_list)):
        print(video_list[video_j])
        video_anom_frame_count = 0
        for nota_i in range(len(video_list[video_j])):
            if not nota_i % 2 and nota_i != 0: #i=2,4,6...
                aux2 = int(video_list[video_j][nota_i])
                dif_aux = aux2-int(video_list[video_j][nota_i-1])
                total_anom_frame_count += dif_aux 
                video_anom_frame_count += dif_aux
        print(video_anom_frame_count,'frames | ', "%.2f"%(video_anom_frame_count/24) ,'secs | ', int(video_list[video_j][-1]),'max anom frame\n')
    
    total_secs = total_anom_frame_count/24
    mean_secs = total_secs / len(video_list)
    mean_frames = total_anom_frame_count / len(video_list)
    print("TOTAL OF ", "%.2f"%(total_anom_frame_count),"frames  "\
            "%.2f"%(total_secs), "secs\n"\
            "MEAN OF", "%.2f"%(mean_frames),"frames  "\
            "%.2f"%(mean_secs), "secs per video\n")