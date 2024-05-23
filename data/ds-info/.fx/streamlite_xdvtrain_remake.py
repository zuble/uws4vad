import streamlit as st
import os , time , subprocess
import numpy as np
import pandas as pd
import plotly.express as px



# ----------------------------------------------------------------- #
## GLOBAL VARS
SERVER_TRAIN_COPY_PATH = '/raid/DATASETS/anomaly/XD_Violence/training_copy'
SERVER_TEST_COPY_PATH =  '/raid/DATASETS/anomaly/XD_Violence/testing_copy'

OUT_BASEPATH_4CUTTER= "/raid/DATASETS/anomaly/XD_Violence/training_copy_alter"
OUT_BASEPATH_4CUTTER_A = OUT_BASEPATH_4CUTTER+'/A'
OUT_BASEPATH_4CUTTER_BG = OUT_BASEPATH_4CUTTER+'/BG'
OUT_BASEPATH_4CUTTER_BG_ALTER = OUT_BASEPATH_4CUTTER+'/BG_alter'

# ----------------------------------------------------------------- #
## SINet FX

def ffprobe_from_mp4(data,printt=False,only_aud_sr=False,only_vid_fps=False):
    ##https://trac.ffmpeg.org/wiki/FFprobeTips#FrameRate
    out,out2=[],[]
    for i in range(len(data)):
        
        ## aud stuff
        output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 '+str('"'+data[i]+'"'), shell=True)
        output = str(output).replace("\\n","").replace("b","").replace("'","").splitlines()[0]
        out.append(output)
        
        ## vid fps
        output2 = subprocess.check_output('ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 '+str('"'+data[i]+'"'), shell=True)
        output2 = str(output2).replace("\\n","").replace("b","").replace("'","").replace("/1","")
        out2.append(output2)
        
        if printt: print(output,output2)
    
    if only_aud_sr:     return int(str(out[0]).split('|')[1])
    elif only_vid_fps:  return int(out2[0])
    else:               return out,out2

# wont work cause steamlite env is py3.11
def FSDSINet(path):
    cmd  ="python3 sinet.py "+str(path)
    os.system(cmd)


# ----------------------------------------------------------------- #
## FFMPEG CUTTER FX
## http://trac.ffmpeg.org/wiki/Seeking
## https://www.arj.no/2018/05/18/trimvideo/
## https://stackoverflow.com/questions/18444194/cutting-the-videos-based-on-start-and-end-time-using-ffmpeg

def video_cutter(in_path,out_path,ss,t,cutdr=True):
    
    print("\n\n******************************* CUUUUUUTTTTTTTTTT ********************************")
    
    in_fn = os.path.basename(in_path)
    print('IN\n',in_fn)
    print(ffprobe_from_mp4([in_path]),'\n')
    
    ## CMD1
    cmd = (f"ffmpeg -ss {ss} -i {in_path} -t {t} "
        f"-c:v copy -c:a copy {out_path}")
    print("\n",cmd)

    if not cutdr: os.system(cmd)
    
    return cmd



# ----------------------------------------------------------------- #
def is_vfn_cut(path):
    ''' return True if video is already cut '''
    fn = os.path.basename(path).split("__")[0]
    paths,fns_cut = get_vpath_cutter_fldr()
    for fn_cut in fns_cut:
        if fn in fn_cut.split("__")[0]:return True
    return False


## GET PATHS FX
def get_vpath_cutter_fldr(auxA=False,auxBG=False):
    patths,fn,fn_id = [],[],[]
    for root, dirs, files in os.walk(OUT_BASEPATH_4CUTTER):
        for file in files:
            if file.find('.mp4') != -1:
                patths.append(os.path.join(root, file))
                fn.append(file)
                fn_id.append(file.split("__")[0])
        break
                
    if auxA:
        for root, dirs, files in os.walk(OUT_BASEPATH_4CUTTER_A):
            for file in files:
                if file.find('.mp4') != -1:
                    patths.append(os.path.join(root, file))
                    fn.append(file)
                    fn_id.append(file.split("__")[0])
                    
    if auxBG:
        for root, dirs, files in os.walk(OUT_BASEPATH_4CUTTER_BG):
            for file in files:
                if file.find('.mp4') != -1:
                    patths.append(os.path.join(root, file))
                    fn.append(file)
                    fn_id.append(file.split("__")[0])
                    
    patths.sort()
    fn.sort()
    fn_id.sort()

    return patths,fn,fn_id


def get_vpath_totfra_fromxdvstatstxt(train_or_test):
    filenames,total_frames = [],[]
    
    if train_or_test == 'train':
        basepath = SERVER_TRAIN_COPY_PATH
        fpp = '/raid/DATASETS/.zuble/vigia/zurgb/dataset-xdv-info/train_sort.txt'
        stopper = 3953
    elif train_or_test == 'test':
        basepath = SERVER_TEST_COPY_PATH
        fpp = '/raid/DATASETS/.zuble/vigia/zurgb/dataset-xdv-info/test_sort.txt'
        stopper = 799
    else: Exception(" 'train' or 'test' ")
    
    with open(fpp, 'r') as f:
        lines = f.readlines()
    
    count=0
    for line in lines:
        if count == stopper: break
        filename = line.rstrip().split(' | ')[-1]  # Get the last element of the split line, which is the filename
        total_frame= line.rstrip().split(' | ')[0].strip("frames")
        filenames.append(os.path.join(basepath,filename+'.mp4'))
        total_frames.append(total_frame)
        count+=1
    #print(filenames)
    return filenames,total_frames

def get_paths(frame_max = 8643):
    ## GET ORIGINALS VIDEO PATHS , FNS , TOTALFRAMES
    mp4_paths,total_frames = get_vpath_totfra_fromxdvstatstxt('train')

    mp4_fns = []
    for i in range(len(mp4_paths)):
        mp4_fns.append(mp4_paths[i].replace("/raid/DATASETS/anomaly/XD_Violence/training_copy/",""))
    
    orig_all_sorted={
        "paths":        mp4_paths,
        "total_frames": total_frames,
        "fns":          mp4_fns
    }    
    
    ## GET FROM ORIGINALS VIDEOS W MORE THAN 4000 FRAMES    
    mp4_paths_wmoreN , total_frames_wmoreN , mp4_fn_wmoreN = [],[],[]
    for i in range(len(total_frames)):
        if int(total_frames[i]) > frame_max:
            mp4_paths_wmoreN.append(mp4_paths[i])
            total_frames_wmoreN.append(total_frames[i])
            mp4_fn_wmoreN.append( mp4_paths[i].replace("/raid/DATASETS/anomaly/XD_Violence/training_copy/","") )
        else:break
        
    print("LAST MP4WMORE4000",mp4_paths_wmoreN[-1],total_frames_wmoreN[-1])
    print("\nshape wmoreframe_max",np.shape(mp4_paths_wmoreN),total_frames_wmoreN[-1])
    
    orig_wmoreN_sorted={
        "paths":        mp4_paths_wmoreN,
        "total_frames": total_frames_wmoreN,
        "fns":          mp4_fn_wmoreN        
    }

    return orig_all_sorted , orig_wmoreN_sorted

orig_all_sorted , orig_wmoreN_sorted = get_paths()

print(orig_all_sorted["fns"][246])


def is_vfnid_in_alterfldr(path):
    ''' return True if video is already cut '''
    fn = os.path.basename(path).split("__")[0]
    paths,fns_cut,fns_cut_id = get_vpath_cutter_fldr()
    for fn_cut in fns_cut:
        if fn in fn_cut.split("__")[0]:return True
    return False


# ----------------------------------------------------------------- #
# copy all label 0 with <= 8643 frames from training_copy in2 training_copy_alter

def lets_finish_this(printt=False):

    if printt: print(np.shape(orig_all_sorted["paths"]),np.shape(orig_all_sorted["total_frames"]),np.shape(orig_all_sorted["fns"]))
    
    paths_cut,fns_cut,fns_cut_id = get_vpath_cutter_fldr()
    if printt: print("alter_flds_len",np.shape(paths_cut),np.shape(fns_cut))
    
    orig_vid_processed , orig_vid_2beprocessed_labelA , orig_vid_2beprocessed_labelBG = [],[],[]
    for i in range(len(orig_all_sorted["paths"])):
        
        fn_id_orig = os.path.basename(orig_all_sorted["paths"][i]).split("__")[0]
        fn_label_orig = os.path.basename(orig_all_sorted["paths"][i]).split("__")[1]

        ## this files i already cut or cp
        if int(orig_all_sorted["total_frames"][i]) > 8643:
            
            orig_vid_processed.append(orig_all_sorted["paths"][i])
            
            ## NN excluded
            if not is_vfnid_in_alterfldr(orig_all_sorted["paths"][i]):
                if printt:print("NN",i,orig_all_sorted["total_frames"][i],orig_all_sorted["fns"][i],'\n')
            
            ## processed and chosen
            else:
                if printt: print(i,is_vfnid_in_alterfldr(orig_all_sorted["paths"][i]),orig_all_sorted["total_frames"][i],orig_all_sorted["fns"][i])
                
                cp_flag = True
                for j in range(len(fns_cut)):
                    if fn_id_orig in fns_cut_id[j] and orig_all_sorted["fns"][i] != fns_cut[j]:
                        if printt:print('\talter@',fns_cut[j]);cp_flag=False
                
                if cp_flag and printt:print("\tCP\n")
                elif printt:print("\tCUT\n")
            
            
        ## this are files i do not processed && are LABEL A  
        elif 'label_A' in fn_label_orig:
            
            orig_vid_2beprocessed_labelA.append(orig_all_sorted["paths"][i])
            
            ## id collision from atual orig vfn and all vfnid in alter
            if is_vfnid_in_alterfldr(orig_all_sorted["paths"][i]):
                
                if printt: print(i,"A ",orig_all_sorted["fns"][i] ,orig_all_sorted["total_frames"][i])
                
                for j in range(len(fns_cut)):
                    if fn_id_orig in fns_cut_id[j] and printt:
                        print('\talter@',fns_cut[j])
                
                ## check for name collsions
                if orig_all_sorted["fns"][i] in fns_cut and printt: print("\tCOLLISION")

                if printt: print("\n")
            
            ## no id collision    
            elif printt:
                print(i,"A ", orig_all_sorted["fns"][i], orig_all_sorted["total_frames"][i])
                print("\tNEW\n")
        
        ## this are files not processed && LABEL BG        
        else:
            orig_vid_2beprocessed_labelBG.append(orig_all_sorted["paths"][i])
            
            ## id collision from atual orig vfn and all vfnid in alter
            if is_vfnid_in_alterfldr(orig_all_sorted["paths"][i]):
                
                if printt: print(i,"BG ",orig_all_sorted["fns"][i],orig_all_sorted["total_frames"][i])
                
                for j in range(len(fns_cut)):
                    if fn_id_orig in fns_cut_id[j] and printt:
                        print('\talter@',fns_cut[j])   
                        
                ## check for name collsions
                if orig_all_sorted["fns"][i] in fns_cut and printt: print("\tCOLLISION")

                if printt: print("\n")
            
            ## no id collision    
            elif printt:
                print(i,"BG ", orig_all_sorted["fns"][i], orig_all_sorted["total_frames"][i])
                print("\tNEW\n")
                
    return orig_vid_processed , orig_vid_2beprocessed_labelA , orig_vid_2beprocessed_labelBG


orig_vid_processed , orig_vid_2beprocessed_labelA , orig_vid_2beprocessed_labelBG = lets_finish_this()
print(np.shape(orig_vid_processed),np.shape(orig_vid_2beprocessed_labelA),np.shape(orig_vid_2beprocessed_labelBG))

# ----------------------------------------------------------------- #
## CREATE tabs
#tabs = ["FFMPEG CUTTER", "XDV TRAIN STATS"]
#selected_tab = st.selectbox("tap a tab", tabs)


# ----------------------------------------------------------------- #
## VIDEO PLAYER && FFMPEG CUTTER
#if selected_tab == "FFMPEG CUTTER":
    
def start():
    
    def time_to_seconds(time_str):
        timee = str(time_str).split(':')
        
        if len(timee) == 2: hours = 00; minutes=int(timee[0]); seconds=int(timee[1]) 
        if len(timee) == 3: hours = int(timee[0]); minutes=int(timee[1]); seconds=int(timee[2])   
        
        total_seconds = (hours * 3600) + (minutes * 60) + seconds
        return total_seconds

    def seconds_to_time(seconds):
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def frames_to_timecode(frames, fps):
        total_seconds = (frames / fps)+1
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return '{:02d}:{:02d}:{:02d}'.format(int(hours), int(minutes), int(seconds))
    

    def get_cut_fn_from_start(start,frame_w_interval,vid_fps,vpath,printt=False):
        ''' start hh:mm:ss '''

        ## takes care os inital 0's in hh:mm
        def refine_start_str(s):
            ss = str(s).split(':')
            if (len(ss))==2:
                if int(ss[0]) < 10:
                    s='00:0'+s
            if (len(ss))==3:
                if int(ss[0]) < 10:
                    s='0'+s                
            return s.replace(':','-')

    
        vfn = os.path.basename(vpath) 
        print('\n#______________________________#')
        print("get_cut_fn_from_start\nvorig_fn"+vfn)
        
        window_len_secs = int(frame_w_interval/vid_fps)+1
        window_len_timecode = seconds_to_time(window_len_secs)
        print("wlen",window_len_secs,window_len_timecode)
        
        ss = start_secs = time_to_seconds(start)
        to = ss + window_len_secs
        print("start_secs",start_secs,"ss",ss,"to",to)
        
        if '__#1_' in vfn:
            print("\ninser time interval in name")
        
            vorig_fnid,vorig_label = vfn.split("__#1")
            if printt: print(vorig_fnid,vorig_label)
            
            vcut_tc_start = refine_start_str(start)
            if printt: print("vcut_tc_start",vcut_tc_start)
            
            vcut_secs_end = start_secs + window_len_secs
            vcut_tc_end = seconds_to_time(vcut_secs_end).replace(":","-")
            if printt: print(vcut_secs_end,vcut_tc_end)
            
            vcut_fn = vorig_fnid+'__#'+vcut_tc_start+'_'+vcut_tc_end+vorig_label
            print("vcut_fn",vcut_fn)
            
        else:
            print("\nadd time interval to fn")
            
            vorig_fnid,vorig_tclabel = vfn.split("__#")
            vorig_tc,vorig_label = vorig_tclabel.split("_label_")
            if printt: print(vorig_tc,vorig_label)
            
            vorig_tc_start,vorig_tc_end = vorig_tc.split("_")
            if printt: print("oric_tc_start",vorig_tc_start)
            if printt: print("oric_tc_end",vorig_tc_end)
            

            vorig_secs_start = time_to_seconds(vorig_tc_start.replace("-",":"))
            vcut_secs_start = vorig_secs_start + start_secs
            
            vcut_tc_start = seconds_to_time(vcut_secs_start).replace(":","-")
            if printt: print("vcut_tc_start",vcut_tc_start)
            
            vcut_secs_end = vcut_secs_start + window_len_secs
            vcut_tc_end = seconds_to_time(vcut_secs_end).replace(":","-")
            if printt: print("vcut_tc_end",vcut_tc_end)
            
            vcut_fn = vorig_fnid+'__#'+vcut_tc_start+'_'+vcut_tc_end+'_label_'+vorig_label
            print("vcut_fn",vcut_fn)
            
        
        return vcut_fn , vcut_tc_start , vcut_tc_end ,  window_len_secs , window_len_timecode , ss , to
    
    
    ## SIDE BAR FILE LIST
    with st.sidebar:
        option = st.selectbox(
        'which file list',
        ('orig_vid_processed' , 'orig_vid_2beprocessed_labelA' , 'orig_vid_2beprocessed_labelBG'))
        if option == 'orig_vid_processed': filepath_sel_xdv = st.radio('ALL',orig_vid_processed)
        if option == 'orig_vid_2beprocessed_labelA': filepath_sel_xdv = st.radio('ALL',orig_vid_2beprocessed_labelA)
        if option == 'orig_vid_2beprocessed_labelBG': filepath_sel_xdv = st.radio('ALL',orig_vid_2beprocessed_labelBG) 
        

    i = orig_all_sorted["paths"].index(filepath_sel_xdv)
    sel_xdv_totframes = orig_all_sorted["total_frames"][i]
    vid_fps = ffprobe_from_mp4([filepath_sel_xdv],only_vid_fps=True)
    
    
    st.write("B1 FIGHT ; B2 SHOOTING ; B4 RIOT ; B5ABUSE ; B6 CARACC ; G EXPLOS")
    st.write(filepath_sel_xdv)
    st.write(sel_xdv_totframes,' frames @',vid_fps,'fps')
    #st.write("#####"+" is this video already in alter fldr ?",str(is_vfn_cut(filepath_sel_xdv)))        
    
    ## VIDEO PLAYER
    v = open(filepath_sel_xdv, 'rb')
    vread = v.read() #reading the file
    video_player = st.video(vread) #displaying the video


    ## FRaME_MAX SLIDER && TIME INTERVALS
    usr_in_fmax = st.slider(label='f_max',min_value=0, max_value=43800, step=100,value=4000)
    #st.write("#####",usr_in_fmax," ",frames_to_timecode(usr_in_fmax,vid_fps))
    
    usr_in_start = st.text_input("write START hh:mm:ss")
    if ( usr_in_start or usr_in_fmax ) and usr_in_start:
        
        vcut_fn , vcut_tc_start , vcut_tc_end ,  window_len_secs , window_len_timecode , ss , to =  get_cut_fn_from_start(usr_in_start.replace(" ",""),usr_in_fmax,vid_fps,filepath_sel_xdv,printt=True)
        
        st.write("#####"+" W @ ",usr_in_fmax," ",window_len_timecode)
        st.write("#####"+" ENDD @ ", seconds_to_time(to))
        st.write("#####"+" ORIG @ "+os.path.basename(filepath_sel_xdv))
        st.write("#####"+" CUTT @ "+vcut_fn)
        st.write("#### ["+str(vcut_tc_start)+' , '+str(vcut_tc_end)+']')
        
        out_path = os.path.join(OUT_BASEPATH_4CUTTER_BG_ALTER,vcut_fn)
        
    
    ## FFMPEG dryrun    
    if st.button("FAKE CUT"):
        cmd=video_cutter(filepath_sel_xdv,out_path,ss,to)
        st.write(cmd)
    
    ## FFMPEG true
    if st.button("TRUE CUT"):
        cmd=video_cutter(filepath_sel_xdv,out_path,ss,to,cutdr=False)
        st.write(cmd) 

    ## JUST COPY TO ALLTER
    if st.button("CP"):
        cmd = (f'cp {filepath_sel_xdv} {out_path}')
        #os.system(cmd)
        st.write(cmd)
        
    ## ALTER VIEWER AND INSANITY CHECKER
    #col1,col2=st.columns(2)
    #with col1:
    #    st.write("## CUTTED")
    #    paths_cutterfld,fn_cutterfld = get_vpath_cutter_fldr()
    #    fn_cut_sel = st.selectbox('_',fn_cutterfld)
    #    #st.write(fn_cut_sel) 
    #        
    #    path_cut_sel= paths_cutterfld[fn_cutterfld.index(fn_cut_sel)]    
    #    
    #    v2 = open(path_cut_sel, 'rb')
    #    vread2 = v2.read() #reading the file
    #    video_player2 = st.video(vread2) #displaying the video
    #    
    #    ## SINET PREDICT
    #    if st.button("SINet predict"):
    #        #FSDSINet(file_xdv_cut)
    #        st.write("llll")
    
            
start()


# ----------------------------------------------------------------- #
## XDV TRAIN STATS
#else:
#
#    step_frame_len = 4000
#    data = {
#        "fn":[],
#        "total_f":[]
#    }
#    for i in range(len(orig_all_sorted["fns"])):
#        data["fn"].append(orig_all_sorted["fns"][i])
#        data["total_f"].append(int(orig_all_sorted["total_frames"][i]))
#    df = pd.DataFrame(data)
#
#
#    ## PREPARATION 
#    # create bins based on the total_frames column
#    bins = np.arange(0, int(df['total_f'].max()) + step_frame_len, step_frame_len)
#    labels = [f'{i}-{i+(step_frame_len-1)}' for i in range(0, int(df['total_f'].max())+1, step_frame_len)]
#    df['frame_interval'] = pd.cut(df['total_f'], bins=bins, labels=labels, include_lowest=True)
#
#
#    ## PLOTTY CHART
#    # group the data by frame_interval and count the number of mp4_fn_wmore4000 in each group
#    counts2 = df.groupby("frame_interval")["fn"].count().reset_index(name="count")
#    # create a histogram using Plotly Express
#    fig = px.histogram(counts2, x="frame_interval", y="count", nbins=len(labels),
#                    labels={"frame_interval": "Total Frames", "count": "Number of Files"},
#                    title="Distribution of Files by Total Frames")
#    fig.update_xaxes(categoryorder="array", categoryarray=labels)  # set the category order for x-axis
#    # display the histogram using Streamlit
#    st.plotly_chart(fig, use_container_width=True)
#
#
#    ## SIMPLE TABLE with fn | total frames
#    df2 = pd.DataFrame(data)
#    st.write(df2.set_index("fn"), use_container_width=True)
