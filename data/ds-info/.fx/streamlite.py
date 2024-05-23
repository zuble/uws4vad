# %%
import streamlit as st
import os
import numpy as np

# %%
SERVER_TEST_PATH = '/raid/DATASETS/anomaly/XD_Violence/testing_copy'
SERVER_TEST_PATH_RWF = '/raid/DATASETS/anomaly/RWF-2000'
SERVER_TEST_AUD_ORIG_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/original'
SERVER_TEST_AUD_MONO_PATH = '/raid/DATASETS/anomaly/XD_Violence/aud/testing/mono'



# %%
def load_xdv_test(accc_path=SERVER_TEST_AUD_MONO_PATH):
    mp4_paths_xdv, mp4_labels = [],[]
    for root, dirs, files in os.walk(SERVER_TEST_PATH):
        for file in files:
            if file.find('.mp4') != -1:
                mp4_paths_xdv.append(os.path.join(root, file))
    mp4_paths_xdv.sort()
    for i in range(len(mp4_paths_xdv)):            
        if 'label_A' in  file:mp4_labels.append(0)
        else:mp4_labels.append(1)
    

    aac_paths, aac_labels = [],[]                            
    for root, dirs, files in os.walk(accc_path):
        for file in files:
            if file.find('.aac') != -1:
                aac_paths.append(os.path.join(root, file))
    aac_paths.sort()
    for i in range(len(aac_paths)):               
        if 'label_A' in  file:aac_labels.append(0)
        else:aac_labels.append(1)                
    
    return mp4_paths_xdv,mp4_labels,aac_paths,aac_labels


def load_rwf_test():
    
    rwf_mp4_path_all , rwf_mp4_path_train_fight , rwf_mp4_path_train_nofight , rwf_mp4_path_val_fight ,rwf_mp4_path_val_nofight = [],[],[],[],[]
    
    for root, dirs, files in os.walk(os.path.join(SERVER_TEST_PATH_RWF,'train/Fight')):
        for file in files:
            if file.find('.avi') != -1:
                rwf_mp4_path_train_fight.append(os.path.join(root, file))
                
    for root, dirs, files in os.walk(os.path.join(SERVER_TEST_PATH_RWF,'train/NonFight')):
        for file in files:
            if file.find('.avi') != -1:
                rwf_mp4_path_train_nofight.append(os.path.join(root, file))
                
    for root, dirs, files in os.walk(os.path.join(SERVER_TEST_PATH_RWF,'val/Fight')):
        for file in files:
            if file.find('.avi') != -1:
                rwf_mp4_path_val_fight.append(os.path.join(root, file))
                
    for root, dirs, files in os.walk(os.path.join(SERVER_TEST_PATH_RWF,'val/NonFight')):
        for file in files:
            if file.find('.avi') != -1:
                rwf_mp4_path_val_nofight.append(os.path.join(root, file))  
                      
    rwf_mp4_path_all =  rwf_mp4_path_train_fight + rwf_mp4_path_train_nofight + rwf_mp4_path_val_fight + rwf_mp4_path_val_nofight               
    
    return rwf_mp4_path_all , rwf_mp4_path_train_fight , rwf_mp4_path_train_nofight , rwf_mp4_path_val_fight ,rwf_mp4_path_val_nofight


mp4_paths_xdv,mp4_labels,aac_paths,aac_labels = load_xdv_test()
rwf_mp4_path_all , rwf_mp4_path_train_fight , rwf_mp4_path_train_nofight , rwf_mp4_path_val_fight ,rwf_mp4_path_val_nofight = load_rwf_test()
print(np.shape(rwf_mp4_path_all))
video = True


# %% MP4 VIEWER

if video:
    
    #XD_VIOLENCE
    # strip paths to have video name per labels
    xdv_mp4_path_all_windex, xdv_mp4B1fight_windex, xdv_mp4B2shoot_windex, xdv_mp4B4riot_windex, xdv_mp4B5abuse_windex, xdv_mp4B6caracc_windex, xdv_mp4Gexplosion_windex = [],[],[],[],[],[],[]
    
    for i in range(len(mp4_paths_xdv)):
        
        mp4_label_strip = os.path.splitext(os.path.basename(mp4_paths_xdv[i]))[0].split('label')[1]
        
        xdv_mp4_path_all_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])) )
        
        if 'B1' in mp4_label_strip: xdv_mp4B1fight_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])))
        if 'B2' in mp4_label_strip: xdv_mp4B2shoot_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])))
        if 'B4' in mp4_label_strip: xdv_mp4B4riot_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])))
        if 'B5' in mp4_label_strip: xdv_mp4B5abuse_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])))
        if 'B6' in mp4_label_strip: xdv_mp4B6caracc_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])))
        if 'G' in  mp4_label_strip: xdv_mp4Gexplosion_windex.append(str(i)+' '+str(os.path.basename(mp4_paths_xdv[i])))
           
    with st.sidebar:

        anom_type = st.selectbox(
            'XDViolenve',
            ('ALL','B1 - FIGHT', 'B2 - SHOOT', 'B4 - RIOT','B5 - ABUSE','B6 - CAR ACIDENTE','G - EXPLSION')
        )
        if anom_type == 'ALL':file_sel_xdv = st.radio('ALL', xdv_mp4_path_all_windex)
        if anom_type == 'B1 - FIGHT':file_sel_xdv = st.radio('B1 - FIGHT',xdv_mp4B1fight_windex)
        if anom_type == 'B2 - SHOOT':file_sel_xdv = st.radio('B2 - SHOOT',xdv_mp4B2shoot_windex)
        if anom_type == 'B4 - RIOT':file_sel_xdv = st.radio('B4 - RIOT',xdv_mp4B4riot_windex)
        if anom_type == 'B5 - ABUSE':file_sel_xdv = st.radio('B5 - ABUSE',xdv_mp4B5abuse_windex)
        if anom_type == 'B6 - CAR ACIDENTE':file_sel_xdv = st.radio('B6 - CAR ACIDENTE',xdv_mp4B6caracc_windex)
        if anom_type == 'G - EXPLSION':file_sel_xdv = st.radio('G - EXPLSION',xdv_mp4Gexplosion_windex)


        rwf_sel = st.selectbox(
            'RWF-2000',
            ('ALL','TRAIN FIGHT', 'TRAIN NO FIGHT', 'VAL FIGHT','VAL NO FIGHT')
        )
        if rwf_sel == 'ALL' :            file_sel_rwf = st.radio('ALL', rwf_mp4_path_all)
        if rwf_sel == 'TRAIN FIGHT' :    file_sel_rwf = st.radio('TRAIN FIGHT', rwf_mp4_path_train_fight)
        if rwf_sel == 'TRAIN NO FIGHT' : file_sel_rwf = st.radio('TRAIN NO FIGHT', rwf_mp4_path_train_nofight)
        if rwf_sel == 'VAL FIGHT' :      file_sel_rwf = st.radio('VAL FIGHT', rwf_mp4_path_val_fight)
        if rwf_sel == 'VAL NO FIGHT' :   file_sel_rwf = st.radio('VAL NO FIGHT', rwf_mp4_path_val_nofight)
     
 
    st.write(file_sel_xdv,file_sel_rwf)
    
    
    with st.columns(2):
        video_fn_xdv = file_sel_xdv.split(' ')[1]
        video_file_xdv = open(SERVER_TEST_PATH+'/'+video_fn_xdv, 'rb')
        video_bytes_xdv = video_file_xdv.read() #reading the file
        st.video(video_bytes_xdv) #displaying the video

        video_fn_rwf = file_sel_rwf.split(' ')[1]
        video_file_rwf = open(SERVER_TEST_PATH_RWF+'/'+video_fn_rwf, 'rb')
        video_bytes_rwf = video_file_rwf.read() #reading the file
        st.video(video_bytes_rwf) #displaying the video

else:
    
    with st.sidebar:
        file_sel_xdv = st.radio('ALL',aac_paths)
        
    st.write(file_sel_xdv)


    aac_file = open(file_sel_xdv, 'rb')
    aac_bytes = aac_file.read() #reading the file
    st.audio(aac_bytes, format='audio/aac')