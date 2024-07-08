import os, subprocess , cv2
from essentia.standard import *

'''
from pylab import plot, show, figure, imshow
#%matplotlib inline
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (15, 6) # set plot sizes to something larger than default

def plot_mfcc_melbands(audio):
    
    w = Windowing(type = 'hann')
    spectrum = Spectrum()  # FFT() would return the complex FFT, here we just want the magnitude spectrum
    mfcc = MFCC()
    
    mfccs = []
    melbands = []
    melbands_log = []

    logNorm = UnaryOperator(type='log')
    
    for frame in FrameGenerator(audio, frameSize=1024, hopSize=512, startFromZero=True):
        mfcc_bands, mfcc_coeffs = mfcc(spectrum(w(frame)))
        mfccs.append(mfcc_coeffs)
        melbands.append(mfcc_bands)
        melbands_log.append(logNorm(mfcc_bands))

    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    mfccs = essentia.array(mfccs).T
    melbands = essentia.array(melbands).T
    melbands_log = essentia.array(melbands_log).T

    # and plot
    imshow(melbands[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Mel band spectral energies in frames")
    show()

    imshow(melbands_log[:,:], aspect = 'auto', origin='lower', interpolation='none')
    plt.title("Log-normalized mel band spectral energies in frames")
    show()

    imshow(mfccs[1:,:], aspect='auto', origin='lower', interpolation='none')
    plt.title("MFCCs in frames")
    show()
'''

def print_acodec_from_mp4(data,printt=False,only_sr=False):
    out=[]
    for i in range(len(data)):
        output = subprocess.check_output('ffprobe -hide_banner -v error -select_streams a:0 -show_entries stream=codec_name,channels,sample_rate,bit_rate -of default=noprint_wrappers=1 -of compact=p=0:nk=1 '+str('"'+data[i]+'"'), shell=True)
        output = str(output).replace("\\n","").replace("b","").replace("'","").splitlines()[0]
        out.append(output)
        if printt:print(output)
    #os.system('ffprobe -hide_banner -show_streams -select_streams a:0 -i '+str('"'+test_fn[1]+'"'))
    #os.system('ffprobe -hide_banner '+str('"'+test_fn[1]+'"'))
    if only_sr:return int(str(out[0]).split('|')[1])
    else: return out