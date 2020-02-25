import subprocess, os
import matplotlib.pyplot as plt
from utils.printProgressBar import printProgressBar

def AddFig2Movie(fig, i, every_n_frames, nframes):
    printProgressBar(i, nframes-1, prefix = 'Progress:', suffix = 'Complete', length=50)
    if i % every_n_frames == 0:
        plt.savefig('{}{}'.format('frame_', i), dpi=100, transparent=False, pad_inches=0.1)
    plt.close(fig)

    if i==nframes-1:
        bashCommand = "ffmpeg -pattern_type glob  -f image2 -r 1/1 -i frame* -vcodec mpeg4 output.mp4"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
    
        files = os.listdir()
        for i in files:
            if 'frame_' in i:
                os.remove(i)