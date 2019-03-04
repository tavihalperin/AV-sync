import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import imageio, subprocess, os
import torch
import python_speech_features
import sys

cuda = torch.cuda.is_available()

SR = 16000
FPS = 25
WINDOW_SIZE = 512
OVERLAP = 256
WINDOWS_PER_SECOND = SR/(WINDOW_SIZE-OVERLAP)
WINDOWS_PER_FRAME = WINDOWS_PER_SECOND/FPS

batch_size = 32
smooth_div = 20
relax_end = 10
MAX_WARP = 200

FNULL = open(os.devnull, 'w')

data_dir = 'data/'
def read_aud(audio_path, verbose=True):
    if not audio_path.endswith('.wav'):
        video_path = audio_path
        audio_path = video_path[:-4] + '.wav'
        command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % 
            (video_path, audio_path)) 
        run_command(command, verbose)    
    
    sr, y = wavfile.read(audio_path)
    assert sr == SR
    return y

def read_vid(video_path):

    reader = imageio.get_reader(video_path)
    assert reader.get_meta_data()['fps'] == FPS, 'when changing fps MAKE SURE no new frames are inserted'

    images = []
    try:
        for f in reader:
            images.append(f) 
    except:
        pass

    print ('number of frames in vid:', len(images))
    return images

def calc_mfcc(aud):
    mfcc = zip(*python_speech_features.mfcc(aud, SR))
    return np.stack([np.array(i) for i in mfcc])

def stabilizeSequence(samples, regularization_lambda=30.0):
    length = samples.size
    if length < 6:
        return samples
    
    # find x that solves laplacian_matrix * x = b
    laplacian_matrix = (
        np.diag((1.0 + 2 * regularization_lambda) * np.ones((length - 2,)), 0)
        + np.diag((-1.0 * regularization_lambda) * np.ones((length - 3,)), 1)
        + np.diag((-1.0 * regularization_lambda) * np.ones((length - 3,)), -1))
    
    constraints = np.zeros(length - 2)
    constraints[0] = (regularization_lambda * samples[0])
    constraints[-1] = (regularization_lambda * samples[-1])
    constraints += samples[1:-1]

    x = samples.copy()
    x[1:-1] = np.linalg.solve(laplacian_matrix, constraints)
    return x

def im2tensor(im):
    im = np.expand_dims(np.stack(im,axis=3), axis=0).transpose((0,3,4,1,2))
    return torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

def aud2tensor(aud):
    mfcc = calc_mfcc(aud)
    cc = np.expand_dims(np.expand_dims(mfcc,axis=0),axis=0)
    return torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

def run_batch_vid(model, im, ii, lastframe):
    im_batch = [ im[:,:,vframe:vframe+5,:,:] for vframe in range(ii,min(lastframe,ii+batch_size)) ]
    im_in = torch.cat(im_batch,0)
    if cuda:
        im_in = im_in.cuda()_
    im_out  = model.__S__.forward_lip(im_in)
    return im_out.data.cpu()
    
def run_aud_batch(model, cc, ii, lastframe):
    cc_batch = [ cc[:,:,:,vframe*4:vframe*4+20] for vframe in range(ii,min(lastframe,ii+batch_size)) ]
    cc_in = torch.cat(cc_batch,0)
    if cuda:
        cc_in = cc_in.cuda()_
    cc_out  = model.__S__.forward_aud(cc_in)
    return cc_out.data.cpu()
    
def aud2emb(aud, model):
    if type(aud) == str:
        aud = read_aud(aud)
    lastframe = int(aud.size/SR*FPS) - 6
    y_tensor = aud2tensor(aud)

    cc_feat = torch.cat(
        [run_aud_batch(model, y_tensor, i, lastframe) 
        for i in range(0, lastframe, batch_size)], 0)
    
    return aud, cc_feat.numpy()

def vid2emb(images, model):
    if type(images) == str:
        images = read_vid(images)
    assert np.all(np.array([224,224,3]) == images[0].shape)
    lastframe = len(images) - 6
    images_tensor = im2tensor(images)

    im_feat = torch.cat(
        [run_batch_vid(model, images_tensor, i, lastframe) 
        for i in range(0,lastframe,batch_size)], 
        0)

    return images, im_feat.numpy()

def run_command(command, verbose=True):
    print('runing: ', command if verbose else '...' , end=' ')
    ret =  subprocess.call(command, shell=True, stdout=FNULL, stderr=FNULL)
    if ret :
        print('\nFailed!!!')
        sys.exit()
    else:
        print('.... OK!')

def aud2spec(aud):
    return signal.stft(aud, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[2]

def spec2aud(spec):
    return signal.istft(spec, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[1]


def warp_spec(spec, warp):
    xx, yy = np.meshgrid(np.arange(warp.size), np.arange(spec.shape[0]))
    coordiantes = [yy, warp+xx]
    warped_real = map_coordinates(spec.real, coordiantes, mode='reflect', order=1)
    warped_imag = map_coordinates(spec.imag, coordiantes, mode='reflect', order=1)
    return warped_real + 1j*warped_imag

def data_dist(d1,d2):
    return np.mean((d1-d2)**2)

def cos(d1,d2):
    d1 = d1 / np.linalg.norm(d1)
    d2 = d2 / np.linalg.norm(d2)
    return (d1 @ d2)

def resize_vec(vec, new_length):
    new_length = int(new_length)
    vec = vec * new_length / vec.size
    coordinates = np.linspace(0, vec.size-1, new_length)
    return map_coordinates(vec, [coordinates], order=1)

def minimal_dist(features1, features2, i, j):
    dists = []
    for f1 in features1:
        for f2 in features2:
            dists.append(data_dist(f1[i], f2[j]))
    return np.min(dists)

def prev(ind):
    return max(0, ind-1)

def dynamic_programming(emb1, emb2, max_warp=MAX_WARP, prefer_delay=False):
    if not type(emb1) == list:
        emb1 = [emb1]
        emb2 = [emb2]
    mat = np.ones((emb1[0].shape[0], emb2[0].shape[0])) * np.inf

    mat[0,:relax_end] = 0
    mat[:relax_end,0] = 0

    orig = np.zeros((emb1[0].shape[0], emb2[0].shape[0], 2), dtype=int)

    for i in range(emb1[0].shape[0]):
        diag_i = i*mat.shape[1]//mat.shape[0]
        for j in range(max(0, diag_i-max_warp), min(diag_i+max_warp, mat.shape[1])):
            if prefer_delay:
                data_cost = np.minimum(minimal_dist(emb1, emb2, i, j),
                                       minimal_dist(emb1, emb2, prev(i), j))
            else:
                data_cost = minimal_dist(emb1, emb2, i, j)
            smooth_costs = np.array([1,0,1])/smooth_div
            prev_costs = [mat[i, prev(j)], mat[prev(i),prev(j)], mat[prev(i), j]]

            cost = data_cost + smooth_costs + prev_costs

            argmin = np.argmin(cost)
            orig[i,j] = [[i, prev(j)], [prev(i),prev(j)], [prev(i), j]][argmin]
            mat[i,j] = cost[argmin]
    return mat, orig

def done(i,j):
    if i == 0 and j < relax_end:
        return True
    if j == 0 and i < relax_end:
        return True
    return False

def find_path(dp_mat, dp_orig):
    amin1 = np.argmin(dp_mat[-1, -relax_end:]) + dp_mat.shape[1]-relax_end
    amin2 = np.argmin(dp_mat[-relax_end:, -1]) + dp_mat.shape[0]-relax_end
    path = []
    if dp_mat[-1, amin1] < dp_mat[amin2, -1]:
        i,j = dp_orig.shape[0]-1, amin1
    else:
        i,j = amin2, dp_orig.shape[1]-1
    cost = dp_mat[i,j]
    while not done(i,j):
        path.append([i,j])
        i,j = dp_orig[i,j]
    path.append([i,j])
    path = path[::-1]
    return np.array(path), cost

def griffin_lim(spec, iterations=100):
    mag = np.abs(spec)
    iterations = max(iterations, 1)
    for i in range(iterations):
        wav = signal.istft(spec, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[1]
        phase = np.angle(signal.stft(wav, nperseg=WINDOW_SIZE, noverlap=OVERLAP)[2])
        spec = mag * np.exp(1j* phase)
    return wav

def phase_vocoder(spec, time_steps):
    hop_length=WINDOW_SIZE-OVERLAP

    xx, yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))
    xx = np.zeros_like(xx) 
    coordiantes = [yy, time_steps+xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # Expected phase advance in each bin
    phi_advance = np.linspace(0, np.pi * hop_length, spec.shape[0])

    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):

        # Store to output array
        warped_spec[:, t] *= np.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step+1] - spec_angle[:, step] - phi_advance)

        # Wrap to -pi:pi range
        dphase = np.mod(dphase-np.pi, 2*np.pi)-np.pi

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return warped_spec
