import os, argparse
from scipy.io import wavfile
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import SyncNetInstance
import avsnap_utils as utils
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--modality', type=str, default='va', 
    help='aa (audio vs. audio) | vv (video vs. video) | av | va | all (i.e audio and video from both videos)')
parser.add_argument('--griffin_lim_iterations', type=int, default=0)
parser.add_argument('--max_diff_smooth', type=float, default=.05, 
    help='maximal difference between smoothed and original signal (seconds) for laplacian smoothing stage')
parser.add_argument('--gaussian_sigma', type=int, default=10, 
    help='sigma for gaussian in second smoothing stage')

parser.add_argument('--prefer_delay', action='store_true', default=False)
parser.add_argument('--save_dp_images', action='store_true', default=False,
                    help='whether to store images showing the dynamic programing weights')

parser.add_argument('vid1', type=str)
parser.add_argument('vid2', type=str)

opt = parser.parse_args()

model = SyncNetInstance.SyncNetInstance()

model.loadParameters(os.path.join(utils.data_dir, "syncnet_v2.model"))
print("Model loaded.")

model.__S__.eval()

max_diff_sync = opt.max_diff_smooth 
gaussian_sigma = opt.gaussian_sigma 


def process_data():
    data = {}

    data['vid1_base_name'] = os.path.basename(opt.vid1)[:-4]
    data['vid2_base_name'] = os.path.basename(opt.vid2)[:-4]

    data['vid1'] = utils.read_vid(opt.vid1)
    data['vid2'] = utils.read_vid(opt.vid2)

    data['aud1'] = utils.read_aud(os.path.join('data/out', data['vid1_base_name'], 'audio.wav'))
    data['aud2'] = utils.read_aud(os.path.join('data/out', data['vid2_base_name'], 'audio.wav'))

    vid1_cropped = os.path.join('data/out', data['vid1_base_name'], 'cropped.avi')
    vid2_cropped = os.path.join('data/out', data['vid2_base_name'], 'cropped.avi')
    if opt.modality == 'all':
        def get_audio_and_video_embeddings(vid, aud):
            vid_emb = utils.vid2emb(vid, model)[1]
            aud_emb = utils.aud2emb(aud, model)[1]
            min_len = min(len(vid_emb), len(aud_emb))
            return [vid_emb[:min_len], aud_emb[:min_len]]

        data['emb1'] = get_audio_and_video_embeddings(vid1_cropped, data['aud1'])
        data['emb2'] = get_audio_and_video_embeddings(vid2_cropped, data['aud2'])

    else:
        if opt.modality[0] == 'v':
            data['emb1'] = utils.vid2emb(vid1_cropped, model)[1]
        else:
            data['emb1'] = utils.aud2emb(data['aud1'], model)[1]
        if opt.modality[1] == 'v':
            data['emb2'] = utils.vid2emb(vid2_cropped, model)[1]
        else:
            data['emb2'] = utils.aud2emb(data['aud2'], model)[1]

    data['spec1'] = utils.aud2spec(data['aud1'])
    data['spec2'] = utils.aud2spec(data['aud2'])
    return data


def write_mix(data, out_folder):
    audio_mix = np.zeros(max(len(data['aud1']), len(data['aud2'])))
    audio_mix[:len(data['aud1'])] += data['aud1']
    audio_mix[:len(data['aud2'])] += data['aud2']
    wavfile.write('tmp.wav', rate=utils.SR, data=audio_mix/audio_mix.max())

    num_frames = max(len(data['vid1']), len(data['vid2']))
    frame_size1 = np.array(data['vid1'][0].shape[:2])
    frame_size2 = np.array(data['vid2'][0].shape[:2])
    pair_frame_size = (max(frame_size1[0], frame_size2[0]), frame_size1[1]+frame_size2[1], 3)
    writer = imageio.get_writer('tmp.mp4', fps=25)
    for i in range(num_frames):
        pair = np.zeros(pair_frame_size, dtype=np.uint8)
        if i < len(data['vid1']):
            pair[:frame_size1[0], :frame_size1[1]] = data['vid1'][i]
        if i < len(data['vid2']):
            pair[:frame_size2[0], frame_size1[1]:frame_size1[1]+frame_size2[1]] = data['vid2'][i]
        writer.append_data(pair)
    writer.close()

    command = (
        'ffmpeg -y -i tmp.mp4 -i tmp.wav -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest %s' %
        os.path.join(out_folder, 'mix.avi'))
    utils.run_command(command)
    os.remove('tmp.wav')
    os.remove('tmp.mp4')


def warp_audio(spec, vid, dp_path, base_file):
    if dp_path.max() >= spec.shape[1]:
        print('flow field reaches outside source for warp')
        spec = np.pad(spec, ((0, 0), (0, int(dp_path.max() - spec.shape[1] + 1))), mode='constant')
    warped_complex_dp = utils.phase_vocoder(spec, dp_path)

    print('applying griffin lim')
    aud_dp_gl = utils.griffin_lim(warped_complex_dp, opt.griffin_lim_iterations)
    print('done')

    warped_aud_file = utils.data_dir + 'tmp_dp.wav'

    wavfile.write(warped_aud_file, rate=utils.SR, data=aud_dp_gl / aud_dp_gl.max())

    command = (
        'ffmpeg -y -i %s -i %s -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest %s' %
        (opt.vid1, warped_aud_file, base_file + '.avi'))
    utils.run_command(command)
    os.remove(warped_aud_file)

    # create baseline stretched audio
    # linear_path = np.linspace(0, spec.shape[1] - 1, dp_path.size)
    # streched_spec = utils.phase_vocoder(spec, linear_path)

    # aud_streched = utils.griffin_lim(streched_spec, opt.griffin_lim_iterations)
    # streched_aud_file = utils.data_dir + 'tmp_streched.wav'
    # wavfile.write(streched_aud_file, rate=utils.SR, data=aud_streched / aud_streched.max())

    # command = (
    #     'ffmpeg -y -i %s -i %s -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -shortest %s' %
    #     (opt.vid1, streched_aud_file, base_file + '_streched' + '.avi'))
    # utils.run_command(command)
    # os.remove(streched_aud_file)



def smooth_flow(path, prefix):

    path = utils.resize_vec(path - np.arange(path.size),
        utils.WINDOWS_PER_FRAME * path.size)

    factor = utils.WINDOWS_PER_SECOND
    max_diff = 0
    sigma = 1
    print ('smooth flow field')
    path_smoothed = path
    while max_diff < max_diff_sync and sigma < 10000:
        sigma *= 1.5 
        path_smoothed = utils.stabilizeSequence(path, sigma)
        max_diff = np.abs(path_smoothed/factor - path/factor).max()
        print ('s',sigma, max_diff)
    print('max smoothness (seconds): ', max_diff)
    print('sigma', sigma)

    path_smoothed = gaussian_filter(path_smoothed, gaussian_sigma)
    print('max path diff in seconds (after smoothing): ', 
        np.abs(path/factor - path_smoothed/factor).max())
    print('(found at)): ', 
        np.argmax(np.abs(path/factor - path_smoothed/factor)))

    plt.figure()
    plt.plot(path / factor)
    plt.plot(path_smoothed / factor)
    plt.savefig(prefix + '_dp_smooth.png', dpi=150)
    print(prefix + '_dp_smooth.png')
    return path_smoothed + np.arange(path_smoothed.size)


def save_dp_images(dp_mat, dp_path, data):
    if opt.modality == 'all':
        dists = [np.linalg.norm(data['emb2'][0][None] - data['emb1'][0][:, None], axis=(2)),
                 np.linalg.norm(data['emb2'][0][None] - data['emb1'][1][:, None], axis=(2)),
                 np.linalg.norm(data['emb2'][1][None] - data['emb1'][0][:, None], axis=(2)),
                 np.linalg.norm(data['emb2'][1][None] - data['emb1'][1][:, None], axis=(2)), ]
        max_shape = np.maximum(np.maximum(dists[0].shape, dists[1].shape),
                               np.maximum(dists[2].shape, dists[3].shape))
        for i in range(4):
            dists[i] = np.pad(dists[i],
                              ((0, max_shape[0] - dists[i].shape[0]),
                               (0, max_shape[1] - dists[i].shape[1])), mode='edge')
        dists = np.minimum(np.minimum(dists[0], dists[1]), np.minimum(dists[2], dists[3]))
    else:
        dists = np.linalg.norm(data['emb2'][None] - data['emb1'][:, None], axis=(2))

    path2_1 = np.array([dp_path[i, 0] for i in range(1, dp_path.shape[0])
                        if dp_path[i, 1] > dp_path[i - 1, 1]])
    path2_1 = np.r_[dp_path[0, 0], path2_1]

    dists_im = dists[dp_path[0, 0]:dp_path[-1, 0] + 1, dp_path[0, 1]:dp_path[-1, 1] + 1]
    dists_im = dists_im / dists_im.max()
    dists_im = np.dstack([dists_im] * 3)

    plt.figure()
    plt.imshow(dp_mat)
    plt.plot(path2_1, 'r')
    plt.savefig(data['out_base_file'] + '_dp_comulative.png', dpi=150)

    plt.figure()
    plt.imshow(dists_im)
    plt.plot(path2_1, 'r')
    plt.savefig(data['out_base_file'] + '_dp.png', dpi=150)


def main():

    data = process_data()

    out_folder = os.path.join(utils.data_dir,
            'out/%s_%s' % (data['vid1_base_name'], data['vid2_base_name']))
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    data['out_base_file'] = os.path.join(out_folder, 'modality_%s' % opt.modality)
    print('out will be saved to ', data['out_base_file'])

    dp_mat, dp_orig = utils.dynamic_programming(data['emb1'], data['emb2'], prefer_delay=opt.prefer_delay)
    path, cost = utils.find_path(dp_mat, dp_orig)

    if opt.save_dp_images:
        save_dp_images(dp_mat, path, data)

    # every audio frame should be associated with a single video frame
    dp_path = [path[0,1]]
    for i in range(1,path.shape[0]):
        if path[i,0] > path[i-1,0]:
            dp_path.append(path[i, 1])
    dp_path = np.array(dp_path) + 2 # embeddings refer to the middle frame
    dp_path = np.concatenate([[0,1], dp_path])
    dp_path = smooth_flow(dp_path, data['out_base_file'])

    write_mix(data, out_folder)
    warp_audio(data['spec2'], data['vid1'], dp_path, data['out_base_file'])


if __name__ == '__main__':
    main()
    utils.FNULL.close()
    print('done')


