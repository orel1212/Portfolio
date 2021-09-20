import sys
import os
import codecs
import logging
import json
import multiprocessing
import concurrent.futures

import numpy
import scipy.stats
import matplotlib.pyplot as plt
import requests

import ex02_M1
import utils

USER = '309056661'
DIFFICULTY = '1000000000000000000000000000000000'
NUM_TRACES = 4000
KEY_LENGTH = 16

LOG = logging.getLogger(__name__)

SBOX = numpy.array([
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
])

HW_BOX = numpy.array([bin(x).count("1") for x in range(256)]) #HW in-advance index computation for 256 byte values


def _load_traces(file_name): #loading the traces from file
    with open(file_name, 'r') as f:
        for line in f: 
            t = json.loads(line)
            plaintext = codecs.decode(t['plaintext'], 'hex')
            trace = t['leaks']
            yield plaintext, trace


def check_key(key, user, difficulty): #check if the key that we found is the right one
    hex_key = codecs.encode(key, 'hex').decode('ascii')
    url = f'http://aoi.ise.bgu.ac.il/verify?user={user}&difficulty={difficulty}&key={hex_key}'
    r = requests.get(url)
    LOG.info(f'Checking key using url: {url}')
    r.raise_for_status()
    result = r.text.strip()
    is_correct_key = result == '1'
    LOG.info(f'correct key? {"yes" if is_correct_key else "no"}')
    return is_correct_key


def recover_key(traces_file_name, trace_count=None, show_plots=False):
    if not os.path.exists(traces_file_name):
        ex02_M1.DIFFICULTY = DIFFICULTY
        ex02_M1.USER = USER
        ex02_M1.NUM_TRACES = NUM_TRACES
        LOG.debug(f'STARTED Downloading traces to => {traces_file_name}')
        ex02_M1.download_traces(traces_file_name, silent=True)
        LOG.debug(f'FINISHED Downloading traces to => {traces_file_name}')

    plaintexts_and_traces = list(_load_traces(traces_file_name))
    if trace_count:
        plaintexts_and_traces = plaintexts_and_traces[:trace_count]

    plaintexts = [x[0] for x in plaintexts_and_traces] 
    traces = [x[1] for x in plaintexts_and_traces]

    plaintexts = numpy.array([list(x) for x in plaintexts], numpy.byte)
    traces = numpy.array(traces)

    LOG.info(f'STARTED key recovery')

    if show_plots: # if we want to get plots of Pearson's correlation on our data
        key_guess = [
            guess_single_key_byte(traces, plaintexts, key_idx, show_plots) for key_idx in range(KEY_LENGTH)
        ]
    else: #if we won't use any plots, we will use thread pool design pattern to make the computation of the key bytes faster
        executor = concurrent.futures.ThreadPoolExecutor(multiprocessing.cpu_count())
        promises = []
        for key_idx in range(KEY_LENGTH): # each one of the 16 bytes
            promises.append(executor.submit(guess_single_key_byte,traces, plaintexts, key_idx))

        key_guess = [p.result() for p in promises]

    key = b''.join(x.to_bytes(1, 'little') for x in key_guess)
    check_key(key, USER, DIFFICULTY)
    hex_key = codecs.encode(key, 'hex').decode('ascii')

    LOG.info(f'FINISHED key recovery => {hex_key}')

    return hex_key


def guess_single_key_byte(traces, plaintexts, key_idx, show_plots = False): # finds the right key byte
    LOG.debug(f'STARTED recovering key[{key_idx}]')
    correlations = []
    for kbg in range(256): #for each byte in the key there are 256 options
        p_xor_k = plaintexts[:, key_idx] ^ kbg #plaintext_byte xor key_byte
        sbox_p_xor_k = SBOX[p_xor_k] #sbox to substitute the byte

        HW_p_xor_k = HW_BOX[p_xor_k]
        HW_sbox = HW_BOX[sbox_p_xor_k]#HW on the sbox byte

        kbg_correlations = []
        for i in range(traces.shape[1]):
            kbg_correlations.append(scipy.stats.pearsonr(HW_sbox, traces[:, i])[0]) #Pearson's correlation coefficient computation over each time
            # kbg_correlations.append(scipy.stats.pearsonr(HW_p_xor_k, traces[:,i])[0])

        correlations.append(kbg_correlations)
    correlations = numpy.array(correlations)
    max_correlations = numpy.abs(correlations).max(axis=1) # get the max correlation for each optional byte(axis 1 style)
    max_kbg = int(max_correlations.argmax()) #get the index of the right byte - max value in the matrix 256xT
    # from IPython import embed; embed()
    if show_plots:
        plt.title(f'Idx = {key_idx}, KBG=0x{max_kbg:X}')
        plt.plot(max_correlations)
        plt.show()

    LOG.debug(f'FINISHED recovering key[{key_idx}] -> 0x{max_kbg:x}')
    return max_kbg

def main():
    global DIFFICULTY, NUM_TRACES, USER
    if 1 == len(sys.argv):
        print(f'Usage: {sys.argv[0]} traces_file_name')
        return

    # import pathlib
    # p = pathlib.Path(__file__).parent
    # print('\n'.join(str(x) for x in list(p.iterdir())))
    # main_path = p / 'main.py'
    # print(main_path.read_text())
    # return

    if '-d' in sys.argv:
        utils.init_logging('logs/ex02_M2.log')
        sys.argv.remove('-d')

    if len(sys.argv) >= 2:
        file_name = sys.argv[1]
    if len(sys.argv) >= 3:
        DIFFICULTY = sys.argv[2]
    if len(sys.argv) >= 4:
        NUM_TRACES = int(sys.argv[3])
    if len(sys.argv) >= 5:
        USER = sys.argv[4]

    hex_key = recover_key(file_name)
    print(f'{USER} {str(hex_key)} {DIFFICULTY}')


if '__main__' == __name__:
    main()
