import sys
import random
import subprocess
import os
import numpy as np
import time

def main():

    FNULL = open(os.devnull, 'w')
    s = time.time()

    r = int(sys.argv[1])

    scores = []
    for it in range(r):
        seed = random.randint(0, 10000)
        seed1 = random.randint(0, 10000)
        proc = subprocess.Popen(['python','grading.py', str(seed), str(seed1)], stdout=subprocess.PIPE, stderr=FNULL)
        stdout = proc.communicate()[0]

        seed, seed1, score = stdout.splitlines()[-1].split(',')

        print(it, seed, seed1, score)
        scores.append(int(score))

    print('average', np.mean(scores), 'std', np.std(scores), 'min', np.min(scores), 'max', np.max(scores), 'time', time.time() - s)


if __name__ == '__main__':
    main()