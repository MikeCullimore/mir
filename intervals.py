import matplotlib.pyplot as plt
import numpy as np

def main():
    # todo: understand/derive/explain these numbers.
    # todo: add x labels unison, minor second etc.
    just = [1, 16/15, 9/8, 6/5, 5/4, 4/3, 45/32, 3/2, 8/5, 5/3, 9/5, 15/8, 2]
    pythag = [1, 2**8/3**5, 3**2/2**3, 2**5/3**3, 3**4/2**6, 2**2/3, 2**10/3**6, 3/2, 2**7/3**4, 3**3/2**4, 2**4/3**2, 3**5/2**7, 2]
    equal = np.logspace(start=0, stop=12, base=2**(1/12), num=13)

    # todo: plot differences from equal temperament.
    # diff_j = equal - just
    # diff_p = equal - pythag
    
    plt.figure()
    plt.plot(just, label='Just intonation')
    plt.plot(pythag, label='Pythagorean')
    plt.plot(equal, label='Equal temperament')
    plt.xlabel('Interval')
    plt.ylabel('Ratio')
    plt.legend()
    plt.show()
    # plt.savefig('Intervals in different tuning systems.png')

if __name__ == '__main__':
    main()