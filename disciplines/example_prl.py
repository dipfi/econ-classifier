import multiprocessing as mp
import time
print('Outside before fct:', __name__)

cores = 2 #mp.cpu_count()

def calc_square(x):
    #time.sleep(0.2)
    print('Within Function :', __name__)
    return x**2

print('Outside after fct:', __name__)

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

results = []

if __name__ == '__main__':
    print('Beginning inside :', __name__)
    pool =  mp.Pool(cores)

    results = pool.map(calc_square, [x for x in numbers])
    print('End inside :', __name__)
    pool.close()
    print('Done inside :', __name__)

print(results)
print('all done')