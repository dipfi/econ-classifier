import multiprocessing as mp

print('Outside before fct:', __name__)

def calc_square(x):
    #time.sleep(0.2)
    print('Within Function :', __name__)
    return x**2

print('Outside after fct:', __name__)

def main():
    cores = mp.cpu_count()
    print(cores)

    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    print('Beginning inside :', __name__)
    pool = mp.Pool(cores)

    results = pool.map(calc_square, numbers)
    print('End inside :', __name__)
    pool.close()
    pool.join()
    print('Done inside :', __name__)
    print(results, " :", __name__)
    return results

def myerror():
    print(bla)


if __name__ == '__main__':
    results = main()
    myerror()

print('all done')

