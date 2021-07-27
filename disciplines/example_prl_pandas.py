import multiprocessing as mp
import pandas as pd
import langdetect

print('Outside before fct:', __name__)

def calc_square(x):
    #time.sleep(0.2)
    print('Within Function :', __name__)
    return x + "N"

def detect_languages(text_series):
    language_series = [langdetect.detect(text_series) if text_series.strip() != "" else ""]
    return language_series

print('Outside after fct:', __name__)

def main():
    cores = 2 #mp.cpu_count()
    print(cores)

    text = pd.Series(["hallo welt wie geht es dir ist alles gut und so",
                      "hola mundo como estas vale yo no",
                      "hello world is my life changing for the better anytime soon",
                      "bonjour monde je suis un homme et je suis le mieux"])
    print(text)

    print('Beginning inside :', __name__)
    pool = mp.Pool(cores)
    print(type(text))
    results = pool.map(detect_languages, text)
    print('End inside :', __name__)
    pool.close()
    pool.join()
    print('Done inside :', __name__)
    print(results, " :", __name__)
    print(type(results))
    return results
'''
def myerror():
    print(bla)
'''



if __name__ == '__main__':
    results = main()
    #myerror()

print('all done')

