import os
import shutil


def clean_tmp():
    path_to_tests = os.path.dirname(os.path.realpath(__file__))
    TMP = os.path.join(path_to_tests, 'data/tmp/')

    if os.path.exists(TMP):
        shutil.rmtree(TMP)
