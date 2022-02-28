# run this script with the command
# $ pytest -q unit_tests.py
import os

from tests.unit.test_bullet_utility import TestBulletUtility


if __name__ == "__main__":
    print("Start running unit tests ...")
    os.system('pytest -q unit_tests.py')