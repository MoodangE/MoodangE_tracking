import pandas as pd
import warnings

# Adjusting for Data Frame Output
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

warnings.filterwarnings('ignore')


def make_corpora_tfid():
    ai = ''
    with open("part_location_data_ver2/AI.txt", 'r') as f:
        data = f.read()
        ai += data + ' '
    ai = ai.replace('\n', ' ')
    f.close()

    art = ''
    with open("part_location_data_ver2/Art.txt", 'r') as f:
        data = f.read()
        art += data + ' '
    art = art.replace('\n', ' ')
    f.close()

    main_lib = ''
    with open("part_location_data_ver2/MainLib.txt", 'r') as f:
        data = f.read()
        main_lib += data + ' '
    main_lib = main_lib.replace('\n', ' ')
    f.close()

    rotary = ''
    with open("part_location_data_ver2/Rotary.txt", 'r') as f:
        data = f.read()
        rotary += data + ' '
    rotary = rotary.replace('\n', ' ')
    f.close()

    education = ''
    with open("part_location_data_ver2/Education.txt", 'r') as f:
        data = f.read()
        education += data + ' '
    education = education.replace('\n', ' ')
    f.close()

    edu_main_lib = ''
    with open("part_location_data_ver2/EduMainLib.txt", 'r') as f:
        data = f.read()
        edu_main_lib += data + ' '
    edu_main_lib = edu_main_lib.replace('\n', ' ')
    f.close()

    main_gate = ''
    with open("part_location_data_ver2/MainGate.txt", 'r') as f:
        data = f.read()
        main_gate += data + ' '
    main_gate = main_gate.replace('\n', ' ')
    f.close()

    student = ''
    with open("part_location_data_ver2/Student.txt", 'r') as f:
        data = f.read()
        student += data + ' '
    student = student.replace('\n', ' ')
    f.close()

    tunnel = ''
    with open("part_location_data_ver2/Tunnel.txt", 'r') as f:
        data = f.read()
        tunnel += data + ' '
    tunnel = tunnel.replace('\n', ' ')
    f.close()

    corpora = [
        # AI, Art, MainLib, Rotary, Education, EduMainLib, MainGate, Student, Tunnel
        ai, art, education, edu_main_lib, main_gate, main_lib, rotary, student, tunnel
    ]
    return corpora
