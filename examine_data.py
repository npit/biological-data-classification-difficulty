from glob import glob
import numpy as np
import sys
import os



from util import read_arff_data

def examine_path(arff_file):
    data = read_arff_data(arff_file)
    data.fillna(0.0)
    first = set(data.values[:,0])
    last = set(data.values[:,-1])
    passed = False
    print("\t", arff_file)
    print("\t", f"{len(first)} distinct values in the first col")
    if len(first) == 2:
        print("\t", first)
        passed = True
    print("\t", f"{len(last)} distinct values in the last col")
    if len(last) == 2:
        print("\t", last)
        passed = True
    return passed

def summarize_path(arff_file):
    data = read_arff_data(arff_file)
    data.fillna(0.0)
    return np.asarray(data.values.shape)


def summarize_folder(representations_folder):
    for folder in glob(representations_folder + "*/"):
        print(folder)
        szs = []
        for path in glob(folder + "*"):
            if os.path.isfile(path):
                sz = summarize_path(path)
            else:
                summarize_folder(path)
                continue
            # print(arff_file, sz)
            szs.append(sz)
        stats = np.vstack(szs)
        print("Mean lens / dims: ", np.mean(stats, axis=0))


def examine_folder(representations_folder):
    problematic_folders = set()
    problematic_files = set()

    for folder in glob(representations_folder + "*/"):
        print(folder)
        for arff_file in glob(folder + "*"):
            print(arff_file)
            passed = examine_path(arff_file)
            if not passed:
                print("\t", "problematic!")
                problematic_files.add(arff_file)
                problematic_folders.add(folder)
            break

    # print(problematic_files)
    print(problematic_folders)


# path = "/home/nik/work/repr/anna/Combined Representations/Tetrahedron/Combined1.arff"
inp = sys.argv[1]
if os.path.isdir(inp):
    summarize_folder(inp)
else:
    summarize_path(inp)
