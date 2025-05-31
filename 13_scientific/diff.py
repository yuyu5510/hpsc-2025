def printdiff(pyfile, cppfile):    
    cpp_strs = []
    py_strs = []

    with open(pyfile, "r") as f:
        py_strs = f.read().split()

    with open(cppfile, "r") as f:
        cpp_strs = f.read().split()

    max_diff = 0
    for (pystr, cppstr) in zip(py_strs, cpp_strs):
        py = float(pystr)
        cpp = float(cppstr)
        max_diff = max(max_diff, abs(py-cpp)) 
    print(f"max_diff: {max_diff}")


if __name__=="__main__":
    printdiff("pyu.dat", "u.dat")
    printdiff("pyv.dat", "v.dat")
    printdiff("pyp.dat", "p.dat")

