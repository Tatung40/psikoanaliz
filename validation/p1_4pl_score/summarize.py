import glob, sys
n = sum(1 for f in glob.glob(sys.argv[2]) for _ in open(f)); print("records", n)
