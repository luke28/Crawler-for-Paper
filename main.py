import os
import sys
import json
import argparse

def main():
    parser = argparse.ArgumentParser(formatter_class = argparse.RawTextHelpFormatter)
    parser.add_argument('--conference', type = str, default = "nips")
    parser.add_argument('--url', type = str, required = True)
    parser.add_argument('--pattern', type = str, required = True)
    parser.add_argument('--filename', type = str, default = "out.md")
    args = parser.parse_args()
    mdl = __import__(args.conference)
    mdl.find_by_re(args.url, args.pattern, args.filename)

if __name__ == '__main__':
    main()
