from mc_lang import run
import sys

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python run_mc.py <file.mc>')
        sys.exit(1)
    path = sys.argv[1]
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    res, err = run(src, path)
    if err:
        print(err)
        sys.exit(1)
    if res is not None:
        print(res)
