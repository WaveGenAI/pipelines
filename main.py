import argparse
from src.manage import Manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file path")
    parser.add_argument("--output", help="output directory path")
    args = parser.parse_args()

    input_file = args.input if args.input else "musics.xml"
    output_dir = args.output if args.output else "outputs"  # /media/works/

    manager = Manager(input_file, output_dir)
    manager.run()
