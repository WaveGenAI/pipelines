import argparse

from src.manage import Manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file path")
    parser.add_argument("--output", help="output directory path")
    parser.add_argument("--llm", help="Pass data to llm", type=bool, default=False)
    parser.add_argument(
        "--transcript", help="Pass data to transcript", type=bool, default=False
    )
    args = parser.parse_args()

    input_file = args.input if args.input else "musics.xml"
    output_dir = args.output if args.output else "outputs"  # /media/works/

    manager = Manager(input_file, output_dir, llm=args.llm, transcript=args.transcript)
    manager.run()
