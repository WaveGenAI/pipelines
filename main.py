import argparse

from src.manage import Manager

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file path", default="musics.xml")
    parser.add_argument("--output", help="output directory path", default="outputs")
    parser.add_argument("--llm", help="Pass data to llm", type=bool, default=False)
    parser.add_argument(
        "--transcript", help="Pass data to transcript", type=bool, default=False
    )
    parser.add_argument(
        "--download", help="Download audio files", type=bool, default=True
    )
    args = parser.parse_args()

    manager = Manager(
        args.input,
        args.output,
        llm=args.llm,
        transcript=args.transcript,
        download=args.download,
    )
    manager.run()
