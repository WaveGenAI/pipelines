from src.manage import Manager

if __name__ == "__main__":
    input_file = "musics.xml"
    output_dir = "output"
    manager = Manager(input_file, output_dir)
    manager.run()
