import argparse


def show_version():
    print('Version: 1.0')


def read_file():
    f = open('text.txt', 'r')
    print(f.read())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Keywords Parser')

    parser.add_argument('-v', action="store_true", help='show version of Keywords Parser tool')
    parser.add_argument('--file', default='text.txt', help='reads file')

    args = parser.parse_args()

    if args.v:
        show_version()

    if args.file:
        read_file()


