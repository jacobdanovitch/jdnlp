from allennlp.data.dataset_readers.dataset_reader import DatasetReader

if __name__ == "__main__":
    import sys

    n = (len(sys.argv) >= 3) and sys.argv[2].strip()
    n = (n and n.isdigit() and int(n)) or 5

    fmt = len(sys.argv) >= 4 and sys.argv[3].strip()

    reader = DatasetReader.by_name(sys.argv[1].strip())(lazy=True)
    reader.preview(n, fmt)