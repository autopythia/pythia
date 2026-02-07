import gzip
import json

from pythia_test.snapshot import snapshot_test

def main():
    test_diff()

@snapshot_test
def test_diff(snapshot):
    import pythia.diff
    with open("cli.py.diff", "r") as file:
        text = file.read()
    result = pythia.diff.parse_diff(text, ".")
    assert snapshot == result

def test_diff_v1():
    # snapshot_path = "pythia_test.testdata/test_diff.json"
    # snapshot_path = "pythia_test.testdata/test_diff.json.gz"
    snapshot_path = "pythia_test.testdata/test_diff.__snapshot__.gz"
    try:
        with gzip.open(snapshot_path, "rb") as file:
            data = file.read()
        snapshot = json.loads(data.decode("utf-8"))
    except OSError:
        snapshot = None
    import pythia.diff
    with open("cli.py.diff", "r") as file:
        text = file.read()
    result = pythia.diff.parse_diff(text, ".")
    if False:
        diff = result["ok"]
        print(len(diff["files"]))
        print(len(diff["files"][0]["hunks"]))
    if snapshot is None:
        data = json.dumps(result).encode("utf-8")
        with gzip.open(snapshot_path, "wb") as file:
            # print(data, end="", file=file)
            wlen = file.write(data)
            assert wlen == len(data)
    else:
        assert snapshot == result
    print("ok")

if __name__ == "__main__":
    main()
