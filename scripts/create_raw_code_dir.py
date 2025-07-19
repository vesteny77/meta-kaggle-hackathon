from pathlib import Path

def main():
    (Path("data") / "raw_code").mkdir(exist_ok=True)

if __name__ == "__main__":
    main() 