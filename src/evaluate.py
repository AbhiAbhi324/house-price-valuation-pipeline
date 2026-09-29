from src.data_pipeline import main as data_pipeline_main

def main() -> None:
    print("Starting evaluation...")
    data_pipeline_main()
    print("Evaluation completed.")

if __name__ == "__main__":
    main()