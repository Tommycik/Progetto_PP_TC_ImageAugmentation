try:
    from Src.app import run_application
except ModuleNotFoundError as error:
    print(f"Missing Python dependency: {error.name}")
    print("Install the packages from requirements.txt using the project interpreter in PyCharm.")
    raise SystemExit(1) from error


if __name__ == "__main__":
    run_application()
