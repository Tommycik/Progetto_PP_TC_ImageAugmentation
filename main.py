# project entry point used by PyCharm

try:
    from Src.app import run_application
except ModuleNotFoundError as error:
    # show the missing package
    print(f"Missing Python dependency: {error.name}")
    print("Install the packages listed in requirements.txt using the PyCharm interpreter.")
    raise SystemExit(1) from error


if __name__ == "__main__":
    run_application()
