# project entry point used by the PyCharm Run button

try:
    # import the menu only after Python starts the main file
    from Src.app import run_application
except ModuleNotFoundError as error:
    # show the missing package instead of a long import traceback
    print(f"Missing Python dependency: {error.name}")
    print("Install the packages listed in requirements.txt using the PyCharm interpreter.")
    raise SystemExit(1) from error


if __name__ == "__main__":
    # start the interactive program only when this file is executed directly
    run_application()
