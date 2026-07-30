
try:
    # import the interactive application menu
    from Src.app import run_application
except ModuleNotFoundError as error:
    print(f"Missing Python dependency: {error.name}")
    print("Install the packages listed in requirements.txt")
    raise SystemExit(1) from error


if __name__ == "__main__":
    # start the interactive console application
    run_application()
