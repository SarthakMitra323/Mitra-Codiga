import subprocess
import sys

language = "python3"
run = "python web_app.py"

def main():
    try:
        subprocess.run([sys.executable, "web_app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running web app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()