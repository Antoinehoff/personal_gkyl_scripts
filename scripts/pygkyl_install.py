import sys
import subprocess
import os
import argparse

postgkyl_repo = "https://github.com/ammarhakim/postgkyl.git"
personal_gkyl_scripts_repo = "https://github.com/Antoinehoff/personal_gkyl_scripts.git"

parser = argparse.ArgumentParser(
    description="""
Install pygkyl and its dependencies (postgkyl and personal_gkyl_scripts).

This script will:
  1. Clone or update the postgkyl repository and install it via pip.
  2. Clone or update the personal_gkyl_scripts repository and install pygkyl from it.
  3. Remove old build artifacts before installation.
  4. Test the installation by importing pygkyl.

USAGE EXAMPLES:
---------------
- Install everything in your home directory (default):
    python pygkyl_install.py

- Install in a custom directory:
    python pygkyl_install.py --path /some/other/path

- Skip pulling repositories (useful when you don't want to update remote changes):
    python pygkyl_install.py --no-pull

PARAMETERS:
-----------
-p, --path     Base path for repositories (default: ~)
--no-pull      Skip running 'git pull' for repositories after cloning/updating
--no-postgkyl-install Skip postgkyl installation
-h, --help     Show this help message and exit

Logs for pip installations are saved in the respective repository directories.
""",
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument("-p", "--path", default="~", help="Base path for repositories (default: ~)")
parser.add_argument("--no-pull", action="store_true", help="Skip pulling (git pull) repositories after cloning/updating")
parser.add_argument("--no-postgkyl-install", action="store_true", help="Skip installing postgkyl")
args = parser.parse_args()

base_path = os.path.expanduser(args.path)
postgkyl_path = os.path.join(base_path, "postgkyl")
personal_gkyl_scripts_path = os.path.join(base_path, "personal_gkyl_scripts")
pygkyl_path = os.path.join(personal_gkyl_scripts_path, "pygkyl")

print("Installing pygkyl and dependencies...")
print(f"-Base path: {base_path}")
print(f"-Postgkyl path: {postgkyl_path}")
print(f"-Personal gkyl scripts path: {personal_gkyl_scripts_path}")
print(f"-Pygkyl path: {pygkyl_path}")
print("\n")

print("1.0 Check if postgkyl repository exists")
if not os.path.exists(postgkyl_path):
	print("Cloning postgkyl repository...")
	subprocess.run(["git", "clone", postgkyl_repo, postgkyl_path], check=True)
	
print("1.1 Pull postgkyl repository")
if not args.no_pull:
	subprocess.run(["git", "-C", postgkyl_path, "pull"], check=True)
else:
	print("Skipping git pull for postgkyl (--no-pull)")

print("1.2 Install postgkyl (required for pygkyl)")
if not args.no_postgkyl_install:
	subprocess.run(["touch", os.path.join(postgkyl_path, "postgkyl_install.log")], check=True)
	with open(os.path.join(postgkyl_path, "postgkyl_install.log"), "w") as logf:
		subprocess.run([sys.executable, "-m", "pip", "install", postgkyl_path], stdout=logf, stderr=subprocess.STDOUT, check=True)
else:
    print("Skipping postgkyl installation (--no_postgkyl_install)")


print("2.0 Check if personal_gkyl_scripts repository exists")
if not os.path.exists(personal_gkyl_scripts_path):
	print("Cloning personal_gkyl_scripts repository...")
	subprocess.run(["git", "clone", personal_gkyl_scripts_repo, personal_gkyl_scripts_path], check=True)

print("2.2 Pull personal_gkyl_scripts repository")
if not args.no_pull:
	subprocess.run(["git", "-C", personal_gkyl_scripts_path, "pull"], check=True)
else:
	print("Skipping git pull for personal_gkyl_scripts (--no-pull)")

print("2.3 Remove old pygkyl egg-info and build directories")
subprocess.run(["rm", "-rf", os.path.join(pygkyl_path, "pygkyl.egg-info")], check=True)
subprocess.run(["rm", "-rf", os.path.join(pygkyl_path, "build")], check=True)

print("2.4 Install pygkyl (personal gkyl scripts)")
subprocess.run(["touch", os.path.join(pygkyl_path, "pygkyl_install.log")], check=True)
with open(os.path.join(pygkyl_path, "pygkyl_install.log"), "w") as logf:
	subprocess.run([sys.executable, "-m", "pip", "install", pygkyl_path], stdout=logf, stderr=subprocess.STDOUT, check=True)

# Import the pygkyl package to test the installation
try:
	import pygkyl
	print("->pygkyl installed successfully")
except ImportError:
	print("->pygkyl installation failed")
	print(f"Please check the installation log at {os.path.join(pygkyl_path, 'pygkyl_install.log')} for details.")