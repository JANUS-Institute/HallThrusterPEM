import hallmd
from git import Repo
from pathlib import Path
import sys

repo = Repo(Path('.'))
repo.git.checkout('main')
tag_message = sys.argv[1] if len(sys.argv) == 2 else ""
new_tag = repo.create_tag(f'v{hallmd.__version__}', message=tag_message)
repo.remotes.origin.push(new_tag)
