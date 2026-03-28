# Git Commands Cheat Sheet

## Page 1: Basic Concepts & Repository Setup

### 1.1 What is Git?
Git is a distributed version control system that tracks changes in any set of computer files, usually used for coordinating work among programmers collaboratively developing source code during software development.

### 1.2 Initializing and Cloning
- `git init`: Initializes a new Git repository in the current directory.
- `git clone [url]`: Clones a repository into a newly created directory.

### 1.3 The Staging Area
- `git status`: Shows the working tree status (which files are tracked, untracked, staged, or modified).
- `git add [file]`: Adds file contents to the staging area.
- `git add .`: Adds all current modifications and new files to the staging area.
- `git commit -m "[message]"`: Records changes to the repository with a descriptive message.

## Page 2: Branching & Merging Strategies

### 2.1 Managing Branches
Branches are highly lightweight and making them is nearly instantaneous.
- `git branch`: Lists all local branches in the current repository.
- `git branch [branch-name]`: Creates a new branch.
- `git checkout [branch-name]`: Switches the working directory to the specified branch.
- `git checkout -b [branch-name]`: Creates a new branch and immediately switches to it.
- `git branch -d [branch-name]`: Deletes a specific branch safely.

### 2.2 Merging Changes
- `git merge [branch]`: Joins the specified branch's history into the current branch.
- Merge Conflicts: When changes happen on the same lines of the same file in different branches, Git pauses the merge. You must manually resolve the conflict, `git add` the resolved file, and then `git commit`.

## Page 3: Inspecting History & Undoing Changes

### 3.1 Viewing Commit History
- `git log`: Shows the commit logs.
- `git log --oneline --graph --decorate`: Shows a condensed, visual graph of commits.
- `git diff`: Shows changes between the working tree and the index (what you haven't staged yet).
- `git diff --staged`: Shows changes between the index and the last commit (what you are about to commit).

### 3.2 Undoing Mistakes
- `git restore [file]`: Discards changes in the working directory.
- `git reset [commit]`: Undoes all commits after `[commit]`, preserving changes locally.
- `git reset --hard [commit]`: Discards all history and changes back to the specified commit. **Warning: This is destructive!**
- `git revert [commit]`: Creates a new commit that undoes all of the changes made in `[commit]`, preserving the project history.

## Page 4: Advanced: Stashing, Rebasing, & Remotes

### 4.1 Stashing Uncommitted Work
- `git stash`: Temporarily shelves changes you've made to your working copy so you can work on something else.
- `git stash pop`: Restores the most recently stashed files and removes them from the stash list.
- `git stash list`: Lists all existing stashes.

### 4.2 Rebasing
- `git rebase [base]`: Reapplies commits on top of another base tip. Commonly used to maintain a linear project history.
- `git rebase -i [commit]`: Interactive rebase allows you to squash, edit, or reorder previous commits.

### 4.3 Working with Remotes
- `git remote add origin [url]`: Connects your local repository to a remote server.
- `git fetch`: Downloads objects and refs from another repository.
- `git pull`: Fetches from and integrates with another repository or a local branch (effectively `fetch` + `merge`).
- `git push`: Updates remote refs along with associated objects.
