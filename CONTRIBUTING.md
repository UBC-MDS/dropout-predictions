# Contribution

We appreciate your interest to support our project. Everyone is encouraged to contribute. You consent to adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) by taking part in this project. 
Please open a GitHub issue with the owners of this repository, who are identified in the [README.md](README.md), before making any changes while contributing to this repository. 
Please adhere to these rules while contributing to this project.

## Core Team Contribution

All core team members should work on their feature branch. All contributions are expected to go through code review via a pull request (PR). A branch protection rule is set in the main branch (currently at least 1 reviewer is required). There shouldn't exist commits in the main branch, except the ones created when the team merged branches. 

Team members should review the changes and approve the PR within 7 days.

Team members should `git pull` to get the latest version of the project before making changes.

General contribution steps:

1. `git pull` to obtain the latest changes from the remote repository
2. `git switch -c <branch_name>`, where branch_name is the name of the new feature branch
3. Modify code & test the code
4. `git add <related_files>`, add all the modified files
5. `git commit -m "msg"`, commit changes with a **meaningful** commit message
6. `git push --set-upstream origin <branch_name>`, pushing local changes with a new branch to the remote
7. Submit a PR
8. Other teammates should review the PR within 7 days, and approve the PR or add comments if necessary
9. Merge branch


## How Can I Contribute?

For non-core team member contributions, please fork the git repository, modify the code and submit a PR. The core team will review the issue and act within 7 days.

### 1. Fixing typos

If you spot any minor typos or grammatical errors in the documentation, you can modify it right away using the GitHub web interface as long as you make the changes in the project's _source_ file.

### 2. Reporting Bugs

  Bugs are tracked as [GitHub issues](https://github.com/UBC-MDS/dropout-predictions/issues). After you've determined which code or module your bug is related to, create an issue on the repository and provide the following information by filling in [the template](https://github.com/atom/.github/blob/master/.github/ISSUE_TEMPLATE/bug_report.md).

#### How Do I Submit A Bug Report?

  The process described here has several goals:
- Maintain dropout-prediction's quality
- Fix problems that are important to users
- Enable a sustainable system for dropout-prediction's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:
1. Follow all instructions in the templates: 
    - [Python](https://peps.python.org/pep-0008/)
    - [R](https://style.tidyverse.org)
2. After you submit your pull request, verify that all [status checks](https://help.github.com/articles/about-status-checks/) are passing 

While the prerequisites above must be satisfied prior to have your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.