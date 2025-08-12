# nreca-repository-template
This is a NRECA-BTS repository template, which is to be used as starting point when any GitHub team/member needs to create a new GitHub repository. Once a repository is created using this template a team member needs to apply following changes to be in compliance.

## Another repo?
Before you go further and setup all the compliance requirements for this new repo, a team must think through the need for this repo and weighs in pros and cons for managing one more repo. To help you make this decision, please take a look at the blog post below.

[Too Many Repositories](https://itwiki.nreca.org/pages/viewpage.action?pageId=229738640)

## Repository Documentation Compliance Requirements
Following are repository documentation related compliance requirements.

1. Follow the guidelines on repository [naming convention requirements](https://itwiki.nreca.org/pages/viewpage.action?pageId=245729144).
2. Follow the guidelines on repository [documentation requirements](https://itwiki.nreca.org/display/DSO/GitHub+Repository+Hygiene#GitHubRepositoryHygiene-DocumentRepositories).
3. New repository visibility must be public. Follow [GitHub guidelines](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/managing-repository-settings/setting-repository-visibility) to make it visible.
4. New repository must have one or more topic(s) applied. Follow the [guidelines](https://itwiki.nreca.org/display/DSO/GitHub+Repository+Hygiene#GitHubRepositoryHygiene-EstablishRepositoryOwnership).


#### .gitignore file
A .gitignore file has already been included in this repository for Visual Studio becasue it was created from NRECA repository template. It is expected from team to keep maintaining this file or have multiple .gitignore files to support their development workflow. 

Here are more resources to know more about .gitignore file. 

[GITHUB Documentation on .gitignore file](https://docs.github.com/en/get-started/getting-started-with-git/ignoring-files).

[GITHUB repository for sample .gitignore file for most frameworks](https://github.com/github/gitignore).

#### README.md file
Teams can find more details on what needs to be included in a ReadMe file by following [documentation requirements](https://itwiki.nreca.org/display/DSO/GitHub+Repository+Hygiene#GitHubRepositoryHygiene-DocumentRepositories).

## Repository Code-Review Compliance Requirement

#### pull_request_template.md file
A pull_request_template.md file has already been included in this repository becasue it was created from NRECA-BTS repository template. It is expected from team to keep using this file to improve their PR review process. Follow the guidelines on [pull_request_template.md](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository)

#### CODEOWNERS file
A CODEOWNERS file has already been included in this repository, which needs to be updated to set repository code owners. Follow the guidelines on [CODEOWNERS](https://itwiki.nreca.org/display/DE/GHEC%3A+Teams+Responsibilities#GHEC:TeamsResponsibilities-ManagingCodeOwnershipandOutsideContributors/Collaborators)

#### Branch Protection Rules
Follow the guidelines on [applying branch protection rules](https://itwiki.nreca.org/display/DSO/Configuring+a+GitHub+Repository+for+Code+Review)

## GitHub Advance Security (GHAS)

Team needs to follow the guidelines on [secrets management](https://itwiki.nreca.org/display/DSO/GitHub+-+Managing+Secrets+in+Repositories).

Protected data should never be committed to a repository. Follow the guidelines on management of [PHI/PII data](https://itwiki.nreca.org/display/DSO/GitHub+PII+Remediation).