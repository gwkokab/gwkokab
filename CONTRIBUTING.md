<!-- Most things are copied from JAX developer documentation https://jax.readthedocs.io/en/latest/contributing.html -->

# Contributing to GWKokab

Everyone can contribute to GWKokab, and we
value everyone's contributions. There are several ways to contribute, including:

- Answering questions on GWKokab's [discussions page](https://github.com/gwkokab/gwkokab/discussions).
- Improving or expanding GWKokab's [documentation](https://gwkokab.readthedocs.io/en/latest/).
- Contributing to GWKokab's [code base](https://github.com/gwkokab/gwkokab).

## Ways to contribute

We welcome pull requests, in particular for those issues marked with
[contributions welcome](https://github.com/gwkokab/gwkokab/issues?q=is%3Aissue+is%3Aopen+label%3A%22contributions+welcome%22)
or [good first issue](https://github.com/gwkokab/gwkokab/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/gwkokab/gwkokab/issues/new/choose) or
[Discussion](https://github.com/gwkokab/gwkokab/discussions) to seek feedback on your
planned contribution.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Install `uv` package manager if you don't have it already.

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   See details at [uv installation guidelines](https://docs.astral.sh/uv/getting-started/installation/).

2. Install `make` if you don't have it already. On Ubuntu/Debian, you can install it via:

   ```bash
   sudo apt-get install make
   ```

3. Fork the GWKokab repository by clicking the Fork button on the
   [repository page](https://github.com/gwkokab/gwkokab). This creates a copy of the
   GWKokab repository in your account.

4. Clone your forked repository,

   ```bash
   git clone https://github.com/YOUR_USERNAME/gwkokab
   cd gwkokab
   ```

5. Create a new virtual environment using `uv` with any python>=3.11,

   ```bash
   uv venv -p 3.12
   source .venv/bin/activate
   ```

   Then install the development dependencies:

   ```bash
   make install --PIP_FLAGS=--upgrade EXTRA=dev,test,docs
   ```

   This allows you to modify the code and immediately test it out.

6. Add the GWKokab repo as an upstream remote, so you can use it to sync your changes.

      ```bash
      git remote add upstream https://www.github.com/gwkokab/gwkokab
      ```

7. Create a new branch for your changes:

   ```bash
   git checkout -b name-of-change
   ```

8. Make sure your code passes GWKokab’s lint and type checks, by running the following
   from the top of the repository:

   ```bash
   pip install prek
   prek run --all-files
   ```

9. Make sure the tests pass by running the following command from the top of the repository:

   ```bash
   pytest -n auto tests/
   ```

   If you know the specific test file that covers your changes, you can limit the tests to that; for example:

   ```bash
   pytest -n auto tests/test_model_transformations.py
   ```

   You can narrow the tests further by using the `pytest -k` flag to match particular test names:

   ```bash
   pytest -n auto tests/test_model_transformations.py -k test_bijective_transforms
   ```

0. Once you are satisfied with your change, create a commit as follows
   ([how to write a commit message](https://cbea.ms/git-commit/)).

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote branch in
   your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

1. Create a pull request from the GWKokab repository and send it for review. When
   preparing your PR consult
   [GitHub Help](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) if you need
   more information on using pull requests.

## Full GitHub test suite

Your PR will automatically be run through a full test suite on GitHub CI, which covers
a range of Python versions, dependency versions, and configuration options. It’s normal
for these tests to turn up failures that you didn’t catch locally; to fix the issues you
can push new commits to your branch.
