# Contribute to 🫧 torchstack

Everyone is welcome to contribute, and we value everybody's contribution. Code
contributions are not the only way to help the community. Answering questions, helping others, and improving the documentation are also immensely valuable.

It also helps us if you spread the word! Reference the library in blog posts
about the awesome projects it made possible, shout out on Twitter every time it has
helped you, or simply ⭐️ the repository to say thank you.

**This guide was heavily inspired by [Transformers guide to contributing](https://github.com/huggingface/transformers/blob/main/CONTRIBUTING.md).**

## Ways to contribute

There are several ways you can contribute to 🫧 torchstack:

* Fix outstanding issues with the existing code.
* Submit issues related to bugs or desired new features.
* Contribute to the examples or to the documentation.

If you don't know where to start, there is a special [Good First
Issue](https://github.com/huggingface/transformers/contribute) listing. It will give you a list of
open issues that are beginner-friendly and help you start contributing to open-source. The best way to do that is to open a Pull Request and link it to the issue that you'd like to work on. We try to give priority to opened PRs as we can easily track the progress of the fix, and if the contributor does not have time anymore, someone else can take the PR over.

For something slightly more challenging, you can also take a look at the [Good Second Issue](https://github.com/huggingface/transformers/labels/Good%20Second%20Issue) list. In general though, if you feel like you know what you're doing, go for it and we'll help you get there! 🚀

> All contributions are equally valuable to the community. 🥰

## Fixing outstanding issues

If you notice an issue with the existing code and have a fix in mind, feel free to [start contributing](#create-a-pull-request) and open a Pull Request!

## Submitting a bug-related issue or feature request

Do your best to follow these guidelines when submitting a bug-related issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The 🫧 torchstack library is robust and reliable thanks to users who report the problems they encounter.

Before you report an issue, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on GitHub under Issues). Your issue should also be related to bugs in the library itself, and not your code. If you're unsure whether the bug is in your code or the library, please ask in the [forum](https://discuss.huggingface.co/) or on our [discord](https://discord.com/invite/hugging-face-879548962464493619) first. This helps us respond quicker to fixing issues related to the library versus general questions.

## Commit Conventions

Here’s a section you can include in your `CONTRIBUTION.md` to guide users on using the Conventional Commits pattern:

---

### Commit Message Guidelines

To maintain a clean and meaningful commit history, please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) standard when making commits. This helps in automating versioning, generating changelogs, and improving collaboration.

A Conventional Commit has the following structure:

```plaintext
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

#### Commit Types
Here are the most common `type` values you can use:
- **feat**: A new feature.
- **fix**: A bug fix.
- **docs**: Documentation-only changes.
- **style**: Changes that do not affect the meaning of the code (e.g., formatting, whitespace, missing semicolons).
- **refactor**: Code changes that neither fix a bug nor add a feature.
- **test**: Adding or updating tests.
- **chore**: Changes to the build process or auxiliary tools and libraries (e.g., CI configuration, package updates).

#### Examples
1. Adding a new feature:
   ```plaintext
   feat(parser): add support for new syntax
   ```

2. Fixing a bug:
   ```plaintext
   fix(api): handle null inputs in endpoint
   ```

3. Updating documentation:
   ```plaintext
   docs(README): update installation instructions
   ```

4. Non-functional code changes:
   ```plaintext
   style: fix linting issues in utils module
   ```

5. Updating tests:
   ```plaintext
   test(api): add edge case tests for parser
   ```

#### Scope (Optional)
The scope indicates what part of the code is affected (e.g., a module, component, or file). Use lowercase and keep it short and specific:
- Examples: `api`, `parser`, `utils`, `cli`

#### Footer (Optional)
Include any references, breaking changes, or metadata in the footer:
- **Breaking changes**: Use `BREAKING CHANGE: <description>` to indicate changes that break backward compatibility.
- **Issue references**: Mention related issues using `#<issue-number>`.

**Example with footer:**
```plaintext
feat(auth): add JWT authentication

This introduces JWT-based authentication for API requests.

BREAKING CHANGE: API endpoints now require a valid JWT token.
```

#### Validation
To ensure your commits follow the pattern, consider using tools like:
- [Commitlint](https://commitlint.js.org/) to validate commit messages.
- [Husky](https://typicode.github.io/husky/) to enforce commit rules during the commit process.

By following these guidelines, you help make the project easier to maintain and collaborate on. Thank you! 😊