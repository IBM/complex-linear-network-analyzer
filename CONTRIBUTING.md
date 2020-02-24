## Contributing In General
Our project welcomes external contributions. If you have an itch, please feel
free to scratch it.

To contribute code or documentation, please submit a [pull request](https://github.com/IBM/complex-linear-network-analyzer/pulls).

A good way to familiarize yourself with the codebase and contribution process is
to look for and tackle low-hanging fruit in the [issue tracker](https://github.com/IBM/complex-linear-network-analyzer/issues).
Before embarking on a more ambitious contribution, please quickly get in touch with us.

**Note: We appreciate your effort, and want to avoid a situation where a contribution
requires extensive rework (by you or by us), sits in backlog for a long time, or
cannot be accepted at all!**

### Proposing new features

If you would like to implement a new feature, please [raise an issue](https://github.com/IBM/complex-linear-network-analyzer/issues)
before sending a pull request so the feature can be discussed. This is to avoid
you wasting your valuable time working on a feature that the project developers
are not interested in accepting into the code base.

### Fixing bugs

If you would like to fix a bug, please [raise an issue](https://github.com/IBM/complex-linear-network-analyzer/issues) before sending a
pull request so it can be tracked.

### Merge approval

Project maintainers use LGTM (Looks Good To Me) in comments on the code
review to indicate acceptance. For small fixes only one project maintainer needs to approve, 
for larger changes two maintainers should approve. 

## Legal

Each source file must include a license header for the Apache
Software License 2.0. Using the SPDX format is the simplest approach.
e.g.

```
/*
Copyright <holder> All Rights Reserved.

SPDX-License-Identifier: Apache-2.0
*/
```

We have tried to make it as easy as possible to make contributions. This
applies to how we handle the legal aspects of contribution. We use the
same approach - the [Developer's Certificate of Origin 1.1 (DCO)](https://github.com/hyperledger/fabric/blob/master/docs/source/DCO1.1.txt) - that the Linux® Kernel [community](https://elinux.org/Developer_Certificate_Of_Origin)
uses to manage code contributions.

We simply ask that when submitting a patch for review, the developer
must include a sign-off statement in the commit message.

Here is an example Signed-off-by line, which indicates that the
submitter accepts the DCO:

```
Signed-off-by: John Doe <john.doe@example.com>
```

## Setup

For new features create a feature branch. Code in the master branch should always be stable and ready for deployment.

## Testing

Please run all unittests in the tests directory before pushing any code changes. 
The tests do not check the correctness of the visualization features. In case you update them, please verify 
the correctness manually. 

## Coding style guidelines

We use reStructured Text formatting for docstrings.