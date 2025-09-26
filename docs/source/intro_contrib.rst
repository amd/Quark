.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quark's ``contrib`` Area
========================

Welcome to the AMD Quark ``contrib`` area! This guide outlines the terms of use for users and the policies for contributing code that is shipped with Quark but is not officially supported by the Quark Core Team. All contributions to this area are maintained and supported by their original authors.

.. _users_contrib_area:

For Users: Understanding the ``contrib`` Area
---------------------------------------------

The ``contrib`` area is a valuable collection of community-authored extensions to Quark. While these components are included in the AMD Quark library for convenience, it is critical to understand their relationship with the Quark Core Team and your responsibility as a user:

* **No Official Support:** Components within the ``contrib`` area are **not officially supported** by the Quark Core Team. This means we cannot provide direct assistance, bug fixes, or guarantees for these modules.

* **Use at Your Own Risk:** You **use these contributions at your own risk**. While we encourage high-quality submissions, the Quark Core Team does not guarantee the stability, security, or future compatibility of ``contrib`` code.

* **Support from the Author:** If you encounter issues, have questions, or require support, please contact the **original author** of the contribution. The author's name and contact information (typically a GitHub profile or link) should be available in the module's documentation.

By using ``contrib`` code, you agree that the Quark Core Team is not responsible for any issues that may arise from its use.

.. _contributors_contrib_area:

For Contributors: How to Contribute
-----------------------------------

.. _contribution_principles:

Contribution Principles
^^^^^^^^^^^^^^^^^^^^^^^

By contributing to the ``contrib`` area, you agree to the following responsibilities:

* **Quark Release Schedule:** Quark's release schedule is determined by the Core Team and is not subject to change based on the readiness or release requirements of ``contrib`` contributions.
* **Legal Responsibility:** All code, including in the ``contrib`` area, must pass a legal review. You are responsible for resolving any legal issues identified by our Legal department before your code can be merged.
* **Code Standards & Maintenance:** Your code must comply with Quark's established code formatting, style, and quality standards. You are the sole maintainer of your contribution, responsible for its long-term health and compatibility with new Quark releases.
* **Full Author Support:** The original author is responsible for all aspects of their contribution's lifecycle, including:
    * Performing code reviews for any proposed changes to your contribution.
    * Fixing bugs and refactoring code as needed.
    * Responding to user and customer support requests.
    * Resolving CI (Continuous Integration) failures related to your code.
    * Ensuring the code meets all required unit test and code coverage policies.
* **Licensing:** Your contribution must be licensed under the MIT license, consistent with the main Quark project.
* **Documentation:** All code must be fully documented using Quark's specified policy and tooling. Your module must include comprehensive usage instructions and API documentation.

.. _how_to_contribute_steps:

How to Contribute
^^^^^^^^^^^^^^^^^

* **File an Intent-to-Contribute Issue:** Before starting any work, you **must** open an issue on the Quark GitHub repository to declare your intent to contribute. This issue should provide a summary, design, and other relevant technical information about your proposed contribution. This step is crucial for preventing work duplication with the Quark Core Team's roadmap and allows for a discussion of the solution before significant time is invested in implementation.

* **Create a ``README.md``:** Each contribution must include a ``README.md`` file within its component directory (e.g., ``quark/contrib/your-component-name/README.md``). This file should provide a short description of your contribution, its purpose, and clear contact information for users to report bugs, provide feedback, or address any other concerns directly to you, the author.

* **Take ownership of your contrib subfolder:** You must update the ``CODEOWNERS`` file to include yourself as a code owner for your contribution. This ensures that you are notified of any changes or issues related to your code. Note that Quark core team will also be allowed to make changes to the code base for maintenance purposes.

E.g., for taking ownership for ``your-component-name``:

.. code-block:: bash

    /quark/contrib/your-component-name @your-github-username

* **Contribution Guidelines:** For general development guidelines, including information on setting up your environment, writing tests, and submitting a pull request, please refer to the main ``CONTRIBUTING.md`` file in the repository.

* **Update the ``.gitignore``:** If your contribution includes files that should not be tracked by Git, you must update the ``.gitignore`` file in your component's directory. This ensures that temporary files, build artifacts, or other non-essential files are not included in the repository.

* **Update the ``requirements.txt``:** If your contribution introduces new Python dependencies, you must reach out to the Quark Core Team to discuss whether it is appropriate to update the main ``requirements.txt``. Otherwise, create a ``requirements.txt`` file in your component's directory to include these dependencies and produce documentation to inform users about the installation process.

.. _testing_contribution:

Testing Your Contribution
^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``contrib`` component is expected to include its own dedicated tests to ensure its functionality and stability.

* **Test Location:** All unit tests for your contribution must reside in a ``tests`` subfolder within your component's directory, i.e., ``quark/contrib/your-component-name/tests/``.

* **Test Quality:** As per the general ``CONTRIBUTING.md`` guidelines and test's README.md (aka ``tests/README.md``)_, your unit tests must be fast, small, and strive to cover most, if not all, of the source code maintained by your contribution.

* **CI/CD Integration:** To integrate your tests into Quark's continuous integration/continuous deployment (CI/CD) workflows and scripts, you **must consult with a DevOps engineer** from the Quark Core Team. They will assist in updating the infrastructure to properly run your tests.

.. _documenting_contribution:

Documenting Your Contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive documentation is vital for the usability and maintainability of your ``contrib`` component.

* **Documentation Location:** All documentation for your contribution must be placed in a ``docs`` subfolder within your component's directory, i.e., ``quark/contrib/your-component-name/docs/``.

* **Format and Content:** Documentation must be written in **ReStructuredText** format. It should include a good conceptual introduction, clear examples, relevant diagrams, and any other information necessary to make the component easy to use, understand, and maintain for other developers.

* **CI/CD Integration:** To integrate your documentation into Quark's build and deployment infrastructure, you **may consult with a DevOps engineer** from the Quark Core Team. They will assist in updating the necessary CI/CD workflows or scripts, if needed. Typically, it is needed to update the ``docs/source/index.rst`` file to add an entry point for your new contribution's documentation. This entry should be placed right below the ``intro_contrib.rst`` page, similar to this example:

    .. code-block:: rst

        .. toctree::
            :hidden:
            :caption: Contributions
            :maxdepth: 1

            Quark ``contrib``s <intro_contrib.rst>
            Your component name <contrib_your_component_name.rst>

* Write and test rendering of the documentation as per the general ``CONTRIBUTING.md`` guidelines.

Thank you for your commitment to expanding the Quark ecosystem.
