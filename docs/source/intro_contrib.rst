.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quark's ``contrib`` Area
========================

Welcome to the AMD Quark ``contrib`` area! We're excited to have you explore this collection of community-authored extensions. This guide outlines the terms of use for users and the policies for contributing code that is shipped with Quark but is not officially supported by the Quark Core Team. All contributions to this area are maintained and supported by their original authors, fostering a collaborative ecosystem that extends Quark's capabilities beyond the core library.

.. _users_contrib_area:

For Users: Understanding the ``contrib`` Area
---------------------------------------------

The ``contrib`` area is a valuable collection of community-authored extensions to Quark that expand functionality beyond the core library. While these components are included in the AMD Quark library for your convenience, it is important to understand their relationship with the Quark Core Team and your responsibilities as a user:

* **No Official Support:** Components within the ``contrib`` area are **not officially supported** by the Quark Core Team. While we facilitate their inclusion in the repository and provide infrastructure support, we cannot provide direct assistance, bug fixes, or guarantees for these modules. For support, please contact the original author.

* **Use at Your Own Risk:** You **use these contributions at your own risk**. While we evaluate all contributions for quality, stability, and alignment with the project's direction before merging, the Quark Core Team does not guarantee the ongoing stability, security, or future compatibility of ``contrib`` code. The original author is responsible for maintaining their contribution.

* **Support from the Author:** If you encounter issues, have questions, or require support, please contact the **original author** of the contribution. The author's name and contact information (typically a GitHub profile or link) will be available in the module's ``README.md`` file and documentation.

By using ``contrib`` code, you acknowledge that the Quark Core Team is not responsible for any issues that may arise from its use, and that support is provided by the original author on a best-effort basis.

.. _contributors_contrib_area:

For Contributors: How to Contribute
-----------------------------------

We appreciate your interest in contributing to the Quark ``contrib`` area! Your contributions help expand the capabilities of Quark and benefit the entire community. To ensure a smooth and successful contribution process, please familiarize yourself with the following principles and guidelines.

.. important::

   **These guidelines are the single source of truth, and they evolve over time.**
   Always follow the guidelines as written here — do not model your contribution on
   an existing ``contrib`` component. Older components were merged under earlier
   versions of these rules and may not yet reflect current requirements; where an
   existing component and this document disagree, **this document wins**. Treating a
   legacy component as precedent for skipping a current requirement is not a valid
   justification during review. We update existing components toward these guidelines
   over time, but they may lag, so trust the guidelines rather than the code you see
   in the tree.

.. _contribution_principles:

Contribution Principles
^^^^^^^^^^^^^^^^^^^^^^^

By contributing to the ``contrib`` area, you agree to the following responsibilities and expectations:

* **Quality & Alignment:** While the quality bar for the ``contrib`` area is lower than for the core library, **merging is not unconditional**. All pull requests are evaluated for stability, user experience (UX), and alignment with Quark's overall direction. Contributions that do not meet these criteria or introduce unnecessary complexity may not be accepted. Additionally, **hardware-specific code is not permitted** in the Quark repository. This includes, but is not limited to, ONNX Runtime operators, NPU kernels, and other hardware-dependent implementations. Quark is a model optimization library, and only contributions that are tightly related to model optimization belong in this repository.

* **Contributor Responsibilities:** As a contributor, you are responsible for:

    * **Providing comprehensive documentation** that enables users to understand and use your contribution effectively.
    * **Writing thorough tests** that validate your contribution's functionality and maintain code quality.
    * **Timely addressing issues** reported by the Quark Core Team, users, or identified through automated testing and security scans.
    * **Maintaining compatibility** with new Quark releases and updating your contribution as needed.
    * **Performing code reviews** for any proposed changes to your contribution.
    * **Responding to user and customer support requests** in a timely manner.

* **Maintenance & Removal:** The Quark Core Team absorbs the overhead for CI/CD infrastructure, security scans, and legal approvals to facilitate your contribution. However, **we rely on you to actively maintain your code**. Failure to maintain a contribution, address feedback from the Quark team, or resolve critical issues in a timely manner **may result in its removal from the repository**. Specifically:

    * **Release Branch Removal:** If a release is in progress and you fail to address any test failures, documentation issues, bugs, or other feedback from the Quark Core Team, **your contribution will be removed from the release branch**.
    * **Main Branch Removal:** If issues persist and you continue to not address the required changes, **your contribution will also be removed from the main branch**.

  We will make reasonable efforts to communicate with you before taking such action, but unmaintained code poses risks to the overall project and may need to be removed to protect the integrity of the Quark ecosystem.

* **Release Schedules:** Quark's official release schedule is determined by the Core Team and **is not subject to change based on the readiness or release requirements of** ``contrib`` **contributions**. If you have a time-sensitive customer need that cannot wait for the next official release, please follow the process for `Creating a Quark Private Drop for Customers` in our Wiki. Users can reach out to the Quark team to discuss how to use our **private drop mechanism** to create a non-official development version. This approach unblocks immediate customer needs while we collaborate to streamline the contribution for inclusion in the next stable release.

* **Legal Responsibility:** All code, including contributions to the ``contrib`` area, must pass a legal review conducted by AMD's Legal department. You are responsible for ensuring your code complies with all applicable licenses and resolving any legal issues identified during the review process before your code can be merged.

* **Code Standards:** Your code must comply with Quark's established code formatting, style, and quality standards as outlined in the main ``CONTRIBUTING.md`` file. This includes passing all linting checks, type checking, and adhering to the project's coding conventions.

* **Licensing:** Your contribution must be licensed under the MIT license, consistent with the main Quark project. By submitting a contribution, you affirm that you have the right to license the code under these terms.

* **CI/CD Integration:** You are responsible for resolving any Continuous Integration (CI) failures related to your code. The Quark Core Team provides the infrastructure, but you must ensure your contribution passes all required checks and tests.

.. _how_to_contribute_steps:

How to Contribute
^^^^^^^^^^^^^^^^^

* **Know where each part of your contribution belongs.** Your component's **package code**, its unit tests (``test/``), its documentation (``docs/``), and any component-scoped ``requirements.txt`` / ``.gitignore`` **must** live under your component's directory, ``quark/contrib/your-component-name/``. Do **not** edit shared documentation files such as ``docs/source/index.rst`` — your docs are wired in automatically (see below). Two categories of assets are deliberate exceptions and live in the shared top-level folders so users can find them alongside the core equivalents:

  * **Runnable examples** go in the repository's root ``examples/`` tree, under ``examples/contrib/your-component-name/``.
  * **Jupyter tutorials** go in the repository's root ``tutorials/`` tree, under ``tutorials/contrib/your-component-name/``.

  The only other files a contribution places outside its own folder are CI/CD workflows and reusable actions (under ``.github/workflows/`` and ``.github/actions/``, see :ref:`the CI/CD section <running_cicd_contribution>`) and your ownership entry in the root ``CODEOWNERS`` file. Keeping ``docs/`` and ``test/`` in the component makes it easy to review, maintain, and remove; keeping ``examples/`` and ``tutorials/`` in the shared trees makes them discoverable next to the core ones.

* **File an Intent-to-Contribute Issue:** Before starting any work, you **must** open an issue on the Quark GitHub repository to declare your intent to contribute. This issue should provide a summary, design, and other relevant technical information about your proposed contribution. This step is crucial for preventing work duplication with the Quark Core Team's roadmap and allows for a discussion of the solution before significant time is invested in implementation.

* **Create a ``README.md``:** Each contribution must include a ``README.md`` file within its component directory (e.g., ``quark/contrib/your-component-name/README.md``). This file should provide a short description of your contribution, its purpose, and clear contact information for users to report bugs, provide feedback, or address any other concerns directly to you, the author.

* **Take ownership of your contrib subfolder:** You must update the ``CODEOWNERS`` file to include yourself as a code owner for your contribution. This ensures that you are notified of any changes or issues related to your code. Note that Quark core team will also be allowed to make changes to the code base for maintenance purposes.

E.g., for taking ownership for ``your-component-name``:

.. code-block:: bash

    /quark/contrib/your-component-name @your-github-username

* **Contribution Guidelines:** For general development guidelines, including information on setting up your environment, writing tests, and submitting a pull request, please refer to the main ``CONTRIBUTING.md`` file in the repository.

* **Contributing an algorithm (optional):** If you are contributing a quantization algorithm for the PyTorch backend, Quark provides ``QuarkAlgorithm`` — a wrapper that lets your code live in your own component directory instead of being threaded into Quark's core files. Follow :doc:`How to Add a New contrib Algorithm <contrib_add_algorithm>`, which explains how to package and register your algorithm.

* **Update the ``.gitignore``:** If your contribution includes files that should not be tracked by Git, you must update the ``.gitignore`` file in your component's directory. This ensures that temporary files, build artifacts, or other non-essential files are not included in the repository.

* **Update the ``requirements.txt``:** If your contribution introduces new Python dependencies, you must reach out to the Quark Core Team to discuss whether it is appropriate to update the main ``requirements.txt``. Otherwise, create a ``requirements.txt`` file in your component's directory to include these dependencies and produce documentation to inform users about the installation process.

.. _testing_contribution:

Testing Your Contribution
^^^^^^^^^^^^^^^^^^^^^^^^^

Each ``contrib`` component is expected to include its own dedicated tests to ensure its functionality and stability.

* **Test Location:** All unit tests for your contribution must reside in a ``test`` subfolder within your component's directory, i.e., ``quark/contrib/your-component-name/test/``. The CI test runner discovers tests only under this ``test/`` folder.

* **Test Quality:** Your unit tests must be fast, small, and achieve the **minimum code coverage threshold required by Quark**. Failure to meet the coverage threshold **will block your contribution from being merged**. Please refer to the general ``CONTRIBUTING.md`` guidelines and test's ``README.md`` (aka ``test/README.md``) for the specific coverage requirements and other testing policies.

* **CI/CD Integration:** To integrate your tests into Quark's continuous integration/continuous deployment (CI/CD) workflows and scripts, you **must consult with a DevOps engineer** from the Quark Core Team. They will assist in updating the infrastructure to properly run your tests.

.. _documenting_contribution:

Documenting Your Contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Comprehensive documentation is vital for the usability and maintainability of your ``contrib`` component.

* **Documentation Location:** All documentation for your contribution must be placed in a ``docs`` subfolder within your component's directory, i.e., ``quark/contrib/your-component-name/docs/``. Your ``docs`` folder **must** contain an ``index.rst`` that serves as the landing page for your component — this is the file the documentation build looks for.

* **Format and Content:** Documentation must be written in **ReStructuredText** format. It should include a clear conceptual introduction, practical examples, relevant diagrams or visualizations where appropriate, and any other information necessary to make the component easy to use, understand, and maintain for other developers. High-quality documentation is essential for user adoption and long-term maintainability. If your landing page has additional sub-pages, add them to a ``toctree`` within your ``docs/index.rst``.

* **Prefer executable tutorials (Jupyter notebooks) over source-code examples:** Whenever possible, teach your component with a runnable **Jupyter notebook** (``.ipynb``) rather than a standalone ``.py`` example plus prose. Notebooks are strongly preferred because they interleave explanation, runnable code, and real output on one page; they are executed by the documentation CI, so they double as living tests that catch API drift; and they give users an interactive, copy-and-run onboarding experience. A plain ``.py`` example is only validated for syntax, not behavior, and tends to rot. Place notebooks under ``tutorials/contrib/your-component-name/`` in the repository's root ``tutorials/`` tree (alongside the core tutorials, with any per-notebook ``requirements.txt`` next to the notebook), and reference them from your ``docs/index.rst`` so they appear under your component's documentation. Runnable ``.py`` examples, if any, go under ``examples/contrib/your-component-name/``.

  A good tutorial follows the notebook **Style Guide** in ``CONTRIBUTING.md``: it opens with a one-line statement of what the reader will achieve, states assumed knowledge, walks through a concrete end-to-end example the reader can run and modify, includes a short troubleshooting subsection, and closes with a summary plus links to further reading. For reference, study these core-repo tutorials as templates:

  * ``tutorials/torch/quickstart_tutorial/quickstart_tutorial.ipynb`` — a minimal, welcoming first-run walkthrough.
  * ``tutorials/torch/llm_ptq/llm_tutorial/llm_tutorial.ipynb`` — a fuller end-to-end PTQ example with explanation interleaved with runnable cells.
  * ``tutorials/onnx/image_classification/onnx_image_classification_tutorial.ipynb`` — a task-oriented tutorial with concrete inputs and outputs.

* **Automatic integration — no core files to edit:** You do **not** need to touch any shared documentation file (such as ``docs/source/index.rst``) to publish your docs. At build time, ``docs/build_docs.sh`` copies every ``quark/contrib/<component>/docs/`` directory into the documentation tree, and ``intro_contrib.rst`` indexes each component's ``index.rst`` automatically via a globbing ``toctree`` under the **Contributions** section. Simply create ``quark/contrib/your-component-name/docs/index.rst`` and your page will appear in the rendered documentation.

* Write and test rendering of the documentation as per the general ``CONTRIBUTING.md`` guidelines.

.. _running_cicd_contribution:

Running CI/CD for Your Contribution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When contributing to the ``contrib`` area, you may need to set up CI/CD workflows to test your contribution. This section outlines the requirements and process for integrating CI/CD workflows for your contribution.

GitHub Actions Runners
""""""""""""""""""""""

**When Runners Are Required:**

Partners contributing to Quark **must provide GitHub Actions Runners** (servers) to run CI/CD workloads in the following scenarios:

* **GPU/Special Hardware Requirements:** Any workflow that requires GPU access or other special hardware (e.g., CUDA, ROCm, specialized accelerators).
* **Long-Running Workflows:** Any workflow that takes more than a few minutes to execute on a standard CPU server.

**Requesting Runners:**

To set up new GitHub Actions runners for your contribution:

1. **Contact Quark DevOps:** Reach out to the Quark DevOps team to request the installation of new runners for your contribution.
2. **Runner Labels:** New runners should use labels prefixed with ``contrib-<component_name>-`` to clearly distinguish them from core Quark runners. For example, if your component is named ``my-component``, your runner labels might be:
   * ``contrib-my-component-cpu``
   * ``contrib-my-component-cuda``
   * ``contrib-my-component-rocm``

This labeling convention ensures clear separation between core Quark infrastructure and contribution-specific infrastructure.

Creating Workflows
"""""""""""""""""""

**Workflow File Naming:**

All workflows for contributions must be created in the ``.github/workflows`` folder and must use the following naming convention:

* **Prefix:** ``contrib__<component_name>__``
* **Format:** ``contrib__<component_name>__<workflow_description>.yml``

For example, if your component is named ``my-component`` and you're creating a test workflow, the file should be named:
``contrib__my-component__test.yml``

This naming convention makes it immediately clear that the workflow belongs to a specific contribution and distinguishes it from core Quark workflows.

**Workflow Approval:**

**All new workflow files must be approved by Quark DevOps** before they can be merged into the repository. When submitting a pull request that includes new workflow files:

1. **Notify DevOps:** Explicitly mention in your pull request that you are adding new CI/CD workflows.
2. **Provide Details:** Include information about:
   * What the workflow does
   * What runners/labels it requires
   * Expected execution time
   * Any special hardware or software requirements
3. **Wait for Approval:** Do not merge the pull request until Quark DevOps has reviewed and approved the workflow configuration.

**Example Workflow Structure:**

Here's an example of how a contribution workflow might be structured:

.. code-block:: yaml

    name: contrib__my-component__test

    on:
      pull_request:
        paths:
          - 'quark/contrib/my-component/**'
      push:
        branches: [main]
        paths:
          - 'quark/contrib/my-component/**'

    jobs:
      test:
        runs-on: contrib-my-component-cpu
        steps:
          - uses: actions/checkout@v4
          # ... your test steps here

Creating Reusable Actions
"""""""""""""""""""""""""

**Action Directory:**

All reusable GitHub Actions for contributions must be stored in the ``.github/actions`` folder. Each reusable action should be placed in its own sub-directory within this folder.

**Action Naming:**

Reusable actions should follow a similar naming convention to workflows:

* **Prefix:** ``contrib__<component_name>__``
* **Format:** ``contrib__<component_name>__<action_description>``

For example, if your component is named ``my-component`` and you're creating a setup action, the directory should be named:
``.github/actions/contrib__my-component__setup``

**Action Approval:**

**The same approval policy applies to reusable actions as workflows.** All new reusable action directories and files must be approved by Quark DevOps before they can be merged into the repository. When submitting a pull request that includes new reusable actions:

1. **Notify DevOps:** Explicitly mention in your pull request that you are adding new reusable actions.
2. **Provide Details:** Include information about:
   * What the action does
   * Which workflows use the action
   * Any dependencies or requirements
   * Expected behavior and outputs
3. **Wait for Approval:** Do not merge the pull request until Quark DevOps has reviewed and approved the action configuration.

**Best Practices:**

* Keep workflows focused and efficient. Avoid unnecessary steps that increase execution time.
* Use appropriate runner labels to ensure your workflows run on the correct hardware.
* Document any special requirements or setup steps needed for your workflows.
* Ensure workflows fail fast and provide clear error messages for debugging.

Conclusion
----------

Thank you for your interest in contributing to the Quark ``contrib`` area! Your contributions play a vital role in expanding Quark's capabilities and serving the diverse needs of our community. We look forward to collaborating with you to build high-quality, well-maintained extensions that benefit all Quark users.

If you have any questions about the contribution process or need assistance, please don't hesitate to reach out to the Quark Core Team through GitHub issues or our community channels. We're here to help you succeed!

.. Each contrib component ships its own docs under quark/contrib/<component>/docs/.
   docs/build_docs.sh copies them to contrib/<component>/ in the build tree, and the
   globbing toctree below auto-indexes every component's index page — contributors
   never edit this file.

.. toctree::
   :caption: Contributions
   :maxdepth: 1
   :glob:

   contrib/*/index
