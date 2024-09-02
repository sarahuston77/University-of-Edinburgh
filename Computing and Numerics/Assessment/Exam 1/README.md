# Computing and Numerics - Assignment A2 [40 marks]

The assignment consists of 3 tasks, in the notebook `A2.ipynb`. Each task may contain coding and/or discussion questions.

If any of the following instructions require clarification, **please ask on Piazza and/or in class!**

A2 is set up exactly like A1, but for completeness here are the details again.

---

### Submission structure

Your submission will consist of the files `A2.ipynb`, `task2.py`, and `task3.py`.

- When you need to **define a function**, do so in the appropriate **module** (see the section on Modules in the Week 5 notebook).
    - Functions for Task 2 should be defined in `task2.py`.
    - Functions for Task 3 should be defined in `task3.py`.
- Each module should **only** contain **function definitions**, and any required `import` statements (for example if you need to import `numpy`).
- You are allowed to define additional functions in the modules at your convenience.
- For each question, a few simple **tests** will be provided in the notebook `A2.ipynb`. You can use these to start checking the basic working of your functions. They work similarly to your Coderunner quizzes. Run the code cells in the notebook to run the tests (and please don't edit them!).
    - In some cases, the tests use `assert` statements.
        - `assert X` will do nothing if `X` is `True`.
        - `assert X` will produce an `AssertionError` if `X` is `False`.
        - NumPy also has a `np.testing` suite of functions which can `assert` statements in a similar way -- you will see some examples in the provided tests. The syntax is slightly different, but should be self-explanatory.
    - The provided tests are minimal (similar to the "Examples" provided for pre-checking in the Coderunner quizzes). To make sure that your function works fully with all possible inputs, you should add more of your own tests.
    - Any tests that you write yourself will not be assessed, and you do not need to submit them.
    - When you submit on Gradescope, more automatic tests will be performed. You will see the results of **some** of these tests when you submit; if any fail, you can try to correct your code, and resubmit as many times as you want (until the deadline).
    - Gradescope will also have hidden tests which will test your functions further (similar to the further tests in the Coderunner quizzes, which are run when you click "Check"). You will not see the results of these tests until the grades and feedback are returned to the class. This means, in particular, that *passing all the visible tests on Gradescope does not guarantee full marks.*
- For **non-code** questions:
    - You can use the **Markdown cells** provided in `A2.ipynb`, indicated by ✏️ . Double-click these cells to edit them, and run them to display your Markdown-formatted text. Please refer to the Week 1 notebook for Markdown syntax.
    - You can also attach a PDF file if you prefer to submit your handwriting. If you do so, make sure to name the file `taskX_Y.pdf`, if this is the answer to question `X.Y`.

---

### Presentation, plots, and code comments

- Your code should be **well-commented**, with a comment explaining each step of your code. All your functions should have **docstrings**. This is not only good practice -- it is also essential for us to assess your understanding of the code you have written.
- Your code should have good **readability** and sensible **structure**. See the Week 3 workshop activity for reference, as well as the CR tasks.
- If you are printing any results, they should be displayed in a **clear and easily readable** manner. For instance, avoid printing values or arrays on their own, without any context.
- Your code must generate and display all relevant output when run. Rerun your code cells after editing your code, to make sure that the output is updated.
- Your discussions in Markdown cells should use appropriate Markdown formatting when necessary.

**Up to half** of the marks for a given question may be deducted for missing, incomplete, or inaccurate code comments, poor code structure or readability, or poorly presented plots and results.

---

### Working on your project with git and GitHub

- While working on the project, **commit and push your changes often** -- every time you make progress on a subtask. If you tend to forget to do this regularly, you could e.g. set a timer to remind you to commit every hour or so. This ensures that
    - you won't have any last-minute technical issues when it comes to submitting, and
    - your progress is **backed up** on GitHub, and not just inside a temporary codespace.
- You will submit the assignment through **Gradescope**, by providing a link to your GitHub repository. You can resubmit to Gradescope as many times as you want before the deadline. (More detailed submission instructions will be available shortly.)

---

### Academic integrity

This is an **individual** assignment -- just like for the Coderunner quizzes, the work your submit must be your own, to reflect and assess your own understanding and knowledge.

#### Collaboration vs. collusion

Collaboration is fine, but collusion is not. Concretely, this means that discussing the assignment **in broad terms** with others students is fine (and encouraged), as well as giving each other hints or general advice on how to approach a problem. You can use Piazza for this, for example -- if you are stuck, then please ask for help!

However, you are **not permitted** to share your working (even partially) with other students -- that includes your code, any detailed description or explanation of code, and any results or analysis you perform.

For example:

- Alice and Bob are discussing the assignment in the library. Bob's code is not working for one of the questions, and he can't figure out why. He asks Alice how she's tackled the problem, and she explains her approach in broad terms. This gives Bob an idea, and he tries it later. *This is all fine.*
- Bob's idea doesn't work out, and he calls Alice later on Teams. He shares his screen with her to show his code. *This is getting dangerous* -- here's why:
    - Alice helps him with understanding the error, and gives him some pointers and ideas to try, without explaining the problem or the solution in much detail. *That would still be fine.*
    - Alice is stuck on the next question, though, and spots a couple of lines of Bob's code at the bottom of the screen. She uses some of that code for the next question in her submission. This is not OK: *both Bob and Alice have now committed misconduct* -- Alice by using Bob's code, and Bob by sharing his screen.
- Bob is still stuck. He posts his code for that question on Piazza. Some students help and also give him some  general advice. Charlie sees the post on Piazza, and didn't know how to start that question. Charlie uses some of Bob's code, with some corrections to fix the problems, and submits it for the assignment. *This is also misconduct* by both Bob and Charlie.
- Bob is still stuck (poor Bob!). It's getting very close to the deadline now, so he asks his friend Dara to *pleaaaase* show their solution, he promises not to copy it. Bob and Dara are really good friends, so Dara finds it difficult to refuse and sends their code. Bob rewrites Dara's code by changing some variable names, rearranging a bit, and paraphrasing the code comments so that they are "in his own words". *This is misconduct* by both Bob and Dara.

Use and trust your own judgement. It's important to understand that even with the best intentions, you expose yourself to academic misconduct as soon as you show your code to another student, and this could have very serious consequences.

#### Providing references

For **every** separate question, **most** of the code you submit must be **authored by you**. That being said, you may use any code from the course material (e.g. workshop tasks, tutorial sheets, videos), without citing it.

You may also use **small pieces of code** (a few lines max at a time) that you found elsewhere -- e.g. examples from the documentation, a textbook, forums, blogs, etc... You may use this code *verbatim* (i.e. almost exactly as you found it), or adapt it to write your own solution.

A programming assignment is just like any other academic assignment -- and therefore, **you must provide a citation for any such code**, whether you use it *verbatim* or adapt it. To do so, include a code comment at the start of your script or notebook cell, indicating:

- the line numbers where the code was used or adapted,
- the URL of the source (or, if it's from a book, a full reference to the book),
- the date you accessed the source,
- the author of the code (if the information is available). **This includes cases where the "author" is an AI.**

You can use this template -- delete one of the URL or book reference lines as appropriate:

```python
# Lines X-Y: Author Name
# URL: http://...
# Book Title, year published, page number.
# Accessed on 30 Feb 2024.
```

You must also provide **detailed code comments** for any such code, in your own words, to demonstrate that you fully understand how it works -- you will lose marks if you use external code without explaining it, even if it's cited correctly.

Your mark will also be negatively affected if a substantial part of your submission has not been authored by you, even if everything has been cited appropriately. The extent of this will depend on the proportion of your submission which you have authored yourself, and the proportion which comes from other sources.

Remember to exercise caution if you use any code from external sources -- there are a lot of blogs and forums out there with very bad code!

With all that, we trust that you'll be able to use your best judgement, and to cite your sources appropriately -- if anything is not clear, please do ask. Note that **all submissions** will be automatically checked (and manually reviewed) for plagiarism and collusion, and [the University's academic misconduct policy](https://www.ed.ac.uk/academic-services/staff/discipline/academic-misconduct) applies.
